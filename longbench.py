#!/usr/bin/env python3
import aiohttp
import asyncio
import json
import sys
import re
import uuid
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

class BenchmarkLogger:
    def __init__(self, output_file=None):
        if output_file is None:
            unique_id = uuid.uuid4().hex[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output_{timestamp}_{unique_id}.jsonl"
        
        self.output_file = output_file
        self.file = open(output_file, 'w', encoding='utf-8')
        print(f"Logging results to: {self.output_file}")

    def log_result(self, result_dict: dict):
        json_line = json.dumps(result_dict, ensure_ascii=False)
        self.file.write(json_line + '\n')
        self.file.flush()
        
        print(f"\nResult saved:")
        print(f"Question ID: {result_dict.get('question_id')}")
        print(f"Extracted Answer: {result_dict.get('extracted_answer')}")
        print(f"Ground Truth: {result_dict.get('ground_truth')}")
        print(f"Correct: {result_dict.get('is_correct')}")
        print("-" * 50)

    def close(self):
        self.file.close()

async def load_dataset(filename: str) -> list:
    try:
        with open(filename, 'r') as file:
            data = [json.loads(line) for line in file]
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading dataset: {e}", file=sys.stderr)
        sys.exit(1)

async def extract_json_from_text(text: str) -> str:
    """Extract JSON response using regex."""
    try:
        # Look for JSON between triple backticks
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # If no triple backticks, look for any JSON object
        json_match = re.search(r'({[^{]*})', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
            
        return None
    except Exception:
        return None

async def send_streaming_request(session: aiohttp.ClientSession, messages: list, max_retries: int = 3) -> str:
    """Send streaming request and return the complete response content."""
    
    # messages.append({
    #     "role": "system",
    #     "content": "Return your final answer as a JSON object with 'answer' field containing just the letter A, B, C, or D, wrapped in triple backticks. Example: ```json{\"answer\": \"A\"}```"
    # })
    
    for attempt in range(max_retries):
        try:
            response_content = ""
            
            async with session.post(
                "http://localhost:8000/v1/chat/completions",
                json={
                    "model": "hello",
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            ) as response:
                async for line in response.content:
                    try:
                        line = line.decode()
                        if line.startswith("data: "):
                            data = line.removeprefix("data: ")
                            if data.strip() == "[DONE]":
                                break
                            
                            chunk_data = json.loads(data)
                            if content := chunk_data.get("choices", [{}])[0].get("delta", {}).get("content"):
                                sys.stdout.write(content)
                                sys.stdout.flush()
                                response_content += content
                                            
                    except json.JSONDecodeError:
                        continue
                
                print()  # Final newline
                return response_content
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error after {max_retries} attempts: {e}", file=sys.stderr)
                return ""
            print(f"Attempt {attempt + 1} failed, retrying...", file=sys.stderr)
            await asyncio.sleep(1)  # Wait before retry
    
    return ""

async def extract_answer(response_content: str) -> str:
    """Extract answer from response content and validate it."""
    if not response_content:
        return "UNCLEAR"
        
    # Try to extract JSON from response
    json_str = await extract_json_from_text(response_content)
    if not json_str:
        return "UNCLEAR"
        
    try:
        answer_data = json.loads(json_str)
        if "answer" in answer_data:
            answer = answer_data["answer"].strip().upper()
            if re.match(r'^[A-D]$', answer):
                return answer
    except json.JSONDecodeError:
        pass
        
    return "UNCLEAR"

def calculate_score(predictions: list, ground_truths: list) -> Tuple[float, dict]:
    if not predictions or not ground_truths:
        return 0.0, {}
    
    total = len(ground_truths)
    correct = 0
    unclear = 0
    incorrect = 0
    confusion_matrix = {'A': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'UNCLEAR': 0},
                       'B': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'UNCLEAR': 0},
                       'C': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'UNCLEAR': 0},
                       'D': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'UNCLEAR': 0}}
    
    for pred, truth in zip(predictions, ground_truths):
        pred = pred.strip().upper()
        truth = truth.strip().upper()
        
        if pred == "UNCLEAR":
            unclear += 1
        elif pred == truth:
            correct += 1
        else:
            incorrect += 1
            
        if truth in confusion_matrix and pred in confusion_matrix[truth]:
            confusion_matrix[truth][pred] += 1

    metrics = {
        'accuracy': correct / total,
        'unclear_rate': unclear / total,
        'incorrect_rate': incorrect / total,
        'confusion_matrix': confusion_matrix,
        'total_questions': total,
        'correct_answers': correct,
        'unclear_answers': unclear,
        'incorrect_answers': incorrect
    }
    
    return metrics['accuracy'], metrics

async def benchmark(session: aiohttp.ClientSession, dataset: list, logger: BenchmarkLogger):
    predictions = []
    ground_truths = []

    for idx, entry in enumerate(dataset):
        try:
            print(f"\nProcessing Question {idx + 1}/{len(dataset)}:")
            print(f"Question: {entry['question']}")
            print(f"Options:")
            print(f"A: {entry['choice_A']}")
            print(f"B: {entry['choice_B']}")
            print(f"C: {entry['choice_C']}")
            print(f"D: {entry['choice_D']}")
            
            # First stage: Reasoning and analysis prompt
            analysis_messages = [
                {"role": "system", "content": f"You are a helpful assistant. Your task is to analyze multiple choice questions based on context.\n\nContext:\n{entry['context']}\n\nEnd of Context."},
                {"role": "user", "content": f"Question: {entry['question']}\n\nOptions:\nA: {entry['choice_A']}\nB: {entry['choice_B']}\nC: {entry['choice_C']}\nD: {entry['choice_D']}\n\nPlease analyze each option step by step, using evidence from the context to support your reasoning. Then return you Final Answer."}
            ]

            print("\nAnalysis Stage:")
            analysis_response = await send_streaming_request(session, analysis_messages)
            
            # Second stage: Final answer selection prompt
            answer_format = '```json{"answer": "X"}```'
            answer_messages = [
                {"role": "system", "content": "Based on the previous analysis, provide your final answer as a single letter choice."},
                {"role": "user", "content": f"Previous analysis:\n{analysis_response}\n\nBased on this analysis, which option (A, B, C, or D) is correct? Respond with ONLY a JSON object containing your answer letter, using this format: {answer_format} where X is your chosen letter."}
            ]

            print("\nModel Response:")
            # Get the final answer from the second stage
            final_response = await send_streaming_request(session, answer_messages)
            extracted_answer = await extract_answer(final_response)
            
            # Combine both responses for logging
            detailed_response = f"ANALYSIS STAGE:\n{analysis_response}\n\nANSWER STAGE:\n{final_response}"
            
            print(f"\nExtracted Answer: {extracted_answer}")
            print(f"Ground Truth: {entry['answer']}")
            
            result = {
                'question_id': idx + 1,
                'is_correct': extracted_answer == entry['answer'],
                'timestamp': datetime.now().isoformat(),
                'question': entry['question'],
                # 'context': entry['context'],
                'choices': {
                    'A': entry['choice_A'],
                    'B': entry['choice_B'],
                    'C': entry['choice_C'],
                    'D': entry['choice_D']
                },
                'detailed_response': detailed_response,
                'extracted_answer': extracted_answer,
                'ground_truth': entry['answer'],
                'domain': entry.get('domain'),
                'sub_domain': entry.get('sub_domain'),
                'difficulty': entry.get('difficulty'),
                'length': entry.get('length')
            }
            
            logger.log_result(result)
            predictions.append(extracted_answer)
            ground_truths.append(entry['answer'])

        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user.")
            break
        except Exception as e:
            print(f"Error processing question {idx + 1}: {e}", file=sys.stderr)
            error_result = {
                'question_id': idx + 1,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'question': entry['question'],
                'ground_truth': entry['answer']
            }
            logger.log_result(error_result)
            continue

    if predictions and ground_truths:
        accuracy, metrics = calculate_score(predictions, ground_truths)
        final_metrics = {
            'timestamp': datetime.now().isoformat(),
            'type': 'final_metrics',
            'accuracy': accuracy,
            'total_questions': metrics['total_questions'],
            'correct_answers': metrics['correct_answers'],
            'unclear_answers': metrics['unclear_answers'],
            'incorrect_answers': metrics['incorrect_answers'],
            'confusion_matrix': metrics['confusion_matrix']
        }
        logger.log_result(final_metrics)
        
        print("\nFinal Benchmark Results:")
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        print(f"Total Questions: {metrics['total_questions']}")
        print(f"Correct Answers: {metrics['correct_answers']}")
        print(f"Unclear Answers: {metrics['unclear_answers']}")
        print(f"Incorrect Answers: {metrics['incorrect_answers']}")
    else:
        print("\nNo predictions were made.")

async def main():
    logger = BenchmarkLogger()
    
    try:
        dataset = await load_dataset("longbench/data.jsonl")
        async with aiohttp.ClientSession() as session:
            await benchmark(session, dataset, logger)
    finally:
        logger.close()

if __name__ == "__main__":
    asyncio.run(main())