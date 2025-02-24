#!/usr/bin/env python3
import aiohttp
import asyncio
import json
import sys
from pathlib import Path

# Load dataset from JSONL file
async def load_dataset(filename: str) -> list:
    """Load dataset from a JSONL file."""
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

# Send request and stream response
async def send_request(session: aiohttp.ClientSession, messages: list) -> str:
    """Send request and stream response."""
    response_content = ""
    async with session.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "hello",
            "messages": messages,
            "stream": True,
            "temperature": 0.1,
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
                    response = json.loads(data)
                    if content := response.get("choices", [{}])[0].get("delta", {}).get("content"):
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        response_content += content
            except json.JSONDecodeError:
                continue
        print()  # Final newline
    return response_content

# Calculate score (e.g., accuracy or F1)
def calculate_score(predictions: list, ground_truths: list) -> float:
    """Calculate accuracy score."""
    correct = sum(1 for pred, truth in zip(predictions, ground_truths) if pred.strip().lower() == truth.strip().lower())
    return correct / len(ground_truths)



# {
#     "_id": "Unique identifier for each piece of data",
#     "domain": "The primary domain category of the data",
#     "sub_domain": "The specific sub-domain category within the domain",
#     "difficulty": "The difficulty level of the task, either 'easy' or 'hard'",
#     "length": "The length category of the task, which can be 'short', 'medium', or 'long'",
#     "question": "The input/command for the task, usually short, such as questions in QA, queries in many-shot learning, etc",
#     "choice_A": "Option A", "choice_B": "Option B", "choice_C": "Option C", "choice_D": "Option D",
#     "answer": "The groundtruth answer, denoted as A, B, C, or D",
#     "context": "The long context required for the task, such as documents, books, code repositories, etc."
# }


# Benchmark function
async def benchmark(session: aiohttp.ClientSession, dataset: list):
    """Run benchmark on the dataset."""
    predictions = []
    ground_truths = []

    for idx, entry in enumerate(dataset):
        try:
            print(f"\nQuestion {idx + 1}/{len(dataset)}:")
            print(f"Question: {entry['question']}")
            print(f"Ground Truth: {entry['answer']}")

            # Prepare messages
            messages = [
                {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer questions.\n\nContext:\n{entry['context']}\n\nEnd of Context."},
                {"role": "user", "content": f"Given the question {entry['question']}\n\n Answer the multiple choice question by giving letter, here are the options A {entry['choice_A']}, B {entry['choice_B']}, C {entry['choice_C']}, or D {entry['choice_D']}. But first, discuss each choice and think step by step. Use citations from the context to support your answer. Then return a final answer."}
            ]

            # display the options


            # Get model response
            print("Model Response:")

            response = await send_request(session, messages)
            print(f"Ground Truth: {entry['answer']}")
            predictions.append(response)
            ground_truths.append(entry['answer'])

        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user.")
            break
        except Exception as e:
            print(f"Error processing question {idx + 1}: {e}", file=sys.stderr)
            continue

    # Calculate and display score
    if predictions and ground_truths:
        score = calculate_score(predictions, ground_truths)
        print(f"\nBenchmark completed. Score: {score * 100:.2f}%")
    else:
        print("\nNo predictions were made.")

# Main function
async def main():
    # Load context from llm.txt
    # context = await read_file("llm.txt")

    # Load dataset from JSONL file
    dataset = await load_dataset("longbench/data.jsonl")

    # Run benchmark
    async with aiohttp.ClientSession() as session:
        await benchmark(session, dataset)

# Helper function to read file
async def read_file(filename: str) -> str:
    """Read content from the specified file."""
    try:
        content = Path(filename).read_text(encoding='utf-8')
        return content
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
