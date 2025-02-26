#!/usr/bin/env python3
import re
import aiohttp
import asyncio
import json
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Configuration
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "hello"
TIMEOUT = 1800  # 30 minutes timeout
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

class DeepDiveGenerator:
    def __init__(self, context_file: str = "llm.txt"):
        self.context = self._read_file(context_file)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.debug = True  # Enable debug mode to get more verbose outputs
        
    def _read_file(self, filename: str) -> str:
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
    
    def _write_file(self, filename: str, content: str) -> None:
        """Write content to the specified file."""
        try:
            Path(filename).write_text(content, encoding='utf-8')
            print(f"Successfully wrote to {filename}")
        except Exception as e:
            print(f"Error writing to file: {e}", file=sys.stderr)
    
    async def send_request(self, session: aiohttp.ClientSession, instruction: str, 
                          collect_response: bool = True, include_context: bool = True) -> str:
        """Send request and stream response, optionally collecting the full response.
        Now with option to exclude context for JSON extraction requests."""
        full_response = ""
        
        # Create the messages array - only include context when needed
        if include_context:
            messages = [
                {"role": "system", "content": f"You are a helpful assistant. You answer questions from the following context\n\n\nContext:{self.context}\n\n\nEnd of Context window for QA assistant"},
                {"role": "user", "content": instruction}
            ]
        else:
            # For JSON extraction, we don't need to send the context
            messages = [
                {"role": "system", "content": "You are a helpful assistant that returns valid JSON."},
                {"role": "user", "content": instruction}
            ]
        
        if self.debug:
            print(f"\nSending request with instruction: {instruction[:100]}...")
            print(f"System message begins with: {messages[0]['content'][:50]}...")
            print(f"Context included: {include_context}")
        
        try:
            async with session.post(
                API_URL,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7,
                    "timeout": TIMEOUT
                },
                timeout=TIMEOUT
            ) as response:
                async for line in response.content:
                    try:
                        line = line.decode()
                        if line.startswith("data: "):
                            data = line.removeprefix("data: ")
                            if data.strip() == "[DONE]":
                                break
                            response_data = json.loads(data)
                            if content := response_data.get("choices", [{}])[0].get("delta", {}).get("content"):
                                sys.stdout.write(content)
                                sys.stdout.flush()
                                if collect_response:
                                    full_response += content
                    except json.JSONDecodeError:
                        continue
                print()  # Final newline
        except Exception as e:
            print(f"Error during API request: {e}", file=sys.stderr)
            
        return full_response
    
    async def generate_with_retry(self, session: aiohttp.ClientSession, instruction: str, include_context: bool = True) -> str:
        """Try to generate content with retries for robustness."""
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.send_request(session, instruction, include_context=include_context)
                return response
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print("Max retries reached. Moving on.")
                    return ""
    
    async def generate_toc(self, session: aiohttp.ClientSession) -> str:
        """Generate the Table of Contents from the context."""
        print("\n=== Generating Table of Contents ===\n")
        
        toc_instruction = """Create a comprehensive Table of Contents (TOC) for a deep dive guide based on the information in the context.
        The TOC should have main sections and subsections that cover all important topics in the content.
        Format the TOC properly with markdown headers and bullet points. Only include the TOC, no introduction or additional text."""
        
        toc_content = await self.generate_with_retry(session, toc_instruction, include_context=True)
        self._write_file(self.output_dir / "toc.md", toc_content)
        return toc_content
    
    async def extract_toc_structure(self, session: aiohttp.ClientSession, toc_content: str) -> List[Dict[str, Any]]:
        """Extract TOC structure as JSON."""
        print("\n=== Extracting TOC Structure ===\n")
        
        # First, check if the TOC content has system tags and clean them
        toc_content = self._clean_system_tags(toc_content)
        
        extraction_instruction = f"""Extract the structure from this Table of Contents and convert it to a JSON format:
        
        {toc_content}
        
        Use the following schema:
        ```json
        [
          {{
            "title": "Main Section Title",
            "level": 1,
            "subsections": [
              {{
                "title": "Subsection Title",
                "level": 2,
                "subsections": []
              }}
            ]
          }}
        ]
        ```
        
        Only return the valid JSON array, no additional text or explanation. Do not include any other text or system information. Return only the JSON array."""
        
        # IMPORTANT: Don't include context when extracting JSON
        json_str = await self.generate_with_retry(session, extraction_instruction, include_context=False)
        
        # Clean up JSON string to ensure it's valid
        json_str = self._clean_json_string(json_str)
        
        try:
            toc_structure = json.loads(json_str)
            self._write_file(self.output_dir / "toc_structure.json", json.dumps(toc_structure, indent=2))
            return toc_structure
        except json.JSONDecodeError as e:
            print(f"Error parsing TOC structure: {e}")
            print(f"Problematic JSON string: {json_str}")
            
            # Fallback: If we still can't parse the JSON, try a more direct approach
            print("Attempting fallback TOC structure generation...")
            fallback_structure = self._generate_fallback_toc_structure(toc_content)
            if fallback_structure:
                self._write_file(self.output_dir / "toc_structure.json", json.dumps(fallback_structure, indent=2))
                return fallback_structure
            
            # If all else fails, return a minimal structure
            return []
    
    def _clean_system_tags(self, text: str) -> str:
        """Remove system tags and compression info from text."""
        # More comprehensive cleaning of system tags
        if "<s>" in text:
            # Extract content between <s> and </s> tags
            s_tag_pattern = re.compile(r'<s>.*?</s>', re.DOTALL)
            text = s_tag_pattern.sub('', text)
            
            # In case there's an unclosed <s> tag
            if "<s>" in text:
                text = text.split("<s>", 1)[0]
        
        # Remove any remaining system-like messages about tokens
        token_pattern = re.compile(r'Initial token count:.*?available space', re.DOTALL)
        text = token_pattern.sub('', text)
        
        # Remove compression task messages
        compress_pattern = re.compile(r'Compression task ID:.*?successfully!', re.DOTALL)
        text = compress_pattern.sub('', text)
            
        return text.strip()
        
    def _clean_json_string(self, json_str: str) -> str:
        """Clean a JSON string to make it valid."""
        if self.debug:
            print(f"Original JSON string (first 100 chars): {json_str[:100]}...")
            
        # Remove system tags and compression info
        json_str = self._clean_system_tags(json_str)
            
        # Remove markdown code blocks
        if "```json" in json_str:
            json_str = json_str.split("```json", 1)[1]
        if "```" in json_str:
            json_str = json_str.split("```", 1)[0]
        
        # Trim whitespace
        json_str = json_str.strip()
        
        # Search for any JSON-like content with more aggressive pattern matching
        if not json_str or not (json_str.startswith('[') or json_str.startswith('{')):
            # Look for JSON array or object pattern
            json_pattern = re.compile(r'(\[|\{).*?(\]|\})', re.DOTALL)
            match = json_pattern.search(json_str)
            if match:
                json_str = match.group(0)
        
        if self.debug:
            print(f"Cleaned JSON string (first 100 chars): {json_str[:100]}...")
            
        return json_str
        
    def _generate_fallback_toc_structure(self, toc_content: str) -> List[Dict[str, Any]]:
        """Generate a fallback TOC structure from markdown content."""
        if self.debug:
            print("Generating fallback TOC structure from markdown...")
            
        # Clean the toc_content first
        toc_content = self._clean_system_tags(toc_content)
        
        # Simple regex-based parser for markdown headers
        structure = []
        current_level = {1: None, 2: None, 3: None}
        current_parents = {1: structure, 2: None, 3: None}
        
        # Match markdown headers (# Header, ## Header, etc.)
        header_pattern = re.compile(r'^(#{1,3})\s+(.+)', re.MULTILINE)
        matches = header_pattern.findall(toc_content)
        for marker, title in matches:
            level = len(marker)
            section = {"title": title.strip(), "level": level, "subsections": []}
            
            # Add to appropriate parent
            if level == 1:
                structure.append(section)
                current_level[1] = section
                current_parents[2] = section["subsections"]
            elif level == 2 and current_parents[2] is not None:
                current_parents[2].append(section)
                current_level[2] = section
                current_parents[3] = section["subsections"]
            elif level == 3 and current_parents[3] is not None:
                current_parents[3].append(section)
                current_level[3] = section
        return structure
    
    
    async def generate_questions(self, session: aiohttp.ClientSession, section: Dict[str, Any]) -> List[str]:
        """Generate deep-dive questions for a section."""
        title = section["title"]
        print(f"\n=== Generating Questions for: {title} ===\n")
        
        questions_instruction = f"""Only write thought-provoking, in-depth questions for this subject: "{title}"
        
        Create 3-5 deep dive questions that:
        1. Explore complex aspects of the topic
        2. Require comprehensive explanations
        3. Cover different perspectives or approaches
        4. Would lead to insightful tutorial content
        
        Format the questions as a numbered list. Only provide the questions, no introductions or explanations."""
        
        questions_content = await self.generate_with_retry(session, questions_instruction, include_context=True)
        return questions_content
    
    async def extract_questions(self, session: aiohttp.ClientSession, questions_content: str, section_title: str) -> List[str]:
        """Extract questions as a JSON list."""
        print(f"\n=== Extracting Questions for: {section_title} ===\n")
        
        # First clean the questions content
        questions_content = self._clean_system_tags(questions_content)
        
        extraction_instruction = f"""Extract the questions from this content and convert them to a JSON array:
        
        {questions_content}
        
        Only return a valid JSON array of strings, with each string being a single question.
        Example:
        ```json
        ["Question 1?", "Question 2?", "Question 3?"]
        ```
        
        Only return the valid JSON array, no additional text or explanation."""
        
        # IMPORTANT: Don't include context when extracting JSON
        json_str = await self.generate_with_retry(session, extraction_instruction, include_context=False)
        json_str = self._clean_json_string(json_str)
        
        try:
            questions = json.loads(json_str)
            return questions
        except json.JSONDecodeError:
            print(f"Error parsing questions JSON. Using fallback extraction method.")
            # Fallback: simple line-by-line extraction
            lines = questions_content.strip().split('\n')
            questions = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Extract the question content, ignoring numbering
                if '. ' in line and line[0].isdigit():
                    question = line.split('. ', 1)[1]
                else:
                    question = line
                questions.append(question)
            return questions
    
    async def generate_article(self, session: aiohttp.ClientSession, question: str, section_title: str) -> str:
        """Generate extremely direct technical content answering a question."""
        print(f"\n=== Generating Technical Answer for: {question} ===\n")
        
        article_instruction = f"""Answer this question with extreme directness: "{question}"
        
        Rules:
        1. NO INTRODUCTION - start immediately with specific technical details
        2. Use confident, authoritative language with zero hedging
        3. Provide exact values, formulas, and code examples where relevant
        4. Structure with headers and bullet points for scannable content
        5. No fluff words, generalizations, or unnecessary context
        6. Skip all conceptual explanations unless absolutely critical
        
        Write as if explaining to an expert who needs answers immediately. Be extremely precise and specific."""
        
        article_content = await self.generate_with_retry(session, article_instruction, include_context=True)
        return article_content
    
    async def extract_article(self, session: aiohttp.ClientSession, article_content: str, question: str) -> Dict[str, str]:
        """Extract article metadata and content."""
        print(f"\n=== Extracting Article Data for: {question} ===\n")
        
        # First clean the article content
        article_content = self._clean_system_tags(article_content)
        
        extraction_instruction = f"""Extract the following information from this article and return it as JSON:
        
        {article_content}
        
        Use this schema:
        ```json
        {{
          "title": "The main title of the article",
          "summary": "A brief 1-2 sentence summary of the content",
          "content": "The full article content in markdown"
        }}
        ```
        
        Only return the valid JSON object, no additional text or explanation."""
        
        # IMPORTANT: Don't include context when extracting JSON
        json_str = await self.generate_with_retry(session, extraction_instruction, include_context=False)
        json_str = self._clean_json_string(json_str)
        
        try:
            article_data = json.loads(json_str)
            return article_data
        except json.JSONDecodeError:
            print(f"Error parsing article JSON. Using fallback extraction method.")
            # Fallback: return original content with minimal structure
            # Try to extract a title from the first lines of the article
            lines = article_content.strip().split('\n')
            title = question
            for line in lines[:5]:  # Look at first 5 lines for a title
                if line.startswith('#'):
                    title = line.lstrip('#').strip()
                    break
            
            return {
                "title": title,
                "summary": f"This article addresses the question: {question}",
                "content": article_content
            }
    
    async def process_section(self, session: aiohttp.ClientSession, section: Dict[str, Any], level: int = 1) -> Dict[str, Any]:
        """Process a section, including generating questions and articles."""
        title = section["title"]
        
        # Generate questions for this section
        questions_content = await self.generate_questions(session, section)
        questions = await self.extract_questions(session, questions_content, title)
        
        # Process each question
        section_articles = []
        for question in questions:
            article_content = await self.generate_article(session, question, title)
            article_data = await self.extract_article(session, article_content, question)
            
            section_articles.append({
                "question": question,
                "article": article_data
            })
        
        # Create the section data
        section_data = {
            "title": title,
            "level": level,
            "questions": questions,
            "articles": section_articles,
            "subsections": []
        }
        
        # Process subsections
        for subsection in section.get("subsections", []):
            subsection_data = await self.process_section(session, subsection, level + 1)
            section_data["subsections"].append(subsection_data)
        
        return section_data
    
    def generate_markdown(self, sections_data: List[Dict[str, Any]]) -> str:
        """Generate the final markdown file from processed data."""
        markdown = "# Deep Dive Guide\n\n"
        
        # Table of Contents
        markdown += "## Table of Contents\n\n"
        markdown += self._generate_toc_markdown(sections_data)
        markdown += "\n\n"
        
        # Content
        markdown += self._generate_content_markdown(sections_data)
        
        return markdown
    
    def _generate_toc_markdown(self, sections: List[Dict[str, Any]], indent: int = 0) -> str:
        """Generate markdown for the table of contents."""
        toc = ""
        for section in sections:
            title = section["title"]
            toc += f"{'  ' * indent}* [{title}](#{self._create_anchor(title)})\n"
            
            # Add questions as sub-items
            for article in section.get("articles", []):
                question = article["question"]
                toc += f"{'  ' * (indent+1)}* [{question}](#{self._create_anchor(question)})\n"
            
            # Process subsections
            if section.get("subsections"):
                toc += self._generate_toc_markdown(section["subsections"], indent + 1)
        
        return toc
    
    def _generate_content_markdown(self, sections: List[Dict[str, Any]], level: int = 1) -> str:
        """Generate markdown content from sections data."""
        content = ""
        for section in sections:
            title = section["title"]
            section_level = "#" * (level + 1)  # +1 because main title is level 1
            
            content += f"{section_level} {title}\n\n"
            
            # Add articles for this section
            for article_data in section.get("articles", []):
                question = article_data["question"]
                article = article_data["article"]
                
                # Article title (using the question)
                content += f"{'#' * (level + 2)} {question}\n\n"
                
                # Article content
                if "content" in article:
                    content += f"{article['content']}\n\n"
                else:
                    content += "Content not available.\n\n"
            
            # Process subsections
            if section.get("subsections"):
                content += self._generate_content_markdown(section["subsections"], level + 1)
        
        return content
    
    def _create_anchor(self, text: str) -> str:
        """Create an anchor from text for internal links."""
        anchor = text.lower()
        anchor = anchor.replace("?", "")
        anchor = anchor.replace(" ", "-")
        anchor = "".join(c for c in anchor if c.isalnum() or c == "-")
        return anchor
    
    async def run(self):
        """Run the full deep dive generation process."""
        async with aiohttp.ClientSession() as session:
            # 1. Generate Table of Contents
            toc_content = await self.generate_toc(session)
            
            # 2. Extract TOC structure
            toc_structure = await self.extract_toc_structure(session, toc_content)
            
            # 3. Process each section and generate content
            all_sections_data = []
            for section in toc_structure:
                section_data = await self.process_section(session, section)
                all_sections_data.append(section_data)
            
            # 4. Generate final markdown
            final_markdown = self.generate_markdown(all_sections_data)
            
            # 5. Save to file
            self._write_file(self.output_dir / "deepdive.md", final_markdown)
            
            # 6. Save all data for potential reuse
            self._write_file(
                self.output_dir / "deepdive_data.json", 
                json.dumps(all_sections_data, indent=2)
            )
            
            print("\n=== Deep Dive Generation Complete ===\n")
            print(f"Results saved to {self.output_dir}")

async def main():
    # Allow specifying a different context file via command line
    context_file = "llm.txt"
    if len(sys.argv) > 1:
        context_file = sys.argv[1]
    
    generator = DeepDiveGenerator(context_file)
    await generator.run()

if __name__ == "__main__":
    asyncio.run(main())