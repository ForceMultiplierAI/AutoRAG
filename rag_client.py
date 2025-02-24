#!/usr/bin/env python3
import aiohttp
import asyncio
import json
import sys
from pathlib import Path

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

async def send_request(session: aiohttp.ClientSession, messages: list) -> None:
    """Send request and stream response."""
    async with session.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "hello",
            "messages": messages,
            "stream": True,
            "temperature": 0.7
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
            except json.JSONDecodeError:
                continue
        print()  # Final newline

async def main():
    # Read the content from llm.txt
    context = await read_file("llm.txt")
    
    async with aiohttp.ClientSession() as session:
        # Initialize conversation with context
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. You answer questions from the following context\n\n\nContext:{context}\n\n\nEnd of Context window for QA assistant"}, #You are a helpful AI assistant. Use the following context to answer questions."},
            # {"role": "user", "content": f"{context}"}
        ]
        
        # Interactive Q&A loop
        while True:
            try:
                # Get user question
                print("\nEnter your question (or press Ctrl+C to exit):", end=" ")
                question = input()
                
                # Add question to messages
                messages.append({"role": "user", "content": question})
                
                # Send request and get response
                await send_request(session, messages)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                break

if __name__ == "__main__":
    asyncio.run(main())