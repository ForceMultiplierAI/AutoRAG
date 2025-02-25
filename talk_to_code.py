#!/usr/bin/env python3
import aiohttp
import asyncio
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import mimetypes

# File extensions to consider as code files
CODE_EXTENSIONS = {
    '.py', #'.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
    # '.cs', '.go', '.rs', '.rb', '.php', '.scala', '.kt', '.swift', '.m',
    # '.html', '.css', '.scss', '.less', '.json', '.yaml', '.yml', '.toml',
    # '.xml', '.md', '.sh', '.bash', '.zsh', '.ps1', '.bat', '.sql', '.r',
    # '.lua', '.pl', '.ex', '.exs', '.erl', '.fs', '.fsx', '.hs', '.dart',
    # '.groovy', '.jl', '.clj', '.nim', '.v', '.zig'
}

def is_code_file(file_path: Path) -> bool:
    """Check if the file is a code file based on extension."""
    return file_path.suffix.lower() in CODE_EXTENSIONS

def get_file_info(file_path: Path, base_dir: Path) -> dict:
    """Get file information including relative path and last modified time."""
    try:
        relative_path = file_path.relative_to(base_dir)
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        return {
            "path": str(relative_path),
            "last_modified": last_modified.strftime("%Y-%m-%d %H:%M:%S"),
            "content": file_path.read_text(encoding='utf-8', errors='replace')
        }
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None

def scan_directory(input_dir: Path, ignore_dirs=None, max_file_size_mb=5) -> list:
    """
    Scan directory for code files and return their contents.
    
    Args:
        input_dir: Directory to scan
        ignore_dirs: List of directory names to ignore
        max_file_size_mb: Maximum file size in MB to include
    
    Returns:
        List of dictionaries with file information
    """
    if ignore_dirs is None:
        ignore_dirs = ['.git', 'node_modules', 'venv', '.env', '__pycache__', 'build', 'dist']
    
    max_file_size = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
    files_info = []
    
    for root, dirs, files in os.walk(input_dir):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs and d != '__pycache__']
        
        for file in files:
            file_path = Path(root) / file
            
            # Skip files that are too large
            if file_path.stat().st_size > max_file_size:
                continue
                
            # Skip non-code files
            if not is_code_file(file_path):
                continue
                
            file_info = get_file_info(file_path, input_dir)
            if file_info:
                files_info.append(file_info)
    
    return files_info

def format_context(files_info: list) -> str:
    """Format file information into a structured context."""
    context = []
    
    for i, file_info in enumerate(files_info):
        file_header = f"File {i+1}, {file_info['path']} (Last modified: {file_info['last_modified']})"
        context.append(f"{file_header}\n=====\n{file_info['content']}\n=====\n")
    
    return "\n".join(context)

async def send_request(session: aiohttp.ClientSession, messages: list, model: str, temperature: float) -> None:
    """Send request and stream response."""
    async with session.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature
        }
    ) as response:
        if response.status != 200:
            error_text = await response.text()
            print(f"Error from server: {response.status} - {error_text}", file=sys.stderr)
            return
            
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
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing response: {e}", file=sys.stderr)
        print()  # Final newline

async def main():
    parser = argparse.ArgumentParser(description="Talk to a codebase using a local LLM API.")
    parser.add_argument("--input-dir", required=True, help="Directory containing code files to analyze")
    parser.add_argument("--ignore-dirs", nargs="+", default=None, help="Directories to ignore")
    parser.add_argument("--max-file-size", type=int, default=5, help="Maximum file size in MB (default: 5)")
    parser.add_argument("--model", default="hello", help="Model to use (default: hello)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for response generation (default: 0.7)")
    parser.add_argument("--system-prompt", type=str, default=None, 
                        help="Custom system prompt (default uses a standard prompt)")
    args = parser.parse_args()
    
    # Convert input directory to absolute path
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Directory '{args.input_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scanning directory: {input_dir}")
    files_info = scan_directory(input_dir, args.ignore_dirs, args.max_file_size)
    
    if not files_info:
        print("No code files found in the specified directory.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(files_info)} code files.")
    context = format_context(files_info)
    
    default_system_prompt = (
        "You are a helpful AI assistant that can answer questions about the provided codebase. "
        "Use the following context containing code files to answer questions. "
        "When referencing files or code, specify the file path. "
        "If you don't know the answer based on the provided context, say so."
    )
    
    system_prompt = args.system_prompt if args.system_prompt else default_system_prompt
    
    async with aiohttp.ClientSession() as session:
        # Initialize conversation with context
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}\n\nEnd of Context window for codebase assistant"}
        ]
        
        print("\nCodebase context loaded. You can now ask questions about the code.")
        print("-------------------------------------------------------------")
        
        # Interactive Q&A loop
        while True:
            try:
                # Get user question
                print("\nEnter your question (or press Ctrl+C to exit):", end=" ")
                question = input()
                
                if not question.strip():
                    continue
                
                # Add question to messages
                messages.append({"role": "user", "content": question})
                
                print("\nResponse:", end=" ")
                # Send request and get response
                await send_request(session, messages, args.model, args.temperature)
                
                # Store assistant's response (we don't have it directly here, so using a placeholder)
                # In a real implementation, you'd capture the response text and add it properly
                messages.append({"role": "assistant", "content": "[Response from assistant]"})
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                break

if __name__ == "__main__":
    asyncio.run(main())
