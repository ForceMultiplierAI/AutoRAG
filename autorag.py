#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys
import re
import hashlib
from typing import AsyncGenerator, Any, Dict, Tuple
from collections import defaultdict

import aiohttp
from aiohttp import web
from llmlingua import PromptCompressor
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

class Config:
    target_url: str = "http://localhost:6002/v1"
    listen_port: int = 8000
    context_limit: int = 32  # Default 32k context window
    compression_buffer: int = 1024  # Reserved tokens for response

config = Config()

def get_target_token_limit() -> int:
    """Calculate target token limit based on context limit."""
    max_tokens = config.context_limit * 1024  # Convert k to actual tokens
    target_tokens = max_tokens - config.compression_buffer
    return int(target_tokens * 0.90)  # 95% of available space to be safe

# Simple cache structure
compression_cache = {}

def calculate_hash(text: str) -> str:
    """Calculate SHA-256 hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]  # Using first 8 chars for brevity

def clean_tags(text: str) -> str:
    """Remove system and thinking tags from text."""
    # Repeatedly remove tags until no more matches are found
    prev_text = None
    while prev_text != text:
        prev_text = text
        text = re.sub(r'<system>(?:[^<]|<(?!/?system))*</system>\n?', '', text, flags=re.DOTALL)
        text = re.sub(r'<thinking>(?:[^<]|<(?!/?thinking))*</thinking>\n?', '', text, flags=re.DOTALL)
    return text

def format_sse_event(content: str) -> bytes:
    """Format content as an SSE event with the expected JSON structure."""
    event_data = {
        "choices": [
            {
                "delta": {"content": content},
                "index": 0
            }
        ]
    }
    return f"data: {json.dumps(event_data)}\n\n".encode()

import torch

# Check if CUDA

# # Check if CUDA is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# Initialize compressor with error handling
try:
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True
    )
    # Test compression with a small sample
    test_result = llm_lingua.compress_prompt("Test compression with a small sample text", rate=0.5)
    print("LLMlingua initialization successful")
except Exception as e:
    print(f"Error initializing LLMlingua: {e}")
    sys.exit(1)


def calculate_tokens(text: str) -> int:
    """Calculate exact token count using tokenizer."""
    return len(tokenizer.encode(text))

async def compress_text(text: str, target_rate: float = 0.75) -> Dict[str, Any]:
    """Compress text using LLMlingua-2."""
    try:
        result = llm_lingua.compress_prompt(
            text,
            rate=target_rate,
            force_tokens=['\n']  # Preserve newlines
        )
        
        return {
            'compressed_text': result['compressed_prompt'],
            'compressed_prompt': result['compressed_prompt'],
            'original_tokens': result['origin_tokens'],
            'compressed_tokens': result['compressed_tokens']
        }
        
    except Exception as e:
        print(f"Fatal compression error: {str(e)}")
        token_limit = get_target_token_limit()
        tokenized = tokenizer.encode(text)
        truncated_tokens = tokenized[:token_limit - 1000]
        return {
            'compressed_text': tokenizer.decode(truncated_tokens),
            'compressed_prompt': tokenizer.decode(truncated_tokens),
            'error': str(e)
        }


async def stream_compression_progress(session: aiohttp.ClientSession, prompt: str, target_tokens: int = None) -> AsyncGenerator[Tuple[bytes, Dict[str, Any]], None]:
    """Stream compression progress and return final result."""
    try:
        yield format_sse_event("<system>\n"), None
        
        # Initial setup
        cleaned_prompt = clean_tags(prompt)
        prompt_hash = calculate_hash(cleaned_prompt)
        
        # Calculate initial token count and required compression
        initial_tokens = calculate_tokens(cleaned_prompt)
        token_limit = get_target_token_limit()
        percentage = (initial_tokens / token_limit) * 100
        
        yield format_sse_event(f"Initial token count: {initial_tokens:,} tokens\n"), None
        yield format_sse_event(f"Token limit: {token_limit:,} tokens\n"), None
        yield format_sse_event(f"Currently using: {percentage:.1f}% of available space\n"), None
        
        result = None
        # Check cache first
        if prompt_hash in compression_cache:
            cached_result = compression_cache[prompt_hash]
            cached_tokens = calculate_tokens(cached_result['compressed_prompt'])
            
            # Only use cache if it's under the token limit
            if cached_tokens <= token_limit:
                yield format_sse_event(f"Using cached version (hash: {prompt_hash})\n"), cached_result
                result = cached_result
            else:
                yield format_sse_event(f"Cached version exceeds token limit, recompressing...\n"), None
        
        # If no cache hit or cached version too large, compress
        if result is None:
            if initial_tokens > token_limit:
                # Add 10% buffer to ensure we get under the limit
                target_size = token_limit * 0.9
                required_rate = target_size / initial_tokens
                yield format_sse_event(f"Token count exceeds limit. Calculating required compression...\n"), None
                yield format_sse_event(f"Required compression rate: {required_rate:.3f}\n"), None
                
                try:
                    result = await compress_text(cleaned_prompt, target_rate=required_rate)
                    result['compressed_prompt'] = result.get('compressed_text', cleaned_prompt)
                    
                    compressed_tokens = calculate_tokens(result['compressed_prompt'])
                    reduction = ((initial_tokens - compressed_tokens) / initial_tokens) * 100
                    compressed_percentage = (compressed_tokens / token_limit) * 100
                    
                    yield format_sse_event(f"Compressed to: {compressed_tokens:,} tokens (reduced by {reduction:.1f}%)\n"), None
                    yield format_sse_event(f"Now using: {compressed_percentage:.1f}% of available space\n"), None
                    
                    # If still over limit, try more aggressive compression
                    if compressed_tokens > token_limit:
                        yield format_sse_event("Still over limit, trying more aggressive compression...\n"), None
                        new_rate = required_rate * 0.8  # 20% more aggressive
                        result = await compress_text(cleaned_prompt, target_rate=new_rate)
                        result['compressed_prompt'] = result.get('compressed_text', cleaned_prompt)
                        
                        compressed_tokens = calculate_tokens(result['compressed_prompt'])
                        reduction = ((initial_tokens - compressed_tokens) / initial_tokens) * 100
                        compressed_percentage = (compressed_tokens / token_limit) * 100
                        
                        yield format_sse_event(f"Compressed to: {compressed_tokens:,} tokens (reduced by {reduction:.1f}%)\n"), None
                        yield format_sse_event(f"Now using: {compressed_percentage:.1f}% of available space\n"), None
                
                except Exception as e:
                    yield format_sse_event(f"Warning: Compression attempt failed: {str(e)}\n"), None
                    result = {
                        'compressed_prompt': cleaned_prompt,
                        'compressed_text': cleaned_prompt
                    }
            else:
                # No compression needed
                result = {
                    'compressed_prompt': cleaned_prompt,
                    'compressed_text': cleaned_prompt
                }
                yield format_sse_event("No compression needed - under token limit\n"), None
            
            # Only cache if under token limit
            if calculate_tokens(result['compressed_prompt']) <= token_limit:
                compression_cache[prompt_hash] = result
        
        final_tokens = calculate_tokens(result['compressed_prompt'])
        yield format_sse_event(f"Final token count: {final_tokens:,}\n"), result
        yield format_sse_event("</system>\n"), None
        
    except Exception as e:
        error_msg = f"Compression error: {e}\n"
        yield format_sse_event(error_msg), {
            "compressed_prompt": cleaned_prompt,
            "compressed_text": cleaned_prompt,
            "hash": prompt_hash,
            "cached": False
        }
        yield format_sse_event("</system>\n"), None

async def stream_thinking_content(session: aiohttp.ClientSession) -> AsyncGenerator[bytes, None]:
    """Stream response from LLM for thinking content."""
    async with session.post(
        f"{config.target_url}/chat/completions",
        json={
            "model": "hello",
            "messages": [{"role": "user", "content": "Say hello world"}],
            "stream": True
        },
        headers={"Content-Type": "application/json"}
    ) as response:
        async for chunk in response.content:
            if chunk:
                chunk_str = chunk.decode()
                if chunk_str.startswith('data: '):
                    try:
                        if chunk_str.strip() == 'data: [DONE]':
                            continue
                        data = json.loads(chunk_str.removeprefix('data: '))
                        if choices := data.get('choices'):
                            if content := choices[0].get('delta', {}).get('content'):
                                yield content.encode()
                    except json.JSONDecodeError:
                        continue

async def proxy_request(request: web.Request) -> web.StreamResponse:
    """Forward request and stream back the response with system message compression."""
    try:
        body = await request.json()
        
        # Setup response
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )
        await response.prepare(request)

        # Use a single session for all requests
        async with aiohttp.ClientSession() as session:
            # Process and stream system message if present
            if messages := body.get('messages'):
                for message in messages:
                    if message.get('role') == 'system' and (content := message.get('content')):
                        target_tokens = get_target_token_limit()
                        compression_result = None
                        
                        async for progress, result in stream_compression_progress(session, content, target_tokens):
                            await response.write(progress)
                            if result is not None:
                                compression_result = result
                        
                        # Update message content with compressed version if we got a result
                        if compression_result and 'compressed_prompt' in compression_result:
                            message['content'] = compression_result['compressed_prompt']
                            assert calculate_tokens(message['content']) <= target_tokens, f"FATAL: Attempted to send {calculate_tokens(message['content']):,} tokens which exceeds limit of {target_tokens:,} tokens. This should never happen - compression failed to reduce tokens sufficiently."
                        else:
                            print("Warning: No compressed content available")
                        break


            thinking = False
            if thinking:
                # Send opening thinking tag
                await response.write(b'data: {"choices":[{"delta":{"content":"<thinking>\\n"},"index":0}]}\n\n')

                # Stream thinking content using the same session
                async for content in stream_thinking_content(session):
                    await response.write(
                        f'data: {{"choices":[{{"delta":{{"content":"{content.decode()}"}}, "index":0}}]}}\n\n'.encode()
                    )

                # Send closing thinking tag
                await response.write(b'data: {"choices":[{"delta":{"content":"\\n</thinking>\\n"},"index":0}]}\n\n')



            # Forward request to target using the same session
            headers = {k: v for k, v in request.headers.items() 
                      if k.lower() not in {'host', 'content-length'}}
                      
            async with session.post(
                f"{config.target_url}/chat/completions",
                json=body,
                headers=headers
            ) as api_response:
                async for chunk in api_response.content:
                    await response.write(chunk)
                await response.write(b'\n')

        await response.write(b'data: [DONE]\n\n')
        await response.write_eof()
        return response

    except Exception as e:
        print(f"Error in proxy_request: {e}", file=sys.stderr)
        return web.Response(
            status=500,
            text=json.dumps({"error": str(e)}),
            content_type='application/json'
        )
def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post('/v1/chat/completions', proxy_request)
    return app

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI API Streaming Proxy with Cached System Message Compression")
    parser.add_argument("--url", default=config.target_url, 
                      help="Target URL (default: %(default)s)")
    parser.add_argument("--listen", type=int, default=config.listen_port,
                      help="Listen port (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=config.context_limit,
                      help="Context window limit in thousands of tokens (default: %(default)sk)")
    parser.add_argument("--buffer", type=int, default=config.compression_buffer,
                      help="Token buffer reserved for response (default: %(default)s)")
    parser.add_argument("--thinking", action='store_true', default=False,
                        help="Enable thinking content streaming (default: %(default)s)")
    args = parser.parse_args()

    config.target_url = args.url.rstrip("/")
    config.listen_port = args.listen
    config.context_limit = args.limit
    config.compression_buffer = args.buffer
    web.run_app(create_app(), port=config.listen_port)

if __name__ == "__main__":
    main()