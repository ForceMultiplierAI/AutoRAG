#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys
import time
from typing import AsyncGenerator, Dict, Tuple, Any

import aiohttp
from aiohttp import web
from compressor import Compressor
from ragqueue import CompressionQueue

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
    return int(target_tokens * 0.90)  # 90% of available space to be safe

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

# Initialize global compressor and queue
compressor = None
compression_queue = None

async def stream_compression_progress(session: aiohttp.ClientSession, prompt: str, target_tokens: int = None) -> AsyncGenerator[Tuple[bytes, Dict[str, Any]], None]:
    """Stream compression progress and return final result."""
    # Initialize cleaned_prompt early to avoid potential UnboundLocalError
    cleaned_prompt = compressor.clean_tags(prompt)
    prompt_hash = compressor.calculate_hash(cleaned_prompt)
    
    try:
        yield format_sse_event("<s>\n"), None
        
        # Calculate initial token count and required compression
        initial_tokens = compressor.calculate_tokens(cleaned_prompt)
        token_limit = get_target_token_limit()
        percentage = (initial_tokens / token_limit) * 100
        
        yield format_sse_event(f"Initial token count: {initial_tokens:,} tokens\n"), None
        yield format_sse_event(f"Token limit: {token_limit:,} tokens\n"), None
        yield format_sse_event(f"Currently using: {percentage:.1f}% of available space\n"), None
        
        result = None
        # Check cache first
        try:
            if prompt_hash in compressor.cache:
                cached_result = compressor.cache[prompt_hash]
                cached_tokens = compressor.calculate_tokens(cached_result['compressed_prompt'])
                
                # Only use cache if it's under the token limit
                if cached_tokens <= token_limit:
                    yield format_sse_event(f"Using cached version (hash: {prompt_hash})\n"), cached_result
                    result = cached_result
                else:
                    yield format_sse_event(f"Cached version exceeds token limit, recompressing...\n"), None
        except KeyError:
            pass
        
        # If no cache hit or cached version too large, compress using the queue
        if result is None:
            if initial_tokens > token_limit:
                # Check if another compression is already in progress
                if compression_queue.processing:
                    yield format_sse_event("Another compression is in progress. Adding to queue...\n"), None
                else:
                    yield format_sse_event("No other tasks in progress. Starting compression immediately...\n"), None
                    
                # Add task to the queue
                target_size = token_limit * 0.9
                required_rate = target_size / initial_tokens
                
                yield format_sse_event(f"Target compression rate: {required_rate:.3f}\n"), None
                
                # Add to queue and get task ID
                task_id = await compression_queue.add_task(
                    prompt=cleaned_prompt,
                    target_tokens=token_limit
                )
                
                yield format_sse_event(f"Compression task ID: {task_id[:8]}\n"), None
                
                # Poll for task completion
                task = None
                attempt = 0
                max_attempts = 600  # 10 minutes max (600 seconds)
                last_status = None
                
                while attempt < max_attempts:
                    task = await compression_queue.get_task(task_id)
                    if not task:
                        await asyncio.sleep(1)
                        attempt += 1
                        continue
                        
                    current_status = task.status
                    
                    # Only send status updates when status changes or every 5 seconds
                    status_changed = current_status != last_status
                    time_interval = attempt % 5 == 0
                    
                    if status_changed or time_interval:
                        if current_status == "completed":
                            yield format_sse_event(f"Compression completed successfully!\n"), None
                            result = task.result
                            break
                        elif current_status == "failed":
                            yield format_sse_event(f"Compression failed: {task.error}\n"), None
                            # Fallback to truncation as last resort
                            result = {
                                'compressed_prompt': cleaned_prompt[:token_limit],
                                'compressed_text': cleaned_prompt[:token_limit]
                            }
                            break
                        elif current_status == "in_progress":
                            yield format_sse_event(f"Compressing text... (elapsed: {attempt}s)\n"), None
                        else:  # pending
                            yield format_sse_event(f"Waiting in queue... (elapsed: {attempt}s)\n"), None
                    
                    last_status = current_status
                    await asyncio.sleep(1)
                    attempt += 1
                
                if attempt >= max_attempts:
                    yield format_sse_event("Compression timed out after 10 minutes\n"), None
                    # Fallback to truncation as last resort
                    result = {
                        'compressed_prompt': cleaned_prompt[:token_limit],
                        'compressed_text': cleaned_prompt[:token_limit]
                    }
                
                if task and task.status == "completed" and result:
                    compressed_tokens = compressor.calculate_tokens(result['compressed_prompt'])
                    reduction = ((initial_tokens - compressed_tokens) / initial_tokens) * 100
                    compressed_percentage = (compressed_tokens / token_limit) * 100
                    
                    if '[...SPLIT...]' in result['compressed_prompt']:
                        yield format_sse_event("Content was split and compressed in parts\n"), None
                    
                    yield format_sse_event(f"Compressed to: {compressed_tokens:,} tokens (reduced by {reduction:.1f}%)\n"), None
                    yield format_sse_event(f"Now using: {compressed_percentage:.1f}% of available space\n"), None
            else:
                # No compression needed
                result = {
                    'compressed_prompt': cleaned_prompt,
                    'compressed_text': cleaned_prompt
                }
                yield format_sse_event("No compression needed - under token limit\n"), None
            
            # Only cache if under token limit and we have a result
            if result and compressor.calculate_tokens(result['compressed_prompt']) <= token_limit:
                compressor.cache[prompt_hash] = result
        
        final_tokens = compressor.calculate_tokens(result['compressed_prompt'])
        yield format_sse_event(f"Final token count: {final_tokens:,}\n"), result
        yield format_sse_event("</s>\n"), None
        
    except Exception as e:
        error_msg = f"Compression error: {e}\n"
        yield format_sse_event(error_msg), {
            "compressed_prompt": cleaned_prompt,
            "compressed_text": cleaned_prompt,
            "hash": prompt_hash,
            "cached": False
        }
        yield format_sse_event("</s>\n"), None

async def stream_thinking_content(session: aiohttp.ClientSession, compressed_context:str = "", prompt: str = "") -> AsyncGenerator[bytes, None]:
    """Stream response from LLM for thinking content."""
    if compressed_context != "" and prompt != "":
        # Use compressed context and prompt
        async with session.post(
            f"{config.target_url}/chat/completions",
            json={
                "model": "hello",
                "messages": [
                    {"role": "system", "content": f"{compressed_context}"},
                    {"role": "user", "content": f"{prompt}"}
                ],
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
    else:
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
        # Periodic cache cleanup (every hour)
        if time.time() - compressor.cache.last_cleanup > 3600:
            removed = compressor.cache.cleanup()
            print(f"Cache cleanup: removed {removed} expired entries")
            print(f"Cache stats: {compressor.cache.stats}")
            
        # Also periodically clean up old compression tasks
        if compression_queue:
            await compression_queue.cleanup_old_tasks()
            
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
            compression_result = None

            # Process and stream system message if present
            if messages := body.get('messages'):
                for i, message in enumerate(messages):
                    if message.get('role') == 'system' and (content := message.get('content')):
                        target_tokens = get_target_token_limit()
                        
                        async for progress, result in stream_compression_progress(session, content, target_tokens):
                            await response.write(progress)
                            if result is not None:
                                compression_result = result
                        
                        # Update message content with compressed version if we got a result
                        if compression_result and 'compressed_prompt' in compression_result:
                            message['content'] = compression_result['compressed_prompt']
                        else:
                            print("Warning: No compressed content available")
                            await response.write(format_sse_event("Warning: Using truncated content\n"))
                        break

            # Enable thinking mode if requested
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
        print(f"Proxy error: {e}")
        return web.Response(status=500, text=str(e))

async def init_app():
    global compressor, compression_queue
    
    # Initialize the compressor instance
    compressor = Compressor()
    
    # Initialize compression queue with reference to compressor
    compression_queue = CompressionQueue(compressor)
    await compression_queue.start_worker()
    
    # Set up the web application
    app = web.Application(client_max_size=100*1024*1024)
    
    # Register routes
    app.router.add_post('/v1/chat/completions', proxy_request)
    app.router.add_get('/', lambda request: web.Response(text="AutoRAG Proxy Server is running"))
    
    return app

def parse_args():
    parser = argparse.ArgumentParser(description="AutoRAG Proxy Server")
    parser.add_argument("--url", help="Target LLM API URL", default="http://localhost:6002/v1")
    parser.add_argument("--listen", type=int, help="Listen port", default=8000)
    parser.add_argument("--limit", type=int, help="Context limit in K tokens", default=32)
    parser.add_argument("--buffer", type=int, help="Reserved tokens for response", default=1024)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Update config with command line arguments
    config.target_url = args.url
    config.listen_port = args.listen
    config.context_limit = args.limit
    config.compression_buffer = args.buffer
    
    print(f"Initializing AutoRAG Proxy with:")
    print(f"  Target URL: {config.target_url}")
    print(f"  Listen port: {config.listen_port}")
    print(f"  Context limit: {config.context_limit}K tokens")
    print(f"  Response buffer: {config.compression_buffer} tokens")
    
    # Start the web application
    web.run_app(init_app(), port=config.listen_port)

if __name__ == "__main__":
    main()