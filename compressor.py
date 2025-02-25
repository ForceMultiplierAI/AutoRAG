#!/usr/bin/env python3
import hashlib
import re
import sys
import time
from typing import Dict, Any, Optional, List

from llmlingua import PromptCompressor
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

# LRU Cache for compression results
class LRUCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self.last_cleanup = time.time()
        
    def __setitem__(self, key, value):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.stats['evictions'] += 1
            
        self.cache[key] = {
            'data': value,
            'timestamp': time.time()
        }
            
    def __getitem__(self, key):
        if key not in self.cache:
            self.stats['misses'] += 1
            raise KeyError(key)
            
        # Access item
        item = self.cache[key]
        self.stats['hits'] += 1
        return item['data']
        
    def __contains__(self, key):
        return key in self.cache
        
    def __len__(self):
        return len(self.cache)
        
    def cleanup(self, max_age=3600):
        """Remove entries older than max_age seconds"""
        now = time.time()
        expired = [
            k for k, v in list(self.cache.items())
            if now - v['timestamp'] > max_age
        ]
        for k in expired:
            self.cache.pop(k)
        self.stats['evictions'] += len(expired)
        self.last_cleanup = now
        return len(expired)

class Compressor:
    def __init__(self):
        self.cache = LRUCache(max_size=1000)
        
        # Initialize compressor with error handling
        try:
            self.llm_lingua = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True
            )
            # Test compression with a small sample
            test_result = self.llm_lingua.compress_prompt("Test compression with a small sample text", rate=0.5)
            print("LLMlingua initialization successful")
        except Exception as e:
            print(f"Error initializing LLMlingua: {e}")
            sys.exit(1)
    
    def split_content(self, text: str) -> List[str]:
        """Split content into roughly equal parts, preserving structure."""
        # Try paragraphs first
        parts = text.split('\n\n')
        if len(parts) < 2:
            # Fall back to sentences if no paragraphs
            parts = text.split('. ')
            if len(parts) < 2:
                # Last resort: split by rough character count
                mid = len(text) // 2
                return [text[:mid], text[mid:]]
        
        # Combine into roughly equal halves
        mid = len(parts) // 2
        return [
            '\n\n'.join(parts[:mid]) if '\n\n' in text else '. '.join(parts[:mid]),
            '\n\n'.join(parts[mid:]) if '\n\n' in text else '. '.join(parts[mid:])
        ]
    
    def calculate_hash(self, text: str) -> str:
        """Calculate SHA-256 hash of text."""
        return hashlib.sha256(text.encode()).hexdigest()[:8]  # Using first 8 chars for brevity
    
    def clean_tags(self, text: str) -> str:
        """Remove system and thinking tags from text."""
        # Repeatedly remove tags until no more matches are found
        prev_text = None
        while prev_text != text:
            prev_text = text
            text = re.sub(r'<s>(?:[^<]|<(?!/?system))*</s>\n?', '', text, flags=re.DOTALL)
            text = re.sub(r'<thinking>(?:[^<]|<(?!/?thinking))*</thinking>\n?', '', text, flags=re.DOTALL)
            text = re.sub(r'<s>(?:[^<]|<(?!/?s))*</s>\n?', '', text, flags=re.DOTALL)
        return text
    
    def calculate_tokens(self, text: str) -> int:
        """Calculate exact token count using tokenizer."""
        return len(tokenizer.encode(text))
    
    async def compress_text(self, text: str, target_rate: float = 0.75, target_tokens: int = None, depth: int = 0) -> Dict[str, Any]:
        """Compress text using LLMlingua-2 with recursion limits and safeguards."""
        try:
            # Guard against excessive recursion
            if depth > 3:  # Limit recursion depth
                print(f"Warning: Max recursion depth ({depth}) reached")
                # Return as-is rather than truncating
                return {
                    'compressed_text': text,
                    'compressed_prompt': text,
                    'original_tokens': self.calculate_tokens(text),
                    'compressed_tokens': self.calculate_tokens(text)
                }

            initial_tokens = self.calculate_tokens(text)
            
            # Guard against zero-length or tiny texts
            if initial_tokens < 10:  # Don't try to compress very small texts
                return {
                    'compressed_text': text,
                    'compressed_prompt': text,
                    'original_tokens': initial_tokens,
                    'compressed_tokens': initial_tokens
                }

            # Calculate target rate safely
            if target_tokens:
                # Ensure we don't divide by zero and have a reasonable minimum rate
                target_rate = max(0.1, min(0.9, target_tokens / max(1, initial_tokens)))

            # First try normal compression
            try:
                result = self.llm_lingua.compress_prompt(
                    text,
                    rate=target_rate,
                    force_tokens=['\n']  # Preserve newlines
                )
                compressed_tokens = self.calculate_tokens(result['compressed_prompt'])
            except Exception as e:
                print(f"LLMlingua compression failed: {str(e)}")
                # Don't fall back to truncation, try splitting instead
                result = None
                compressed_tokens = initial_tokens

            # If compression failed or still too large, split and recurse
            if (not result or (target_tokens and compressed_tokens > target_tokens)) and depth < 3:
                parts = self.split_content(text)
                compressed_parts = []
                
                if len(parts) <= 1:  # If splitting failed, return original
                    return {
                        'compressed_text': text,
                        'compressed_prompt': text,
                        'original_tokens': initial_tokens,
                        'compressed_tokens': initial_tokens
                    }
                
                # Calculate safe target for parts
                part_target = (target_tokens or initial_tokens) // max(1, len(parts))
                part_rate = target_rate * 0.8  # More aggressive for parts
                
                # Compress each part
                for part in parts:
                    if not part.strip():  # Skip empty parts
                        continue
                        
                    part_result = await self.compress_text(
                        part,
                        target_rate=part_rate,
                        target_tokens=part_target,
                        depth=depth + 1
                    )
                    compressed_parts.append(part_result['compressed_prompt'])
                
                # Combine compressed parts
                final_text = '\n\n[...SPLIT...]\n\n'.join(
                    part for part in compressed_parts if part.strip()
                )
                
                return {
                    'compressed_text': final_text,
                    'compressed_prompt': final_text,
                    'original_tokens': initial_tokens,
                    'compressed_tokens': self.calculate_tokens(final_text)
                }
                
            # If normal compression succeeded
            if result:
                return {
                    'compressed_text': result['compressed_prompt'],
                    'compressed_prompt': result['compressed_prompt'],
                    'original_tokens': result['origin_tokens'],
                    'compressed_tokens': result['compressed_tokens']
                }
                
            # If all else fails, return original
            return {
                'compressed_text': text,
                'compressed_prompt': text,
                'original_tokens': initial_tokens,
                'compressed_tokens': initial_tokens
            }
            
        except Exception as e:
            print(f"Compression error: {str(e)}")
            # Return original text rather than truncating
            return {
                'compressed_text': text,
                'compressed_prompt': text,
                'original_tokens': self.calculate_tokens(text),
                'compressed_tokens': self.calculate_tokens(text)
            }