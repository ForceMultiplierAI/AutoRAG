#!/usr/bin/env python3
import asyncio
import time
import uuid
import logging
from typing import Optional, Dict, Any, List

# Set up logging
logger = logging.getLogger(__name__)

class CompressionTask:
    """Represents a single compression task in the queue"""
    def __init__(self, id: str, prompt: str, target_tokens: Optional[int] = None):
        self.id = id
        self.prompt = prompt
        self.target_tokens = target_tokens
        self.status = "pending"  # pending, in_progress, completed, failed
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.attempts = 0
        self.max_attempts = 3

    def __repr__(self):
        return f"CompressionTask(id={self.id[:8]}, status={self.status}, attempts={self.attempts})"

class CompressionQueue:
    """Manages a queue of compression tasks and processes them asynchronously"""
    def __init__(self, compressor=None):
        self.queue = asyncio.Queue()
        self.tasks = {}  # task_id -> CompressionTask
        self.worker_task = None
        self.processing = False
        self.compressor = compressor
        self.stats = {
            "completed": 0,
            "failed": 0,
            "retried": 0,
            "total": 0
        }
    
    async def start_worker(self):
        """Start the worker that processes tasks from the queue"""
        if self.worker_task is not None:
            logger.warning("Worker already running, not starting another one")
            return
            
        self.worker_task = asyncio.create_task(self._worker())
        # Yield control to ensure task starts
        await asyncio.sleep(0)
        logger.info("Started compression worker")
    
    async def _worker(self):
        """Worker that processes tasks from the queue"""
        logger.info("Worker started and waiting for compression tasks")
        while True:
            task_id = None
            try:
                # Get task from queue
                task_id = await self.queue.get()
                task = self.tasks.get(task_id)
                
                if not task:
                    logger.warning(f"Task {task_id[:8]} not found in registry")
                    self.queue.task_done()
                    continue
                
                logger.info(f"Processing task {task_id[:8]} (attempt {task.attempts + 1})")
                
                # Mark as processing and record start time
                self.processing = True
                task.status = "in_progress"
                task.started_at = time.time()
                task.attempts += 1
                
                # Process task (without holding any locks)
                try:
                    # Check if compressor is available
                    if not self.compressor:
                        raise RuntimeError("Compressor not available")
                        
                    result = await self.compressor.compress_text(
                        task.prompt, 
                        target_tokens=task.target_tokens
                    )
                    
                    # Validate the result
                    if not result or not isinstance(result, dict) or 'compressed_prompt' not in result:
                        raise ValueError("Invalid compression result")
                    
                    task.result = result
                    task.status = "completed"
                    task.completed_at = time.time()
                    self.stats["completed"] += 1
                    
                    # Cache the result if not already in cache
                    prompt_hash = self.compressor.calculate_hash(task.prompt)
                    if prompt_hash not in self.compressor.cache and 'compressed_prompt' in result:
                        self.compressor.cache[prompt_hash] = result
                    
                    logger.info(f"Task {task_id[:8]} completed successfully")
                    
                except Exception as e:
                    logger.error(f"Error processing task {task_id[:8]}: {str(e)}")
                    task.error = str(e)
                    
                    # Handle retries
                    if task.attempts < task.max_attempts:
                        logger.info(f"Retrying task {task_id[:8]} (attempt {task.attempts}/{task.max_attempts})")
                        task.status = "pending"
                        self.stats["retried"] += 1
                        # Re-queue the task with a small delay
                        await asyncio.sleep(1)  # Wait a bit before retrying
                        await self.queue.put(task_id)
                    else:
                        task.status = "failed"
                        task.completed_at = time.time()
                        self.stats["failed"] += 1
                        logger.warning(f"Task {task_id[:8]} failed after {task.attempts} attempts")
            
            except asyncio.CancelledError:
                logger.info("Worker task cancelled")
                break
                
            except Exception as e:
                logger.error(f"Unhandled worker error: {str(e)}")
                if task_id and task_id in self.tasks:
                    self.tasks[task_id].status = "failed"
                    self.tasks[task_id].error = f"Unhandled error: {str(e)}"
                    self.tasks[task_id].completed_at = time.time()
                
                # Add a small delay to prevent tight loops on persistent errors
                await asyncio.sleep(1)
            
            finally:
                # Always mark task as done and reset processing flag
                if task_id:
                    self.queue.task_done()
                    self.processing = False
    
    async def add_task(self, prompt: str, target_tokens: Optional[int] = None) -> str:
        """
        Add a task to the queue and return its ID
        
        Args:
            prompt: The text to compress
            target_tokens: Optional target token count
            
        Returns:
            task_id: Unique ID for the compression task
        """
        # First check if we already have this in the cache
        if self.compressor:
            prompt_hash = self.compressor.calculate_hash(prompt)
            
            try:
                if prompt_hash in self.compressor.cache:
                    cached_result = self.compressor.cache[prompt_hash]
                    
                    # Create a pre-completed task
                    task_id = str(uuid.uuid4())
                    task = CompressionTask(
                        id=task_id,
                        prompt=prompt,
                        target_tokens=target_tokens
                    )
                    task.status = "completed"
                    task.result = cached_result
                    task.created_at = time.time()
                    task.started_at = time.time()
                    task.completed_at = time.time()
                    
                    # Store the task but don't queue it
                    self.tasks[task_id] = task
                    logger.info(f"Task {task_id[:8]} created with cached result (hash: {prompt_hash[:8]})")
                    return task_id
            except Exception as e:
                logger.warning(f"Cache lookup failed: {str(e)}")
                # Continue with normal task creation
        
        # Create a new task and add it to the queue
        task_id = str(uuid.uuid4())
        task = CompressionTask(
            id=task_id,
            prompt=prompt,
            target_tokens=target_tokens
        )
        self.tasks[task_id] = task
        self.stats["total"] += 1
        
        # Ensure worker is running
        if not self.worker_task or self.worker_task.done():
            await self.start_worker()
            
        await self.queue.put(task_id)
        logger.info(f"Added task {task_id[:8]} to queue (queue size: {self.queue.qsize()})")
        return task_id
    
    async def get_task(self, task_id: str) -> Optional[CompressionTask]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the queue"""
        return {
            "queue_size": self.queue.qsize(),
            "task_count": len(self.tasks),
            "processing": self.processing,
            "stats": self.stats
        }
    
    async def cleanup_old_tasks(self, max_age: int = 3600) -> int:
        """
        Remove old tasks from the registry
        
        Args:
            max_age: Maximum age in seconds for completed/failed tasks
            
        Returns:
            count: Number of tasks removed
        """
        now = time.time()
        removed = []
        
        for task_id, task in list(self.tasks.items()):
            # Remove completed or failed tasks that are older than max_age
            if task.status in ("completed", "failed") and task.completed_at and (now - task.completed_at) > max_age:
                del self.tasks[task_id]
                removed.append(task_id)
        
        if removed:
            logger.info(f"Cleaned up {len(removed)} old tasks")
            
        return len(removed)
    
    async def stop(self):
        """Stop the worker task"""
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
            logger.info("Worker stopped")