#!/usr/bin/env python3
import asyncio
import time
import uuid
from typing import Optional, Dict

class CompressionTask:
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

class CompressionQueue:
    def __init__(self, compressor=None):
        self.queue = asyncio.Queue()
        self.tasks = {}  # task_id -> CompressionTask
        self.worker_task = None
        self.processing = False
        self.compressor = compressor
    
    async def start_worker(self):
        """Start the worker that processes tasks from the queue"""
        self.worker_task = asyncio.create_task(self._worker())
        # Yield control to ensure task starts
        await asyncio.sleep(0)
        print("Started compression worker")
    
    async def _worker(self):
        """Worker that processes tasks from the queue"""
        print("Worker started and waiting for compression tasks")
        while True:
            try:
                # Get task from queue
                task_id = await self.queue.get()
                print(f"Processing task {task_id[:8]}")
                
                # Mark as processing and record start time
                self.processing = True
                task = self.tasks.get(task_id)
                if not task:
                    self.queue.task_done()
                    self.processing = False
                    continue
                
                task.status = "in_progress"
                task.started_at = time.time()
                
                # Process task (without holding any locks)
                try:
                    result = await self.compressor.compress_text(
                        task.prompt, 
                        target_tokens=task.target_tokens
                    )
                    task.result = result
                    task.status = "completed"
                    print(f"Task {task_id[:8]} completed")
                except Exception as e:
                    print(f"Error processing task {task_id[:8]}: {str(e)}")
                    task.error = str(e)
                    task.status = "failed"
                
                task.completed_at = time.time()
            
            except Exception as e:
                print(f"Worker error: {str(e)}")
            
            finally:
                # Always mark task as done and reset processing flag
                self.queue.task_done()
                self.processing = False
    
    async def add_task(self, prompt: str, target_tokens: Optional[int] = None) -> str:
        """Add a task to the queue and return its ID"""
        task_id = str(uuid.uuid4())
        task = CompressionTask(
            id=task_id,
            prompt=prompt,
            target_tokens=target_tokens
        )
        self.tasks[task_id] = task
        await self.queue.put(task_id)
        return task_id
    
    async def get_task(self, task_id: str) -> Optional[CompressionTask]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
    
    async def cleanup_old_tasks(self, max_age: int = 3600) -> int:
        """Remove old tasks from the registry"""
        now = time.time()
        removed = []
        for task_id, task in list(self.tasks.items()):
            if task.status in ("completed", "failed") and task.completed_at and (now - task.completed_at) > max_age:
                del self.tasks[task_id]
                removed.append(task_id)
        return len(removed)