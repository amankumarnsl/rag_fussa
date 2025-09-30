#!/usr/bin/env python3
"""
Celery task monitoring script for RAG FUSSA API
"""
import os
import sys
import time
from celery import Celery

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.celery_app import celery_app

def monitor_tasks():
    """Monitor active tasks"""
    print("🔍 RAG FUSSA Task Monitor")
    print("=" * 50)
    
    # Get active tasks
    inspect = celery_app.control.inspect()
    
    # Active tasks
    active_tasks = inspect.active()
    if active_tasks:
        print("\n📊 Active Tasks:")
        for worker, tasks in active_tasks.items():
            print(f"  Worker: {worker}")
            for task in tasks:
                print(f"    - {task['name']} (ID: {task['id']})")
                if 'args' in task:
                    print(f"      Args: {task['args']}")
    else:
        print("\n✅ No active tasks")
    
    # Scheduled tasks
    scheduled_tasks = inspect.scheduled()
    if scheduled_tasks:
        print("\n⏰ Scheduled Tasks:")
        for worker, tasks in scheduled_tasks.items():
            print(f"  Worker: {worker}")
            for task in tasks:
                print(f"    - {task['name']} (ID: {task['id']})")
    else:
        print("\n✅ No scheduled tasks")
    
    # Registered tasks
    registered_tasks = inspect.registered()
    if registered_tasks:
        print("\n📋 Registered Tasks:")
        for worker, tasks in registered_tasks.items():
            print(f"  Worker: {worker}")
            for task in tasks:
                print(f"    - {task}")
    
    # Worker stats
    stats = inspect.stats()
    if stats:
        print("\n📈 Worker Stats:")
        for worker, stat in stats.items():
            print(f"  Worker: {worker}")
            print(f"    - Total tasks: {stat.get('total', 'N/A')}")
            print(f"    - Pool processes: {stat.get('pool', {}).get('processes', 'N/A')}")
    
    print("\n" + "=" * 50)

def check_task_status(task_id: str):
    """Check status of a specific task"""
    result = celery_app.AsyncResult(task_id)
    
    print(f"🔍 Task Status: {task_id}")
    print("=" * 50)
    print(f"State: {result.state}")
    print(f"Ready: {result.ready()}")
    print(f"Successful: {result.successful()}")
    print(f"Failed: {result.failed()}")
    
    if result.state == "PROGRESS":
        print(f"Progress: {result.info}")
    elif result.state == "SUCCESS":
        print(f"Result: {result.result}")
    elif result.state == "FAILURE":
        print(f"Error: {result.info}")
    
    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
        check_task_status(task_id)
    else:
        monitor_tasks()
