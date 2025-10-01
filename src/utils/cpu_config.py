"""
CPU configuration and multiprocessing utilities
"""
import os
import multiprocessing
import asyncio
from typing import Callable, Any, Tuple, Dict


# CPU Core Configuration
TOTAL_CORES = multiprocessing.cpu_count()
CPU_CORES = int(os.getenv("CPU_CORES", max(1, TOTAL_CORES // 2)))  # 50% default
MAX_CONCURRENT_CPU_TASKS = int(os.getenv("MAX_CONCURRENT_CPU_TASKS", 4))

# Semaphore to limit concurrent CPU tasks
cpu_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CPU_TASKS)


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU configuration information."""
    return {
        "total_cores": TOTAL_CORES,
        "allocated_cores": CPU_CORES,
        "max_concurrent_tasks": MAX_CONCURRENT_CPU_TASKS,
        "utilization_percentage": round((CPU_CORES / TOTAL_CORES) * 100, 1)
    }


async def run_cpu_task(func: Callable, *args, **kwargs) -> Any:
    """
    Run a CPU-intensive task using multiprocessing with semaphore limiting.
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Exception: If both multiprocessing and single-threaded execution fail
    """
    async with cpu_semaphore:
        try:
            # Try multiprocessing first
            with multiprocessing.Pool(processes=CPU_CORES) as pool:
                result = await asyncio.to_thread(pool.apply, func, args, kwargs)
            return result
            
        except Exception as e:
            # Fallback to single-threaded execution
            try:
                result = await asyncio.to_thread(func, *args, **kwargs)
                return result
            except Exception as fallback_error:
                raise Exception(f"Both multiprocessing and single-threaded execution failed. "
                              f"Multiprocessing error: {str(e)}, "
                              f"Fallback error: {str(fallback_error)}")


def debug_print_cpu_info():
    """Print CPU configuration for debugging."""
    import os
    DEBUG_PRINT = os.getenv("DEBUG_PRINT", "false").lower() == "true"
    
    if DEBUG_PRINT:
        info = get_cpu_info()
        print(f"🔧 CPU Configuration:")
        print(f"  Total cores: {info['total_cores']}")
        print(f"  Allocated cores: {info['allocated_cores']}")
        print(f"  Utilization: {info['utilization_percentage']}%")
        print(f"  Max concurrent CPU tasks: {info['max_concurrent_tasks']}")
