import subprocess
import os

def get_optimal_gpu() -> int:
    """
    Queries nvidia-smi to find the GPU index with the highest available VRAM.
    Returns: int (GPU index)
    """
    try:
        # Query free memory in MB, strictly formatted for easy parsing
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"], 
            encoding="utf-8"
        )
        
        free_memory = [int(x) for x in result.strip().split('\n')]
        best_gpu_index = free_memory.index(max(free_memory))
        
        return best_gpu_index

    except FileNotFoundError:
        raise RuntimeError("nvidia-smi executable not found in PATH.")
    except Exception as e:
        raise RuntimeError(f"GPU query failed: {e}")