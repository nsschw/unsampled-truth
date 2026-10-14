import yaml
import subprocess
import itertools
from pathlib import Path
import time

def launch_matrix():
    # 1. Define the models you want to sweep
    models = [
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it"
    ]
    
    # 2. Grab your existing baseline persona configs (e.g., antonia.yaml)
    config_dir = Path("configs")
    base_configs = list(config_dir.glob("*.yaml")) 
    
    temp_dir = Path("configs/temp_runs")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    active_processes = []
    
    # Set to your hardware limit
    MAX_CONCURRENT_JOBS = 4

    for model_id, base_cfg_path in itertools.product(models, base_configs):
        if "temp_runs" in str(base_cfg_path) or "default" in base_cfg_path.name:
            continue
            
        with open(base_cfg_path, "r") as f:
            run_cfg = yaml.safe_load(f)
            
        short_model = model_id.split('/')[-1]
        baseline_persona = base_cfg_path.stem
        experiment_name = f"{baseline_persona}_{short_model}"
        
        # 3. Dynamically override the configuration
        run_cfg["model_id"] = model_id
        run_cfg["experiment_name"] = experiment_name
        
        tmp_yaml_path = temp_dir / f"{experiment_name}.yaml"
        with open(tmp_yaml_path, "w") as f:
            yaml.dump(run_cfg, f, sort_keys=False)

        print(f"Launching Job: {experiment_name} (Contains {len(run_cfg.get('persona_wordings', []))} persona wordings)")
        
        # 4. Dispatch the Job
        cmd = ["python", "scripts/run_inference.py", "--config", str(tmp_yaml_path)]
        p = subprocess.Popen(cmd)
        active_processes.append(p)
        
        # ANTI-RACE CONDITION DELAY: 
        # Wait 15 seconds to allow PyTorch to allocate VRAM on the chosen GPU 
        # before the next job calls get_optimal_gpu()
        time.sleep(15)

        # 5. Queue management
        while len(active_processes) >= MAX_CONCURRENT_JOBS:
            active_processes = [p for p in active_processes if p.poll() is None]
            if len(active_processes) >= MAX_CONCURRENT_JOBS:
                time.sleep(5) 

    # Wait for the final batch to finish
    for p in active_processes:
        p.wait()
        
    print("\nMatrix Sweep Complete.")

if __name__ == "__main__":
    launch_matrix()