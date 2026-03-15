import torch
import time
import os
import json
from pathlib import Path
from datetime import datetime

LOG_FILE = "training_monitor_log.json"
CKPT_DIR = Path("checkpoints/genesis_720_extended")
TARGET_ITER = 5000
CHECK_INTERVAL = 60  # seconds

def get_gpu_stats():
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        parts = result.stdout.strip().split(", ")
        return {
            "gpu_util": int(parts[0]),
            "mem_used_mb": int(parts[1]),
            "mem_total_mb": int(parts[2]),
            "temp_c": int(parts[3]),
            "power_w": float(parts[4])
        }
    except:
        return {}

def get_latest_checkpoint():
    ckpts = sorted(CKPT_DIR.glob("checkpoint_*.pt"))
    if not ckpts:
        return None, 0, []
    latest = ckpts[-1]
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    return latest.name, ckpt.get("iteration", 0), ckpt.get("eval_results", [])

def main():
    start_time = time.time()
    log_entries = []
    
    print(f"=== Training Monitor Started @ {datetime.now().strftime('%H:%M:%S')} ===")
    print(f"Target: {TARGET_ITER} iterations")
    print(f"Checking every {CHECK_INTERVAL}s")
    print()
    
    last_iter = 0
    last_time = start_time
    
    while True:
        ckpt_name, current_iter, eval_results = get_latest_checkpoint()
        gpu = get_gpu_stats()
        now = time.time()
        
        # Calculate speed
        if current_iter > last_iter:
            iter_diff = current_iter - last_iter
            time_diff = now - last_time
            ms_per_iter = (time_diff * 1000) / iter_diff if iter_diff > 0 else 0
            last_iter = current_iter
            last_time = now
        else:
            ms_per_iter = 0
        
        # Get CLIP-IQA
        clip_iqa = eval_results[-1]["clip_iqa"] if eval_results else 0
        
        # Calculate ETA
        remaining = TARGET_ITER - current_iter
        if ms_per_iter > 0:
            eta_sec = (remaining * ms_per_iter) / 1000
            eta_str = f"{eta_sec/60:.1f} min"
        else:
            eta_str = "calculating..."
        
        # Log entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": current_iter,
            "checkpoint": ckpt_name,
            "clip_iqa": clip_iqa,
            "ms_per_iter": ms_per_iter,
            "gpu_util": gpu.get("gpu_util", 0),
            "mem_used_mb": gpu.get("mem_used_mb", 0),
            "temp_c": gpu.get("temp_c", 0),
            "power_w": gpu.get("power_w", 0),
            "elapsed_min": (now - start_time) / 60
        }
        log_entries.append(entry)
        
        # Print status
        progress = (current_iter / TARGET_ITER) * 100
        print(f"[{datetime.now().strftime('%H:%M:%S')}] iter={current_iter}/{TARGET_ITER} ({progress:.0f}%) | "
              f"CLIP-IQA={clip_iqa:.4f} | GPU={gpu.get('gpu_util', 0)}% {gpu.get('mem_used_mb', 0)}MB | "
              f"ETA={eta_str}")
        
        # Check completion
        if current_iter >= TARGET_ITER:
            print()
            print("=== TRAINING COMPLETE ===")
            total_time = (now - start_time) / 60
            print(f"Total time: {total_time:.1f} minutes")
            print(f"Final CLIP-IQA: {clip_iqa:.4f}")
            print(f"Avg GPU util: {sum(e['gpu_util'] for e in log_entries)/len(log_entries):.0f}%")
            print(f"Peak memory: {max(e['mem_used_mb'] for e in log_entries)}MB")
            
            # Save log
            with open(LOG_FILE, "w") as f:
                json.dump({
                    "summary": {
                        "total_time_min": total_time,
                        "final_clip_iqa": clip_iqa,
                        "total_iterations": current_iter,
                        "avg_gpu_util": sum(e['gpu_util'] for e in log_entries)/len(log_entries),
                        "peak_memory_mb": max(e['mem_used_mb'] for e in log_entries)
                    },
                    "entries": log_entries
                }, f, indent=2)
            print(f"Log saved to {LOG_FILE}")
            break
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
