#!/usr/bin/env python3
"""Monitor OpenVid HD download progress"""

import os
import time
from pathlib import Path
from datetime import datetime

def check_openvid_status():
    """Check if OpenVid HD Part 1 is downloaded"""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--nkp37--OpenVid-1M"

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking OpenVid download status...")
    print(f"Cache path: {hf_cache}\n")

    if not hf_cache.exists():
        print("Status: Cache directory not yet created")
        print("→ Download likely initializing or not started")
        return False

    # Look for zip files
    zip_files = list(hf_cache.glob("**/OpenVidHD_part_*.zip"))

    if not zip_files:
        print("Status: Cache exists but no zip files found yet")
        print("→ Metadata downloaded, waiting for file chunks...")
        return False

    print("Status: Download in progress\n")
    for zip_file in sorted(zip_files):
        size_gb = zip_file.stat().st_size / (1024**3)
        print(f"  {zip_file.name}: {size_gb:.2f} GB")

    # Check if complete (should be ~48GB)
    total_size = sum(f.stat().st_size for f in zip_files) / (1024**3)
    if total_size > 40:
        print(f"\n✓ Download nearly complete: {total_size:.2f} GB")
        return True
    else:
        print(f"\nPartial download: {total_size:.2f} GB (target ~48GB)")
        return False

if __name__ == "__main__":
    # Run checks every 30 seconds
    try:
        while True:
            complete = check_openvid_status()
            if complete:
                print("\n✓ Download complete! Ready to use for training.")
                break
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
