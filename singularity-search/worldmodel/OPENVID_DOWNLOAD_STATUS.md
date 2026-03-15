# OpenVid HD Download Status

## Task Initiated: 2026-02-05

### Download Details
- **Dataset**: OpenVid-1M (nkp37 HuggingFace)
- **File**: `OpenVidHD/OpenVidHD_part_1.zip`
- **Type**: Dataset (gated access)
- **Expected Size**: ~48 GB
- **Status**: DOWNLOAD IN PROGRESS

### Process Information
- **Background Task ID**: bdc8deb
- **Process**: Python script using `huggingface_hub`
- **Start Time**: 2026-02-05 (exact time logged in background task)

### What This Provides
- Native 1080p video clips (1920x1080)
- 433K high-quality video samples
- Replaces JAT 64x64 upscaled data (which hit 0.287 CLIP-IQA ceiling)
- Enable 720p Genesis training with >0.50 CLIP-IQA target

### Monitoring

Use the monitoring script to check progress:
```bash
python B:\M\ArtificialArchitecture\worldmodel\monitor_openvid_download.py
```

Or manually check HuggingFace cache:
```python
from pathlib import Path
hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--nkp37--OpenVid-1M"
if hf_cache.exists():
    for f in hf_cache.rglob("*.zip"):
        print(f"{f.name}: {f.stat().st_size / (1024**3):.2f} GB")
```

### Expected Timeline
- **Full download**: 4-8 hours (depends on network speed)
- **After download**: Extract (1-2 hours) → Use in training

### Next Steps (After Download Complete)
1. Verify file integrity
2. Extract to appropriate location
3. Integrate with Genesis training pipeline:
   ```bash
   python scripts/genesis_experiment.py --mode train \
     --data-mode webvid --image-size 720 \
     --data-path /path/to/OpenVidHD_part_1 \
     --use-fsq --use-perceptual
   ```

### Cache Location
- **Path**: `~/.cache/huggingface/hub/datasets--nkp37--OpenVid-1M/`
- **Windows**: `C:\Users\<username>\.cache\huggingface\hub\datasets--nkp37--OpenVid-1M\`

### Notes
- HuggingFace Hub will automatically resume if interrupted
- No need to restart download if connection drops
- File will be symlinked/cached for future use
- Storage required: ~50-60 GB (including overhead)
