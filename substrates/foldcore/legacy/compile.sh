#!/bin/bash
# FluxCore CUDA compilation script
# Targets sm_75 (RTX 4070 Ti)
set -e

FLUXCORE_DIR="/mnt/b/M/avir/research/fluxcore"
OUTPUT_DIR="$FLUXCORE_DIR/results"
mkdir -p "$OUTPUT_DIR"

echo "=== FluxCore CUDA Compilation ===" | tee "$OUTPUT_DIR/compile_log.txt"
echo "Target: sm_89 (RTX 4090)" | tee -a "$OUTPUT_DIR/compile_log.txt"
echo "Date: $(date)" | tee -a "$OUTPUT_DIR/compile_log.txt"
echo "" | tee -a "$OUTPUT_DIR/compile_log.txt"

cd "$FLUXCORE_DIR"

# Check nvcc
echo "--- nvcc version ---" | tee -a "$OUTPUT_DIR/compile_log.txt"
nvcc --version 2>&1 | tee -a "$OUTPUT_DIR/compile_log.txt"
echo "" | tee -a "$OUTPUT_DIR/compile_log.txt"

# Compile fluxcore_entity_impl.cu (standalone impl)
echo "--- Compiling fluxcore_entity_impl.cu ---" | tee -a "$OUTPUT_DIR/compile_log.txt"
nvcc -O3 -arch=sm_89 fluxcore_entity_impl.cu -o fluxcore_entity 2>&1 | tee -a "$OUTPUT_DIR/compile_log.txt"

if [ $? -eq 0 ]; then
    echo "COMPILE SUCCESS: fluxcore_entity" | tee -a "$OUTPUT_DIR/compile_log.txt"
else
    echo "COMPILE FAILED: fluxcore_entity_impl.cu" | tee -a "$OUTPUT_DIR/compile_log.txt"
    exit 1
fi

echo "" | tee -a "$OUTPUT_DIR/compile_log.txt"
echo "=== Done ===" | tee -a "$OUTPUT_DIR/compile_log.txt"
