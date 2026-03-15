#!/bin/bash
set -e

cd /mnt/b/M/avir/research/fluxcore

./fluxcore_entity > results/cuda_baseline.txt 2>&1
exit $?
