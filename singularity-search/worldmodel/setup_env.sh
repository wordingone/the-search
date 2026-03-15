#!/bin/bash
# Source this file before running Genesis scripts to use local cache
# Usage: source setup_env.sh

export HF_HOME="B:/M/ArtificialArchitecture/worldmodel/.cache/huggingface"
export HF_DATASETS_CACHE="B:/M/ArtificialArchitecture/worldmodel/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="B:/M/ArtificialArchitecture/worldmodel/.cache/huggingface/hub"
export TORCH_HOME="B:/M/ArtificialArchitecture/worldmodel/.cache/torch"

echo "Genesis environment configured:"
echo "  HF_HOME=$HF_HOME"
echo "  TORCH_HOME=$TORCH_HOME"
