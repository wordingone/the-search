#!/usr/bin/env bash
# setup.sh — one-command environment setup for the-search
# Run from repo root: bash setup.sh
# Then activate: source .venv/Scripts/activate  (Windows)
#               source .venv/bin/activate       (Linux/Mac)

set -e

echo "=== the-search environment setup ==="

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    python -m venv .venv
else
    echo ".venv already exists, skipping creation"
fi

# Detect activate script (Windows vs Unix)
if [ -f ".venv/Scripts/activate" ]; then
    ACTIVATE=".venv/Scripts/activate"
    PIP=".venv/Scripts/pip"
else
    ACTIVATE=".venv/bin/activate"
    PIP=".venv/bin/pip"
fi

echo "Installing dependencies..."
"$PIP" install --upgrade pip
"$PIP" install -r requirements.txt

echo "Installing Atari ROMs..."
"$PIP" install autorom
.venv/Scripts/autorom --accept-license 2>/dev/null || .venv/bin/autorom --accept-license 2>/dev/null || echo "autorom: skipped (run manually if needed)"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $ACTIVATE"
echo "Run experiment: PYTHONPATH=. python experiments/step0720_cifar_dynamic_r3.py"
