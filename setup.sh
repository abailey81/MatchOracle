#!/bin/bash
# MatchOracle — Setup Script
# Creates virtual environment and installs all dependencies

set -e

echo "=========================================="
echo "  MatchOracle — Environment Setup"
echo "=========================================="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment ..."
    python3 -m venv venv
    echo "  Done."
else
    echo "Virtual environment already exists."
fi

# Activate
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip ..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies ..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "  Activate:    source venv/bin/activate"
echo ""
echo "  Usage:"
echo "    1. Fetch data:    python data/generator.py"
echo "    2. Features:      python features/engine.py"
echo "    3. Predictions:   python models/run_pipeline.py"
echo ""
echo "  Fast mode (skip slow APIs):"
echo "    python data/generator.py --fast"
echo ""
