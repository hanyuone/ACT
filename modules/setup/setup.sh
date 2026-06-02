#!/bin/bash
set -e

# Configuration flags and parameters
ACT_CI_MODE=${ACT_CI_MODE:-false}
COMPONENT=${1:-"main"}  # Default to main component if no argument provided

# Function to show usage
show_usage() {
    echo "Usage: source setup.sh [COMPONENT]"
    echo "COMPONENT can be: main"
    echo "Set ACT_CI_MODE=true for CI installation"
    echo ""
    echo "Examples:"
    echo "  source setup.sh              # Install main environment (local mode)"
    echo "  source setup.sh main         # Install only main environment"
    echo "  ACT_CI_MODE=true source setup.sh main  # Install main for CI"
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

if [ "$ACT_CI_MODE" = "true" ]; then
    echo "[ACT-CI] Setting up ACT environment for CI (component: $COMPONENT)..."
else
    echo "[ACT] Setting up ACT environment (component: $COMPONENT)..."
fi

# Step 1: Conda check
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda not found on this system."
    echo "[INFO] Please install Miniconda or Anaconda first from:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo "[ABORT] Exiting setup..."
    exit 1
fi

# Step 2: Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Function to setup main environment
setup_main() {
    echo "[ACT] Setting up main environment..."
    
    # Step 3: Create and activate main environment (act-py312)
    if ! conda env list | grep -q "^act-py312 "; then
        echo "[ACT] Creating conda env: act-py312..."
        conda create -y -n act-py312 python=3.12 pip
    else
        echo "[ACT] Conda env 'act-py312' already exists."
    fi

    echo "[ACT] Activating ACT-main environment..."
    conda activate act-py312

    echo "[ACT] Installing ACT requirements..."
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r main_requirements.txt

    # Step 4: Install gurobi via conda for ACT main environment solving
    if [ "$ACT_CI_MODE" = "true" ]; then
        echo "[ACT-CI] Skipping Gurobi installation for CI..."
    else
        echo "[ACT] Installing Gurobi for act-py312 environment..."
        conda config --add channels http://conda.anaconda.org/gurobi
        conda install -y gurobi 
    fi
}

# Main setup logic based on component selection
case "$COMPONENT" in
    "main")
        setup_main
        ;;
    *)
        echo "[ERROR] Unknown component: $COMPONENT"
        show_usage
        exit 1
        ;;
esac

# Final setup steps (for all or when not in CI mode)
if [ "$ACT_CI_MODE" = "false" ]; then
    export ACTHOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    export GRB_LICENSE_FILE=$ACTHOME/modules/gurobi/gurobi.lic
    echo "[ACT] Gurobi license path configured for this shell: $GRB_LICENSE_FILE"
fi

echo "[ACT] Setup complete for component: $COMPONENT"
if [ "$ACT_CI_MODE" = "false" ]; then
    echo "[ACT] Now you can run with 'conda activate act-py312' to start."
fi
