#!/usr/bin/env bash
# EczemaFlow: Master Execution Pipeline
# This script runs the entire remedied pipeline as requested by the peer reviewer.

set -e

echo "=========================================="
echo "1. Generating Canonical Results"
echo "=========================================="
export PYTHONPATH=$(pwd)
source venv/bin/activate
python scripts/generate_canonical_results.py

echo "=========================================="
echo "2. Validating Topology Pipeline (TDA)"
echo "=========================================="
python scripts/validate_topology.py

echo "=========================================="
echo "3. Running External Validation (GSE197023)"
echo "=========================================="
# This runs patient-level zero-shot evaluation and reports metrics
python run_external_validation.py

echo "=========================================="
echo "4. Running Full Cross-Validation (GSE206391)"
echo "=========================================="
# WARNING: This step will take hours/days depending on hardware.
# It uses the library-size normalized datasets and unmocked routines.
python train.py

echo "Pipeline execution complete."
