#!/usr/bin/env bash
# Reproduce all experiments and figures of the paper.
# Usage: scripts/reproduce.sh [--quick]
set -euo pipefail
OUT="runs/$(date +%Y%m%d-%H%M%S)"
python -m emb.experiments --out "$OUT" "$@"
echo "Figures and JSON written to $OUT"
