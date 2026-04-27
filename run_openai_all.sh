#!/bin/bash
# Runs all remaining OpenAI (gpt-5.4-mini) experiments sequentially.
# A/house-price already done. spaceship-titanic uses 50 steps (needs more room).
set -e
cd "$(dirname "$0")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== OpenAI experiment batch starting ==="

# --- Condition A ---
log "A / spaceship-titanic / gpt-5.4-mini (50 steps)"
python3 evaluation/run_experiment.py --condition A --task spaceship-titanic --model gpt-5.4-mini --steps 50

log "A / vectorization / gpt-5.4-mini"
python3 evaluation/run_experiment.py --condition A --task vectorization --model gpt-5.4-mini --steps 30

log "A / feedback / gpt-5.4-mini"
python3 evaluation/run_experiment.py --condition A --task feedback --model gpt-5.4-mini --steps 30

# --- Condition B ---
log "B / house-price / gpt-5.4-mini"
python3 evaluation/run_experiment.py --condition B --task house-price --model gpt-5.4-mini --steps 30 --attempts 3

log "B / spaceship-titanic / gpt-5.4-mini (50 steps)"
python3 evaluation/run_experiment.py --condition B --task spaceship-titanic --model gpt-5.4-mini --steps 50 --attempts 3

log "B / vectorization / gpt-5.4-mini"
python3 evaluation/run_experiment.py --condition B --task vectorization --model gpt-5.4-mini --steps 30 --attempts 3

log "B / feedback / gpt-5.4-mini"
python3 evaluation/run_experiment.py --condition B --task feedback --model gpt-5.4-mini --steps 30 --attempts 3

# --- Condition C ---
log "C / house-price / gpt-5.4-mini"
python3 evaluation/run_experiment.py --condition C --task house-price --model gpt-5.4-mini --steps 30 --attempts 3

log "C / spaceship-titanic / gpt-5.4-mini (50 steps)"
python3 evaluation/run_experiment.py --condition C --task spaceship-titanic --model gpt-5.4-mini --steps 50 --attempts 3

log "C / vectorization / gpt-5.4-mini"
python3 evaluation/run_experiment.py --condition C --task vectorization --model gpt-5.4-mini --steps 30 --attempts 3

log "C / feedback / gpt-5.4-mini"
python3 evaluation/run_experiment.py --condition C --task feedback --model gpt-5.4-mini --steps 30 --attempts 3

log "=== All OpenAI experiments complete ==="
