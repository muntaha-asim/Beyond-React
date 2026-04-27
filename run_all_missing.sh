#!/bin/bash
# Runs all missing B+C experiments for gpt-5.4-mini and claude-sonnet-4-6.
# GPT and Claude batches run in parallel; within each model tasks run sequentially.
# Logs to run_gpt.log and run_claude.log.  Safe to re-run — skips existing results.

set -uo pipefail
cd "$(dirname "$0")"
source implementation/venv2/bin/activate

RESULTS="results"

skip_if_done() {
    local path="$1"
    if [ -f "$path" ]; then
        echo "[SKIP] $path already exists"
        return 0
    fi
    return 1
}

run_one() {
    local cond="$1" task="$2" model="$3" steps="${4:-30}" attempts="${5:-3}"
    echo ""
    echo "========================================"
    echo "[$(date '+%H:%M:%S')] START  cond=$cond  task=$task  model=$model"
    echo "========================================"
    python3 evaluation/run_experiment.py \
        --condition "$cond" --task "$task" --model "$model" \
        --steps "$steps" --attempts "$attempts"
    local rc=$?
    echo "[$(date '+%H:%M:%S')] DONE   cond=$cond  task=$task  model=$model  exit=$rc"
    return $rc
}

# ── GPT batch ────────────────────────────────────────────────────────────────
run_gpt() {
    echo "[GPT batch starting at $(date)]"

    # Condition B
    skip_if_done "$RESULTS/B/house-price_gpt54mini.json"       || run_one B house-price       gpt-5.4-mini 30 3
    skip_if_done "$RESULTS/B/spaceship-titanic_gpt54mini.json" || run_one B spaceship-titanic gpt-5.4-mini 50 3
    skip_if_done "$RESULTS/B/vectorization_gpt54mini.json"     || run_one B vectorization     gpt-5.4-mini 30 3
    skip_if_done "$RESULTS/B/feedback_gpt54mini.json"          || run_one B feedback          gpt-5.4-mini 30 3

    # Condition C
    skip_if_done "$RESULTS/C/house-price_gpt54mini.json"       || run_one C house-price       gpt-5.4-mini 30 3
    skip_if_done "$RESULTS/C/spaceship-titanic_gpt54mini.json" || run_one C spaceship-titanic gpt-5.4-mini 50 3
    skip_if_done "$RESULTS/C/vectorization_gpt54mini.json"     || run_one C vectorization     gpt-5.4-mini 30 3
    skip_if_done "$RESULTS/C/feedback_gpt54mini.json"          || run_one C feedback          gpt-5.4-mini 30 3

    echo "[GPT batch finished at $(date)]"
}

# ── Claude batch ──────────────────────────────────────────────────────────────
run_claude() {
    echo "[Claude batch starting at $(date)]"

    # Condition B
    skip_if_done "$RESULTS/B/house-price_sonnet46.json"       || run_one B house-price       claude-sonnet-4-6 30 3
    skip_if_done "$RESULTS/B/spaceship-titanic_sonnet46.json" || run_one B spaceship-titanic claude-sonnet-4-6 50 3
    skip_if_done "$RESULTS/B/vectorization_sonnet46.json"     || run_one B vectorization     claude-sonnet-4-6 30 3
    skip_if_done "$RESULTS/B/feedback_sonnet46.json"          || run_one B feedback          claude-sonnet-4-6 30 3

    # Condition C
    skip_if_done "$RESULTS/C/house-price_sonnet46.json"       || run_one C house-price       claude-sonnet-4-6 30 3
    skip_if_done "$RESULTS/C/spaceship-titanic_sonnet46.json" || run_one C spaceship-titanic claude-sonnet-4-6 50 3
    skip_if_done "$RESULTS/C/vectorization_sonnet46.json"     || run_one C vectorization     claude-sonnet-4-6 30 3
    skip_if_done "$RESULTS/C/feedback_sonnet46.json"          || run_one C feedback          claude-sonnet-4-6 30 3

    echo "[Claude batch finished at $(date)]"
}

# ── Launch both in parallel ───────────────────────────────────────────────────
echo "[Master] Starting at $(date) — GPT and Claude running in parallel"
run_gpt  > >(tee run_gpt.log)  2>&1 &
GPT_PID=$!
run_claude > >(tee run_claude.log) 2>&1 &
CLAUDE_PID=$!

wait $GPT_PID
GPT_EXIT=$?
wait $CLAUDE_PID
CLAUDE_EXIT=$?

echo ""
echo "[Master] All done at $(date)"
echo "[Master] GPT exit=$GPT_EXIT  Claude exit=$CLAUDE_EXIT"

# Quick summary of what landed
echo ""
echo "=== Results on disk ==="
for cond in A B C; do
    for task in house-price spaceship-titanic vectorization feedback; do
        for slug in gemini25flash gpt54mini sonnet46; do
            f="$RESULTS/$cond/${task}_${slug}.json"
            [ -f "$f" ] && echo "  ✓ $cond/$task/$slug" || echo "  ✗ $cond/$task/$slug"
        done
    done
done
