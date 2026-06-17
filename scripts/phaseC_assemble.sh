#!/usr/bin/env bash
# Phase C - assemble each seed's AUTHORITATIVE 30-cell JSON, sequential & single-writer.
# Reloads the 30 cached model zips per seed (no retraining) and writes the complete
# outputs/seeds/results_rl_booking_seed<s>.json in one pass. These - not the partial
# JSONs from Phase B - are the files we average.
# Mirrors docs/reseed_plan_updated.pdf Phase C / reseed_teammate_prompt.md Step 6.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (this script lives in scripts/)
# Same CPU device as training, so reloaded models evaluate identically.
export CUDA_VISIBLE_DEVICES=""

for s in 1 3 7 202606; do
  echo "=== Phase C assembling seed $s ($(date +%H:%M:%S)) ==="
  nice -n 10 .venv/bin/python scripts/reseed_booking_grid.py --seed "$s" \
    2>&1 | tee "outputs/reseed_assembly_seed${s}.log"
done

echo "=== Phase C log scan (cache miss / retrain / errors should be ABSENT) ==="
grep -nEi "cache miss|retrain|traceback|exception|error|failed|nan|inf" \
  outputs/reseed_assembly_seed*.log | tail -100 || true
# A healthy assembly reuses every cell: expect 30 'reuse' lines per seed, 0 'training'.
echo "=== reuse vs training line counts per seed (expect reuse=30, training=0) ==="
for s in 1 3 7 202606; do
  r=$(grep -c "reuse" "outputs/reseed_assembly_seed${s}.log" 2>/dev/null || echo 0)
  t=$(grep -c "training" "outputs/reseed_assembly_seed${s}.log" 2>/dev/null || echo 0)
  echo "seed $s: reuse=$r training=$t"
done
echo "=== Phase C done ($(date +%H:%M:%S)) ==="
