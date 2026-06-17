#!/usr/bin/env bash
# Phase B - train all 120 booking seed-cells in parallel (12-wide pool).
# 4 seeds x 5 arms x 2 profiles x 3 crowds = 120 jobs, one cell per process.
# The DURABLE output of this phase is the 120 cached model zips in
# outputs/models/seeds/. The per-seed JSONs written here are PARTIAL and are NOT
# the final results - Phase C reassembles each seed's authoritative JSON.
# Mirrors docs/reseed_plan_updated.pdf Phase B / reseed_teammate_prompt.md Step 5.
set -o pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (this script lives in scripts/)
# Force CPU (see Phase A2): deterministic + faster for tiny MLPs + lets 12 workers
# share 32 cores instead of contending over one GPU. Inherited by all xargs children.
export CUDA_VISIBLE_DEVICES=""
mkdir -p outputs

for s in 1 3 7 202606; do
  for arm in ppo_base dqn_base mppo_int ppo_int dqn_int; do
    for prof in art tourist; do
      for c in 500 2500 5000; do
        echo "$s $arm $prof $c"
      done
    done
  done
# -P 16: 16 workers x 2 torch threads = 32 cores exactly on this box (measured
# under-utilization at -P 12). Correctness-neutral - A2 proved parallelism degree
# does not change results. ~13 GB RAM, fits the 20 GB available.
done | xargs -P 16 -L 1 bash -c \
  'nice -n 10 .venv/bin/python scripts/reseed_booking_grid.py \
    --seed "$1" --arms "$2" --profiles "$3" --crowds "$4"' _ \
  > outputs/reseed_multiseed.log 2>&1

status=$?
echo "Phase B parallel training exit status: $status"
if [ "$status" -ne 0 ]; then
  echo "---- tail of outputs/reseed_multiseed.log ----"
  tail -200 outputs/reseed_multiseed.log
  exit "$status"
fi

echo "---- post-run error grep (false positives possible; inspect any hits) ----"
grep -nEi "traceback|exception|error|failed|nan|inf" outputs/reseed_multiseed.log | tail -100 || true
echo "---- cached model zip count ----"
ls outputs/models/seeds/*.zip 2>/dev/null | wc -l
echo "=== Phase B done ($(date +%H:%M:%S)) ==="
