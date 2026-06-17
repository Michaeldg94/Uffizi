#!/usr/bin/env bash
# Phase A2 - sequential-vs-parallel smoke check (the gate before Phase B).
# Trains 4 sampled seed-cells in two ISOLATED repo copies: once sequentially,
# once through the xargs parallel path. Step 4 (canonical-JSON compare) is run
# separately by the caller. Follows docs/reseed_teammate_prompt.md Steps 0-3 verbatim.
set -uo pipefail

ORIG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (this script lives in scripts/)
cd "$ORIG"
# Force CPU training: the driver is CPU-designed (set_num_threads(2)) and the seed-42
# models are CPU-trained. Hiding the GPU makes training deterministic (so this gate is
# meaningful) and faster for these tiny MLPs. Changes no hyperparameter/env/seed.
export CUDA_VISIBLE_DEVICES=""
PY="$ORIG/.venv/bin/python"
CHECK_ROOT="$ORIG/outputs/checks/sequential_vs_parallel"
mkdir -p "$CHECK_ROOT"

# --- Step 0: the 4 sampled cells (one distinct seed each) ---
cat > "$CHECK_ROOT/cells.txt" <<'EOF'
1 ppo_base art 500
7 dqn_base tourist 2500
3 mppo_int art 5000
202606 ppo_int tourist 500
EOF
echo "=== A2 cells ==="; cat "$CHECK_ROOT/cells.txt"

# --- Step 1: two clean source copies (exclude .git/.venv/outputs) ---
echo "=== A2 building isolated repo_seq / repo_par ($(date +%H:%M:%S)) ==="
rm -rf "$CHECK_ROOT/repo_seq" "$CHECK_ROOT/repo_par"
rsync -a --delete --exclude '.git' --exclude '.venv' --exclude 'outputs' ./ "$CHECK_ROOT/repo_seq/"
rsync -a --delete --exclude '.git' --exclude '.venv' --exclude 'outputs' ./ "$CHECK_ROOT/repo_par/"

# --- Step 2: run the sampled cells SEQUENTIALLY ---
echo "=== A2 SEQUENTIAL run ($(date +%H:%M:%S)) ==="
while read -r s arm prof crowd; do
  [ -z "$s" ] && continue
  (
    cd "$CHECK_ROOT/repo_seq"
    PYTHONPATH="$PWD" nice -n 10 "$PY" scripts/reseed_booking_grid.py \
      --seed "$s" --arms "$arm" --profiles "$prof" --crowds "$crowd"
  )
done < "$CHECK_ROOT/cells.txt" 2>&1 | tee "$CHECK_ROOT/sequential.log"

# --- Step 3: run the SAME cells through the PARALLEL path (xargs -P 4) ---
echo "=== A2 PARALLEL run ($(date +%H:%M:%S)) ==="
export PY CHECK_ROOT
cat "$CHECK_ROOT/cells.txt" | xargs -P 4 -L 1 bash -c '
  cd "$CHECK_ROOT/repo_par"
  PYTHONPATH="$PWD" nice -n 10 "$PY" scripts/reseed_booking_grid.py \
    --seed "$1" --arms "$2" --profiles "$3" --crowds "$4"
' _ 2>&1 | tee "$CHECK_ROOT/parallel.log"

echo "=== A2 training phase done ($(date +%H:%M:%S)) ==="
