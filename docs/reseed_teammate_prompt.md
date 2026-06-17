# Uffizi RL - Add One Training Seed (Teammate Instructions)

> **Coordinator setup (before sending):** Replace **every** `<SEED>` below with the unique
> integer you assign this teammate - one of the target new seeds `1`, `3`, `7`, or `202606`, and **not `42`** (seed 42 is already
> committed as the first seed). Then hand them everything from the line below down, verbatim.

> **Solo / many-core option:** You do not have to distribute this work. On a many-core machine,
> one operator can run the four new seeds (`1`, `3`, `7`, and `202606`) locally and average them. That plan is slightly different
> from the teammate protocol: it parallelizes independent seed-cell jobs, then rebuilds one
> authoritative JSON per seed in a final single-writer pass. See **`docs/reseed_plan.pdf`**.

---

## Role

You are helping with the **BSE Reinforcement Learning Uffizi project**. We need **additional
random TRAINING seeds** for our RL models so we can report averages instead of single runs.

**Your job:** retrain everything under ONE assigned training seed and produce two small results
files. Do **not** change any environment, reward, room capacity, the crowd simulator, or any
hyperparameter. Run the provided scripts exactly as written. **If anything errors, STOP and
report the full error verbatim** rather than working around it.

## Your assigned training seed

```bash
<SEED>
```

This seed is unique to you. Do **not** use `42`.

## Context (understand this - do not act beyond the Steps)

- The models were each trained with a single training seed (`42`), already committed as the
  first seed. We now want four additional seeds in total (`1`, `3`, `7`, and `202606`), then we average them with seed `42` for `n = 5`.
- Evaluation uses six **fixed** common-random-number crowd seeds (`900000..900005`). Those are
  **not** training seeds and must stay fixed. You only vary the **training** seed via `--seed`.
- The driver `scripts/reseed_booking_grid.py` retrains the full 30-cell matrix
  (5 algorithms x 2 visitor profiles x 3 crowds) and saves **seed-stamped** models, so it never
  collides with the committed seed-42 models.
- **Do not** run pipeline scripts `07` / `08` / `09` directly: they reuse the committed
  seed-42 models and would retrain nothing.

## Rules (read before you start)

- Change **nothing** in the envs / rewards / capacities / simulator / hyperparameters.
- Do **not** change the eval seeds (`900000..900005`) or the training budgets.
- Run only **one booking driver process for your assigned seed at a time**.
- If you use the optional split commands below, run them **sequentially**, not concurrently.
- On any error: **STOP and report it verbatim.**

## Steps

**1. Confirm prerequisites.** `git` and `uv` must be installed (uv: <https://docs.astral.sh/uv/>):

```bash
git --version
uv --version
```

**2. Update your local clone to the latest `mocl` branch:**

```bash
cd Uffizi
git pull origin mocl
```

**3. Install the locked environment:**

```bash
uv sync
```

**4. Confirm the driver exists.** If it is missing, **STOP and report** (do not improvise):

```bash
ls scripts/reseed_booking_grid.py
```

**5. HEAVY STEP - retrain the full matrix under YOUR seed.** Niced and CPU-capped. It is about
3.6M training steps, so plan for several hours - likely overnight. It is **per-cell resumable**:
if you stop it (Ctrl-C) and rerun the same command, it skips cells already done. This full command
is the preferred route because it leaves one complete results JSON for your seed.

```bash
nice -n 10 uv run python scripts/reseed_booking_grid.py --seed <SEED>
```

**Optional split mode (only if your machine is slow or we deliberately split the work):** run the
pieces **one after another**, not in parallel for the same seed. These commands cache the relevant
per-cell models. After all split pieces finish, run the full command once more to rebuild the
complete authoritative JSON for the seed.

```bash
# by profile - run sequentially
nice -n 10 uv run python scripts/reseed_booking_grid.py --seed <SEED> --profiles art
nice -n 10 uv run python scripts/reseed_booking_grid.py --seed <SEED> --profiles tourist

# or by crowd - run sequentially
nice -n 10 uv run python scripts/reseed_booking_grid.py --seed <SEED> --crowds 500 2500
nice -n 10 uv run python scripts/reseed_booking_grid.py --seed <SEED> --crowds 5000

# final assembly pass after any split-mode run
nice -n 10 uv run python scripts/reseed_booking_grid.py --seed <SEED>
```

Why this final pass matters: partial split runs may rewrite the seed-level JSON using only the
cells requested in that run. The last full pass reloads the cached models and writes the complete
30-cell results file with a single writer.

**6. Retrain the toy tabular Q-learning under YOUR seed** (fast - a couple of minutes), then
copy its result to a seed-stamped name:

```bash
uv run python uffizi_rl/pipeline/02_train_q_learning.py --seed <SEED> --episodes 25000
mkdir -p outputs/seeds
cp outputs/02_q_learning.json outputs/seeds/q_learning_seed<SEED>.json
```

**7. Sanity-check both files.** First check that both files are valid JSON and do not contain
obvious NaN/Inf strings:

```bash
python -m json.tool outputs/seeds/results_rl_booking_seed<SEED>.json > /dev/null
python -m json.tool outputs/seeds/q_learning_seed<SEED>.json > /dev/null

grep -iE 'nan|inf' outputs/seeds/results_rl_booking_seed<SEED>.json \
  outputs/seeds/q_learning_seed<SEED>.json || true
```

Then inspect the files and confirm that `mppo_int` check-ins are between 0 and 3:

```bash
cat outputs/seeds/results_rl_booking_seed<SEED>.json
cat outputs/seeds/q_learning_seed<SEED>.json
```

**8. Send the two files back.** They are small; the model zips are large and **not** needed.
Either send Marco both files, or push only them on a per-seed branch:

```bash
git checkout -b seed-<SEED>
git add outputs/seeds/results_rl_booking_seed<SEED>.json outputs/seeds/q_learning_seed<SEED>.json
git commit -m "RL results for training seed <SEED>"
git push origin seed-<SEED>
```

Do **not** commit the model zips (`outputs/models/seeds/`), and do **not** overwrite anyone
else's seed file or the committed seed-42 files.

## Definition of done

- [ ] `outputs/seeds/results_rl_booking_seed<SEED>.json` exists, is valid JSON, has no NaN/Inf,
      and `mppo_int` check-ins are in `[0, 3]`.
- [ ] `outputs/seeds/q_learning_seed<SEED>.json` exists, is valid JSON, and has no NaN/Inf.
- [ ] Both files sent to Marco (or pushed on branch `seed-<SEED>`); no model zips committed; no
      existing seed files overwritten.

---

## For the coordinator only (do not send to teammates)

Once a teammate's `results_rl_booking_seed<N>.json` and `q_learning_seed<N>.json` are in your
`outputs/seeds/` (fetch their `seed-<N>` branch and copy the files in, or just drop the files
they send you), run:

```bash
uv run python scripts/average_seed_runs.py
```

It averages every seed file in `outputs/seeds/`, including the seed-42 files already on git, and
prints the two grids (30-cell matrix with mean +/- std and an `n` column, plus the toy Q vs
Double-Q line). With seed `42` plus new seeds `1`, `3`, `7`, and `202606`, the final expected count is `n = 5`. No `--include-canonical` is needed, since seed 42 now lives in `outputs/seeds/`
as a normal seed file.

## Coordinator-only many-core plan and sequential-vs-parallel check

Use this section only for the local many-core run. The teammate protocol above remains conservative:
one assigned seed, one booking driver process at a time.

The many-core target is four new training seeds:

```bash
1 3 7 202606
```

Together with committed seed `42`, this gives five total training seeds (`n = 5`).

### Safety idea

The full parallel run is acceptable only if we first verify that the parallel execution path does not
change the result relative to a sequential execution path. We do **not** test every cell, because that
would duplicate the whole experiment. Instead, we run one representative cell per new seed in two
isolated temporary copies:

- `outputs/checks/sequential_vs_parallel/repo_seq/` for the sequential reference;
- `outputs/checks/sequential_vs_parallel/repo_par/` for the parallel-path check.

The sampled sequential and parallel JSON outputs must match exactly after canonical JSON formatting.
If they do not match, stop and diagnose before using the many-core plan. Zip-file checksums are not
the acceptance criterion because zip metadata can differ; the metric JSONs are the project outputs we
compare.

### Step 0 - create the check folder and sampled cells

Run this from the repository root after `uv sync`:

```bash
set -euo pipefail
ORIG="$PWD"
PY="$ORIG/.venv/bin/python"
CHECK_ROOT="$ORIG/outputs/checks/sequential_vs_parallel"
mkdir -p "$CHECK_ROOT"

cat > "$CHECK_ROOT/cells.txt" <<'EOF'
1 ppo_base art 500
7 dqn_base tourist 2500
3 mppo_int art 5000
202606 ppo_int tourist 500
EOF
```

### Step 1 - build isolated sequential and parallel workspaces

This prevents cached models or partial JSONs in the real `outputs/` directory from making the check
pass by accident.

```bash
rm -rf "$CHECK_ROOT/repo_seq" "$CHECK_ROOT/repo_par"
rsync -a --delete --exclude '.git' --exclude '.venv' --exclude 'outputs' \
  ./ "$CHECK_ROOT/repo_seq/"
rsync -a --delete --exclude '.git' --exclude '.venv' --exclude 'outputs' \
  ./ "$CHECK_ROOT/repo_par/"
```

### Step 2 - run the sampled cells sequentially

```bash
while read -r s arm prof crowd; do
  (
    cd "$CHECK_ROOT/repo_seq"
    PYTHONPATH="$PWD" nice -n 10 "$PY" scripts/reseed_booking_grid.py \
      --seed "$s" --arms "$arm" --profiles "$prof" --crowds "$crowd"
  )
done < "$CHECK_ROOT/cells.txt" 2>&1 | tee "$CHECK_ROOT/sequential.log"
```

### Step 3 - run the same sampled cells through the parallel path

```bash
export PY CHECK_ROOT
cat "$CHECK_ROOT/cells.txt" | xargs -P 4 -L 1 bash -c '
  cd "$CHECK_ROOT/repo_par"
  PYTHONPATH="$PWD" nice -n 10 "$PY" scripts/reseed_booking_grid.py \
    --seed "$1" --arms "$2" --profiles "$3" --crowds "$4"
' _ 2>&1 | tee "$CHECK_ROOT/parallel.log"
```

### Step 4 - compare sequential and parallel JSON outputs

```bash
python - <<'PY'
import difflib
import json
from pathlib import Path

root = Path("outputs/checks/sequential_vs_parallel")
seeds = [1, 3, 7, 202606]
failed = False

def canonical(path):
    with path.open() as f:
        obj = json.load(f)
    return json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)

for seed in seeds:
    seq = root / "repo_seq" / "outputs" / "seeds" / f"results_rl_booking_seed{seed}.json"
    par = root / "repo_par" / "outputs" / "seeds" / f"results_rl_booking_seed{seed}.json"
    if not seq.exists() or not par.exists():
        print(f"Missing check output for seed {seed}: {seq} or {par}")
        failed = True
        continue

    a = canonical(seq)
    b = canonical(par)
    if a != b:
        diff_path = root / f"diff_seed{seed}.txt"
        diff = difflib.unified_diff(
            a.splitlines(), b.splitlines(),
            fromfile=f"sequential_seed{seed}",
            tofile=f"parallel_seed{seed}",
            lineterm="",
        )
        diff_path.write_text("\n".join(diff))
        print(f"Mismatch for seed {seed}; see {diff_path}")
        failed = True
    else:
        print(f"Seed {seed}: sequential and parallel JSONs match exactly.")

if failed:
    raise SystemExit("Sequential-vs-parallel smoke check failed. Stop and diagnose.")
print("Sequential-vs-parallel smoke check passed for all sampled cells.")
PY
```

### Step 5 - only after the check passes, run the full 120-cell parallel training pass

```bash
set -o pipefail
mkdir -p outputs

for s in 1 3 7 202606; do
  for arm in ppo_base dqn_base mppo_int ppo_int dqn_int; do
    for prof in art tourist; do
      for c in 500 2500 5000; do
        echo "$s $arm $prof $c"
      done
    done
  done
done | xargs -P 12 -L 1 bash -c \
  'nice -n 10 .venv/bin/python scripts/reseed_booking_grid.py \
    --seed "$1" --arms "$2" --profiles "$3" --crowds "$4"' _ \
  > outputs/reseed_multiseed.log 2>&1

status=$?
echo "Parallel training exit status: $status"
if [ "$status" -ne 0 ]; then
  tail -200 outputs/reseed_multiseed.log
  exit "$status"
fi

grep -nEi "traceback|exception|error|failed|nan|inf" \
  outputs/reseed_multiseed.log | tail -100 || true
```

### Step 6 - rebuild the authoritative full JSON for each seed

Keep this step sequential per seed. These final JSONs, not the partial JSONs from the parallel phase,
are the files to average.

```bash
for s in 1 3 7 202606; do
  nice -n 10 uv run python scripts/reseed_booking_grid.py --seed "$s" \
    2>&1 | tee "outputs/reseed_assembly_seed${s}.log"
done

grep -nEi "cache miss|retrain|traceback|exception|error|failed|nan|inf" \
  outputs/reseed_assembly_seed*.log | tail -100 || true
```

### Step 7 - average all five seeds

```bash
uv run python scripts/average_seed_runs.py
```

The expected averaged output should report `n = 5`: committed seed `42` plus new seeds `1`, `3`, `7`,
and `202606`.

### Step 8 - make it reproducible (the last step)

Package everything so a fresh clone can regenerate the whole reseed with one command.

```bash
# Reproduce the full reseed from scratch (Phases A->E). Per-cell resumable: a re-run
# reuses cached model zips, so it is cheap once the heavy training has been done once.
uv run python run_project.py --reseed

# Handy variants:
uv run python run_project.py --reseed --skip-train     # only assemble/sanity/average from cache
uv run python run_project.py --reseed --seeds 7         # add a single extra training seed
uv run python run_project.py --reseed --workers 8       # cap the parallel CPU pool
```

What this step provides:

- **One portable entry point.** `run_project.py --reseed` delegates to `scripts/reseed_reproduce.py`,
  which runs Phase A (toy Q per seed) -> Phase B (the 120 booking cells in a CPU process pool) ->
  Phase C (single-writer assembly per seed) -> Phase D (sanity) -> Phase E (grand average, `n = 5`).
- **Deterministic.** Training is pinned to CPU via `CUDA_VISIBLE_DEVICES=""`. On a machine with a
  GPU, Stable-Baselines3's default `device="auto"` would otherwise train on the GPU, which is both
  slower for these small MLPs and not bit-reproducible. CPU makes the sequential-vs-parallel gate
  pass and the numbers repeatable. No hyperparameter, env, reward, budget, or eval seed changes.
- **Self-rooting scripts.** Every helper (`scripts/reseed_reproduce.py`, `scripts/phase*_*.{sh,py}`)
  derives the repo root from its own location, so there are no hardcoded paths.
- **What is committed.** The new scripts, the `run_project.py --reseed` wiring, the docs, and the
  eight small result JSONs in `outputs/seeds/` go into the repo. The large model zips
  (`outputs/models/seeds/`) and the throwaway smoke-check workspace (`outputs/checks/`) are
  `.gitignore`d and regenerate on demand - never commit them.

