"""Phase A2 - Step 4: compare the sequential vs parallel sampled JSON outputs.

The sampled sequential and parallel JSONs must match exactly after canonical JSON
formatting (sort_keys, allow_nan=False). A mismatch means the parallel execution path
is not reproducible -> stop and diagnose before the many-core Phase B run.
Mirrors docs/reseed_teammate_prompt.md Step 4.
"""
import difflib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "outputs" / "checks" / "sequential_vs_parallel"
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
