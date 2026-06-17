"""Phase D - sanity-check every reseed result file.

For each seed (1, 3, 7, 42, 202606) — all five, including the committed seed 42:
  booking file outputs/seeds/results_rl_booking_seed<s>.json
    - valid JSON, no NaN/Inf
    - train_seed matches, both profiles present, all 3 crowds, all 5 arms with numeric points
    - mppo_int checkins_of_3 in [0, 3] for every profile x crowd cell
  toy file    outputs/seeds/q_learning_seed<s>.json
    - valid JSON, no NaN/Inf
    - seed matches, episodes == 25000, q/double-q mean_return present and finite
Mirrors docs/reseed_plan_updated.pdf Phase D. Exits non-zero on any failure.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1, 3, 7, 42, 202606]
PROFILES = ["art", "tourist"]
CROWDS = ["500", "2500", "5000"]
ARMS = ["ppo_base", "dqn_base", "mppo_int", "ppo_int", "dqn_int"]

problems = []


def nonfinite_paths(obj, path="$"):
    """Yield JSON paths whose value is a non-finite float (NaN/Inf)."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            yield f"{path} = {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from nonfinite_paths(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from nonfinite_paths(v, f"{path}[{i}]")


def check_booking(s):
    f = ROOT / "outputs" / "seeds" / f"results_rl_booking_seed{s}.json"
    if not f.exists():
        problems.append(f"[booking seed {s}] MISSING {f}")
        return
    try:
        data = json.loads(f.read_text())
    except Exception as e:  # noqa: BLE001
        problems.append(f"[booking seed {s}] invalid JSON: {e}")
        return

    bad = list(nonfinite_paths(data))
    if bad:
        problems.append(f"[booking seed {s}] non-finite values: {bad[:5]}")

    if str(data.get("train_seed")) != str(s):
        problems.append(f"[booking seed {s}] train_seed={data.get('train_seed')} != {s}")

    profs = data.get("profiles", {})
    cells_ok = 0
    for prof in PROFILES:
        if prof not in profs:
            problems.append(f"[booking seed {s}] missing profile {prof}")
            continue
        for c in CROWDS:
            if c not in profs[prof]:
                problems.append(f"[booking seed {s}] missing cell {prof}/{c}")
                continue
            arms = profs[prof][c].get("arms", {})
            for a in ARMS:
                pts = arms.get(a, {}).get("points")
                if not isinstance(pts, (int, float)):
                    problems.append(f"[booking seed {s}] {prof}/{c}/{a} points not numeric: {pts}")
            ci = arms.get("mppo_int", {}).get("checkins_of_3")
            if ci is None or not (0 <= ci <= 3):
                problems.append(f"[booking seed {s}] {prof}/{c} mppo_int checkins={ci} not in [0,3]")
            cells_ok += 1
    if cells_ok == 6 and not any(f"seed {s}]" in p for p in problems):
        print(f"[booking seed {s}] OK: 6 cells x 5 arms, mppo_int checkins in [0,3], no NaN/Inf")


def check_toy(s):
    f = ROOT / "outputs" / "seeds" / f"q_learning_seed{s}.json"
    if not f.exists():
        problems.append(f"[toy seed {s}] MISSING {f}")
        return
    try:
        data = json.loads(f.read_text())
    except Exception as e:  # noqa: BLE001
        problems.append(f"[toy seed {s}] invalid JSON: {e}")
        return
    bad = list(nonfinite_paths(data))
    if bad:
        problems.append(f"[toy seed {s}] non-finite values: {bad[:5]}")
    if str(data.get("seed")) != str(s):
        problems.append(f"[toy seed {s}] seed={data.get('seed')} != {s}")
    if data.get("episodes") != 25000:
        problems.append(f"[toy seed {s}] episodes={data.get('episodes')} != 25000")
    for key in ("q_learning_eval", "double_q_learning_eval"):
        mr = data.get(key, {}).get("mean_return")
        if not isinstance(mr, (int, float)) or not math.isfinite(mr):
            problems.append(f"[toy seed {s}] {key}.mean_return invalid: {mr}")
    if not any(f"toy seed {s}]" in p for p in problems):
        q = data["q_learning_eval"]["mean_return"]
        dq = data["double_q_learning_eval"]["mean_return"]
        print(f"[toy seed {s}] OK: q={q:.2f} double_q={dq:.2f}, episodes=25000, no NaN/Inf")


for s in SEEDS:
    check_booking(s)
    check_toy(s)

print()
if problems:
    print("PHASE D FAILED:")
    for p in problems:
        print("  -", p)
    raise SystemExit(1)
print(f"PHASE D PASSED: all {len(SEEDS) * 2} files valid, no NaN/Inf, mppo_int check-ins in [0,3].")
