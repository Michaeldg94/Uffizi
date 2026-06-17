"""Phase B2 - verify cached-model coverage with EXACT filenames.

The reseed driver saves each cell as
    outputs/models/seeds/{arm}_{profile}_{crowd}_seed{seed}.zip
(see scripts/reseed_booking_grid.py train_arm()). We check that all 120 expected
zips exist by exact name. This is stricter than the loose glob patterns in the plan
PDF (e.g. '*500*' also matches 5000), which can produce a false PASS.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "outputs" / "models" / "seeds"

seeds = [1, 3, 7, 202606]
arms = ["ppo_base", "dqn_base", "mppo_int", "ppo_int", "dqn_int"]
profiles = ["art", "tourist"]
crowds = [500, 2500, 5000]

expected = [
    f"{arm}_{prof}_{crowd}_seed{s}.zip"
    for s in seeds for arm in arms for prof in profiles for crowd in crowds
]

missing = [name for name in expected if not (MODELS / name).exists()]
present = len(expected) - len(missing)

print(f"Cached-model coverage: {present}/{len(expected)} expected zips present in {MODELS}")
if missing:
    print("MISSING:")
    for name in missing:
        print(f"  {name}")
    raise SystemExit(f"Cached-model coverage check FAILED: {len(missing)} missing.")
print("Cached model coverage check passed: 120 expected cells found.")
