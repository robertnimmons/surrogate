"""Predict with a trained surrogate for a single design point.

Examples:
    python predict.py --plate-width-mm 100 --hole-diameter-mm 30 --thickness-mm 8 --load-kn 40 --yield-strength-mpa 350
    python predict.py --json '{"plate_width_mm": 100, "hole_diameter_mm": 30, "thickness_mm": 8, "load_kn": 40, "yield_strength_mpa": 350}'
    python predict.py --json-file point.json

Any input not given on the command line is an error. When the model was trained on the
plate-with-hole problem, the analytical answer is printed next to the prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src import physics
from src.inference import Surrogate

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--json", type=str, default=None, help="inputs as an inline JSON object")
    p.add_argument("--json-file", type=Path, default=None, help="inputs as a JSON file")
    # One flag per known input. Extra inputs from a CSV-trained model can be passed via --json.
    for col in physics.INPUT_COLUMNS:
        p.add_argument(f"--{col.replace('_', '-')}", dest=col, type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir: Path = args.models_dir
    if not (model_dir / "best_model.pt").exists():
        sys.exit(f"No trained model in {model_dir}. Run `python train.py` first.")
    surrogate = Surrogate.load(model_dir)

    row: dict[str, float] = {}
    if args.json_file:
        row.update(json.loads(args.json_file.read_text()))
    if args.json:
        row.update(json.loads(args.json))
    row.update({c: getattr(args, c) for c in physics.INPUT_COLUMNS if getattr(args, c) is not None})

    missing = [n for n in surrogate.input_names if n not in row]
    if missing:
        sys.exit(f"Missing inputs: {missing}. The model expects: {surrogate.input_names}")
    row = {n: float(row[n]) for n in surrogate.input_names}

    is_plate = set(physics.INPUT_COLUMNS) <= set(row)
    if is_plate:
        try:
            physics.validate_geometry(row)
        except ValueError as exc:
            sys.exit(f"Invalid input: {exc}")

    surrogate.predict_one(row)                     # warm-up, so the timing below is steady-state
    t = time.perf_counter()
    pred = surrogate.predict_one(row)
    elapsed = time.perf_counter() - t

    print("Inputs:")
    for k, v in row.items():
        print(f"  {k:<22} {v:g}")
    outside = surrogate.scaler.out_of_range([row[n] for n in surrogate.input_names])
    if outside:
        print(f"\nWARNING: outside the training range for {outside}. This is extrapolation; "
              f"expect larger errors.")

    truth = physics.analytical_targets(row) if is_plate else {}
    print(f"\n{'output':<22} {'surrogate':>12} {'analytical':>12} {'error':>9}")
    for name, value in pred.items():
        if name in truth:
            err = 100 * (value - truth[name]) / truth[name]
            print(f"{name:<22} {value:>12.4g} {truth[name]:>12.4g} {err:>+8.3f}%")
        else:
            print(f"{name:<22} {value:>12.4g} {'n/a':>12} {'':>9}")
    print(f"\nInference time: {elapsed * 1e3:.3f} ms (CPU, single point, after one warm-up call)")


if __name__ == "__main__":
    main()
