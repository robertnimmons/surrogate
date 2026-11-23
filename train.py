"""Train the Surrogate MLP end to end: data -> training -> test evaluation -> plots -> summary.

Usage:
    python train.py                                  # synthetic plate-with-hole data, all defaults
    python train.py --data-config data/example_config.json   # train from a CSV
    python train.py --help
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import physics
from src.data import DataConfig, Scaler, load_dataset, split_indices
from src.fea import solve_plate_with_hole
from src.inference import Surrogate
from src.metrics import regression_metrics
from src.model import SurrogateMLP, count_parameters
from src.plotting import plot_loss_curves, plot_parity
from src.trainer import TrainConfig, train
from src.utils import select_device, set_seed, sync

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--data-config", type=Path, default=None,
                   help="JSON file describing a CSV source and its columns (default: synthetic data)")
    g.add_argument("--n-samples", type=int, default=20_000, help="synthetic dataset size")
    g.add_argument("--split", type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    g = p.add_argument_group("model")
    g.add_argument("--hidden", type=int, default=128, help="width of each hidden layer")
    g.add_argument("--layers", type=int, default=3, help="number of hidden layers")
    g.add_argument("--activation", choices=["gelu", "relu"], default="gelu")
    g.add_argument("--dropout", type=float, default=0.0)
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=1000, help="upper bound; early stopping usually ends sooner")
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--patience", type=int, default=50, help="early-stopping patience in epochs")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    g = p.add_argument_group("benchmark")
    g.add_argument("--fea-samples", type=int, default=10,
                   help="test points to re-solve with the FE model for the speedup baseline (0 to skip)")
    g = p.add_argument_group("output")
    g.add_argument("--models-dir", type=Path, default=ROOT / "models")
    g.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return p.parse_args()


def time_inference(surrogate: Surrogate, x_test: np.ndarray, n_single: int = 2000) -> dict[str, float]:
    """Wall-clock inference cost, measured two ways.

    single: one design point at a time, end to end (numpy in -> scale -> model -> unscale -> numpy out).
            This is what a designer calling predict.py in a loop would see.
    batched: the whole test set in one forward pass, divided by the number of rows.
    """
    rows = [dict(zip(surrogate.input_names, r)) for r in x_test[:n_single]]
    for r in rows[:50]:                                   # warm-up (CUDA context, kernel selection)
        surrogate.predict_one(r)
    sync(surrogate.device)
    times = []
    for r in rows:
        t = time.perf_counter()
        surrogate.predict_one(r)
        sync(surrogate.device)
        times.append(time.perf_counter() - t)

    surrogate.predict_array(x_test)
    sync(surrogate.device)
    reps = 20
    t = time.perf_counter()
    for _ in range(reps):
        surrogate.predict_array(x_test)
    sync(surrogate.device)
    batched = (time.perf_counter() - t) / (reps * len(x_test))
    return {"single_median_s": float(np.median(times)), "single_p95_s": float(np.percentile(times, 95)),
            "batched_per_sample_s": batched}


def fea_benchmark(test_df: pd.DataFrame, n: int, seed: int) -> dict[str, float] | None:
    """Re-solve n test points with the in-repo FE model: times it and cross-checks the labels."""
    if n <= 0:
        return None
    pts = test_df.sample(n=min(n, len(test_df)), random_state=seed)
    times, rel_err, n_dof = [], [], 0
    solve_plate_with_hole(100.0, 40.0, 10.0, 50.0)       # warm-up (imports, allocator)
    for _, r in pts.iterrows():
        t = time.perf_counter()
        res = solve_plate_with_hole(r.plate_width_mm, r.hole_diameter_mm, r.thickness_mm, r.load_kn)
        times.append(time.perf_counter() - t)
        rel_err.append(abs(res.max_stress_mpa / r.max_von_mises_mpa - 1))
        n_dof = res.n_dof

    # Closed-form cost, for transparency: the formula itself is faster than any network.
    reps = 20_000
    t = time.perf_counter()
    for _ in range(reps):
        physics.max_stress_mpa(100.0, 40.0, 10.0, 50.0)
    formula_s = (time.perf_counter() - t) / reps
    return {"median_s": float(np.median(times)), "n_points": len(times), "n_dof": n_dof,
            "max_rel_diff_vs_analytical_pct": float(100 * max(rel_err)), "formula_s": formula_s}


def fmt_time(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.1f} µs"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # ---- data -------------------------------------------------------------------------
    data_cfg = DataConfig.from_json(args.data_config) if args.data_config else DataConfig()
    if data_cfg.path and not Path(data_cfg.path).is_absolute() and not Path(data_cfg.path).exists():
        data_cfg.path = str(ROOT / data_cfg.path)        # allow repo-relative paths from any cwd
    df = load_dataset(data_cfg, args.n_samples, args.seed)
    if data_cfg.source == "synthetic":
        df.to_csv(ROOT / "data" / "plate_with_hole.csv", index=False)
    splits = split_indices(len(df), tuple(args.split), args.seed)
    x_all = df[data_cfg.inputs].to_numpy(np.float64)
    y_all = df[data_cfg.targets].to_numpy(np.float64)
    scaler = Scaler.fit(x_all[splits["train"]], y_all[splits["train"]], data_cfg.inputs, data_cfg.targets,
                        data_cfg.target_transform)
    scaler.save(args.models_dir / "scaler.json")
    print(f"[data] {data_cfg.source}: {len(df)} rows, {len(data_cfg.inputs)} inputs -> {data_cfg.targets} | "
          f"train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])}")

    def to_dev(idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(scaler.transform_x(x_all[idx]), dtype=torch.float32, device=device)
        y = torch.tensor(scaler.transform_y(y_all[idx]), dtype=torch.float32, device=device)
        return x, y

    # ---- model + training -------------------------------------------------------------
    model_kwargs = {"n_inputs": len(data_cfg.inputs), "n_outputs": len(data_cfg.targets), "hidden": args.hidden,
                    "n_layers": args.layers, "dropout": args.dropout, "activation": args.activation}
    model = SurrogateMLP(**model_kwargs).to(device)
    n_params = count_parameters(model)
    print(f"[model] MLP {args.layers} x {args.hidden} {args.activation.upper()}, {n_params:,} trainable parameters")

    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            patience=args.patience, log_every=args.log_every)
    ckpt_path = args.models_dir / "best_model.pt"
    try:
        x_tr, y_tr = to_dev(splits["train"])
        x_va, y_va = to_dev(splits["val"])
        result = train(model, x_tr, y_tr, x_va, y_va, train_cfg, ckpt_path,
                       {"model_kwargs": model_kwargs, "data_config": data_cfg.to_dict(), "seed": args.seed},
                       args.seed)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        sys.exit(f"\n[error] CUDA ran out of memory with --batch-size {args.batch_size} and --hidden {args.hidden}.\n"
                 f"        Try: python train.py --batch-size {max(16, args.batch_size // 2)}  "
                 f"(or a smaller --hidden / --n-samples), or close other GPU programs.\n"
                 f"        As a last resort: python train.py --device cpu")
    if device.type == "cuda":
        print(f"[device] peak VRAM allocated during training: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

    # ---- evaluate the best checkpoint on the held-out test set ------------------------
    surrogate = Surrogate.load(args.models_dir, device)
    test_df = df.iloc[splits["test"]].reset_index(drop=True)
    x_test, y_test = x_all[splits["test"]], y_all[splits["test"]]
    y_pred = surrogate.predict_array(x_test)
    metrics = {name: regression_metrics(y_test[:, i], y_pred[:, i]) for i, name in enumerate(data_cfg.targets)}

    pred_df = test_df.copy()
    for i, name in enumerate(data_cfg.targets):
        pred_df[f"{name}_pred"] = y_pred[:, i]
        pred_df[f"{name}_rel_error_pct"] = 100 * np.abs(y_pred[:, i] - y_test[:, i]) / np.abs(y_test[:, i])
    pred_df.to_csv(args.results_dir / "test_predictions.csv", index=False)

    timing = time_inference(surrogate, x_test)
    # The FE baseline only makes sense when the data is the plate-with-hole problem.
    is_plate = set(physics.INPUT_COLUMNS[:4] + ["max_von_mises_mpa"]) <= set(data_cfg.inputs + data_cfg.targets)
    fea = fea_benchmark(test_df, args.fea_samples, args.seed) if is_plate else None

    hist = pd.DataFrame(result.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(args.results_dir / "history.csv", index=False)
    plot_loss_curves(result.history, result.best_epoch, args.results_dir / "loss_curves.png")
    plot_parity(y_test, y_pred, data_cfg.targets, {k: v["r2"] for k, v in metrics.items()},
                log_axes=data_cfg.target_transform == "log", path=args.results_dir / "pred_vs_actual.png")

    # ---- summary ----------------------------------------------------------------------
    hw = torch.cuda.get_device_name(0) if device.type == "cuda" else f"CPU ({platform.processor() or platform.machine()})"
    summary = {
        "hardware": hw, "torch": torch.__version__, "seed": args.seed,
        "dataset_rows": len(df), "split_sizes": {k: len(v) for k, v in splits.items()},
        "model_parameters": n_params, "epochs_run": len(result.history["train_loss"]),
        "best_epoch": result.best_epoch, "best_val_loss_normalised_mse": result.best_val_loss,
        "train_seconds": result.train_seconds, "test_metrics": metrics, "inference": timing, "fea": fea,
    }
    with open(args.results_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    w = 48
    line = "=" * 72
    print(f"\n{line}\n SURROGATE SUMMARY  ({hw})\n{line}")
    print(f" {'Dataset size':<{w}}{len(df):,} rows ({len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])})")
    print(f" {'Model parameters':<{w}}{n_params:,}")
    print(f" {'Training time':<{w}}{fmt_time(result.train_seconds)} ({len(result.history['train_loss'])} epochs, best {result.best_epoch})")
    print(f" {'Best validation loss (normalised MSE)':<{w}}{result.best_val_loss:.3e}")
    for name, m in metrics.items():
        unit = " MPa" if name.endswith("_mpa") else ""
        print(f" --- test: {name}")
        print(f" {'  MAE':<{w}}{m['mae']:.4g}{unit}")
        print(f" {'  RMSE':<{w}}{m['rmse']:.4g}{unit}")
        print(f" {'  Max absolute error':<{w}}{m['max_abs_error']:.4g}{unit}")
        print(f" {'  R²':<{w}}{m['r2']:.6f}")
        print(f" {'  Mean / p99 / max relative error':<{w}}{m['mean_rel_error_pct']:.3f}% / "
              f"{m['p99_rel_error_pct']:.3f}% / {m['max_rel_error_pct']:.3f}%")
    print(f" {'Inference, single point (median, end to end)':<{w}}{fmt_time(timing['single_median_s'])}")
    print(f" {'Inference, batched (per sample)':<{w}}{fmt_time(timing['batched_per_sample_s'])}")
    if fea:
        print(f" {'FE solve, in-repo Q4 model (median)':<{w}}{fmt_time(fea['median_s'])} "
              f"({fea['n_dof']:,} DOF, {fea['n_points']} test points)")
        print(f" {'  FE vs analytical label, max difference':<{w}}{fea['max_rel_diff_vs_analytical_pct']:.2f}%")
        print(f" {'Speedup vs FE, single point':<{w}}{fea['median_s'] / timing['single_median_s']:,.0f}x")
        print(f" {'Speedup vs FE, batched':<{w}}{fea['median_s'] / timing['batched_per_sample_s']:,.0f}x")
        print(f" {'(Closed-form formula, for reference)':<{w}}{fmt_time(fea['formula_s'])}")
    print(line)
    print(f" Saved: {ckpt_path.relative_to(ROOT) if ckpt_path.is_relative_to(ROOT) else ckpt_path}, "
          f"scaler.json, results/loss_curves.png, results/pred_vs_actual.png, "
          f"results/test_predictions.csv, results/history.csv, results/summary.json")


if __name__ == "__main__":
    main()
