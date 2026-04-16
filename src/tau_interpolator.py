import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SELECTIVITY_RE = re.compile(r"0p\d+(?:_\d+)?|1p0+|1p\d+")


def decode_selectivity(token: str) -> float:
    return float(token.replace("p", "."))


def extract_selectivity(path: Path) -> float:
    match = SELECTIVITY_RE.search(path.stem)
    if not match:
        raise ValueError(f"Could not extract selectivity token from {path}")
    return decode_selectivity(match.group(0))


def find_summary_files(root: Path) -> list[Path]:
    files = sorted(root.glob("**/tau/*_summary.csv"))
    if files:
        return files
    # Fallback for older output layout.
    return sorted(root.glob("**/tau_estimation_*_summary.csv"))


def load_tau_points(root: Path, metric: str) -> pd.DataFrame:
    rows = []
    for path in find_summary_files(root):
        try:
            selectivity = extract_selectivity(path)
        except ValueError:
            continue

        df = pd.read_csv(path)
        required = {"metric", "tau", "accuracy", "balanced_accuracy", "macro_f1"}
        if not required.issubset(df.columns):
            continue

        metric_df = df[df["metric"] == metric].copy()
        if metric_df.empty:
            continue

        row = metric_df.iloc[0]
        rows.append(
            {
                "source": str(path),
                "selectivity": float(selectivity),
                "tau": float(row["tau"]),
                "accuracy": float(row["accuracy"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "macro_f1": float(row["macro_f1"]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No tau summary rows found for metric={metric!r} under {root}")
    out = out.sort_values("selectivity").drop_duplicates(subset=["selectivity"], keep="last")
    return out.reset_index(drop=True)


def fit_tau_curve(points_df: pd.DataFrame, degree: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.log10(points_df["selectivity"].to_numpy(dtype=float))
    y = points_df["tau"].to_numpy(dtype=float)
    deg = min(int(degree), max(0, len(x) - 1))
    coeffs = np.polyfit(x, y, deg=deg)
    return coeffs, np.poly1d(coeffs)


def build_interpolation_table(poly, selectivities: np.ndarray) -> pd.DataFrame:
    log_s = np.log10(selectivities.astype(float))
    tau = poly(log_s)
    return pd.DataFrame(
        {
            "selectivity": selectivities.astype(float),
            "log10_selectivity": log_s,
            "interpolated_tau": tau.astype(float),
        }
    )


def plot_interpolation(
    points_df: pd.DataFrame,
    interp_df: pd.DataFrame,
    metric: str,
    degree: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax0, ax1 = axes

    ax0.scatter(
        points_df["selectivity"],
        points_df["tau"],
        s=70,
        color="#1f77b4",
        label="Observed tau",
        zorder=3,
    )
    ax0.plot(
        interp_df["selectivity"],
        interp_df["interpolated_tau"],
        color="#d62728",
        linewidth=2,
        label=f"Polynomial fit (deg={degree})",
    )
    ax0.set_xscale("log")
    ax0.set_title(f"Tau vs Selectivity ({metric})")
    ax0.set_xlabel("Selectivity")
    ax0.set_ylabel("Tau")
    ax0.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax0.legend(frameon=False)

    ax1.plot(
        interp_df["log10_selectivity"],
        interp_df["interpolated_tau"],
        color="#d62728",
        linewidth=2,
        label="Interpolated tau",
    )
    ax1.scatter(
        np.log10(points_df["selectivity"]),
        points_df["tau"],
        s=70,
        color="#1f77b4",
        label="Observed tau",
        zorder=3,
    )
    ax1.set_title(f"Tau vs log10(Selectivity) ({metric})")
    ax1.set_xlabel("log10(Selectivity)")
    ax1.set_ylabel("Tau")
    ax1.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax1.legend(frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(
    out_path: Path,
    metric: str,
    degree: int,
    points_df: pd.DataFrame,
    coeffs: np.ndarray,
    interp_df: pd.DataFrame,
) -> None:
    lines = [
        "# Tau Interpolation Summary",
        "",
        f"- Metric: `{metric}`",
        f"- Polynomial degree: `{degree}`",
        f"- Fitted on `{len(points_df)}` selectivity points",
        "",
        "## Fit Coefficients",
        "",
        f"`{coeffs.tolist()}`",
        "",
        "## Observed Points",
        "",
        "| Selectivity | Tau | Accuracy | Balanced Accuracy | Macro F1 | Source |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in points_df.iterrows():
        lines.append(
            f"| {row['selectivity']:.6f} | {row['tau']:.6f} | {row['accuracy']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | {row['source']} |"
        )

    lines.extend(
        [
            "",
            "## Interpolated Table",
            "",
            "| Selectivity | log10(Selectivity) | Interpolated Tau |",
            "|---:|---:|---:|",
        ]
    )
    for _, row in interp_df.iterrows():
        lines.append(
            f"| {row['selectivity']:.6f} | {row['log10_selectivity']:.6f} | {row['interpolated_tau']:.6f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolate tau as a function of selectivity from tau summary tables."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="out",
        help="Root directory containing experiment output folders.",
    )
    parser.add_argument(
        "--metric",
        choices=["true", "heuristic", "tanh", "pgap"],
        default="tanh",
        help="Metric whose tau values should be interpolated.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Polynomial degree in log10(selectivity) space.",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="0.001,0.01,0.1,0.5",
        help="Comma-separated selectivities to evaluate in the interpolation table.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory. Default: <root>/tau.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (root / "tau")
    out_dir.mkdir(parents=True, exist_ok=True)

    points_df = load_tau_points(root, metric=args.metric)
    coeffs, poly = fit_tau_curve(points_df, degree=args.degree)

    grid_values = np.array([float(x.strip()) for x in args.grid.split(",") if x.strip()], dtype=float)
    interp_df = build_interpolation_table(poly, grid_values)

    csv_path = out_dir / f"tau_interpolation_{args.metric}.csv"
    md_path = out_dir / f"tau_interpolation_{args.metric}.md"
    plot_path = out_dir / f"tau_interpolation_{args.metric}.png"
    points_csv_path = out_dir / f"tau_points_{args.metric}.csv"

    points_df.to_csv(points_csv_path, index=False)
    interp_df.to_csv(csv_path, index=False)
    plot_interpolation(points_df, interp_df, metric=args.metric, degree=min(args.degree, len(points_df) - 1), out_path=plot_path)
    write_markdown_summary(md_path, args.metric, min(args.degree, len(points_df) - 1), points_df, coeffs, interp_df)

    print(f"Saved observed tau points to {points_csv_path}")
    print(f"Saved interpolated tau table to {csv_path}")
    print(f"Saved interpolation summary to {md_path}")
    print(f"Saved interpolation plot to {plot_path}")


if __name__ == "__main__":
    main()
