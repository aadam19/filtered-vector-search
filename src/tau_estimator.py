import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_ORDER = ["negative", "random", "positive"]
METRIC_COLUMNS = [
    ("true_correlation", "true"),
    ("heuristic_correlation", "heuristic"),
    ("tanh_correlation", "tanh"),
    ("pgap_correlation", "pgap"),
]


def classify_score(score: float, tau: float) -> str:
    if score > tau:
        return "positive"
    if score < -tau:
        return "negative"
    return "random"


def load_comparison_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"query_id", "correlation_type", "true_correlation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    out = df.copy()
    out["intended_class"] = out["correlation_type"].str.lower()
    out = out[out["intended_class"].isin(CLASS_ORDER)].copy()

    metric_cols = [col for col, _ in METRIC_COLUMNS if col in out.columns]
    for col in metric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[np.isfinite(out["true_correlation"])].copy()
    if out.empty:
        raise ValueError(f"No finite true_correlation rows found in {path}")
    return out


def confusion_counts(actual: np.ndarray, predicted: np.ndarray) -> dict:
    counts = {}
    for actual_class in CLASS_ORDER:
        counts[actual_class] = {}
        for predicted_class in CLASS_ORDER:
            counts[actual_class][predicted_class] = int(
                np.sum((actual == actual_class) & (predicted == predicted_class))
            )
    return counts


def macro_recall(actual: np.ndarray, predicted: np.ndarray) -> float:
    recalls = []
    for cls in CLASS_ORDER:
        mask = actual == cls
        denom = int(np.sum(mask))
        if denom == 0:
            continue
        recalls.append(float(np.sum(predicted[mask] == cls)) / float(denom))
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    f1s = []
    for cls in CLASS_ORDER:
        tp = int(np.sum((actual == cls) & (predicted == cls)))
        fp = int(np.sum((actual != cls) & (predicted == cls)))
        fn = int(np.sum((actual == cls) & (predicted != cls)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0.0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else float("nan")


def summarize_class_distribution(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows = []
    for cls in CLASS_ORDER:
        vals = df.loc[df["intended_class"] == cls, metric_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append(
            {
                "class": cls,
                "count": int(len(vals)),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "q10": float(np.quantile(vals, 0.10)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
                "q90": float(np.quantile(vals, 0.90)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        )
    return pd.DataFrame(rows)


def sweep_tau(scores: np.ndarray, actual: np.ndarray, num_points: int) -> pd.DataFrame:
    finite_scores = scores[np.isfinite(scores)]
    max_abs = float(np.max(np.abs(finite_scores))) if len(finite_scores) else 0.0
    taus = np.linspace(0.0, max_abs, max(2, int(num_points)))

    rows = []
    for tau in taus:
        predicted = np.array([classify_score(x, float(tau)) for x in scores], dtype=object)
        accuracy = float(np.mean(predicted == actual))
        bal_acc = macro_recall(actual, predicted)
        macrof1 = macro_f1(actual, predicted)
        rows.append(
            {
                "tau": float(tau),
                "accuracy": accuracy,
                "balanced_accuracy": bal_acc,
                "macro_f1": macrof1,
            }
        )
    return pd.DataFrame(rows)


def pick_best_tau(sweep_df: pd.DataFrame, objective: str) -> float:
    best_value = float(sweep_df[objective].max())
    best_rows = sweep_df[np.isclose(sweep_df[objective], best_value)].copy()
    return float(best_rows.sort_values("tau").iloc[0]["tau"])


def evaluate_metric(df: pd.DataFrame, metric_col: str, objective: str, num_points: int) -> dict:
    metric_df = df[np.isfinite(df[metric_col])].copy()
    if metric_df.empty:
        raise ValueError(f"No finite values found for metric column {metric_col!r}")

    actual = metric_df["intended_class"].to_numpy(dtype=object)
    scores = metric_df[metric_col].to_numpy(dtype=float)
    sweep_df = sweep_tau(scores, actual, num_points=num_points)
    tau = pick_best_tau(sweep_df, objective=objective)
    predicted = np.array([classify_score(x, tau) for x in scores], dtype=object)
    summary_df = summarize_class_distribution(metric_df, metric_col)
    best_row = sweep_df[np.isclose(sweep_df["tau"], tau)].iloc[0]
    return {
        "metric_col": metric_col,
        "tau": tau,
        "actual": actual,
        "predicted": predicted,
        "sweep_df": sweep_df,
        "summary_df": summary_df,
        "accuracy": float(best_row["accuracy"]),
        "balanced_accuracy": float(best_row["balanced_accuracy"]),
        "macro_f1": float(best_row["macro_f1"]),
        "metric_df": metric_df,
    }


def plot_metric_tau_estimation(
    metric_df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    sweep_df: pd.DataFrame,
    tau: float,
    objective: str,
    out_path: Path,
) -> None:
    colors = {"negative": "#d62728", "random": "#2ca02c", "positive": "#1f77b4"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax0, ax1 = axes

    all_vals = metric_df[metric_col].to_numpy(dtype=float)
    bins = np.linspace(float(all_vals.min()), float(all_vals.max()), 30)
    for cls in CLASS_ORDER:
        vals = metric_df.loc[metric_df["intended_class"] == cls, metric_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        ax0.hist(vals, bins=bins, density=True, alpha=0.45, color=colors[cls], label=cls.capitalize())
    ax0.axvline(-tau, color="#444444", linestyle="--", linewidth=1.5, label=f"-tau = {-tau:.4f}")
    ax0.axvline(tau, color="#444444", linestyle="--", linewidth=1.5, label=f"+tau = {tau:.4f}")
    ax0.axvline(0.0, color="#888888", linestyle=":", linewidth=1.2)
    ax0.set_title(f"{metric_label.title()} Distribution by Intended Class")
    ax0.set_xlabel(metric_label.replace("_", " ").title())
    ax0.set_ylabel("Density")
    ax0.legend(frameon=False)
    ax0.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)

    ax1.plot(sweep_df["tau"], sweep_df["accuracy"], label="Accuracy", color="#4c78a8", linewidth=2)
    ax1.plot(sweep_df["tau"], sweep_df["balanced_accuracy"], label="Balanced Accuracy", color="#f58518", linewidth=2)
    ax1.plot(sweep_df["tau"], sweep_df["macro_f1"], label="Macro F1", color="#54a24b", linewidth=2)
    ax1.axvline(tau, color="#222222", linestyle="--", linewidth=1.5, label=f"Chosen tau = {tau:.4f}")
    ax1.set_title(f"{metric_label.title()} Threshold Sweep ({objective.replace('_', ' ').title()})")
    ax1.set_xlabel("Tau")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax1.legend(frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_metric_markdown_summary(
    input_path: Path,
    out_path: Path,
    metric_label: str,
    objective: str,
    result: dict,
) -> None:
    confusion = confusion_counts(result["actual"], result["predicted"])
    lines = [
        "# Tau Estimation Summary",
        "",
        f"- Input: `{input_path}`",
        f"- Metric: `{metric_label}`",
        f"- Objective: `{objective}`",
        f"- Estimated tau: `{result['tau']:.6f}`",
        f"- Accuracy at tau: `{result['accuracy']:.4f}`",
        f"- Balanced accuracy at tau: `{result['balanced_accuracy']:.4f}`",
        f"- Macro F1 at tau: `{result['macro_f1']:.4f}`",
        "",
        "## Class Distribution",
        "",
        "| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result["summary_df"].iterrows():
        lines.append(
            f"| {row['class']} | {int(row['count'])} | {row['mean']:.6f} | {row['median']:.6f} | "
            f"{row['q10']:.6f} | {row['q25']:.6f} | {row['q75']:.6f} | {row['q90']:.6f} | "
            f"{row['min']:.6f} | {row['max']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Confusion Matrix at Estimated Tau",
            "",
            "| Actual \\ Predicted | Negative | Random | Positive |",
            "|---|---:|---:|---:|",
        ]
    )
    for cls in CLASS_ORDER:
        counts = confusion[cls]
        lines.append(f"| {cls} | {counts['negative']} | {counts['random']} | {counts['positive']} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_base(input_path: Path, out_dir: Path | None) -> Path:
    base_dir = out_dir if out_dir is not None else (input_path.parent / "tau")
    stem = input_path.stem.replace("correlation_comparison", "tau_estimation")
    return base_dir / stem


def sanitize_label(label: str) -> str:
    return label.replace(" ", "_").replace("-", "_").lower()


def write_overall_summary(
    input_path: Path,
    out_path: Path,
    objective: str,
    results: list[tuple[str, dict]],
) -> None:
    lines = [
        "# Tau Estimation Overview",
        "",
        f"- Input: `{input_path}`",
        f"- Objective: `{objective}`",
        "",
        "| Metric | Tau | Accuracy | Balanced Accuracy | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric_label, result in results:
        lines.append(
            f"| {metric_label} | {result['tau']:.6f} | {result['accuracy']:.4f} | "
            f"{result['balanced_accuracy']:.4f} | {result['macro_f1']:.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate symmetric taus for correlation metrics against intended classes."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to correlation_comparison_*.csv")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory. Default: same directory as the input CSV.",
    )
    parser.add_argument(
        "--objective",
        choices=["accuracy", "balanced_accuracy", "macro_f1"],
        default="balanced_accuracy",
        help="Objective used to select tau.",
    )
    parser.add_argument(
        "--metric",
        choices=["all", "true", "heuristic", "tanh", "pgap"],
        default="all",
        help="Estimate tau for one metric or all available metrics.",
    )
    parser.add_argument("--num-points", type=int, default=400, help="Number of tau values in the sweep.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None

    df = load_comparison_csv(input_path)
    available = [(col, label) for col, label in METRIC_COLUMNS if col in df.columns]
    if args.metric != "all":
        available = [(col, label) for col, label in available if label == args.metric]
    if not available:
        raise ValueError("No requested metric columns are available in the input CSV.")

    output_base = default_output_base(input_path, out_dir)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    summary_results = []

    for metric_col, metric_label in available:
        result = evaluate_metric(df, metric_col, objective=args.objective, num_points=args.num_points)
        label_tag = sanitize_label(metric_label)

        sweep_csv_path = output_base.parent / f"{output_base.name}_{label_tag}_sweep.csv"
        summary_md_path = output_base.parent / f"{output_base.name}_{label_tag}.md"
        plot_path = output_base.parent / f"{output_base.name}_{label_tag}.png"

        result["sweep_df"].to_csv(sweep_csv_path, index=False)
        plot_metric_tau_estimation(
            result["metric_df"],
            metric_col,
            metric_label,
            result["sweep_df"],
            result["tau"],
            args.objective,
            plot_path,
        )
        write_metric_markdown_summary(
            input_path=input_path,
            out_path=summary_md_path,
            metric_label=metric_label,
            objective=args.objective,
            result=result,
        )

        combined_rows.append(
            {
                "metric": metric_label,
                "tau": result["tau"],
                "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "macro_f1": result["macro_f1"],
            }
        )
        summary_results.append((metric_label, result))

        print(f"[{metric_label}] tau={result['tau']:.6f}")
        print(f"[{metric_label}] saved sweep CSV to {sweep_csv_path}")
        print(f"[{metric_label}] saved summary Markdown to {summary_md_path}")
        print(f"[{metric_label}] saved plot to {plot_path}")

    combined_df = pd.DataFrame(combined_rows)
    combined_csv_path = output_base.parent / f"{output_base.name}_summary.csv"
    combined_md_path = output_base.parent / f"{output_base.name}_summary.md"
    combined_df.to_csv(combined_csv_path, index=False)
    write_overall_summary(input_path, combined_md_path, args.objective, summary_results)
    print(f"Saved overall summary CSV to {combined_csv_path}")
    print(f"Saved overall summary Markdown to {combined_md_path}")


if __name__ == "__main__":
    main()
