import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper_funcs import (
    choose_bins,
    compute_cluster_stats,
    compute_correlation_components,
    import_dataset,
)
from tau_interpolator import build_interpolation_table, fit_tau_curve, plot_interpolation


TANH_SCALE = 1.0
CLUSTER_RADIUS_SCALE = 1.3
METRIC_SPECS = [
    {
        "metric_key": "true_correlation",
        "abs_key": "abs_true_correlation",
        "metric_label": "True",
        "metric_token": "true",
        "symbol": "|c_true^null|",
    },
    {
        "metric_key": "heuristic_correlation",
        "abs_key": "abs_heuristic_correlation",
        "metric_label": "Heuristic",
        "metric_token": "heuristic",
        "symbol": "|c_heuristic^null|",
    },
    {
        "metric_key": "tanh_correlation",
        "abs_key": "abs_tanh_correlation",
        "metric_label": "Tanh",
        "metric_token": "tanh",
        "symbol": "|c_tanh^null|",
    },
    {
        "metric_key": "pgap_correlation",
        "abs_key": "abs_pgap_correlation",
        "metric_label": "PGap",
        "metric_token": "pgap",
        "symbol": "|c_pgap^null|",
    },
]


class NumpyFlatL2Index:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, queries, k):
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        max_k = min(int(k), int(len(self.vectors)))
        diff = queries[:, None, :] - self.vectors[None, :, :]
        dists = np.sum(diff * diff, axis=2, dtype=np.float32)
        order = np.argsort(dists, axis=1)[:, :max_k]
        sorted_dists = np.take_along_axis(dists, order, axis=1)
        return sorted_dists, order


def build_cluster_state(vecs, labels):
    class DummyKMeans:
        pass

    fitted_vecs = DummyKMeans()
    fitted_vecs.labels_ = np.asarray(labels, dtype=np.int64).flatten()

    n_clusters = int(np.max(fitted_vecs.labels_)) + 1
    dim = int(vecs.shape[1])
    centroids = np.zeros((n_clusters, dim), dtype=np.float32)

    for cluster_id in range(n_clusters):
        cluster_points = vecs[fitted_vecs.labels_ == cluster_id]
        if len(cluster_points) > 0:
            centroids[cluster_id] = cluster_points.mean(axis=0)

    fitted_vecs.cluster_centers_ = centroids
    return fitted_vecs, centroids


def safe_sel_token(selectivity):
    return f"{float(selectivity):.8f}".rstrip("0").rstrip(".").replace(".", "p")


def compute_selectivity(sorted_values, lo, hi):
    left_idx = np.searchsorted(sorted_values, lo, side="left")
    right_idx = np.searchsorted(sorted_values, hi, side="right")
    return (right_idx - left_idx) / len(sorted_values)


def sample_random_query_range(
    sorted_attr,
    selectivity=0.01,
    rng=None,
    max_tries=64,
    relative_tolerance=0.05,
):
    """Sample a random range on attr while keeping actual selectivity close to target."""
    n_total = len(sorted_attr)
    target = max(1, min(n_total, int(np.floor(selectivity * n_total))))
    if rng is None:
        rng = np.random.default_rng()

    tolerance = max(1.0 / n_total, float(selectivity) * float(relative_tolerance))
    best_range = None
    best_actual_selectivity = None
    best_error = float("inf")

    for _ in range(int(max_tries)):
        start = int(rng.integers(0, n_total - target + 1))
        end = start + target - 1
        query_range = np.array(
            [int(sorted_attr[start]), int(sorted_attr[end])],
            dtype=np.int64,
        )
        actual_selectivity = float(
            compute_selectivity(sorted_attr, query_range[0], query_range[1])
        )
        error = abs(actual_selectivity - float(selectivity))
        if error < best_error:
            best_range = query_range
            best_actual_selectivity = actual_selectivity
            best_error = error
        if error <= tolerance:
            break

    return best_range, float(best_actual_selectivity)


def sampled_true_query_correlation(
    query_range,
    sorted_attr,
    sorted_idx,
    k,
    all_dists,
    n_random_subsets=32,
    rng=None,
):
    """Approximate the true correlation with Monte Carlo random subsets."""
    if int(n_random_subsets) <= 0:
        raise ValueError(
            f"n_random_subsets must be > 0, got {n_random_subsets}"
        )

    lo = int(query_range[0])
    hi = int(query_range[1])
    left = int(np.searchsorted(sorted_attr, lo, side="left"))
    right = int(np.searchsorted(sorted_attr, hi, side="right"))
    subset_size = int(right - left)
    if subset_size <= 0:
        return {
            "true_correlation": float("nan"),
            "abs_true_correlation": float("nan"),
            "expected_random_gk": float("nan"),
            "admissible_gk": float("nan"),
            "subset_size": 0,
            "k_used": 0,
            "monte_carlo_samples": 0,
            "expected_random_gk_stderr": float("nan"),
        }

    all_dists = np.asarray(all_dists, dtype=np.float32)
    admissible_idx = sorted_idx[left:right]
    k_used = min(int(k), subset_size)
    admissible_dists = all_dists[admissible_idx]
    admissible_gk = float(
        np.mean(np.partition(admissible_dists, k_used - 1)[:k_used])
    )

    total = int(all_dists.size)
    if subset_size >= total:
        normalized_true_correlation = (
            0.0 if admissible_gk != 0.0 else float("nan")
        )
        return {
            "true_correlation": normalized_true_correlation,
            "abs_true_correlation": abs(normalized_true_correlation),
            "expected_random_gk": admissible_gk,
            "admissible_gk": admissible_gk,
            "subset_size": subset_size,
            "k_used": k_used,
            "monte_carlo_samples": 1,
            "expected_random_gk_stderr": 0.0,
        }

    if rng is None:
        rng = np.random.default_rng()

    sample_count = int(n_random_subsets)
    sampled_gk = np.empty(sample_count, dtype=np.float64)
    for sample_idx in range(sample_count):
        sampled_idx = rng.choice(
            total,
            size=subset_size,
            replace=False,
            shuffle=False,
        )
        sampled_subset = all_dists[sampled_idx]
        sampled_gk[sample_idx] = float(
            np.mean(np.partition(sampled_subset, k_used - 1)[:k_used])
        )

    expected_random_gk = float(np.mean(sampled_gk))
    if sample_count > 1:
        sample_stderr = float(
            np.std(sampled_gk, ddof=1) / np.sqrt(sample_count)
        )
    else:
        sample_stderr = 0.0

    if expected_random_gk == 0.0:
        normalized_true_correlation = float("nan")
    else:
        normalized_true_correlation = (
            expected_random_gk - admissible_gk
        ) / expected_random_gk

    return {
        "true_correlation": normalized_true_correlation,
        "abs_true_correlation": abs(normalized_true_correlation),
        "expected_random_gk": expected_random_gk,
        "admissible_gk": admissible_gk,
        "subset_size": subset_size,
        "k_used": k_used,
        "monte_carlo_samples": sample_count,
        "expected_random_gk_stderr": sample_stderr,
    }


def write_results_csv(rows, out_path):
    fieldnames = [
        "query_id",
        "range_lo",
        "range_hi",
        "actual_selectivity",
        "subset_size",
        "k_used",
        "true_correlation",
        "abs_true_correlation",
        "heuristic_correlation",
        "abs_heuristic_correlation",
        "tanh_correlation",
        "abs_tanh_correlation",
        "pgap_correlation",
        "abs_pgap_correlation",
        "js_divergence",
        "lift",
        "p_c",
        "p_g",
        "expected_random_gk",
        "admissible_gk",
        "expected_random_gk_stderr",
        "monte_carlo_samples",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(config_rows, metric_rows, top_rows_by_metric, out_path):
    lines = [
        "# Null Correlation Summary",
        "",
        "## Run Configuration",
        "",
        "| Setting | Value |",
        "|---|---:|",
    ]
    for label, value in config_rows:
        lines.append(f"| {label} | {value} |")

    lines.extend(
        [
            "",
            "## Per-Metric Null Quantiles",
            "",
            "| Metric | Finite Results | Median Abs | Q90 | Q95 | Q99 | Max Abs |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        lines.append(
            f"| {row['metric_label']} | {row['finite_results']} | "
            f"{row['median_abs']:.6f} | {row['q90']:.6f} | {row['q95']:.6f} | "
            f"{row['q99']:.6f} | {row['max_abs']:.6f} |"
        )

    for metric_row in metric_rows:
        metric_token = metric_row["metric_token"]
        metric_label = metric_row["metric_label"]
        metric_key = metric_row["metric_key"]
        abs_key = metric_row["abs_key"]
        top_rows = top_rows_by_metric[metric_token]

        lines.extend(
            [
                "",
                f"## Largest Absolute Null Scores: {metric_label}",
                "",
                "| Query ID | abs(score) | score | actual selectivity | subset size |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in top_rows:
            lines.append(
                f"| {row['query_id']} | {row[abs_key]:.6f} | "
                f"{row[metric_key]:.6f} | {row['actual_selectivity']:.6f} | "
                f"{row['subset_size']} |"
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_abs_null_distribution(abs_values, tau_q99, out_path, metric_label, symbol):
    sorted_abs = np.sort(abs_values)
    cdf = np.arange(1, len(sorted_abs) + 1, dtype=np.float64) / float(len(sorted_abs))
    q90 = float(np.quantile(abs_values, 0.90))
    q95 = float(np.quantile(abs_values, 0.95))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.25))
    ax0, ax1 = axes

    ax0.hist(abs_values, bins=40, color="#4c78a8", edgecolor="#1f2933", alpha=0.85)
    ax0.axvline(q90, color="#54a24b", linestyle="--", linewidth=1.6, label=f"Q90 = {q90:.4f}")
    ax0.axvline(q95, color="#f58518", linestyle="--", linewidth=1.6, label=f"Q95 = {q95:.4f}")
    ax0.axvline(tau_q99, color="#e45756", linestyle="--", linewidth=1.8, label=f"Q99 = {tau_q99:.4f}")
    ax0.set_title(f"Absolute Null {metric_label} Correlation")
    ax0.set_xlabel(symbol)
    ax0.set_ylabel("Queries")
    ax0.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax0.legend(frameon=False)

    ax1.plot(sorted_abs, cdf, color="#4c78a8", linewidth=2.0)
    ax1.axhline(0.99, color="#999999", linestyle=":", linewidth=1.2)
    ax1.axvline(tau_q99, color="#e45756", linestyle="--", linewidth=1.8, label=f"Q99 = {tau_q99:.4f}")
    ax1.set_title(f"Empirical CDF of {symbol}")
    ax1.set_xlabel(symbol)
    ax1.set_ylabel("CDF")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax1.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tau_convergence(history_rows, out_path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {
        "true": "#4c78a8",
        "heuristic": "#f58518",
        "tanh": "#54a24b",
        "pgap": "#e45756",
    }

    x = np.asarray(
        [row["processed_queries"] for row in history_rows],
        dtype=np.float64,
    )
    for spec in METRIC_SPECS:
        metric_token = spec["metric_token"]
        y = np.asarray(
            [row.get(metric_token, float("nan")) for row in history_rows],
            dtype=np.float64,
        )
        finite_mask = np.isfinite(y)
        if not np.any(finite_mask):
            continue

        ax.plot(
            x[finite_mask],
            y[finite_mask],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=colors.get(metric_token),
            label=spec["metric_label"],
        )

    ax.set_title("Monte Carlo Tau Estimate vs Processed Queries")
    ax.set_xlabel("Processed Queries")
    ax.set_ylabel("Estimated Tau (Q99 of Absolute Null Score)")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_experiment_dir(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = 1
    while True:
        experiment_dir = base_dir / f"experiment{experiment_id}"
        if not experiment_dir.exists():
            experiment_dir.mkdir()
            return experiment_dir
        experiment_id += 1


def finite_abs_values(rows, abs_key):
    return np.asarray(
        [row[abs_key] for row in rows if np.isfinite(row[abs_key])],
        dtype=np.float64,
    )


def summarize_metric_outputs(results):
    metric_rows = []
    top_rows_by_metric = {}
    tau_by_metric = {}
    abs_values_by_metric = {}

    for spec in METRIC_SPECS:
        abs_values = finite_abs_values(results, spec["abs_key"])
        if abs_values.size == 0:
            continue

        tau_q99 = float(np.quantile(abs_values, 0.99))
        tau_by_metric[spec["metric_token"]] = tau_q99
        abs_values_by_metric[spec["metric_token"]] = abs_values
        metric_rows.append(
            {
                "metric_key": spec["metric_key"],
                "abs_key": spec["abs_key"],
                "metric_label": spec["metric_label"],
                "metric_token": spec["metric_token"],
                "finite_results": int(abs_values.size),
                "median_abs": float(np.median(abs_values)),
                "q90": float(np.quantile(abs_values, 0.90)),
                "q95": float(np.quantile(abs_values, 0.95)),
                "q99": tau_q99,
                "max_abs": float(np.max(abs_values)),
            }
        )
        top_rows_by_metric[spec["metric_token"]] = sorted(
            [row for row in results if np.isfinite(row[spec["abs_key"]])],
            key=lambda row: row[spec["abs_key"]],
            reverse=True,
        )[:10]

    return metric_rows, top_rows_by_metric, tau_by_metric, abs_values_by_metric


def write_tau_history_csv(history_rows, out_path):
    fieldnames = ["processed_queries", "elapsed_seconds"] + [
        spec["metric_token"] for spec in METRIC_SPECS
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


def write_overall_summary_csv(rows, out_path):
    fieldnames = [
        "selectivity",
        "selectivity_token",
        "queries",
        "k_eval",
        "monte_carlo_subsets",
        "progress_every",
        "actual_selectivity_mean",
        "actual_selectivity_min",
        "actual_selectivity_max",
        "elapsed_seconds",
    ] + [
        f"{spec['metric_token']}_tau_q99" for spec in METRIC_SPECS
    ] + [
        "selectivity_dir",
        "results_csv",
        "summary_md",
        "tau_history_csv",
        "tau_convergence_plot",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_overall_summary_markdown(rows, out_path):
    lines = [
        "# Null Tau Summary Across Selectivities",
        "",
        "| Selectivity | Mean Actual | Min Actual | Max Actual | Elapsed (s) | True Q99 | Heuristic Q99 | Tanh Q99 | PGap Q99 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['selectivity']:.6f} | {row['actual_selectivity_mean']:.6f} | "
            f"{row['actual_selectivity_min']:.6f} | {row['actual_selectivity_max']:.6f} | "
            f"{row['elapsed_seconds']:.2f} | {row['true_tau_q99']:.6f} | "
            f"{row['heuristic_tau_q99']:.6f} | {row['tanh_tau_q99']:.6f} | "
            f"{row['pgap_tau_q99']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Selectivity Artifacts",
            "",
            "| Selectivity | Directory | Results CSV | Summary MD | Tau History CSV | Tau Convergence Plot |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['selectivity']:.6f} | {row['selectivity_dir']} | {row['results_csv']} | "
            f"{row['summary_md']} | {row['tau_history_csv']} | {row['tau_convergence_plot']} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_null_interpolation_markdown(
    out_path,
    metric_label,
    degree,
    coeffs,
    points_df,
    interp_df,
):
    lines = [
        "# Null Tau Interpolation Summary",
        "",
        f"- Metric: `{metric_label}`",
        f"- Polynomial degree: `{degree}`",
        f"- Fitted on `{len(points_df)}` selectivity points",
        "",
        "## Fit Coefficients",
        "",
        f"`{coeffs.tolist()}`",
        "",
        "## Observed Tau Points",
        "",
        "| Selectivity | Tau Q99 | Mean Actual | Elapsed (s) | Source |",
        "|---:|---:|---:|---:|---|",
    ]
    for _, row in points_df.iterrows():
        lines.append(
            f"| {row['selectivity']:.6f} | {row['tau']:.6f} | "
            f"{row['actual_selectivity_mean']:.6f} | {row['elapsed_seconds']:.2f} | "
            f"{row['source']} |"
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
            f"| {row['selectivity']:.6f} | {row['log10_selectivity']:.6f} | "
            f"{row['interpolated_tau']:.6f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_all_metric_interpolations(points_by_metric, interp_by_metric, out_path):
    colors = {
        "true": "#4c78a8",
        "heuristic": "#f58518",
        "tanh": "#54a24b",
        "pgap": "#e45756",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.25))
    ax0, ax1 = axes

    for spec in METRIC_SPECS:
        metric_token = spec["metric_token"]
        if metric_token not in points_by_metric or metric_token not in interp_by_metric:
            continue

        color = colors.get(metric_token, "#333333")
        points_df = points_by_metric[metric_token]
        interp_df = interp_by_metric[metric_token]

        ax0.scatter(
            points_df["selectivity"],
            points_df["tau"],
            s=60,
            color=color,
            zorder=3,
        )
        ax0.plot(
            interp_df["selectivity"],
            interp_df["interpolated_tau"],
            color=color,
            linewidth=2.0,
            label=spec["metric_label"],
        )

        ax1.scatter(
            np.log10(points_df["selectivity"]),
            points_df["tau"],
            s=60,
            color=color,
            zorder=3,
        )
        ax1.plot(
            interp_df["log10_selectivity"],
            interp_df["interpolated_tau"],
            color=color,
            linewidth=2.0,
            label=spec["metric_label"],
        )

    ax0.set_xscale("log")
    ax0.set_title("Interpolated Null Tau Curves")
    ax0.set_xlabel("Selectivity")
    ax0.set_ylabel("Tau")
    ax0.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax0.legend(frameon=False)

    ax1.set_title("Interpolated Null Tau Curves vs log10(Selectivity)")
    ax1.set_xlabel("log10(Selectivity)")
    ax1.set_ylabel("Tau")
    ax1.grid(True, linestyle="--", linewidth=0.8, alpha=0.3)
    ax1.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_selectivities(raw_value):
    selectivities = []
    for part in str(raw_value).split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"selectivity must be in (0, 1], got {value}")
        selectivities.append(value)
    if not selectivities:
        raise ValueError("at least one selectivity must be provided")
    return sorted(selectivities)


def run_selectivity_experiment(
    selectivity,
    experiment_dir,
    num_samples,
    k_eval,
    monte_carlo_subsets,
    distance_batch_size,
    progress_every,
    vecs32,
    queries32,
    vec_norms,
    sorted_attr,
    sorted_idx,
    centroid_idx,
    cluster_stats,
    global_hist,
    global_cdf,
    bin_edges,
    sample_seed,
    correlation_seed,
):
    safe_sel = safe_sel_token(selectivity)
    selectivity_dir = experiment_dir / f"sel{safe_sel}"
    selectivity_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tau_null_sel{safe_sel}_q{num_samples}_topk{k_eval}"

    sample_rng = np.random.default_rng(sample_seed)
    correlation_rng = np.random.default_rng(correlation_seed)

    print(
        f"[RUN] Starting null tau estimation for selectivity={selectivity:.6f} "
        f"into {selectivity_dir}",
        flush=True,
    )

    replace_queries = num_samples > len(queries32)
    sampled_indices = sample_rng.choice(
        len(queries32),
        num_samples,
        replace=replace_queries,
    )
    sampled_queries = queries32[sampled_indices]
    sampled_ranges_meta = [
        sample_random_query_range(sorted_attr, selectivity=selectivity, rng=sample_rng)
        for _ in range(num_samples)
    ]
    sampled_ranges = [
        np.asarray(query_range, dtype=np.int64)
        for query_range, _ in sampled_ranges_meta
    ]
    actual_selectivities = np.asarray(
        [actual_sel for _, actual_sel in sampled_ranges_meta],
        dtype=np.float64,
    )

    results = []
    tau_history_rows = []
    start_time = time.perf_counter()

    for batch_start in range(0, num_samples, distance_batch_size):
        batch_end = min(batch_start + distance_batch_size, num_samples)
        q_batch = sampled_queries[batch_start:batch_end]
        q_norms = np.einsum("ij,ij->i", q_batch, q_batch, dtype=np.float32)[:, None]

        dists_batch = q_batch @ vecs32.T
        dists_batch *= -2.0
        dists_batch += vec_norms[None, :]
        dists_batch += q_norms
        np.maximum(dists_batch, 0.0, out=dists_batch)
        np.sqrt(dists_batch, out=dists_batch)

        for row, query_idx in enumerate(range(batch_start, batch_end)):
            true_corr = sampled_true_query_correlation(
                sampled_ranges[query_idx],
                sorted_attr,
                sorted_idx,
                k_eval,
                dists_batch[row],
                n_random_subsets=monte_carlo_subsets,
                rng=correlation_rng,
            )
            comp = compute_correlation_components(
                query_vector=q_batch[row : row + 1],
                query_range=sampled_ranges[query_idx],
                index=centroid_idx,
                clust_stats=cluster_stats,
                glob_hist=global_hist,
                glob_cdf=global_cdf,
                bin_edges=bin_edges,
            )
            lift = float(comp["lift"])
            heuristic_correlation = float(comp["heuristic_correlation"])
            tanh_correlation = float(
                comp["js_divergence"] * np.tanh(lift * TANH_SCALE)
            )
            pgap_correlation = float(comp["pgap_correlation"])
            results.append(
                {
                    "query_id": query_idx,
                    "range_lo": int(sampled_ranges[query_idx][0]),
                    "range_hi": int(sampled_ranges[query_idx][1]),
                    "actual_selectivity": float(actual_selectivities[query_idx]),
                    **true_corr,
                    "heuristic_correlation": heuristic_correlation,
                    "abs_heuristic_correlation": abs(heuristic_correlation),
                    "tanh_correlation": tanh_correlation,
                    "abs_tanh_correlation": abs(tanh_correlation),
                    "pgap_correlation": pgap_correlation,
                    "abs_pgap_correlation": abs(pgap_correlation),
                    "js_divergence": float(comp["js_divergence"]),
                    "lift": lift,
                    "p_c": float(comp["p_c"]),
                    "p_g": float(comp["p_g"]),
                }
            )

        if batch_end == num_samples or batch_end % progress_every == 0:
            elapsed = time.perf_counter() - start_time
            history_row = {
                "processed_queries": int(batch_end),
                "elapsed_seconds": float(elapsed),
            }
            tau_so_far_parts = []
            for spec in METRIC_SPECS:
                finite_abs = finite_abs_values(results, spec["abs_key"])
                tau_so_far = (
                    float(np.quantile(finite_abs, 0.99))
                    if finite_abs.size
                    else float("nan")
                )
                history_row[spec["metric_token"]] = tau_so_far
                tau_so_far_parts.append(
                    f"{spec['metric_token']}={tau_so_far:.6f}"
                )
            tau_history_rows.append(history_row)
            print(
                f"[RUN] sel={selectivity:.6f} | {batch_end}/{num_samples} queries | "
                f"elapsed={elapsed:.2f}s | provisional Q99s: "
                + ", ".join(tau_so_far_parts),
                flush=True,
            )

    elapsed = time.perf_counter() - start_time
    config_rows = [
        ("target selectivity", f"{selectivity:.6f}"),
        ("queries", str(num_samples)),
        ("k_eval", str(k_eval)),
        ("monte carlo subsets", str(monte_carlo_subsets)),
        ("tau snapshot interval", str(progress_every)),
        ("mean actual selectivity", f"{float(np.mean(actual_selectivities)):.6f}"),
        ("min actual selectivity", f"{float(np.min(actual_selectivities)):.6f}"),
        ("max actual selectivity", f"{float(np.max(actual_selectivities)):.6f}"),
        ("elapsed seconds", f"{elapsed:.2f}"),
    ]

    metric_rows, top_rows_by_metric, tau_by_metric, abs_values_by_metric = summarize_metric_outputs(results)

    csv_path = selectivity_dir / f"{stem}.csv"
    md_path = selectivity_dir / f"{stem}.md"
    history_csv_path = selectivity_dir / f"{stem}_tau_convergence.csv"
    convergence_plot_path = selectivity_dir / f"{stem}_tau_convergence.png"

    write_results_csv(results, csv_path)
    write_summary_markdown(config_rows, metric_rows, top_rows_by_metric, md_path)
    if tau_history_rows:
        write_tau_history_csv(tau_history_rows, history_csv_path)
        plot_tau_convergence(tau_history_rows, convergence_plot_path)

    for spec in METRIC_SPECS:
        metric_token = spec["metric_token"]
        if metric_token not in abs_values_by_metric:
            continue
        plot_path = selectivity_dir / f"{stem}_{metric_token}.png"
        plot_abs_null_distribution(
            abs_values_by_metric[metric_token],
            tau_by_metric[metric_token],
            plot_path,
            spec["metric_label"],
            spec["symbol"],
        )
        print(
            f"[DONE] sel={selectivity:.6f} | Q99 {spec['symbol']} = "
            f"{tau_by_metric[metric_token]:.6f} | plot={plot_path}",
            flush=True,
        )

    summary_row = {
        "selectivity": float(selectivity),
        "selectivity_token": safe_sel,
        "queries": int(num_samples),
        "k_eval": int(k_eval),
        "monte_carlo_subsets": int(monte_carlo_subsets),
        "progress_every": int(progress_every),
        "actual_selectivity_mean": float(np.mean(actual_selectivities)),
        "actual_selectivity_min": float(np.min(actual_selectivities)),
        "actual_selectivity_max": float(np.max(actual_selectivities)),
        "elapsed_seconds": float(elapsed),
        "selectivity_dir": str(selectivity_dir),
        "results_csv": str(csv_path),
        "summary_md": str(md_path),
        "tau_history_csv": str(history_csv_path) if tau_history_rows else "",
        "tau_convergence_plot": str(convergence_plot_path) if tau_history_rows else "",
    }
    for spec in METRIC_SPECS:
        summary_row[f"{spec['metric_token']}_tau_q99"] = float(
            tau_by_metric.get(spec["metric_token"], float("nan"))
        )

    print(f"[DONE] sel={selectivity:.6f} | CSV: {csv_path}", flush=True)
    print(f"[DONE] sel={selectivity:.6f} | Summary: {md_path}", flush=True)
    if tau_history_rows:
        print(
            f"[DONE] sel={selectivity:.6f} | Tau convergence plot: "
            f"{convergence_plot_path}",
            flush=True,
        )
    return summary_row


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate null tau thresholds across one or more selectivities and "
            "fit interpolation curves t(s) for the true/heuristic/tanh/pgap metrics."
        )
    )
    parser.add_argument(
        "--selectivities",
        type=str,
        default="0.001,0.01,0.1,0.5",
        help="Comma-separated selectivities to evaluate.",
    )
    parser.add_argument("--num-samples", type=int, default=10000, help="Queries per selectivity.")
    parser.add_argument("--k-eval", type=int, default=100, help="Top-k used by true correlation.")
    parser.add_argument(
        "--monte-carlo-subsets",
        type=int,
        default=32,
        help="Random subsets used per query for true correlation.",
    )
    parser.add_argument(
        "--distance-batch-size",
        type=int,
        default=8,
        help="Queries per distance-computation batch.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Record provisional tau values every N processed queries.",
    )
    parser.add_argument(
        "--interp-degree",
        type=int,
        default=2,
        help="Polynomial degree in log10(selectivity) space.",
    )
    parser.add_argument(
        "--interp-points",
        type=int,
        default=200,
        help="Number of points used to plot each interpolated curve.",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError(f"--num-samples must be > 0, got {args.num_samples}")
    if args.k_eval <= 0:
        raise ValueError(f"--k-eval must be > 0, got {args.k_eval}")
    if args.monte_carlo_subsets <= 0:
        raise ValueError(
            f"--monte-carlo-subsets must be > 0, got {args.monte_carlo_subsets}"
        )
    if args.distance_batch_size <= 0:
        raise ValueError(
            f"--distance-batch-size must be > 0, got {args.distance_batch_size}"
        )
    if args.progress_every <= 0:
        raise ValueError(f"--progress-every must be > 0, got {args.progress_every}")
    if args.interp_points < 2:
        raise ValueError(f"--interp-points must be >= 2, got {args.interp_points}")

    selectivities = parse_selectivities(args.selectivities)
    local_cache_dir = Path("/tmp/fvs_cache")
    local_cache_dir.mkdir(exist_ok=True)

    reports_root = Path(__file__).resolve().parent / "out" / "tau_null_reports"
    experiment_dir = create_experiment_dir(reports_root)
    tau_dir = experiment_dir / "tau"
    tau_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] Loading cached random attributes...", flush=True)
    attr = np.load(local_cache_dir / "rand_cached_attr.npy")
    labels = np.load(local_cache_dir / "cached_labels.npy")
    sorted_idx = np.argsort(attr).astype(np.int64)
    sorted_attr = attr[sorted_idx]

    print("[LOAD] Loading dataset...", flush=True)
    vecs, queries, _ = import_dataset()
    vecs32 = np.asarray(vecs, dtype=np.float32)
    queries32 = np.asarray(queries, dtype=np.float32)
    vec_norms = np.einsum("ij,ij->i", vecs32, vecs32, dtype=np.float32)

    print("[SETUP] Reconstructing cluster state for heuristic null scores...", flush=True)
    fitted_vecs, centroids = build_cluster_state(vecs32, labels)
    centroid_idx = NumpyFlatL2Index(centroids)
    bin_edges = choose_bins(attr)
    cluster_stats, global_hist, global_cdf = compute_cluster_stats(
        fitted_vecs,
        vecs32,
        attr,
        CLUSTER_RADIUS_SCALE,
        bin_edges,
    )

    selectivity_rows = []
    for idx, selectivity in enumerate(selectivities):
        selectivity_rows.append(
            run_selectivity_experiment(
                selectivity=selectivity,
                experiment_dir=experiment_dir,
                num_samples=args.num_samples,
                k_eval=args.k_eval,
                monte_carlo_subsets=args.monte_carlo_subsets,
                distance_batch_size=args.distance_batch_size,
                progress_every=args.progress_every,
                vecs32=vecs32,
                queries32=queries32,
                vec_norms=vec_norms,
                sorted_attr=sorted_attr,
                sorted_idx=sorted_idx,
                centroid_idx=centroid_idx,
                cluster_stats=cluster_stats,
                global_hist=global_hist,
                global_cdf=global_cdf,
                bin_edges=bin_edges,
                sample_seed=1000 + idx,
                correlation_seed=2000 + idx,
            )
        )

    selectivity_rows = sorted(selectivity_rows, key=lambda row: row["selectivity"])
    summary_csv_path = experiment_dir / "tau_null_summary.csv"
    summary_md_path = experiment_dir / "tau_null_summary.md"
    write_overall_summary_csv(selectivity_rows, summary_csv_path)
    write_overall_summary_markdown(selectivity_rows, summary_md_path)

    interp_selectivities = np.logspace(
        np.log10(min(selectivities)),
        np.log10(max(selectivities)),
        int(args.interp_points),
    )
    points_by_metric = {}
    interp_by_metric = {}

    for spec in METRIC_SPECS:
        metric_token = spec["metric_token"]
        point_rows = []
        for row in selectivity_rows:
            tau_value = float(row[f"{metric_token}_tau_q99"])
            if not np.isfinite(tau_value):
                continue
            point_rows.append(
                {
                    "selectivity": float(row["selectivity"]),
                    "tau": tau_value,
                    "actual_selectivity_mean": float(row["actual_selectivity_mean"]),
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                    "source": str(row["summary_md"]),
                }
            )

        if not point_rows:
            continue

        points_df = pd.DataFrame(point_rows).sort_values("selectivity").reset_index(drop=True)
        degree_used = min(int(args.interp_degree), max(0, len(points_df) - 1))
        coeffs, poly = fit_tau_curve(points_df, degree=degree_used)
        interp_df = build_interpolation_table(poly, interp_selectivities)

        points_csv_path = tau_dir / f"tau_points_{metric_token}.csv"
        interp_csv_path = tau_dir / f"tau_interpolation_{metric_token}.csv"
        interp_md_path = tau_dir / f"tau_interpolation_{metric_token}.md"
        interp_plot_path = tau_dir / f"tau_interpolation_{metric_token}.png"

        points_df.to_csv(points_csv_path, index=False)
        interp_df.to_csv(interp_csv_path, index=False)
        plot_interpolation(
            points_df,
            interp_df,
            metric=metric_token,
            degree=degree_used,
            out_path=interp_plot_path,
        )
        write_null_interpolation_markdown(
            interp_md_path,
            spec["metric_label"],
            degree_used,
            coeffs,
            points_df,
            interp_df,
        )
        points_by_metric[metric_token] = points_df
        interp_by_metric[metric_token] = interp_df

        print(f"[DONE] [{metric_token}] observed tau points: {points_csv_path}", flush=True)
        print(f"[DONE] [{metric_token}] interpolated table: {interp_csv_path}", flush=True)
        print(f"[DONE] [{metric_token}] interpolation summary: {interp_md_path}", flush=True)
        print(f"[DONE] [{metric_token}] interpolation plot: {interp_plot_path}", flush=True)

    combined_plot_path = tau_dir / "tau_interpolation_all_metrics.png"
    if points_by_metric and interp_by_metric:
        plot_all_metric_interpolations(points_by_metric, interp_by_metric, combined_plot_path)
        print(f"[DONE] Combined interpolation plot: {combined_plot_path}", flush=True)

    print(f"[DONE] Overall summary CSV: {summary_csv_path}", flush=True)
    print(f"[DONE] Overall summary MD: {summary_md_path}", flush=True)


if __name__ == "__main__":
    main()
