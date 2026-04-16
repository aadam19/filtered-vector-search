import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist
import time
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import pandas as pd
import sys
import os
import ctypes
from pathlib import Path
import argparse

acorn_path = os.path.expanduser("~/filtered-vector-search/ACORN/build/faiss/python")
acorn_lib = os.path.expanduser("~/filtered-vector-search/ACORN/build/faiss/libfaiss.so")
acorn_callbacks = os.path.expanduser("~/filtered-vector-search/ACORN/build/faiss/python/libfaiss_python_callbacks.so")
sys.path.insert(0, acorn_path)

if os.path.exists(acorn_lib):
    ctypes.CDLL(acorn_lib, mode=ctypes.RTLD_GLOBAL)
if os.path.exists(acorn_callbacks):
    ctypes.CDLL(acorn_callbacks, mode=ctypes.RTLD_GLOBAL)

import faiss

from helper_funcs import *

out_base = Path("out")
out_base.mkdir(exist_ok=True)

x = 1
while True:
    exp_dir = out_base / f"experiment{x}"
    if not exp_dir.exists():
        exp_dir.mkdir()
        break
    x += 1

print(f"Output directory created at: {exp_dir}")

vecs, queries, gts = import_dataset()

print(f"Dimensions of base: {vecs.shape}    Type: {vecs.dtype}")
print(f"Dimensions of query: {queries.shape}    Type: {queries.dtype}")
print(f"Dimensions of truth: {gts.shape}    Type: {gts.dtype}")

import os
from pathlib import Path

k = np.int32(np.sqrt(vecs.shape[0]))

parser = argparse.ArgumentParser(description="Run Filtered Vector Search Benchmark")
parser.add_argument("--plot-attr", action="store_true", help="Plot the PCA attribute divisions")
parser.add_argument("--sel", required=False, type=float)
args = parser.parse_args()
selectivity = args.sel if args.sel is not None else 0.01
if not (0.0 < selectivity <= 1.0):
    raise ValueError(f"--sel must be in (0, 1], got {selectivity}")
safe_sel = f"{selectivity:.8f}".rstrip("0").rstrip(".").replace(".", "p")
if not safe_sel:
    safe_sel = "0"

local_cache_dir = Path("/tmp/fvs_cache")
local_cache_dir.mkdir(exist_ok=True)

attr_cache_path = local_cache_dir / "cached_attr.npy"
labels_cache_path = local_cache_dir / "cached_labels.npy"
attr_cache_version_path = local_cache_dir / "cached_attr.version"

cached_attr_version = (
    attr_cache_version_path.read_text().strip()
    if attr_cache_version_path.exists()
    else None
)

if (
    attr_cache_path.exists()
    and labels_cache_path.exists()
    and cached_attr_version == ATTRIBUTE_CACHE_VERSION
):
    print(f"Loading pre-computed {ATTRIBUTE_CACHE_VERSION} attributes from {local_cache_dir}...")
    attr = np.load(attr_cache_path)
    labels = np.load(labels_cache_path)
else:
    if attr_cache_path.exists() or labels_cache_path.exists():
        print(
            "Cached correlated attributes are stale or from an older recipe; "
            "regenerating with the KMeans Gaussian-kernel construction..."
        )
    else:
        print("Generating correlated attributes with KMeans Gaussian-kernel construction...")
    attr, labels = generate_correlated_attribute(
        vecs,
        k,
    )
    
    # Force the terminal to print this immediately before it starts saving
    print(f"Saving {attr.nbytes / 1e6:.2f} MB of attributes to local disk...", flush=True)
    
    np.save(attr_cache_path, attr)
    np.save(labels_cache_path, labels)
    attr_cache_version_path.write_text(f"{ATTRIBUTE_CACHE_VERSION}\n")
    
    print("Saved successfully!", flush=True)
    
# ==========================================
# Build the global exact index ONCE
# ==========================================
print("Building global exact FAISS index...")
exact_index = faiss.IndexFlatL2(int(vecs.shape[1]))
exact_index.add(vecs.astype(np.float32))

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run Filtered Vector Search Benchmark")
parser.add_argument("--plot-attr", action="store_true", help="Plot the PCA attribute divisions")
parser.add_argument("--sel", required=False, type=float)
parser.add_argument(
    "--pca-dim",
    type=int,
    choices=[2, 3],
    default=2,
    help="Number of PCA components to use when plotting attributes (2 or 3)",
)
args = parser.parse_args()
selectivity = args.sel if args.sel is not None else 0.01
if not (0.0 < selectivity <= 1.0):
    raise ValueError(f"--sel must be in (0, 1], got {selectivity}")

print("Sampling data for neighbor stats...")
rng = np.random.default_rng(42)
sample_idx = rng.choice(vecs.shape[0], size=10000, replace=False)

sample_vecs = vecs[sample_idx]
sample_attr = attr[sample_idx]
sample_labels = labels[sample_idx]

stats = compute_neighbor_stats(sample_vecs, sample_attr, attr, exact_index)
print(f"Average attribute diff with neighbors: {stats['neighbor_diff_mean']}")

if args.plot_attr:
    print("Running PCA and plotting correlated attributes...")
    sample_vecs_pca2 = PCA(n_components=2).fit_transform(sample_vecs)
    sample_vecs_pca3 = PCA(n_components=3).fit_transform(sample_vecs)
    plot_attribute_pca_pair(
        sample_vecs_pca2,
        sample_vecs_pca3,
        sample_attr,
        save_path=exp_dir / "correlated_attr_pca_pair.png",
        show=False,
    )
    plot_attribute_stats(
        sample_attr,
        stats["neighbor_mean"],
        sample_labels,
        save_path=exp_dir / "correlated_attr_stats.png",
        show=False,
    )
#####################################
rand_attr_cache_path = local_cache_dir / "rand_cached_attr.npy"

if rand_attr_cache_path.exists():
    print(f"Loading pre-computed random attributes from {local_cache_dir}...")
    # Load into rand_attr so we don't overwrite the correlated ones
    rand_attr = np.load(rand_attr_cache_path)
    rand_labels = None
else:
    print("Generating random attributes (this should be instant)...")
    # Call the random function
    rand_attr, rand_labels = generate_random_attribute(vecs)
    
    print(f"Saving random attributes to local disk...", flush=True)
    np.save(rand_attr_cache_path, rand_attr)
    print("Saved successfully!", flush=True)
    
rand_sample_attr = rand_attr[sample_idx]
rand_sample_labels = None  # Random doesn't use cluster labels

rand_stats = compute_neighbor_stats(sample_vecs, rand_sample_attr, rand_attr, exact_index)
print(f"Average attribute diff with neighbors: {rand_stats['neighbor_diff_mean']}")

if args.plot_attr:
    print("Running PCA and plotting random attributes...")
    plot_attribute_pca_pair(
        sample_vecs_pca2,
        sample_vecs_pca3,
        rand_sample_attr,
        save_path=exp_dir / "random_attr_pca_pair.png",
        show=False,
    )
    plot_attribute_stats(
        rand_sample_attr,
        rand_stats["neighbor_mean"],
        rand_sample_labels,
        save_path=exp_dir / "random_attr_stats.png",
        show=False,
    )

print("Generating exact query ranges...")
pos_ranges, neg_ranges = generate_query_ranges(queries, attr, exact_index, k, selectivity=selectivity)
rand_ranges = generate_random_query_ranges(rand_attr, len(queries), selectivity=selectivity)

pos_overlap = topk_interval_overlap(exact_index, queries, attr, pos_ranges, topk=100)
neg_overlap = topk_interval_overlap(exact_index, queries, attr, neg_ranges, topk=100)
rand_overlap = topk_interval_overlap(exact_index, queries, rand_attr, rand_ranges, topk=100)
print(
    "Top-100 neighbor interval overlap | "
    f"positive={pos_overlap.mean():.4f}, "
    f"negative={neg_overlap.mean():.4f}, "
    f"random={rand_overlap.mean():.4f}"
)

df_pos = pd.DataFrame({
    "query_id": range(len(queries)),
    "range": pos_ranges,
    "label": "positive",
    "top100_overlap": pos_overlap,
})

df_neg = pd.DataFrame({
    "query_id": range(len(queries)),
    "range": neg_ranges,
    "label": "negative",
    "top100_overlap": neg_overlap,
})

df_rand = pd.DataFrame({
    "query_id": range(len(queries)),
    "range": rand_ranges,
    "label": "random",
    "top100_overlap": rand_overlap,
})

df = pd.concat([df_pos, df_neg, df_rand], ignore_index=True)

print(df)

# ==========================================
# INSTANT KMEANS RECONSTRUCTION
# ==========================================
print("\nReconstructing clusters from cached labels...")

class DummyKMeans: pass
fitted_vecs = DummyKMeans()
# Use the labels we already loaded from cache at the top of the script!
fitted_vecs.labels_ = labels.flatten()

# Calculate the centroids instantly using basic math
k_int = int(k)
d = int(vecs.shape[1])
centroids = np.zeros((k_int, d), dtype=np.float32)

for i in range(k_int):
    cluster_points = vecs[fitted_vecs.labels_ == i]
    if len(cluster_points) > 0:
        centroids[i] = cluster_points.mean(axis=0)

fitted_vecs.cluster_centers_ = centroids

threshold = 1.3 # 10% larger radius

print("Computing cluster stats (Correlated)...")
bin_edges = choose_bins(attr)
cluster_stats, global_hist, global_cdf = compute_cluster_stats(fitted_vecs, vecs, attr, threshold, bin_edges)

print("Computing cluster stats (Random)...")
rand_bin_edges = choose_bins(rand_attr)
rand_stats, rand_glob_hist, rand_glob_cdf = compute_cluster_stats(fitted_vecs, vecs, rand_attr, threshold, rand_bin_edges)

print("Building HNSW Centroid Index...")
d = centroids.shape[1]
M = 32
centroid_idx = faiss.IndexHNSWFlat(d, M)
centroid_idx.add(centroids)

comparison_queries = 10
comparison_rng = np.random.default_rng(42)
comparison_count = min(comparison_queries, len(queries))
comparison_idx = np.sort(
    comparison_rng.choice(len(queries), size=comparison_count, replace=False)
)
qs = queries[comparison_idx]
pos = [pos_ranges[int(i)] for i in comparison_idx]
neg = [neg_ranges[int(i)] for i in comparison_idx]
rand = [rand_ranges[int(i)] for i in comparison_idx]

print("Sorting attributes...")
sorted_attr = np.sort(attr)
sorted_attr_idx = np.argsort(attr).astype(np.int64)
sorted_random_attr = np.sort(rand_attr)
sorted_random_attr_idx = np.argsort(rand_attr).astype(np.int64)
vecs_f32 = np.asarray(vecs, dtype=np.float32)
print('Computing correlations...')
tanh_scale = 1.0
comparison_rows = []
for idx, q in enumerate(qs):
    neg_comp = compute_correlation_components(q, neg[idx], centroid_idx, cluster_stats, global_hist, global_cdf, bin_edges)
    pos_comp = compute_correlation_components(q, pos[idx], centroid_idx, cluster_stats, global_hist, global_cdf, bin_edges)
    rand_comp = compute_correlation_components(q, rand[idx], centroid_idx, rand_stats, rand_glob_hist, rand_glob_cdf, rand_bin_edges)
    c, j, l = neg_comp["heuristic_correlation"], neg_comp["js_divergence"], neg_comp["lift"]
    corr, js, lift = pos_comp["heuristic_correlation"], pos_comp["js_divergence"], pos_comp["lift"]
    rand_corr, rand_js, rand_lift = rand_comp["heuristic_correlation"], rand_comp["js_divergence"], rand_comp["lift"]
    neg_tanh_corr = j * np.tanh(l* tanh_scale)
    pos_tanh_corr = js * np.tanh(lift * tanh_scale)
    rand_tanh_corr = rand_js * np.tanh(tanh_scale * rand_lift)
    neg_pgap_corr = neg_comp["pgap_correlation"]
    pos_pgap_corr = pos_comp["pgap_correlation"]
    rand_pgap_corr = rand_comp["pgap_correlation"]
    all_dists = np.linalg.norm(vecs_f32 - np.asarray(q, dtype=np.float32).reshape(1, -1), axis=1)
    sorted_all = np.sort(all_dists.astype(np.float64))
    neg_true = true_query_correlation(q, neg[idx], vecs, attr, k=100, all_dists=all_dists, sorted_all=sorted_all)
    pos_true = true_query_correlation(q, pos[idx], vecs, attr, k=100, all_dists=all_dists, sorted_all=sorted_all)
    rand_true = true_query_correlation(q, rand[idx], vecs, rand_attr, k=100, all_dists=all_dists, sorted_all=sorted_all)

    print(f'Query {idx}:')
    print(
        f'    Positive | Heuristic Corr: {corr}     Tanh Corr: {pos_tanh_corr}     PGap Corr: {pos_pgap_corr}     JS: {js}    Lift: {lift}    '
        f'True Corr: {pos_true["true_correlation"]}    '
        f'E[g_k(R_p)]: {pos_true["expected_random_gk"]}    g_k(X_p): {pos_true["admissible_gk"]}'
    )
    print(
        f'    Negative | Heuristic Corr: {c}     Tanh Corr: {neg_tanh_corr}     PGap Corr: {neg_pgap_corr}     JS: {j}    Lift: {l}    '
        f'True Corr: {neg_true["true_correlation"]}    '
        f'E[g_k(R_p)]: {neg_true["expected_random_gk"]}    g_k(X_p): {neg_true["admissible_gk"]}'
    )
    print(
        f'    Random   | Heuristic Corr: {rand_corr}     Tanh Corr: {rand_tanh_corr}     PGap Corr: {rand_pgap_corr}     JS: {rand_js}    Lift: {rand_lift}    '
        f'True Corr: {rand_true["true_correlation"]}    '
        f'E[g_k(R_p)]: {rand_true["expected_random_gk"]}    g_k(X_p): {rand_true["admissible_gk"]}'
    )
    print('')
    print("correlated pos range selectivity:", compute_selectivity(sorted_attr, pos[idx][0], pos[idx][1]))
    print("correlated neg range selectivity:", compute_selectivity(sorted_attr, neg[idx][0], neg[idx][1]))
    print("random range selectivity:", compute_selectivity(sorted_random_attr, rand[idx][0], rand[idx][1]))
    comparison_rows.extend(
        [
            {
                "query_id": idx,
                "correlation_type": "Positive",
                "heuristic_correlation": corr,
                "tanh_correlation": pos_tanh_corr,
                "pgap_correlation": pos_pgap_corr,
                "js_divergence": js,
                "lift": lift,
                "p_c": pos_comp["p_c"],
                "p_g": pos_comp["p_g"],
                "true_correlation": pos_true["true_correlation"],
                "expected_random_gk": pos_true["expected_random_gk"],
                "admissible_gk": pos_true["admissible_gk"],
                "subset_size": pos_true["subset_size"],
                "k_used": pos_true["k_used"],
            },
            {
                "query_id": idx,
                "correlation_type": "Negative",
                "heuristic_correlation": c,
                "tanh_correlation": neg_tanh_corr,
                "pgap_correlation": neg_pgap_corr,
                "js_divergence": j,
                "lift": l,
                "p_c": neg_comp["p_c"],
                "p_g": neg_comp["p_g"],
                "true_correlation": neg_true["true_correlation"],
                "expected_random_gk": neg_true["expected_random_gk"],
                "admissible_gk": neg_true["admissible_gk"],
                "subset_size": neg_true["subset_size"],
                "k_used": neg_true["k_used"],
            },
            {
                "query_id": idx,
                "correlation_type": "Random",
                "heuristic_correlation": rand_corr,
                "tanh_correlation": rand_tanh_corr,
                "pgap_correlation": rand_pgap_corr,
                "js_divergence": rand_js,
                "lift": rand_lift,
                "p_c": rand_comp["p_c"],
                "p_g": rand_comp["p_g"],
                "true_correlation": rand_true["true_correlation"],
                "expected_random_gk": rand_true["expected_random_gk"],
                "admissible_gk": rand_true["admissible_gk"],
                "subset_size": rand_true["subset_size"],
                "k_used": rand_true["k_used"],
            },
        ]
    )

comparison_df = pd.DataFrame(comparison_rows)
comparison_csv_path = exp_dir / f"correlation_comparison_{safe_sel}.csv"
comparison_md_path = exp_dir / f"correlation_comparison_{safe_sel}.md"
comparison_plot_path = exp_dir / f"correlation_comparison_{safe_sel}.png"
classification_csv_path = exp_dir / f"correlation_classification_{safe_sel}.csv"
classification_md_path = exp_dir / f"correlation_classification_{safe_sel}.md"
classification_matrix_csv_path = exp_dir / f"correlation_classification_matrices_{safe_sel}.csv"
comparison_df.to_csv(comparison_csv_path, index=False)
comparison_md_lines = [
    "| Query ID | Correlation Type | Heuristic Corr | Tanh Corr | PGap Corr | JS Divergence | Lift | p_c | p_g | True Corr | E[g_k(R_p)] | g_k(X_p) | Subset Size | k Used |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in comparison_df.itertuples(index=False):
    comparison_md_lines.append(
        f"| {row.query_id} | {row.correlation_type} | {row.heuristic_correlation:.6f} | "
        f"{row.tanh_correlation:.6f} | {row.pgap_correlation:.6f} | {row.js_divergence:.6f} | {row.lift:.6f} | "
        f"{row.p_c:.6f} | {row.p_g:.6f} | {row.true_correlation:.6f} | {row.expected_random_gk:.6f} | "
        f"{row.admissible_gk:.6f} | {row.subset_size} | {row.k_used} |"
    )
comparison_md_path.write_text("\n".join(comparison_md_lines) + "\n", encoding="utf-8")
print("\nCorrelation comparison table:")
print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print(f"Saved correlation comparison CSV to {comparison_csv_path}")
print(f"Saved correlation comparison Markdown to {comparison_md_path}")

def interpolated_tanh_tau(selectivity):
    log_sel = np.log10(float(selectivity))
    return (
        -0.03652986870695888 * (log_sel ** 2)
        - 0.12008418389936595 * log_sel
        - 0.0020866309341351086
    )


tau = float(interpolated_tanh_tau(selectivity))

def classify_score(score, tau):
    if score > tau:
        return "positive"
    if score < -tau:
        return "negative"
    return "random"

metric_columns = [
    ("heuristic_correlation", "heuristic"),
    ("tanh_correlation", "tanh"),
    ("pgap_correlation", "pgap"),
]

comparison_df["theoretical_class"] = comparison_df["true_correlation"].apply(lambda x: classify_score(x, tau))

classification_rows = []
class_order = ["negative", "random", "positive"]
for metric_col, metric_name in metric_columns:
    predicted_classes = comparison_df[metric_col].apply(lambda x: classify_score(x, tau))
    metric_accuracy = float((predicted_classes == comparison_df["theoretical_class"]).mean())
    for actual_class in class_order:
        for predicted_class in class_order:
            count = int(
                (
                    (comparison_df["theoretical_class"] == actual_class)
                    & (predicted_classes == predicted_class)
                ).sum()
            )
            classification_rows.append(
                {
                    "metric": metric_name,
                    "tau": tau,
                    "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "count": count,
                    "metric_accuracy": metric_accuracy,
                }
            )

classification_df = pd.DataFrame(classification_rows)
classification_df.to_csv(classification_csv_path, index=False)

matrix_rows = []
for _, metric_name in metric_columns:
    metric_df = classification_df[classification_df["metric"] == metric_name]
    accuracy = metric_df["metric_accuracy"].iloc[0] if not metric_df.empty else float("nan")
    for actual_class in class_order:
        row = {"metric": metric_name, "tau": tau, "actual_class": actual_class, "metric_accuracy": accuracy}
        for predicted_class in class_order:
            count = int(
                metric_df[
                    (metric_df["actual_class"] == actual_class)
                    & (metric_df["predicted_class"] == predicted_class)
                ]["count"].iloc[0]
            )
            row[predicted_class] = count
        matrix_rows.append(row)

classification_matrix_df = pd.DataFrame(matrix_rows)
classification_matrix_df.to_csv(classification_matrix_csv_path, index=False)

classification_md_lines = [
    f"Classification threshold tau = {tau:.2f}",
    "",
]
for _, metric_name in metric_columns:
    metric_df = classification_df[classification_df["metric"] == metric_name]
    accuracy = metric_df["metric_accuracy"].iloc[0] if not metric_df.empty else float("nan")
    classification_md_lines.extend(
        [
            f"## {metric_name}",
            f"",
            f"Accuracy: {accuracy:.4f}",
            "",
            "| Actual \\ Predicted | Negative | Random | Positive |",
            "|---|---:|---:|---:|",
        ]
    )
    for actual_class in class_order:
        row = metric_df[metric_df["actual_class"] == actual_class]
        counts = {
            predicted_class: int(
                row[row["predicted_class"] == predicted_class]["count"].iloc[0]
            )
            for predicted_class in class_order
        }
        classification_md_lines.append(
            f"| {actual_class} | {counts['negative']} | {counts['random']} | {counts['positive']} |"
        )
    classification_md_lines.append("")

classification_md_path.write_text("\n".join(classification_md_lines), encoding="utf-8")
print("\nCorrelation classification table:")
print(classification_df.to_string(index=False))
print(f"Saved correlation classification CSV to {classification_csv_path}")
print(f"Saved correlation classification matrix CSV to {classification_matrix_csv_path}")
print(f"Saved correlation classification Markdown to {classification_md_path}")

fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(22, 6))
comparison_colors = {
    "Positive": "#1f77b4",
    "Negative": "#d62728",
    "Random": "#2ca02c",
}
plot_specs = [
    ("heuristic_correlation", "Theoretical vs Heuristic Correlation", "Heuristic Correlation"),
    ("tanh_correlation", f"Theoretical vs JS*tanh({tanh_scale:g}*lift)", "Tanh Correlation"),
    ("pgap_correlation", "Theoretical vs JS*(p_c - p_g)", "PGap Correlation"),
]
for ax_cmp, (col, title, ylabel) in zip(axes_cmp, plot_specs):
    for corr_name in ["Positive", "Negative", "Random"]:
        subset = comparison_df[comparison_df["correlation_type"] == corr_name]
        if subset.empty:
            continue
        ax_cmp.scatter(
            subset["true_correlation"],
            subset[col],
            s=64,
            alpha=0.85,
            color=comparison_colors[corr_name],
            label=corr_name,
        )

    x_vals = comparison_df["true_correlation"].to_numpy(dtype=float)
    y_vals = comparison_df[col].to_numpy(dtype=float)
    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    if np.any(finite_mask):
        x_vals = x_vals[finite_mask]
        y_vals = y_vals[finite_mask]
        x_min = float(np.min(x_vals))
        x_max = float(np.max(x_vals))
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        lo = min(x_min, y_min)
        hi = max(x_max, y_max)
        if lo == hi:
            pad = 1.0 if lo == 0 else abs(lo) * 0.1
            lo -= pad
            hi += pad
        ax_cmp.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="black", alpha=0.6, label="y = x")
        ax_cmp.set_xlim(lo, hi)
        ax_cmp.set_ylim(lo, hi)

        if len(x_vals) >= 2:
            pearson = float(np.corrcoef(x_vals, y_vals)[0, 1])
            ax_cmp.text(
                0.03,
                0.97,
                f"Pearson r = {pearson:.4f}",
                transform=ax_cmp.transAxes,
                ha="left",
                va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            )

    ax_cmp.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.5)
    ax_cmp.axvline(0.0, color="#888888", linewidth=1.0, alpha=0.5)
    ax_cmp.set_title(title)
    ax_cmp.set_xlabel("True Correlation")
    ax_cmp.set_ylabel(ylabel)
    ax_cmp.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax_cmp.legend(frameon=False)
fig_cmp.tight_layout()
fig_cmp.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
plt.close(fig_cmp)
print(f"Saved correlation comparison plot to {comparison_plot_path}")

# Setup indexes

## PRE-FILTER
print('Building or loading indexes...')

## 1. C++ Pre/Post indexes
cpp_corr_index_dir = ensure_cpp_index(attr, tag="correlated")
cpp_rand_index_dir = ensure_cpp_index(rand_attr, tag="random")

## 2. ACORN Indexes
acorn_cache_path = local_cache_dir / "acorn_correlated.faiss"
acorn_rand_cache_path = local_cache_dir / "acorn_random.faiss"

if acorn_cache_path.exists() and acorn_rand_cache_path.exists():
    print("Loading pre-built ACORN indexes from disk...")
    # FAISS requires string paths, not pathlib objects
    acorn_index = faiss.read_index(str(acorn_cache_path))
    acorn_index_rand = faiss.read_index(str(acorn_rand_cache_path))
else:
    print("Building ACORN indexes (this will take a while)...")
    acorn_index, _ = build_acorn_index(vecs, attr)
    acorn_index_rand, _ = build_acorn_index(vecs, rand_attr)
    print("Saving ACORN indexes to disk...")
    faiss.write_index(acorn_index, str(acorn_cache_path))
    faiss.write_index(acorn_index_rand, str(acorn_rand_cache_path))

# Set up ground truths
print('Computing or loading ground truths...')
k_eval = 100

gt_cache_dir = local_cache_dir / "ground_truths"
gt_cache_dir.mkdir(exist_ok=True)

sel_tag = safe_sel
if not sel_tag:
    sel_tag = "0"
gt_cache_path = gt_cache_dir / f"ground_truth_{sel_tag}.npz"

if gt_cache_path.exists():
    print("Loading pre-computed ground truths from disk...")
    # np.load for .npz returns a dictionary-like object
    loaded_gts = np.load(gt_cache_path, allow_pickle=True)
    gt_pos = loaded_gts['gt_pos']
    gt_neg = loaded_gts['gt_neg']
    gt_rand = loaded_gts['gt_rand']
else:
    print("Computing ground truths (this will be fast now)...")
    gt_pos  = compute_ground_truth(vecs, attr, pos_ranges, queries, k_eval, exact_index=exact_index)
    gt_neg  = compute_ground_truth(vecs, attr, neg_ranges, queries, k_eval, exact_index=exact_index)
    gt_rand = compute_ground_truth(vecs, rand_attr, rand_ranges, queries, k_eval, exact_index=exact_index)
    
    print("Saving ground truths to disk...")
    # Ground-truth rows can be ragged when a filter matches < k vectors.
    # Save as object arrays so low-selectivity runs do not fail.
    np.savez(
        gt_cache_path,
        gt_pos=np.array(gt_pos, dtype=object),
        gt_neg=np.array(gt_neg, dtype=object),
        gt_rand=np.array(gt_rand, dtype=object),
    )
    
    
print('Running experiments...')

def print_result(dt, qps, recall):
    print(f'         Finished: {dt:.6f}s | QPS: {qps:.2f} | Recall: {recall:.4f}')

print('Positive:')
print('     ACORN: Running...', flush=True)
pos_dt_acorn, pos_qps_acorn, pos_recall_acorn = run_acorn(acorn_index, queries, attr, pos_ranges, k_eval, gt_pos)
print_result(pos_dt_acorn, pos_qps_acorn, pos_recall_acorn)
print('     Post-filter: Running...', flush=True)
pos_dt_post, pos_qps_post, pos_recall_post = run_post(cpp_corr_index_dir, queries, attr, sorted_attr, sorted_attr_idx, pos_ranges, k_eval, gt_pos)
print_result(pos_dt_post, pos_qps_post, pos_recall_post)
print('     Pre-filter: Running...', flush=True)
pos_dt_pre, pos_qps_pre, pos_recall_pre = run_pre(cpp_corr_index_dir, sorted_attr, sorted_attr_idx, queries, attr, pos_ranges, k_eval, gt_pos)
print_result(pos_dt_pre, pos_qps_pre, pos_recall_pre)

print('Negative:')
print('     ACORN: Running...', flush=True)
neg_dt_acorn, neg_qps_acorn, neg_recall_acorn = run_acorn(acorn_index, queries, attr, neg_ranges, k_eval, gt_neg)
print_result(neg_dt_acorn, neg_qps_acorn, neg_recall_acorn)
print('     Post-filter: Running...', flush=True)
neg_dt_post, neg_qps_post, neg_recall_post = run_post(cpp_corr_index_dir, queries, attr, sorted_attr, sorted_attr_idx, neg_ranges, k_eval, gt_neg)
print_result(neg_dt_post, neg_qps_post, neg_recall_post)
print('     Pre-filter: Running...', flush=True)
neg_dt_pre, neg_qps_pre, neg_recall_pre = run_pre(cpp_corr_index_dir, sorted_attr, sorted_attr_idx, queries, attr, neg_ranges, k_eval, gt_neg)
print_result(neg_dt_pre, neg_qps_pre, neg_recall_pre)

print('Random:')
print('     ACORN: Running...', flush=True)
rand_dt_acorn, rand_qps_acorn, rand_recall_acorn = run_acorn(acorn_index_rand, queries, rand_attr, rand_ranges, k_eval, gt_rand)
print_result(rand_dt_acorn, rand_qps_acorn, rand_recall_acorn)
print('     Post-filter: Running...', flush=True)
rand_dt_post, rand_qps_post, rand_recall_post = run_post(cpp_rand_index_dir, queries, rand_attr, sorted_random_attr, sorted_random_attr_idx, rand_ranges, k_eval, gt_rand)
print_result(rand_dt_post, rand_qps_post, rand_recall_post)
print('     Pre-filter: Running...', flush=True)
rand_dt_pre, rand_qps_pre, rand_recall_pre = run_pre(cpp_rand_index_dir, sorted_random_attr, sorted_random_attr_idx, queries, rand_attr, rand_ranges, k_eval, gt_rand)
print_result(rand_dt_pre, rand_qps_pre, rand_recall_pre)

correlations = ["Positive", "Negative", "Random"]
strategies = ["ACORN", "Post-filter", "Pre-filter"]
# Color-blind-friendly palette: ACORN=blue, Post=red, Pre=green
strategy_colors = ["#1f77b4", "#d62728", "#2ca02c"]

# Recall
recall = np.array([
    [pos_recall_acorn, neg_recall_acorn, rand_recall_acorn],
    [pos_recall_post,  neg_recall_post,  rand_recall_post],
    [pos_recall_pre,   neg_recall_pre,   rand_recall_pre],
])

# QPS
qps = np.array([
    [pos_qps_acorn, neg_qps_acorn, rand_qps_acorn],
    [pos_qps_post,  neg_qps_post,  rand_qps_post],
    [pos_qps_pre,   neg_qps_pre,   rand_qps_pre],
])

# Time
times = np.array([
    [pos_dt_acorn, neg_dt_acorn, rand_dt_acorn],
    [pos_dt_post,  neg_dt_post,  rand_dt_post],
    [pos_dt_pre,   neg_dt_pre,   rand_dt_pre],
])

summary_df = pd.DataFrame(
    [
        {
            "correlation": "Positive",
            "strategy": "ACORN",
            "recall": pos_recall_acorn,
            "qps": pos_qps_acorn,
            "time_s": pos_dt_acorn,
        },
        {
            "correlation": "Positive",
            "strategy": "Post-filter",
            "recall": pos_recall_post,
            "qps": pos_qps_post,
            "time_s": pos_dt_post,
        },
        {
            "correlation": "Positive",
            "strategy": "Pre-filter",
            "recall": pos_recall_pre,
            "qps": pos_qps_pre,
            "time_s": pos_dt_pre,
        },
        {
            "correlation": "Negative",
            "strategy": "ACORN",
            "recall": neg_recall_acorn,
            "qps": neg_qps_acorn,
            "time_s": neg_dt_acorn,
        },
        {
            "correlation": "Negative",
            "strategy": "Post-filter",
            "recall": neg_recall_post,
            "qps": neg_qps_post,
            "time_s": neg_dt_post,
        },
        {
            "correlation": "Negative",
            "strategy": "Pre-filter",
            "recall": neg_recall_pre,
            "qps": neg_qps_pre,
            "time_s": neg_dt_pre,
        },
        {
            "correlation": "Random",
            "strategy": "ACORN",
            "recall": rand_recall_acorn,
            "qps": rand_qps_acorn,
            "time_s": rand_dt_acorn,
        },
        {
            "correlation": "Random",
            "strategy": "Post-filter",
            "recall": rand_recall_post,
            "qps": rand_qps_post,
            "time_s": rand_dt_post,
        },
        {
            "correlation": "Random",
            "strategy": "Pre-filter",
            "recall": rand_recall_pre,
            "qps": rand_qps_pre,
            "time_s": rand_dt_pre,
        },
    ]
)

summary_csv_path = exp_dir / f"performance_metrics_{safe_sel}.csv"
summary_md_path = exp_dir / f"performance_metrics_{safe_sel}.md"
summary_df.to_csv(summary_csv_path, index=False)

md_lines = [
    "| Correlation | Strategy | Recall | QPS | Time (s) |",
    "|---|---|---:|---:|---:|",
]
for row in summary_df.itertuples(index=False):
    md_lines.append(
        f"| {row.correlation} | {row.strategy} | {row.recall:.4f} | {row.qps:.2f} | {row.time_s:.6f} |"
    )
summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print("\nPerformance metrics table:")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print(f"Saved CSV table to {summary_csv_path}")
print(f"Saved Markdown table to {summary_md_path}")

x = np.arange(len(correlations))

fig, axes = plt.subplots(1,3, figsize=(18,5))

metrics = [recall, qps, times]
titles = ["Recall@k", "Queries per Second", "Query Time (s)"]

for ax, metric, title in zip(axes, metrics, titles):
    for i in range(3):
        ax.plot(
            x,
            metric[i],
            marker="o",
            linewidth=2,
            label=strategies[i],
            color=strategy_colors[i],
        )

    if title == "Recall@k":
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.linspace(0, 1.0, 11))
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y:,.0f}" if y >= 1 else f"{y:.2g}")
        )

    ax.set_xticks(x)
    ax.set_xticklabels(correlations)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.grid(True, which="minor", axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.legend(frameon=False)

plt.tight_layout()

save_path = exp_dir / f"performance_metrics_{safe_sel}.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved main plot to {save_path}")
# ----------------------------

plt.show()
