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
import pickle
from helper_funcs import *

out_base = Path("test")
out_base.mkdir(exist_ok=True)

#############################################
#                CONFIG
#############################################
k_center = 10
k_eval = 10
############################################
#                IMPORTS
############################################
local_cache_dir = Path("/tmp/fvs_cache")
local_cache_dir.mkdir(exist_ok=True)

attr_cache_path = local_cache_dir / "cached_attr.npy"
labels_cache_path = local_cache_dir / "cached_labels.npy"
exact_index_path = local_cache_dir / "exact_index.faiss"

print(f"Loading dataset...")
vecs, queries, gts = import_dataset()

query_limit_raw = os.environ.get("FVS_QUERY_LIMIT")
if query_limit_raw is not None:
    query_limit = int(query_limit_raw)
    if query_limit <= 0:
        raise ValueError(f"FVS_QUERY_LIMIT must be > 0, got {query_limit}")
    original_n_queries = len(queries)
    query_limit = min(query_limit, original_n_queries)
    queries = queries[:query_limit]
    gts = gts[:query_limit]
    print(
        f"Limiting workload to first {query_limit} / {original_n_queries} "
        "queries via FVS_QUERY_LIMIT"
    )

print(f"Loading pre-computed {ATTRIBUTE_CACHE_VERSION} attributes from {local_cache_dir}...")
attr = np.load(attr_cache_path)
labels = np.load(labels_cache_path)

print("Loading exact FAISS index...")
exact_index = faiss.read_index(str(exact_index_path))
############################################
#                INDEXES
############################################

def sample_random_query_range(
    sorted_attr,
    selectivity=0.01,
    rng=None,
    max_tries=64,
    relative_tolerance=0.05,
):
    """Sample a random range on attr while keeping actual selectivity close to target."""
    N = len(sorted_attr)
    target = max(1, min(N, int(np.floor(selectivity * N))))
    if rng is None:
        rng = np.random.default_rng()

    tolerance = max(1.0 / N, float(selectivity) * float(relative_tolerance))
    best_range = None
    best_actual_selectivity = None
    best_error = float("inf")

    for _ in range(int(max_tries)):
        start = int(rng.integers(0, N - target + 1))
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

def compute_ground_truth_for_query(sorted_attr, sorted_idx, query, query_range, exact_index, k=10):
    lo, hi = query_range
    left = np.searchsorted(sorted_attr, lo, side="left")
    right = np.searchsorted(sorted_attr, hi, side="right")

    if right <= left:
        return np.array([], dtype=np.int64)

    candidate_ids = np.ascontiguousarray(sorted_idx[left:right], dtype=np.int64)

    params = faiss.SearchParameters()
    params.sel = faiss.IDSelectorArray(candidate_ids)

    _, I = exact_index.search(query, k, params=params)
    ids = I[0]
    return ids[ids != -1]

print("Sorting attributes once for random range generation...")
sorted_idx = np.argsort(attr).astype(np.int64)
sorted_attr = attr[sorted_idx]
queries32 = np.ascontiguousarray(queries, dtype=np.float32)
rng = np.random.default_rng(0)

cache_path = Path(
    f"query_workload_cache_random_attr_q{len(queries)}_topk{k_eval}.pkl"
)
cache_path = local_cache_dir / cache_path

if cache_path.exists():
    print(f"Loading cached workload from {cache_path} ...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    target_selectivities = cache["target_selectivities"]
    actual_selectivities = cache["actual_selectivities"]
    ranges_all = cache["ranges_all"]
    gt_all = cache["gt_all"]

else:
    print("Computing random-on-attr workload and caching it...")

    start = time.perf_counter()

    n_queries = len(queries)
    target_selectivities = rng.uniform(0.01, 0.5, size=n_queries).astype(np.float32)

    actual_selectivities = np.empty(n_queries, dtype=np.float32)
    ranges_all = np.empty((n_queries, 2), dtype=np.int64)
    gt_all = [None] * n_queries

    for qi, target_sel in enumerate(target_selectivities):
        query_range, actual_sel = sample_random_query_range(
            sorted_attr,
            selectivity=float(target_sel),
            rng=rng,
        )

        if qi < 10 or (qi + 1) % 100 == 0 or qi + 1 == n_queries:
            print(
                f"[{qi + 1}/{n_queries}] "
                f"target_sel={float(target_sel):.4f} "
                f"actual_sel={actual_sel:.4f}"
            )

        query = queries32[qi : qi + 1]
        gt = compute_ground_truth_for_query(
            sorted_attr,
            sorted_idx,
            query,
            query_range,
            exact_index,
            k=k_eval,
        )

        actual_selectivities[qi] = actual_sel
        ranges_all[qi] = query_range
        gt_all[qi] = gt

    cache = {
        "n_queries": len(queries),
        "k_eval": k_eval,
        "target_selectivities": target_selectivities,
        "actual_selectivities": actual_selectivities,
        "ranges_all": ranges_all,
        "gt_all": gt_all,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    end = time.perf_counter()
    print(f"Cached workload in {end - start:.4f} seconds")

########################################
#              SETUP
########################################

print("[SETUP] Compute cluster stats...")
# Get centroids
class DummyKMeans: pass
fitted_vecs = DummyKMeans()
fitted_vecs.labels_ = labels.flatten()

n = np.max(labels) + 1
d = int(vecs.shape[1])
centroids = np.zeros((n, d), dtype=np.float32)

for i in range(n):
    mask = (labels == i)
    if np.any(mask):
        centroids[i] = vecs[mask].mean(axis=0)
    else:
        centroids[i] = vecs[np.random.randint(len(vecs))]

fitted_vecs.cluster_centers_ = centroids
bin_edges = choose_bins(attr)
td = 1.3

cluster_stats, global_hist, global_cdf = compute_cluster_stats(fitted_vecs, vecs, attr, td, bin_edges)

print("[SETUP] Build/load planner indexes...")
cpp_corr_index_dir = ensure_cpp_index(attr, tag="correlated")
acorn_cache_path = local_cache_dir / "acorn_correlated.faiss"
if acorn_cache_path.exists():
    print(f"Loading cached ACORN index from {acorn_cache_path}...")
    acorn_index = faiss.read_index(str(acorn_cache_path))
else:
    print("Building correlated ACORN index from scratch...")
    acorn_index, _ = build_acorn_index(vecs, attr)
    faiss.write_index(acorn_index, str(acorn_cache_path))

centroid_idx = faiss.IndexFlatL2(int(centroids.shape[1]))
centroid_idx.add(np.ascontiguousarray(centroids, dtype=np.float32))
cluster_cdfs = np.stack(
    [np.asarray(stats["cdf"], dtype=np.float64) for stats in cluster_stats],
    axis=0,
)
planner_eps = 1e-9
cluster_js = np.array(
    [
        jensenshannon(
            np.asarray(stats["hist"], dtype=np.float64) + planner_eps,
            np.asarray(global_hist, dtype=np.float64) + planner_eps,
            base=2,
        )
        for stats in cluster_stats
    ],
    dtype=np.float64,
)

############################################
#             PLANNER
############################################

def _subset_gt(gt, ids):
    return [gt[int(i)] for i in np.asarray(ids, dtype=np.int64)]


def run_strategy(name, queries, metadata, ranges, sorted_attr, sorted_idx, gt, k_eval):
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    ranges = np.asarray(ranges, dtype=np.int64)

    if name == "PRE":
        dt, qps, recall = run_pre(
            cpp_corr_index_dir,
            sorted_attr,
            sorted_idx,
            queries,
            metadata,
            ranges,
            k_eval,
            gt,
        )
    elif name == "POST":
        dt, qps, recall = run_post(
            cpp_corr_index_dir,
            queries,
            metadata,
            sorted_attr,
            sorted_idx,
            ranges,
            k_eval,
            gt,
        )
    elif name == "ACORN":
        dt, qps, recall = run_acorn(
            acorn_index,
            queries,
            metadata,
            ranges,
            k_eval,
            gt,
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")

    return {
        "strategy": name,
        "dt": float(dt),
        "qps": float(qps),
        "recall": float(recall),
    }


def _interpolated_metric_tau_array(selectivities, metric):
    metric_key = str(metric).strip().lower()
    if metric_key not in TAU_POINTS_BY_METRIC:
        valid = ", ".join(sorted(TAU_POINTS_BY_METRIC))
        raise ValueError(f"Unknown metric {metric!r}; expected one of: {valid}")

    values = np.asarray(selectivities, dtype=np.float64)
    if np.any((values <= 0.0) | (values > 1.0)):
        raise ValueError("selectivities must all be in (0, 1]")

    tau_points = np.asarray(TAU_POINTS_BY_METRIC[metric_key], dtype=np.float64)
    log_points = np.log10(np.asarray(TAU_SELECTIVITY_POINTS, dtype=np.float64))
    log_values = np.log10(values)
    return np.interp(
        log_values,
        log_points,
        tau_points,
        left=tau_points[0],
        right=tau_points[-1],
    )


def analyze_queries(queries, ranges, sorted_attr, pre_selectivity_threshold=0.04):
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    ranges = np.asarray(ranges, dtype=np.int64)
    total_queries = len(queries)

    if total_queries == 0:
        empty_ids = np.array([], dtype=np.int64)
        empty_float = np.array([], dtype=np.float64)
        return {
            "pre_ids": empty_ids,
            "post_ids": empty_ids,
            "acorn_ids": empty_ids,
            "plan_counts": {"PRE": 0, "POST": 0, "ACORN": 0},
            "corr_type_counts": {"positive": 0, "random": 0, "negative": 0},
            "selectivities": empty_float,
            "taus": empty_float,
            "correlations": empty_float,
        }

    _, idxs = centroid_idx.search(queries, 1)
    clusters = idxs[:, 0].astype(np.int64)

    left_idx = np.searchsorted(sorted_attr, ranges[:, 0], side="left")
    right_idx = np.searchsorted(sorted_attr, ranges[:, 1], side="right")
    selectivities = (right_idx - left_idx).astype(np.float64) / float(len(sorted_attr))
    taus = _interpolated_metric_tau_array(selectivities, "heuristic")

    lo_bin = np.searchsorted(bin_edges, ranges[:, 0], side="right") - 1
    hi_bin = np.searchsorted(bin_edges, ranges[:, 1], side="right") - 1
    lo_bin = np.clip(lo_bin, 0, len(bin_edges) - 2)
    hi_bin = np.clip(hi_bin, 0, len(bin_edges) - 2)

    p_c_hi = cluster_cdfs[clusters, hi_bin]
    p_c_lo = np.zeros(total_queries, dtype=np.float64)
    valid_lo = lo_bin > 0
    p_c_lo[valid_lo] = cluster_cdfs[clusters[valid_lo], lo_bin[valid_lo] - 1]
    p_c = p_c_hi - p_c_lo

    global_cdf_values = np.asarray(global_cdf, dtype=np.float64)
    p_g_hi = global_cdf_values[hi_bin]
    p_g_lo = np.zeros(total_queries, dtype=np.float64)
    p_g_lo[valid_lo] = global_cdf_values[lo_bin[valid_lo] - 1]
    p_g = p_g_hi - p_g_lo

    lift = np.log((p_c + planner_eps) / (p_g + planner_eps))
    correlations = cluster_js[clusters] * np.sign(lift)

    negative_mask = correlations <= -taus
    positive_mask = correlations >= taus
    random_mask = ~(negative_mask | positive_mask)

    pre_mask = selectivities < float(pre_selectivity_threshold)
    acorn_mask = (~pre_mask) & negative_mask
    post_mask = (~pre_mask) & (~negative_mask)

    pre_ids = np.flatnonzero(pre_mask).astype(np.int64)
    post_ids = np.flatnonzero(post_mask).astype(np.int64)
    acorn_ids = np.flatnonzero(acorn_mask).astype(np.int64)

    return {
        "pre_ids": pre_ids,
        "post_ids": post_ids,
        "acorn_ids": acorn_ids,
        "plan_counts": {
            "PRE": int(pre_ids.size),
            "POST": int(post_ids.size),
            "ACORN": int(acorn_ids.size),
        },
        "corr_type_counts": {
            "positive": int(np.count_nonzero(positive_mask)),
            "random": int(np.count_nonzero(random_mask)),
            "negative": int(np.count_nonzero(negative_mask)),
        },
        "selectivities": selectivities,
        "taus": taus,
        "correlations": correlations,
    }


def run_planner(
    queries,
    metadata,
    ranges,
    sorted_attr,
    sorted_idx,
    gt,
    k_eval,
    pre_selectivity_threshold=0.04,
):
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    ranges = np.asarray(ranges, dtype=np.int64)
    total_queries = len(queries)

    if total_queries == 0:
        return {
            "routing_dt": 0.0,
            "execution_dt": 0.0,
            "dt": 0.0,
            "qps": float("inf"),
            "recall": 1.0,
            "plan_counts": {"PRE": 0, "POST": 0, "ACORN": 0},
            "corr_type_counts": {"positive": 0, "random": 0, "negative": 0},
        }

    print("[PLANNER] Computing correlation for all queries...")
    routing_t0 = time.perf_counter()
    analysis = analyze_queries(
        queries=queries,
        ranges=ranges,
        sorted_attr=sorted_attr,
        pre_selectivity_threshold=pre_selectivity_threshold,
    )
    routing_dt = time.perf_counter() - routing_t0
    pre_ids = analysis["pre_ids"]
    post_ids = analysis["post_ids"]
    acorn_ids = analysis["acorn_ids"]
    plan_counts = analysis["plan_counts"]
    corr_type_counts = analysis["corr_type_counts"]

    print(
        "[PLANNER] Routing done | "
        f"PRE={plan_counts['PRE']} | POST={plan_counts['POST']} | ACORN={plan_counts['ACORN']} | "
        f"corr(+/0/-)=({corr_type_counts['positive']}/{corr_type_counts['random']}/{corr_type_counts['negative']})"
    )

    execution_results = []
    execution_dt = 0.0

    if pre_ids.size > 0:
        dt, _, recall = run_pre(
            cpp_corr_index_dir,
            sorted_attr,
            sorted_idx,
            queries[pre_ids],
            metadata,
            ranges[pre_ids],
            k_eval,
            _subset_gt(gt, pre_ids),
        )
        execution_dt += dt
        execution_results.append({"dt": dt, "recall": recall, "nq": len(pre_ids), "name": "PRE"})

    if post_ids.size > 0:
        dt, _, recall = run_post(
            cpp_corr_index_dir,
            queries[post_ids],
            metadata,
            sorted_attr,
            sorted_idx,
            ranges[post_ids],
            k_eval,
            _subset_gt(gt, post_ids),
        )
        execution_dt += dt
        execution_results.append({"dt": dt, "recall": recall, "nq": len(post_ids), "name": "POST"})

    if acorn_ids.size > 0:
        dt, _, recall = run_acorn(
            acorn_index,
            queries[acorn_ids],
            metadata,
            ranges[acorn_ids],
            k_eval,
            _subset_gt(gt, acorn_ids),
        )
        execution_dt += dt
        execution_results.append({"dt": dt, "recall": recall, "nq": len(acorn_ids), "name": "ACORN"})

    total_dt = routing_dt + execution_dt
    weighted_recall = (
        sum(item["recall"] * item["nq"] for item in execution_results) / total_queries
        if execution_results
        else 1.0
    )
    qps = total_queries / total_dt if total_dt > 0 else float("inf")

    print(
        "[PLANNER] Finished | "
        f"routing={routing_dt:.6f}s | execution={execution_dt:.6f}s | total={total_dt:.6f}s | "
        f"qps={qps:.2f} | recall={weighted_recall:.4f}"
    )

    return {
        "routing_dt": routing_dt,
        "execution_dt": execution_dt,
        "dt": total_dt,
        "qps": qps,
        "recall": weighted_recall,
        "plan_counts": plan_counts,
        "corr_type_counts": corr_type_counts,
        "selectivities": analysis["selectivities"],
        "taus": analysis["taus"],
        "correlations": analysis["correlations"],
    }


def plot_strategy_comparison(results, out_path):
    strategies = [item["strategy"] for item in results]
    times = [item["dt"] for item in results]
    qps = [item["qps"] for item in results]
    recalls = [item["recall"] for item in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]

    axes[0].bar(strategies, times, color=colors[: len(strategies)])
    axes[0].set_title("Total Time")
    axes[0].set_ylabel("Seconds")

    axes[1].bar(strategies, qps, color=colors[: len(strategies)])
    axes[1].set_title("QPS")
    axes[1].set_ylabel("Queries / second")

    axes[2].bar(strategies, recalls, color=colors[: len(strategies)])
    axes[2].set_title("Recall")
    axes[2].set_ylabel("Recall@k")
    axes[2].set_ylim(0.0, 1.05)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved strategy comparison plot to {out_path}")


def plot_planner_routing(planner_results, out_path):
    labels = ["PRE", "POST", "ACORN"]
    counts = [planner_results["plan_counts"][label] for label in labels]
    colors = ["#2ca02c", "#d62728", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, counts, color=colors)
    ax.set_title("Planner Routing Counts")
    ax.set_ylabel("Queries")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved planner routing plot to {out_path}")


print("[BASELINES] Running PRE / POST / ACORN on the same workload...")
pre_results = run_strategy(
    "PRE",
    queries=queries32,
    metadata=attr,
    ranges=ranges_all,
    sorted_attr=sorted_attr,
    sorted_idx=sorted_idx,
    gt=gt_all,
    k_eval=k_eval,
)
post_results = run_strategy(
    "POST",
    queries=queries32,
    metadata=attr,
    ranges=ranges_all,
    sorted_attr=sorted_attr,
    sorted_idx=sorted_idx,
    gt=gt_all,
    k_eval=k_eval,
)
acorn_results = run_strategy(
    "ACORN",
    queries=queries32,
    metadata=attr,
    ranges=ranges_all,
    sorted_attr=sorted_attr,
    sorted_idx=sorted_idx,
    gt=gt_all,
    k_eval=k_eval,
)

print("[PLANNER] Running heuristic planner on random-on-attr workload...")
planner_results = run_planner(
    queries=queries32,
    metadata=attr,
    ranges=ranges_all,
    sorted_attr=sorted_attr,
    sorted_idx=sorted_idx,
    gt=gt_all,
    k_eval=k_eval,
)

comparison_results = [
    pre_results,
    post_results,
    acorn_results,
    {
        "strategy": "PLANNER",
        "dt": planner_results["dt"],
        "qps": planner_results["qps"],
        "recall": planner_results["recall"],
    },
]

comparison_plot_path = out_base / f"strategy_comparison_q{len(queries32)}_topk{k_eval}.png"
routing_plot_path = out_base / f"planner_routing_q{len(queries32)}_topk{k_eval}.png"
plot_strategy_comparison(comparison_results, comparison_plot_path)
plot_planner_routing(planner_results, routing_plot_path)
