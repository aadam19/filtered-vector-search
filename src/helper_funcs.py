import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from scipy.spatial.distance import jensenshannon
import faiss
import scipy.cluster.hierarchy as hierarchy
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import math


def _faiss_thread_count():
    value = os.environ.get("FVS_NUM_THREADS")
    if value is not None:
        try:
            threads = int(value)
        except ValueError as exc:
            raise ValueError(f"FVS_NUM_THREADS must be an integer, got {value!r}") from exc
        if threads <= 0:
            raise ValueError(f"FVS_NUM_THREADS must be > 0, got {threads}")
        return threads
    return multiprocessing.cpu_count()


faiss.omp_set_num_threads(_faiss_thread_count())

try:
    import filter_kernel
except ImportError:
    filter_kernel = None

MAX_RANGE = 200
ATTRIBUTE_CACHE_VERSION = "kmeans_rbf_anchor_v4"
BASE_FVECS_PATH = Path(os.path.expanduser("~/filtered-vector-search/sift/sift_base.fvecs"))
CPP_DIR = Path(__file__).resolve().parent / "cpp"
CPP_BUILD_DIR = CPP_DIR / "build"
CPP_BUILD_BIN = CPP_BUILD_DIR / "build_indexes"
CPP_SEARCH_BIN = CPP_BUILD_DIR / "search_filters"
SETUP_KERNEL_PY = Path(__file__).resolve().parent / "setup.py"
FAISS_LIB_DIR = Path(os.path.expanduser("~/filtered-vector-search/ACORN/build/faiss"))

def _cpp_sources_newer_than_binaries():
    binaries = [CPP_BUILD_BIN, CPP_SEARCH_BIN]
    if not all(path.exists() for path in binaries):
        return True

    newest_binary_mtime = min(path.stat().st_mtime for path in binaries)
    watched = [
        CPP_DIR / "CMakeLists.txt",
        CPP_DIR / "build_indexes.cpp",
        CPP_DIR / "search_filters.cpp",
        CPP_DIR / "fvs_common.cpp",
        CPP_DIR / "fvs_common.h",
        SETUP_KERNEL_PY,
    ]
    return any(path.exists() and path.stat().st_mtime > newest_binary_mtime for path in watched)

def import_dataset():
    
    def read_ivecs(fname: str) -> np.ndarray:
        with open(fname, "rb") as f:
            data = np.fromfile(f, dtype=np.int32)

        dim = data[0]
        data = data.reshape(-1, dim + 1)

        return data[:, 1:]

    def read_fvecs(fname: str) -> np.ndarray:
        with open(fname, "rb") as f:
            data = np.fromfile(f, dtype=np.int32)

        dim = data[0]
        data = data.reshape(-1, dim + 1)

        # reinterpret everything except first column as float32
        vectors = data[:, 1:].view(np.float32)

        return vectors
    
    BASE = os.path.expanduser("~/filtered-vector-search/sift/sift_base.fvecs")
    QUERY = os.path.expanduser("~/filtered-vector-search/sift/sift_query.fvecs")
    TRUTH = os.path.expanduser("~/filtered-vector-search/sift/sift_groundtruth.ivecs")

    base_vectors = read_fvecs(BASE)
    query_vectors = read_fvecs(QUERY)
    truth_vectors = read_ivecs(TRUTH)
    
    return base_vectors, query_vectors, truth_vectors


def _format_preview(values: np.ndarray, precision: int = 4, max_items: int = 12) -> str:
    values = np.asarray(values)
    if values.size <= max_items:
        return np.array2string(values, precision=precision, separator=", ")

    head_count = max_items // 2
    tail_count = max_items - head_count
    head = np.array2string(values[:head_count], precision=precision, separator=", ")
    tail = np.array2string(values[-tail_count:], precision=precision, separator=", ")
    return f"{head} ... {tail}"


def assign_uniform_anchors(centroids: np.ndarray) -> np.ndarray:
    """Return uniformly spaced centroid anchors in [0, 1] by centroid index."""
    centroids = np.asarray(centroids, dtype=np.float32)
    k = int(len(centroids))

    if k <= 0:
        raise ValueError("centroids must contain at least one row")

    if k == 1:
        print("Uniform anchor assignment | k=1, assigning anchor [0.0]")
        return np.array([0.0], dtype=np.float32)

    anchors = np.linspace(0.0, 1.0, k, dtype=np.float32)
    print(f"Uniform anchor assignment | anchors: {_format_preview(anchors)}")
    return anchors

def generate_correlated_attribute(
    vecs,
    k=10,
    seed=0,
    max_range=200,
    sigma_scale=1,
):
    """Assign correlated integer attributes via the KMeans-anchor RBF construction."""

    if int(k) <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    print("Fitting MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(
        n_clusters=int(k),
        batch_size=10000,
        random_state=seed,
        n_init="auto",
    ).fit(vecs)
    centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    anchors = assign_uniform_anchors(centroids)

    n_samples = len(vecs)
    rng = np.random.default_rng(seed)

    print("Estimating sigma from a random subsample...")
    sample_idx = rng.choice(n_samples, size=min(10000, n_samples), replace=False)
    sample_dists = cdist(vecs[sample_idx], centroids, metric="euclidean")
    sigma = float(np.median(sample_dists) * float(sigma_scale))
    sigma = max(sigma, 1e-12)

    print("Computing continuous attributes in batches...")
    attr_cont = np.zeros(n_samples, dtype=np.float32)
    batch_size = 10000

    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        dists = cdist(vecs[i:end], centroids, metric="euclidean")
        weights = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))
        weight_sums = weights.sum(axis=1)
        weight_sums[weight_sums == 0] = 1e-12
        attr_cont[i:end] = (weights @ anchors) / weight_sums

    print("Normalizing and casting to int32...")
    min_attr = float(attr_cont.min())
    max_attr = float(attr_cont.max())
    span = max_attr - min_attr
    if span == 0.0:
        attr_norm = np.full_like(attr_cont, 0.5, dtype=np.float32)
    else:
        attr_norm = (attr_cont - min_attr) / span

    attr = np.round(float(max_range) * attr_norm).astype(np.int32)
    return attr, np.asarray(kmeans.labels_, dtype=np.int32)

def generate_random_attribute(vecs, seed=42):
    rng = np.random.default_rng(seed)
    attr = rng.integers(0, MAX_RANGE + 1, size=len(vecs), dtype=np.int32)
    return attr, None

def compute_neighbor_stats(query_vecs, query_attr, global_attr, index, n_neighbors=10):
    # Search the global index using our exact 10k sample
    dist, idx = index.search(query_vecs.astype(np.float32), n_neighbors)

    # Look up the attributes of the found neighbors using the GLOBAL 1M array
    neighbor_attr = global_attr[idx]
    
    # Compare them against the attributes of our 10k QUERY vectors
    diff = np.abs(query_attr[:, None] - neighbor_attr)
    neighbor_mean = neighbor_attr.mean(axis=1)

    return {
        "neighbor_mean": neighbor_mean,
        "neighbor_diff_mean": diff.mean()
    }


def topk_interval_overlap(index, queries, attr, ranges, topk=100):
    """Measure how many of the true top-k neighbors fall inside each interval."""
    _, idx = index.search(np.ascontiguousarray(queries, dtype=np.float32), int(topk))
    overlaps = []
    for neighbors, (lo, hi) in zip(idx, ranges):
        inside = ((attr[neighbors] >= int(lo)) & (attr[neighbors] <= int(hi))).sum()
        overlaps.append(float(inside) / float(topk))
    return np.asarray(overlaps, dtype=np.float32)
    
    
def plot_attribute_analysis(vecs, attr, neighbor_mean, clusters=None, save_path=None, show=True):
    """Visualize PCA embeddings (supports 2D or 3D) alongside attribute stats.

    show: if False, skip plt.show() to avoid blocking (useful in batch runs).
    """
    if vecs.ndim != 2 or vecs.shape[1] not in (2, 3):
        raise ValueError(f"vecs must be (n,2) or (n,3); got {vecs.shape}")

    fig = plt.figure(figsize=(18, 5))

    # --- PCA colored by attribute ---
    if vecs.shape[1] == 3:
        ax0 = fig.add_subplot(1, 3, 1, projection="3d")
        sc = ax0.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2], c=attr, s=5, cmap="viridis")
        ax0.set_zlabel("PCA3")
    else:
        ax0 = fig.add_subplot(1, 3, 1)
        sc = ax0.scatter(vecs[:, 0], vecs[:, 1], c=attr, s=5, cmap="viridis")
    ax0.set_title("Vectors colored by attribute")
    ax0.set_xlabel("PCA1")
    ax0.set_ylabel("PCA2")
    fig.colorbar(sc, ax=ax0, label="Attribute")

    # --- Cluster distribution if clusters exist ---
    ax1 = fig.add_subplot(1, 3, 2)
    if clusters is not None:
        df = pd.DataFrame({"cluster": clusters, "attr": attr})
        cluster_means = df.groupby("cluster")["attr"].mean()
        cluster_std = df.groupby("cluster")["attr"].std()

        ax1.errorbar(range(len(cluster_means)), cluster_means, yerr=cluster_std, fmt="o")
        ax1.set_title("Cluster attribute distribution")
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Attribute value")
    else:
        ax1.hist(attr, bins=30)
        ax1.set_title("Attribute distribution")
        ax1.set_xlabel("Attribute")
        ax1.set_ylabel("Count")

    # --- neighbor correlation ---
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.scatter(attr, neighbor_mean, s=5)
    ax2.set_title("Attribute vs neighbor mean")
    ax2.set_xlabel("Point attribute")
    ax2.set_ylabel("neighbor mean attribute")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved attribute analysis plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_attribute_pca_pair(vecs_2d, vecs_3d, attr, save_path=None, show=True):
    """2-panel layout: 2D and 3D PCA colored by attribute."""
    if vecs_2d.shape[1] != 2 or vecs_3d.shape[1] != 3:
        raise ValueError("vecs_2d must be (n,2) and vecs_3d must be (n,3)")

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    sc1 = ax1.scatter(vecs_2d[:, 0], vecs_2d[:, 1], c=attr, s=5, cmap="viridis")
    ax1.set_title("PCA 2D colored by attribute")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sc2 = ax2.scatter(vecs_3d[:, 0], vecs_3d[:, 1], vecs_3d[:, 2], c=attr, s=5, cmap="viridis")
    ax2.set_title("PCA 3D colored by attribute")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")

    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([])
    # Place colorbar between the two axes (to the left of the 3D plot)
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    cbar_width = 0.015
    gap_center = (pos1.x1 + pos2.x0) / 2.0
    cbar_x0 = gap_center - cbar_width / 2.0
    cbar = fig.add_axes([cbar_x0, pos1.y0, cbar_width, pos1.height])
    fig.colorbar(sm, cax=cbar, label="Attribute")

    fig.subplots_adjust(wspace=0.25, left=0.07, right=0.95, top=0.92, bottom=0.1)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved attribute PCA pair plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_attribute_stats(attr, neighbor_mean, clusters=None, save_path=None, show=True):
    """2-panel layout: attribute distribution and attr vs neighbor mean."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1, ax2 = axes
    if clusters is not None:
        df = pd.DataFrame({"cluster": clusters, "attr": attr})
        cluster_means = df.groupby("cluster")["attr"].mean()
        cluster_std = df.groupby("cluster")["attr"].std()
        ax1.errorbar(range(len(cluster_means)), cluster_means, yerr=cluster_std, fmt="o")
        ax1.set_title("Cluster attribute distribution")
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Attribute value")
    else:
        ax1.hist(attr, bins=30)
        ax1.set_title("Attribute distribution")
        ax1.set_xlabel("Attribute")
        ax1.set_ylabel("Count")

    ax2.scatter(attr, neighbor_mean, s=5)
    ax2.set_title("Attribute vs neighbor mean")
    ax2.set_xlabel("Point attribute")
    ax2.set_ylabel("Neighbor mean attribute")

    fig.subplots_adjust(wspace=0.25, left=0.08, right=0.97, top=0.9, bottom=0.12)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved attribute stats plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    
def generate_query_ranges(queries, attr, index, k=50, selectivity=0.01, seed=0):
    """Generate positive/negative query ranges from neighbor-attribute medians."""
    k = int(k)
    N = len(attr)
    target = max(1, min(N, int(np.floor(selectivity * N))))
    
    # Sort the int32 attributes
    sorted_attr = np.sort(attr)

    # Search the global index using the query vectors
    _, idx = index.search(queries.astype(np.float32), k)

    pos_ranges = []
    neg_ranges = []

    # Opposite tails of the global attribute distribution for anti-correlation.
    left_tail = (int(sorted_attr[0]), int(sorted_attr[min(target - 1, N - 1)]))
    right_tail = (int(sorted_attr[max(0, N - target)]), int(sorted_attr[-1]))
    global_median = np.median(sorted_attr)

    for neighbors in idx:
        vals = attr[neighbors]
        center = np.median(vals)

        pos = np.searchsorted(sorted_attr, center)
        half = target // 2
        lo_idx = max(0, pos - half)
        hi_idx = min(N - 1, pos + half)
        a_min = int(sorted_attr[lo_idx])
        a_max = int(sorted_attr[hi_idx])
        pos_ranges.append((a_min, a_max))

        if center > global_median:
            neg_ranges.append(left_tail)
        else:
            neg_ranges.append(right_tail)

    return pos_ranges, neg_ranges


def generate_random_query_ranges(attr, n_queries, selectivity=0.01, seed=0):
    rng = np.random.default_rng(seed)

    N = len(attr)
    target = max(1, min(N, int(np.floor(selectivity * N))))
    sorted_attr = np.sort(attr)

    ranges = []

    for _ in range(n_queries):
        start = int(rng.integers(0, N - target + 1))
        a_min = int(sorted_attr[start])
        a_max = int(sorted_attr[min(N - 1, start + target)])
        ranges.append((a_min, a_max))

    return ranges

def choose_bins(values, target_per_bin=50, min_bins=20, max_bins=200):
    """
    Pick bin edges for histograms used in JS / lift.

    Improvements vs the old heuristic:
    - Use Freedman–Diaconis to adapt bin width to data spread & outliers.
    - Clamp bin count to avoid > max_bins and < min_bins.
    - Fall back to a target-per-bin estimate when IQR is tiny (near-constant data).
    """
    n = len(values)
    vmin, vmax = values.min(), values.max()
    span = vmax - vmin
    if span == 0:
        return np.array([vmin, vmax + 1])

    # Freedman–Diaconis bin width: 2 * IQR / n^(1/3)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    fd_width = 2 * iqr / (n ** (1 / 3)) if iqr > 0 else 0

    if fd_width > 0:
        fd_bins = int(np.ceil(span / fd_width))
    else:
        fd_bins = 0

    # Fallback to target_per_bin when FD suggests too many/few or width is 0
    fallback_bins = int(np.ceil(n / target_per_bin)) if target_per_bin > 0 else min_bins

    est_bins = fd_bins if fd_bins else fallback_bins
    est_bins = max(min_bins, min(max_bins, est_bins))

    return np.linspace(vmin, vmax, est_bins + 1)


def compute_cluster_stats(fitted_vecs, vecs, attr, td, bins, smooth=1e-3):
    centroids = fitted_vecs.cluster_centers_
    
    stats = []
 
    global_hist, _ = np.histogram(attr, bins=bins, density=True)
    global_hist = global_hist + smooth
    global_hist = global_hist / global_hist.sum()
    global_cdf = np.cumsum(global_hist) / np.sum(global_hist)
    # ----------------------------------------------------------
    
    for cluster_id, centroid in enumerate(centroids):
        # 1. Get points assigned to this cluster
        cluster_idx = np.where(fitted_vecs.labels_ == cluster_id)[0]
        cluster_points = vecs[cluster_idx]
        
        # 2. Compute distances from centroid
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        
        # 3. Determine cluster radius and include neighbors slightly outside
        cluster_radius = distances.max() * td
        selected_idx = cluster_idx[distances <= cluster_radius]
        
        selected_vals = attr[selected_idx]

        # 4. Compute histogram & cdf
        hist, _ = np.histogram(selected_vals, bins=bins, density=True)
        hist = hist + smooth
        hist = hist / hist.sum()
        cdf = np.cumsum(hist) / np.sum(hist)

        stats.append({"hist": hist, "cdf": cdf})

    return stats, global_hist, global_cdf

def compute_selectivity(sorted_values, lo, hi):
    # searchsorted finds the exact index boundaries in O(log N) time (microseconds)
    left_idx = np.searchsorted(sorted_values, lo, side='left')
    right_idx = np.searchsorted(sorted_values, hi, side='right')
    
    count = right_idx - left_idx
    return count / len(sorted_values)


def compute_correlation_components(
    query_vector,
    query_range,
    index,
    clust_stats,
    glob_hist,
    glob_cdf,
    bin_edges,
    eps=1e-9,
):
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1).astype(np.float32)

    _, idxs = index.search(query_vector, 1)
    clust = int(idxs[0][0])

    clust_hist = clust_stats[clust]["hist"]
    clust_cdf = clust_stats[clust]["cdf"]

    js = jensenshannon(clust_hist + eps, glob_hist + eps, base=2)

    lo, hi = query_range
    lo_bin = np.searchsorted(bin_edges, lo, side="right") - 1
    hi_bin = np.searchsorted(bin_edges, hi, side="right") - 1
    lo_bin = np.clip(lo_bin, 0, len(bin_edges) - 2)
    hi_bin = np.clip(hi_bin, 0, len(bin_edges) - 2)

    p_c = clust_cdf[hi_bin] - (clust_cdf[lo_bin - 1] if lo_bin > 0 else 0.0)
    p_g = glob_cdf[hi_bin] - (glob_cdf[lo_bin - 1] if lo_bin > 0 else 0.0)

    lift = np.log((p_c + eps) / (p_g + eps))
    return {
        "heuristic_correlation": js * np.sign(lift),
        "js_divergence": js,
        "lift": lift,
        "p_c": p_c,
        "p_g": p_g,
        "pgap_correlation": js * (p_c - p_g),
    }


def compute_correlation(query_vector, query_range, index, clust_stats, glob_hist, glob_cdf, bin_edges, eps=1e-9):
    components = compute_correlation_components(
        query_vector=query_vector,
        query_range=query_range,
        index=index,
        clust_stats=clust_stats,
        glob_hist=glob_hist,
        glob_cdf=glob_cdf,
        bin_edges=bin_edges,
        eps=eps,
    )
    return (
        components["heuristic_correlation"],
        components["js_divergence"],
        components["lift"],
    )


def _log_comb(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def true_query_correlation(
    query_vector,
    query_range,
    vecs,
    attr,
    k,
    all_dists=None,
    sorted_all=None,
):
    """Compute the correlation from the paper definition.

    C(q, p) = E[g_k(q, R_p)] - g_k(q, X_p)
    where R_p is a uniformly sampled subset of X with |R_p| = |X_p|.
    """
    q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    attr = np.asarray(attr)

    mask = (attr >= int(query_range[0])) & (attr <= int(query_range[1]))
    admissible_idx = np.flatnonzero(mask)
    subset_size = int(admissible_idx.size)
    if subset_size <= 0:
        return {
            "true_correlation": float("nan"),
            "expected_random_gk": float("nan"),
            "admissible_gk": float("nan"),
            "subset_size": 0,
            "k_used": 0,
        }

    k_used = min(int(k), subset_size)

    if all_dists is None:
        all_dists = np.linalg.norm(np.asarray(vecs, dtype=np.float32) - q, axis=1)
    else:
        all_dists = np.asarray(all_dists, dtype=np.float64)
    admissible_gk = float(np.mean(np.partition(all_dists[admissible_idx], k_used - 1)[:k_used]))

    if sorted_all is None:
        sorted_all = np.sort(all_dists.astype(np.float64))
    else:
        sorted_all = np.asarray(sorted_all, dtype=np.float64)
    total = sorted_all.size
    log_denom = _log_comb(total, subset_size)

    expected_sum = 0.0
    for rank_1based, dist in enumerate(sorted_all, start=1):
        better = rank_1based - 1
        worse = total - rank_1based
        max_better_selected = min(k_used - 1, better, subset_size - 1)
        prob_in_topk = 0.0
        for t in range(max_better_selected + 1):
            remaining = subset_size - 1 - t
            if remaining < 0 or remaining > worse:
                continue
            log_prob = _log_comb(better, t) + _log_comb(worse, remaining) - log_denom
            prob_in_topk += math.exp(log_prob)
        expected_sum += float(dist) * prob_in_topk

    expected_random_gk = expected_sum / float(k_used)
    if expected_random_gk == 0.0:
        normalized_true_correlation = float("nan")
    else:
        normalized_true_correlation = (expected_random_gk - admissible_gk) / expected_random_gk
    return {
        "true_correlation": normalized_true_correlation,
        "expected_random_gk": expected_random_gk,
        "admissible_gk": admissible_gk,
        "subset_size": subset_size,
        "k_used": k_used,
    }


# =====================================
# SEARCH STRATEGIES
# =====================================

def build_prefilter_indexes(vecs, attr, num_bins=20):
    """
    Build hybrid pre-filter structures:
    - Global HNSW index for high-selectivity ranges.
    - Bucketed HNSW indexes for low-selectivity ranges.
    """
    sorted_idx = np.argsort(attr).astype(np.int64)
    sorted_attr = attr[sorted_idx]

    dim = vecs.shape[1]
    global_index = faiss.IndexHNSWFlat(dim, 32)
    global_index.add(np.ascontiguousarray(vecs, dtype=np.float32))
    global_index.hnsw.efSearch = 256

    N = len(attr)
    num_bins = max(1, int(num_bins))
    bin_size = max(1, int(np.ceil(N / num_bins)))

    bucket_ranges = []
    bucket_indexes = []
    bucket_global_ids = []
    for i in range(num_bins):
        start = i * bin_size
        if start >= N:
            break
        end = min(N, (i + 1) * bin_size)
        gids = sorted_idx[start:end]
        if len(gids) == 0:
            continue
        b_lo = int(sorted_attr[start])
        b_hi = int(sorted_attr[end - 1])
        b_index = faiss.IndexHNSWFlat(dim, 16)
        b_index.add(np.ascontiguousarray(vecs[gids], dtype=np.float32))
        b_index.hnsw.efSearch = 128
        bucket_ranges.append((b_lo, b_hi))
        bucket_indexes.append(b_index)
        bucket_global_ids.append(gids)

    pre_state = {
        "global_index": global_index,
        "bucket_ranges": bucket_ranges,
        "bucket_indexes": bucket_indexes,
        "bucket_global_ids": bucket_global_ids,
    }

    return pre_state, sorted_attr, sorted_idx

def prefilter_search(
    pre_state,
    sorted_attr,
    sorted_idx,
    queries,
    attr,
    ranges,
    k=10,
    high_sel_threshold=0.20,
    oversample=1.15,
    max_steps=4,
    base_ef_search=48,
    max_ef_search=384,
    ef_search_mult=1.0,
    batch_size=512,
    bucket_oversample=2.0,
    max_fetch_k=768,
):
    nq = len(queries)

    D = np.full((nq, k), np.inf, dtype="float32")
    I = np.full((nq, k), -1, dtype="int64")

    global_index = pre_state["global_index"]
    bucket_ranges = pre_state["bucket_ranges"]
    bucket_indexes = pre_state["bucket_indexes"]
    bucket_global_ids = pre_state["bucket_global_ids"]

    faiss.omp_set_num_threads(_faiss_thread_count())
    params_unfiltered = faiss.SearchParametersHNSW()
    params_bucket = faiss.SearchParametersHNSW()
    xq = np.ascontiguousarray(queries, dtype=np.float32)
    n_total = len(attr)

    q_los = np.array([int(r[0]) for r in ranges], dtype=np.int32)
    q_his = np.array([int(r[1]) for r in ranges], dtype=np.int32)
    q_left = np.searchsorted(sorted_attr, q_los, side="left")
    q_right = np.searchsorted(sorted_attr, q_his, side="right")
    q_counts = q_right - q_left
    q_sel = q_counts / n_total

    high_qids = np.where((q_counts > 0) & (q_sel >= high_sel_threshold))[0]
    low_qids = np.where((q_counts > 0) & (q_sel < high_sel_threshold))[0]

    # High-selectivity path: global ANN + filter, batched by estimated fetch depth.
    fetch_cap = min(global_index.ntotal, int(max_fetch_k))
    high_fetch = np.ceil((k / np.maximum(q_sel[high_qids], 1e-4)) * oversample).astype(np.int32)
    high_fetch = np.clip(high_fetch, k, fetch_cap)
    # Round fetch depth to reduce number of groups while avoiding worst-case overfetch.
    high_fetch = ((high_fetch + 31) // 32) * 32
    high_fetch = np.clip(high_fetch, k, fetch_cap)

    fetch_to_qids = {}
    for qi, fk in zip(high_qids.tolist(), high_fetch.tolist()):
        fetch_to_qids.setdefault(int(fk), []).append(int(qi))

    retry_qids = []
    retry_fetch = []

    for fetch_k, qids in fetch_to_qids.items():
        ef_search = int(min(max_ef_search, max(base_ef_search, np.ceil(fetch_k * ef_search_mult))))
        params_unfiltered.efSearch = ef_search
        qids = np.asarray(qids, dtype=np.int64)

        for s in range(0, len(qids), batch_size):
            bqids = qids[s:s + batch_size]
            if len(bqids) == 0:
                continue
            D_b, I_b = global_index.search(xq[bqids], int(fetch_k), params=params_unfiltered)
            for row, qi in enumerate(bqids):
                lo = int(q_los[qi])
                hi = int(q_his[qi])
                ids = I_b[row]
                dists = D_b[row]
                if filter_kernel is not None:
                    best_dists, best_ids = filter_kernel.filter_topk(ids, dists, attr, lo, hi, int(k))
                else:
                    valid = ids != -1
                    ids = ids[valid]
                    dists = dists[valid]
                    mask = (attr[ids] >= lo) & (attr[ids] <= hi)
                    best_ids = ids[mask][:k]
                    best_dists = dists[mask][:k]
                take = min(k, len(best_ids))
                if take > 0:
                    D[qi, :take] = best_dists[:take]
                    I[qi, :take] = best_ids[:take]
                if take < k and fetch_k < fetch_cap:
                    retry_qids.append(int(qi))
                    retry_fetch.append(int(min(fetch_cap, max(fetch_k + 1, fetch_k * 2))))

    # Accuracy recovery pass: only retry queries that failed to fill k.
    if retry_qids:
        retry_to_qids = {}
        for qi, fk in zip(retry_qids, retry_fetch):
            fk = ((int(fk) + 31) // 32) * 32
            fk = min(fetch_cap, max(k, fk))
            retry_to_qids.setdefault(fk, []).append(qi)

        for fetch_k, qids in retry_to_qids.items():
            ef_search = int(min(max_ef_search, max(base_ef_search, np.ceil(fetch_k * ef_search_mult))))
            params_unfiltered.efSearch = ef_search
            qids = np.asarray(qids, dtype=np.int64)

            for s in range(0, len(qids), batch_size):
                bqids = qids[s:s + batch_size]
                D_b, I_b = global_index.search(xq[bqids], int(fetch_k), params=params_unfiltered)
                for row, qi in enumerate(bqids):
                    lo = int(q_los[qi])
                    hi = int(q_his[qi])
                    ids = I_b[row]
                    dists = D_b[row]
                    if filter_kernel is not None:
                        best_dists, best_ids = filter_kernel.filter_topk(ids, dists, attr, lo, hi, int(k))
                    else:
                        valid = ids != -1
                        ids = ids[valid]
                        dists = dists[valid]
                        mask = (attr[ids] >= lo) & (attr[ids] <= hi)
                        best_ids = ids[mask][:k]
                        best_dists = dists[mask][:k]
                    take = min(k, len(best_ids))
                    if take > 0:
                        D[qi, :take] = best_dists[:take]
                        I[qi, :take] = best_ids[:take]

    # Low-selectivity path: bucketed ANN indexes.
    for qi in low_qids:
        lo = int(q_los[qi])
        hi = int(q_his[qi])
        overlaps = []
        for bi, (b_lo, b_hi) in enumerate(bucket_ranges):
            if b_hi >= lo and b_lo <= hi:
                overlaps.append(bi)
        if not overlaps:
            continue

        per_bucket_k = max(1, int(np.ceil((k / len(overlaps)) * bucket_oversample)))
        cand_ids = []
        cand_dists = []
        params_bucket.efSearch = int(min(max_ef_search, max(base_ef_search, np.ceil(per_bucket_k * ef_search_mult))))

        for bi in overlaps:
            b_index = bucket_indexes[bi]
            gids = bucket_global_ids[bi]
            fk = min(b_index.ntotal, per_bucket_k)
            D_b, I_b = b_index.search(xq[qi:qi + 1], fk, params=params_bucket)
            lids = I_b[0]
            dists = D_b[0]
            valid = lids != -1
            if not np.any(valid):
                continue
            g = gids[lids[valid]]
            dd = dists[valid]
            m = (attr[g] >= lo) & (attr[g] <= hi)
            if np.any(m):
                cand_ids.append(g[m])
                cand_dists.append(dd[m])

        if cand_ids:
            all_ids = np.concatenate(cand_ids)
            all_dists = np.concatenate(cand_dists)
            order = np.argsort(all_dists)
            take = min(k, len(order))
            D[qi, :take] = all_dists[order[:take]]
            I[qi, :take] = all_ids[order[:take]]

    return D, I

###########################
# ACORN
##########################

def build_acorn_index(vecs, metadata, M=32, gamma=12, M_beta=32):
    metadata_vec = faiss.Int32Vector()
    faiss.copy_array_to_vector(metadata, metadata_vec)
    
    d = vecs.shape[1]
    index = faiss.IndexACORNFlat(int(d), M, gamma, metadata_vec, M_beta)
    index.add(vecs.astype('float32'))
    
    return index, metadata

def build_filter_map(metadata, ranges):
    """
    metadata: np.ndarray(N, )
    ranges: np.ndarray(nq, 2) --> [lo, hi] per query
    """
    metadata = np.asarray(metadata)
    nq = len(ranges)
    N = len(metadata)
    
    filter_map = np.empty(nq * N, dtype=np.int8)
    
    # Reuse buffers to avoid allocation churn
    mask = np.empty(N, dtype=bool)
    temp = np.empty(N, dtype=bool)
    
    for qi, (lo, hi) in enumerate(ranges):
        np.greater_equal(metadata, lo, out=mask)
        np.less_equal(metadata, hi, out=temp)
        np.logical_and(mask, temp, out=filter_map[qi * N : (qi + 1) * N].view(bool))
        
    return filter_map


def acorn_search(index, queries, filter_map, k=10, efSearch=256):
    nq = len(queries)
    xq = np.ascontiguousarray(queries, dtype='float32')

    D = np.empty((nq, k), dtype="float32")
    I = np.empty((nq, k), dtype=np.int64)
    
    params = faiss.SearchParametersACORN()
    params.efSearch = efSearch
    params.check_relative_distance = False
    
    faiss._swigfaiss.IndexACORN_search(
        index,
        int(nq),
        faiss.swig_ptr(xq),
        int(k),
        faiss.swig_ptr(D),
        faiss.swig_ptr(I),
        faiss.swig_ptr(filter_map),
        params
    )

    return D, I    
    
############################
# POST
############################   

def postfilter_search(
    index,
    attr,
    sorted_attr,
    sorted_idx,
    queries,
    ranges,
    k=10,
    base_ef_search=48,
    max_ef_search=384,
    ef_search_mult=1.0,
    selector_cache_max=4096,
    high_sel_threshold=0.08,
    oversample=1.15,
    max_steps=4,
    batch_size=512,
    max_fetch_k=768,
    selector_max_candidates=30000,
):
    nq = len(queries)
    D_out = np.full((nq, k), np.inf, dtype="float32")
    I_out = np.full((nq, k), -1, dtype="int64")

    faiss.omp_set_num_threads(_faiss_thread_count())

    params = faiss.SearchParametersHNSW()
    params_unfiltered = faiss.SearchParametersHNSW()
    selector_cache = {}
    n_total = len(sorted_attr)
    xq = np.ascontiguousarray(queries, dtype=np.float32)

    def get_selector_and_count(lo, hi):
        key = (int(lo), int(hi))
        cached = selector_cache.get(key)
        if cached is not None:
            return cached

        left = np.searchsorted(sorted_attr, lo, side="left")
        right = np.searchsorted(sorted_attr, hi, side="right")
        count = int(right - left)
        if count <= 0:
            result = (None, 0)
        else:
            candidate_ids = np.ascontiguousarray(sorted_idx[left:right], dtype=np.int64)
            result = (faiss.IDSelectorArray(candidate_ids), count)

        if len(selector_cache) < selector_cache_max:
            selector_cache[key] = result
        return result

    q_los = np.array([int(r[0]) for r in ranges], dtype=np.int32)
    q_his = np.array([int(r[1]) for r in ranges], dtype=np.int32)
    q_left = np.searchsorted(sorted_attr, q_los, side="left")
    q_right = np.searchsorted(sorted_attr, q_his, side="right")
    q_counts = q_right - q_left
    q_sel = q_counts / n_total

    # Route large candidate sets to the batched unfiltered+mask branch, which is
    # far cheaper than per-query selector searches at medium/high selectivity.
    high_mask = (q_counts > 0) & ((q_sel >= high_sel_threshold) | (q_counts >= selector_max_candidates))
    high_qids = np.where(high_mask)[0]
    low_qids = np.where((q_counts > 0) & (~high_mask))[0]

    # High-selectivity path batched by estimated fetch depth.
    high_fetch = np.ceil((k / np.maximum(q_sel[high_qids], 1e-4)) * oversample).astype(np.int32)
    fetch_cap = min(index.ntotal, int(max_fetch_k))
    high_fetch = np.clip(high_fetch, k, fetch_cap)
    high_fetch = ((high_fetch + 31) // 32) * 32
    high_fetch = np.clip(high_fetch, k, fetch_cap)

    fetch_to_qids = {}
    for qi, fk in zip(high_qids.tolist(), high_fetch.tolist()):
        fetch_to_qids.setdefault(int(fk), []).append(int(qi))

    for fetch_k, qids in fetch_to_qids.items():
        ef_search = int(min(max_ef_search, max(base_ef_search, np.ceil(fetch_k * ef_search_mult))))
        params_unfiltered.efSearch = ef_search
        qids = np.asarray(qids, dtype=np.int64)

        for s in range(0, len(qids), batch_size):
            bqids = qids[s:s + batch_size]
            if len(bqids) == 0:
                continue
            D_b, I_b = index.search(xq[bqids], int(fetch_k), params=params_unfiltered)
            for row, qi in enumerate(bqids):
                lo = int(q_los[qi])
                hi = int(q_his[qi])
                ids = I_b[row]
                dists = D_b[row]
                if filter_kernel is not None:
                    best_dists, best_ids = filter_kernel.filter_topk(ids, dists, attr, lo, hi, int(k))
                else:
                    valid = ids != -1
                    ids = ids[valid]
                    dists = dists[valid]
                    mask = (attr[ids] >= lo) & (attr[ids] <= hi)
                    best_ids = ids[mask][:k]
                    best_dists = dists[mask][:k]
                take = min(k, len(best_ids))
                if take > 0:
                    D_out[qi, :take] = best_dists[:take]
                    I_out[qi, :take] = best_ids[:take]

    # Low-selectivity selector path, grouped+batched by (lo, hi) to amortize selector cost.
    low_groups = {}
    for qi in low_qids:
        key = (int(q_los[qi]), int(q_his[qi]))
        low_groups.setdefault(key, []).append(int(qi))

    for (lo, hi), qids in low_groups.items():
        selector, candidate_count = get_selector_and_count(lo, hi)
        if selector is None:
            continue

        target_ef = int(np.ceil(k * ef_search_mult))
        ef_search = int(min(max_ef_search, max(base_ef_search, min(candidate_count, target_ef))))
        params.efSearch = ef_search
        params.sel = selector

        qids = np.asarray(qids, dtype=np.int64)
        for s in range(0, len(qids), batch_size):
            bqids = qids[s:s + batch_size]
            D_b, I_b = index.search(xq[bqids], k, params=params)
            D_out[bqids, :] = D_b
            I_out[bqids, :] = I_b

    return D_out, I_out

#############################

def compute_ground_truth(vecs, attr, ranges, queries, k, exact_index=None, batch_size=512):
    print("Computing exact ground truths (FAISS grouped by range)...")
    nq = len(queries)
    gts = [np.array([], dtype=np.int64) for _ in range(nq)]

    # Reuse caller-provided exact index if available.
    if exact_index is None:
        exact_index = faiss.IndexFlatL2(int(vecs.shape[1]))
        exact_index.add(np.ascontiguousarray(vecs, dtype=np.float32))

    xq = np.ascontiguousarray(queries, dtype=np.float32)
    sorted_idx = np.argsort(attr).astype(np.int64)
    sorted_attr = attr[sorted_idx]

    # Group queries by identical range to amortize selector construction.
    range_to_qids = {}
    for qi, (lo, hi) in enumerate(ranges):
        key = (int(lo), int(hi))
        if key not in range_to_qids:
            range_to_qids[key] = []
        range_to_qids[key].append(qi)

    params = faiss.SearchParameters()

    for (lo, hi), qids in range_to_qids.items():
        left = np.searchsorted(sorted_attr, lo, side="left")
        right = np.searchsorted(sorted_attr, hi, side="right")
        if right <= left:
            continue

        candidate_ids = np.ascontiguousarray(sorted_idx[left:right], dtype=np.int64)
        params.sel = faiss.IDSelectorArray(candidate_ids)

        qids = np.asarray(qids, dtype=np.int64)
        for s in range(0, len(qids), batch_size):
            bqids = qids[s:s + batch_size]
            D, I = exact_index.search(xq[bqids], k, params=params)
            for row, qi in enumerate(bqids):
                ids = I[row]
                gts[int(qi)] = ids[ids != -1]

    return gts


def recall_at_k(results, ground_truth):
    recalls = []

    for res, gt in zip(results, ground_truth):

        res = np.asarray(res).flatten()
        gt  = np.asarray(gt).flatten()

        if len(gt) == 0:
            recalls.append(1.0 if len(res) == 0 else 0.0)
            continue

        hits = len(set(res.tolist()) & set(gt.tolist()))

        recalls.append(hits / len(gt))

    return float(np.mean(recalls))


######################################
# EXPERIMENTS
######################################
import time

_METRIC_RE = re.compile(
    r"\[(pre|post)\]\s+total_time_s=([0-9.eE+-]+),\s+qps=([0-9.eE+-]+),\s+queries=(\d+)"
)


def _write_fvecs(path, vectors):
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    n, d = x.shape
    with open(path, "wb") as f:
        dim = np.array([d], dtype=np.int32)
        for i in range(n):
            dim.tofile(f)
            x[i].tofile(f)


def _write_ranges(path, ranges):
    with open(path, "w", encoding="utf-8") as f:
        for lo, hi in ranges:
            f.write(f"{int(lo)} {int(hi)}\n")


def _ensure_cpp_tools():
    if not _cpp_sources_newer_than_binaries():
        return
    cmd = [sys.executable, str(SETUP_KERNEL_PY), "build_ext", "--inplace"]
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parent))


def ensure_cpp_index(attr, tag="correlated"):
    _ensure_cpp_tools()
    out_dir = Path("/tmp/fvs_cache/cpp_indexes") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    attr_path = out_dir / "attr.npy"
    np.save(attr_path, np.asarray(attr, dtype=np.int32))

    required = [
        out_dir / "manifest.txt",
        out_dir / "pre_hnsw.faiss",
        out_dir / "post_hnsw.faiss",
        out_dir / "attr.i32",
        out_dir / "sorted_attr.i32",
        out_dir / "sorted_ids.i64",
        out_dir / "base_vectors.f32",
    ]
    needs_build = not all(p.exists() for p in required)
    if needs_build:
        cmd = [
            str(CPP_BUILD_BIN),
            "--base", str(BASE_FVECS_PATH),
            "--attr", str(attr_path),
            "--out-dir", str(out_dir),
        ]
        env = os.environ.copy()
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{FAISS_LIB_DIR}:{old}" if old else str(FAISS_LIB_DIR)
        subprocess.run(cmd, check=True, env=env)

    return out_dir


def _run_cpp_search(index_dir, queries, ranges, mode, k_eval):
    _ensure_cpp_tools()
    with tempfile.TemporaryDirectory(prefix=f"fvs_{mode}_") as td:
        td = Path(td)
        q_path = td / "queries.fvecs"
        r_path = td / "ranges.txt"
        i_path = td / "ids.i64"

        _write_fvecs(q_path, queries)
        _write_ranges(r_path, ranges)

        cmd = [
            str(CPP_SEARCH_BIN),
            "--index-dir", str(index_dir),
            "--queries", str(q_path),
            "--ranges", str(r_path),
            "--mode", mode,
            "--k", str(int(k_eval)),
            "--output-ids", str(i_path),
        ]

        if mode == "post":
            post_env_options = [
                ("FVS_POST_EF_SEARCH", "--post-ef-search"),
                ("FVS_POST_MAX_FETCH_K", "--post-max-fetch-k"),
                ("FVS_POST_MAX_RETRIES", "--post-max-retries"),
                ("FVS_POST_FETCH_MULT", "--post-fetch-mult"),
                ("FVS_POST_MIN_VALID_MULT", "--post-min-valid-mult"),
            ]
            for env_name, arg_name in post_env_options:
                value = os.environ.get(env_name)
                if value:
                    cmd.extend([arg_name, value])

        env = os.environ.copy()
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{FAISS_LIB_DIR}:{old}" if old else str(FAISS_LIB_DIR)
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)

        dt = None
        qps = None
        for line in proc.stdout.splitlines():
            m = _METRIC_RE.search(line.strip())
            if m and m.group(1) == mode:
                dt = float(m.group(2))
                qps = float(m.group(3))
                break
        if dt is None or qps is None:
            raise RuntimeError(f"Failed to parse {mode} metrics from C++ output:\n{proc.stdout}")

        ids = np.fromfile(i_path, dtype=np.int64)
        expected = len(queries) * int(k_eval)
        if ids.size != expected:
            raise RuntimeError(
                f"C++ {mode} ids size mismatch: got {ids.size}, expected {expected}"
            )
        I = ids.reshape(len(queries), int(k_eval))
        return dt, qps, I

def run_acorn(index, queries, metadata, ranges, k_eval, gt, batch_size=100):
    t0 = time.perf_counter()
    
    nq = len(queries)
    all_I = []
    
    for i in range(0, nq, batch_size):
        end = min(i + batch_size, nq)
        filter_map = build_filter_map(metadata, ranges[i:end])
        _, I = acorn_search(index, queries[i:end], filter_map, k=k_eval)
        all_I.append(I)
    
    dt = time.perf_counter() - t0

    results = [row for row in np.vstack(all_I)]

    recall = recall_at_k(results, gt)
    qps = len(queries) / dt
    return (dt, qps, recall)

def run_post(index, queries, attr, sorted_attr, sorted_idx, ranges, k_eval, gt):
    dt, qps, I = _run_cpp_search(index, queries, ranges, mode="post", k_eval=k_eval)
    results = [row for row in I]
    recall = recall_at_k(results, gt)
    return (dt, qps, recall)

def run_pre(index, sorted_attr, sorted_idx, queries, attr, ranges, k_eval, gt):
    dt, qps, I_pref = _run_cpp_search(index, queries, ranges, mode="pre", k_eval=k_eval)
    recall = recall_at_k(I_pref, gt)
    return (dt, qps, recall)
