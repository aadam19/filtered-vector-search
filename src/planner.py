import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

SIFT_DATASET = "sift"
AMAZON_DATASET = "amazon"
PRE_FILTER = "PRE"
POST_FILTER = "POST"
ACORN_FILTER = "ACORN"
PLANNER_STRATEGY = "PLANNER"


def _sel_tag(selectivity: float) -> str:
    tag = f"{selectivity:.8f}".rstrip("0").rstrip(".").replace(".", "p")
    return tag if tag else "0"


def _write_fvecs(path: Path, vectors: np.ndarray) -> None:
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    n, d = x.shape
    with path.open("wb") as handle:
        dim = np.array([d], dtype=np.int32)
        for i in range(n):
            dim.tofile(handle)
            x[i].tofile(handle)


def _default_mixed_batch_path(cache_dir: Path, selectivity: float, n_queries: int) -> Path:
    return cache_dir / f"mixed_test_set_sel_{_sel_tag(selectivity)}_q{int(n_queries)}.npz"


def _default_amazon_batch_path(
    cache_dir: Path,
    metadata_column: str,
    selectivity: float,
    n_queries: int,
    seed: int,
) -> Path:
    return cache_dir / (
        f"amazon_{metadata_column}_mixed_sel_{_sel_tag(selectivity)}"
        f"_q{int(n_queries)}_seed{seed}.npz"
    )


def _next_planner_experiment_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    x = 1
    while True:
        exp_dir = out_root / f"experiment{x}"
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True, exist_ok=True)
            return exp_dir
        x += 1


def _default_backend_names(labels: np.ndarray) -> np.ndarray:
    if np.any(np.asarray(labels) == 2):
        return np.array(["corr", "rand"])
    return np.array(["corr"])


def _build_backend_groups(mixed, state):
    backend_ids = mixed["backend_ids"]
    backend_names = list(mixed["backend_names"])
    groups = []
    for backend_id, backend_name in enumerate(backend_names):
        qids = np.where(backend_ids == backend_id)[0]
        if len(qids) == 0:
            continue
        backend = state[backend_name]
        groups.append((backend_name, qids, backend))
    return groups


def _compute_ground_truth_subset(state, backend, queries, ranges, k_eval):
    return state["compute_ground_truth"](
        state["vecs"],
        backend["metadata"],
        ranges,
        queries,
        k_eval,
        exact_index=state.get("exact_index"),
    )


def choose_plan(
    selectivity,
    correlation,
    tau,
    selectivity_threshold,
    negative_corr_selectivity_threshold=0.30,
):
    if not (0.0 <= selectivity <= 1.0):
        raise ValueError(f"selectivity must be in [0, 1], got {selectivity}")

    if selectivity <= float(selectivity_threshold):
        return PRE_FILTER
    if selectivity <= negative_corr_selectivity_threshold and correlation <= -float(tau):
        return PRE_FILTER
    return ACORN_FILTER


def plan_query(
    query_vector,
    query_range,
    sorted_values,
    index,
    cluster_stats,
    global_hist,
    global_cdf,
    bin_edges,
):
    from src.helper_funcs import compute_correlation, compute_selectivity, interpolated_metric_tau

    selectivity = compute_selectivity(sorted_values, query_range[0], query_range[1])
    correlation, js_divergence, lift = compute_correlation(
        query_vector=query_vector,
        query_range=query_range,
        index=index,
        clust_stats=cluster_stats,
        glob_hist=global_hist,
        glob_cdf=global_cdf,
        bin_edges=bin_edges,
    )
    selectivity_threshold = interpolated_metric_tau(selectivity, "true")
    tau = interpolated_metric_tau(selectivity, "heuristic")
    plan = choose_plan(
        selectivity,
        correlation,
        tau=tau,
        selectivity_threshold=selectivity_threshold,
    )
    return {
        "plan": plan,
        "selectivity": float(selectivity),
        "correlation": float(correlation),
        "tau": float(tau),
        "selectivity_threshold": float(selectivity_threshold),
        "js_divergence": float(js_divergence),
        "lift": float(lift),
    }


def plan_queries(
    queries,
    ranges,
    sorted_values,
    index,
    cluster_stats,
    global_hist,
    global_cdf,
    bin_edges,
):
    plans = []
    for query_vector, query_range in zip(queries, ranges):
        plans.append(
            plan_query(
                query_vector=query_vector,
                query_range=query_range,
                sorted_values=sorted_values,
                index=index,
                cluster_stats=cluster_stats,
                global_hist=global_hist,
                global_cdf=global_cdf,
                bin_edges=bin_edges,
            )
        )
    return plans


def _split_query_budget(total_queries: int, buckets: int) -> list[int]:
    total_queries = int(total_queries)
    buckets = int(buckets)
    if total_queries <= 0:
        raise ValueError(f"total_queries must be > 0, got {total_queries}")
    if buckets <= 0:
        raise ValueError(f"buckets must be > 0, got {buckets}")

    base = total_queries // buckets
    remainder = total_queries % buckets
    return [base + (1 if idx < remainder else 0) for idx in range(buckets)]


def build_mixed_test_set(
    selectivity: float,
    n_queries: int,
    seed: int,
    cache_dir: Path,
    out_path: Path | None = None,
):
    faiss = _import_faiss()

    from src.helper_funcs import (
        generate_query_ranges,
        generate_random_query_ranges,
        import_dataset,
    )

    if not (0.0 < selectivity <= 1.0):
        raise ValueError(f"selectivity must be in (0, 1], got {selectivity}")

    vecs, queries, _ = import_dataset()

    corr_attr_path = cache_dir / "cached_attr.npy"
    rand_attr_path = cache_dir / "rand_cached_attr.npy"
    if not corr_attr_path.exists():
        raise FileNotFoundError(f"Missing cached correlated attributes: {corr_attr_path}")
    if not rand_attr_path.exists():
        raise FileNotFoundError(f"Missing cached random attributes: {rand_attr_path}")

    attr = np.load(corr_attr_path)
    rand_attr = np.load(rand_attr_path)

    exact_index = faiss.IndexFlatL2(int(vecs.shape[1]))
    exact_index.add(vecs.astype(np.float32))

    k = np.int32(np.sqrt(vecs.shape[0]))
    pos_ranges, neg_ranges = generate_query_ranges(
        queries, attr, exact_index, k, selectivity=selectivity
    )
    rand_ranges = generate_random_query_ranges(
        rand_attr, len(queries), selectivity=selectivity
    )

    n_total = len(queries)
    rng = np.random.default_rng(seed)
    n_positive, n_negative, n_random = _split_query_budget(n_queries, 3)

    if n_positive + n_negative + n_random > n_total:
        raise ValueError(
            "Requested mixed batch is larger than available queries: "
            f"{n_positive + n_negative + n_random} > {n_total}"
        )

    p_ids = rng.choice(n_total, size=n_positive, replace=False)
    remaining = np.setdiff1d(np.arange(n_total), p_ids, assume_unique=False)
    n_ids = rng.choice(remaining, size=n_negative, replace=False)
    remaining = np.setdiff1d(remaining, n_ids, assume_unique=False)
    r_ids = rng.choice(remaining, size=n_random, replace=False)

    rows = []
    for qi in p_ids:
        lo, hi = pos_ranges[int(qi)]
        rows.append((int(qi), 0, int(lo), int(hi)))
    for qi in n_ids:
        lo, hi = neg_ranges[int(qi)]
        rows.append((int(qi), 1, int(lo), int(hi)))
    for qi in r_ids:
        lo, hi = rand_ranges[int(qi)]
        rows.append((int(qi), 2, int(lo), int(hi)))

    rng.shuffle(rows)

    query_ids = np.array([r[0] for r in rows], dtype=np.int32)
    labels = np.array([r[1] for r in rows], dtype=np.int32)  # 0=pos,1=neg,2=rand
    ranges = np.array([[r[2], r[3]] for r in rows], dtype=np.int32)
    xq = queries[query_ids].astype(np.float32)
    backend_ids = np.array([0 if label != 2 else 1 for label in labels], dtype=np.int32)

    if out_path is None:
        out_path = _default_mixed_batch_path(cache_dir, selectivity, n_queries)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        dataset=np.array([SIFT_DATASET]),
        query_ids=query_ids,
        queries=xq,
        ranges=ranges,
        labels=labels,
        label_names=np.array(["positive", "negative", "random"]),
        backend_ids=backend_ids,
        backend_names=np.array(["corr", "rand"]),
        selectivity=np.array([selectivity], dtype=np.float32),
    )
    return out_path, len(rows)


def load_mixed_test_set(path):
    print(f"Loading mixed test set from: {path}", flush=True)
    data = np.load(path, allow_pickle=True)
    labels = data["labels"].astype(np.int32)
    backend_ids = (
        data["backend_ids"].astype(np.int32)
        if "backend_ids" in data
        else np.array([0 if label != 2 else 1 for label in labels], dtype=np.int32)
    )
    backend_names = (
        data["backend_names"]
        if "backend_names" in data
        else _default_backend_names(labels)
    )
    dataset = (
        str(data["dataset"][0])
        if "dataset" in data
        else SIFT_DATASET
    )
    mixed = {
        "dataset": dataset,
        "query_ids": data["query_ids"].astype(np.int32),
        "queries": data["queries"].astype(np.float32),
        "ranges": data["ranges"].astype(np.int32),
        "labels": labels,
        "label_names": data["label_names"],
        "backend_ids": backend_ids,
        "backend_names": backend_names,
        "selectivity": float(data["selectivity"][0]),
    }
    if "metadata_column" in data:
        mixed["metadata_column"] = str(data["metadata_column"][0])
    if "metadata_kind" in data:
        mixed["metadata_kind"] = str(data["metadata_kind"][0])
    if "query_titles" in data:
        mixed["query_titles"] = data["query_titles"]
    if "query_metadata" in data:
        mixed["query_metadata"] = data["query_metadata"]
    if "query_filter_selectivities" in data:
        mixed["query_filter_selectivities"] = data["query_filter_selectivities"].astype(np.float32)
    if "metadata_scale" in data:
        mixed["metadata_scale"] = float(data["metadata_scale"][0])
    if "state_config_json" in data:
        mixed["state_config"] = json.loads(str(data["state_config_json"][0]))

    label_counts = np.bincount(mixed["labels"], minlength=len(mixed["label_names"]))
    label_summary = ", ".join(
        f"{str(name)}={int(label_counts[idx])}"
        for idx, name in enumerate(mixed["label_names"])
    )
    backend_counts = np.bincount(mixed["backend_ids"], minlength=len(mixed["backend_names"]))
    backend_summary = ", ".join(
        f"{str(name)}={int(backend_counts[idx])}"
        for idx, name in enumerate(mixed["backend_names"])
    )
    extra = f" | metadata={mixed['metadata_column']}" if "metadata_column" in mixed else ""
    print(
        "Loaded mixed batch: "
        f"dataset={mixed['dataset']} | queries={len(mixed['queries'])} | "
        f"{label_summary} | backends={backend_summary} | "
        f"selectivity={mixed['selectivity']:.6f}{extra}",
        flush=True,
    )
    return mixed


def _import_faiss():
    acorn_path = os.path.expanduser("~/filtered-vector-search/ACORN/build/faiss/python")
    acorn_lib = os.path.expanduser("~/filtered-vector-search/ACORN/build/faiss/libfaiss.so")
    acorn_callbacks = os.path.expanduser(
        "~/filtered-vector-search/ACORN/build/faiss/python/libfaiss_python_callbacks.so"
    )

    if acorn_path not in sys.path:
        sys.path.insert(0, acorn_path)
    if os.path.exists(acorn_lib):
        ctypes.CDLL(acorn_lib, mode=ctypes.RTLD_GLOBAL)
    if os.path.exists(acorn_callbacks):
        ctypes.CDLL(acorn_callbacks, mode=ctypes.RTLD_GLOBAL)

    import faiss

    return faiss


def _build_cluster_state(vecs, labels):
    class DummyKMeans:
        pass

    fitted_vecs = DummyKMeans()
    fitted_vecs.labels_ = labels.flatten()

    k_int = int(np.max(fitted_vecs.labels_)) + 1
    d = int(vecs.shape[1])
    centroids = np.zeros((k_int, d), dtype=np.float32)

    for i in range(k_int):
        cluster_points = vecs[fitted_vecs.labels_ == i]
        if len(cluster_points) > 0:
            centroids[i] = cluster_points.mean(axis=0)

    fitted_vecs.cluster_centers_ = centroids
    return fitted_vecs, centroids


def _default_amazon_cluster_count(n_rows: int) -> int:
    return max(1, min(256, max(32, int(np.sqrt(max(1, n_rows))))))


def _resolve_amazon_cache_dir(cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir).expanduser()

    if cache_dir.name == "amazon":
        return cache_dir
    if cache_dir.name == "cache" and cache_dir.parent.name == "amazon":
        return cache_dir

    return cache_dir / "amazon"


def _prepare_amazon_hybrid_data(
    data_dir: Path,
    cache_dir: Path,
    metadata_column: str,
    max_rows: int | None,
    embedding_batch_size: int,
    use_clean_cache: bool,
    use_embedding_cache: bool,
):
    from src.amazon_dataset import (
        EMBEDDING_MODEL_NAME,
        build_embedding_cache_key,
        create_embeddings,
        extract_filter_metadata,
        load_or_clean_data,
    )

    amazon_cache_dir = _resolve_amazon_cache_dir(cache_dir)
    amazon_cache_dir.mkdir(parents=True, exist_ok=True)
    cleaned_df, _, data_cache_key = load_or_clean_data(
        data_dir,
        max_rows,
        cache_dir=amazon_cache_dir,
        use_cache=use_clean_cache,
    )
    if len(cleaned_df) < 2:
        raise ValueError("Need at least 2 valid Amazon rows after cleaning.")

    embedding_cache_key = build_embedding_cache_key(
        cleaned_df,
        data_dir=data_dir,
        max_rows=max_rows,
        model_name=EMBEDDING_MODEL_NAME,
    )
    embeddings = create_embeddings(
        cleaned_df["text_for_embedding"],
        batch_size=embedding_batch_size,
        cache_dir=amazon_cache_dir,
        cache_key=embedding_cache_key,
        use_cache=use_embedding_cache,
    )

    valid_mask, metadata, meta_info = extract_filter_metadata(cleaned_df, metadata_column)
    filtered_df = cleaned_df.loc[valid_mask].reset_index(drop=True)
    filtered_embeddings = embeddings[valid_mask].astype(np.float32, copy=False)
    metadata_display = (
        filtered_df[meta_info["display_column"]].astype(str).to_numpy()
        if meta_info["metadata_kind"] == "categorical"
        else filtered_df[meta_info["display_column"]].to_numpy()
    )

    payload = {
        "data_cache_key": data_cache_key,
        "embedding_cache_key": embedding_cache_key,
        "metadata_column": meta_info["metadata_column"],
        "metadata_kind": meta_info["metadata_kind"],
        "max_rows": max_rows,
        "rows": int(len(filtered_df)),
    }
    dataset_key = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "df": filtered_df,
        "embeddings": filtered_embeddings,
        "metadata": metadata,
        "metadata_display": metadata_display,
        "meta_info": meta_info,
        "dataset_key": dataset_key,
        "data_dir": data_dir,
        "max_rows": max_rows,
        "embedding_batch_size": embedding_batch_size,
    }


def _split_amazon_indices(total_rows: int, query_pool_size: int, seed: int):
    if total_rows < 2:
        raise ValueError("Need at least 2 rows to split Amazon data into base/query sets.")
    if query_pool_size <= 0:
        raise ValueError(f"query_pool_size must be > 0, got {query_pool_size}")
    if query_pool_size >= total_rows:
        raise ValueError(
            f"query_pool_size must be smaller than total_rows ({query_pool_size} >= {total_rows})"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_rows)
    query_pool_ids = np.sort(perm[:query_pool_size])
    base_ids = np.sort(perm[query_pool_size:])
    return base_ids, query_pool_ids


def _compute_filter_selectivities(sorted_values: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    if len(ranges) == 0:
        return np.empty(0, dtype=np.float32)

    total = float(len(sorted_values))
    selectivities = np.empty(len(ranges), dtype=np.float32)
    for idx, (lo, hi) in enumerate(ranges):
        left = np.searchsorted(sorted_values, int(lo), side="left")
        right = np.searchsorted(sorted_values, int(hi), side="right")
        selectivities[idx] = float(right - left) / total
    return selectivities


def _generate_random_numeric_ranges(
    metadata: np.ndarray,
    n_queries: int,
    selectivity: float,
    seed: int,
) -> np.ndarray:
    metadata = np.asarray(metadata, dtype=np.int32)
    n_queries = int(n_queries)
    if n_queries < 0:
        raise ValueError(f"n_queries must be >= 0, got {n_queries}")
    if n_queries == 0:
        return np.empty((0, 2), dtype=np.int32)
    if len(metadata) == 0:
        raise ValueError("metadata must contain at least one value")

    rng = np.random.default_rng(seed)
    total = len(metadata)
    target = max(1, min(total, int(np.floor(float(selectivity) * total))))
    sorted_metadata = np.sort(metadata)
    starts = rng.integers(0, total - target + 1, size=n_queries)

    ranges = np.empty((n_queries, 2), dtype=np.int32)
    for idx, start in enumerate(starts):
        lo = int(sorted_metadata[int(start)])
        hi = int(sorted_metadata[min(total - 1, int(start) + target)])
        ranges[idx] = (lo, hi)
    return ranges


def _generate_random_category_ranges(
    metadata: np.ndarray,
    n_queries: int,
    selectivity: float,
    seed: int,
    relative_tolerance: float = 0.35,
    min_candidates: int = 8,
    max_candidates: int = 64,
) -> np.ndarray:
    metadata = np.asarray(metadata, dtype=np.int32)
    n_queries = int(n_queries)
    if n_queries < 0:
        raise ValueError(f"n_queries must be >= 0, got {n_queries}")
    if n_queries == 0:
        return np.empty((0, 2), dtype=np.int32)
    if len(metadata) == 0:
        raise ValueError("metadata must contain at least one value")

    rng = np.random.default_rng(seed)
    values, counts = np.unique(metadata, return_counts=True)
    freqs = counts.astype(np.float64) / float(len(metadata))
    target = float(selectivity)
    distances = np.abs(freqs - target)
    tolerance = max(1.0 / float(len(metadata)), target * float(relative_tolerance))
    candidate_mask = distances <= tolerance

    candidate_values = values[candidate_mask]
    candidate_distances = distances[candidate_mask]
    if candidate_values.size == 0:
        order = np.argsort(distances, kind="stable")
        keep = min(
            len(values),
            max(int(min_candidates), min(int(max_candidates), int(np.ceil(np.sqrt(len(values)))))),
        )
        candidate_values = values[order[:keep]]
    elif candidate_values.size > int(max_candidates):
        order = np.argsort(candidate_distances, kind="stable")
        candidate_values = candidate_values[order[: int(max_candidates)]]

    chosen = candidate_values[
        rng.choice(len(candidate_values), size=n_queries, replace=True)
    ].astype(np.int32, copy=False)
    return np.column_stack((chosen, chosen)).astype(np.int32, copy=False)


def _generate_amazon_random_filters(
    metadata: np.ndarray,
    metadata_kind: str,
    selectivity: float,
    n_queries: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    metadata_kind = str(metadata_kind).strip().lower()
    if metadata_kind == "numeric":
        ranges = _generate_random_numeric_ranges(metadata, n_queries, selectivity, seed)
    elif metadata_kind == "categorical":
        ranges = _generate_random_category_ranges(metadata, n_queries, selectivity, seed)
    else:
        raise ValueError(f"Unsupported metadata kind: {metadata_kind!r}")

    sorted_metadata = np.sort(np.asarray(metadata, dtype=np.int32))
    selectivities = _compute_filter_selectivities(sorted_metadata, ranges)
    return ranges, selectivities


def build_amazon_mixed_test_set(
    selectivity: float,
    n_queries: int,
    seed: int,
    cache_dir: Path,
    data_dir: Path,
    metadata_column: str,
    max_rows: int | None = None,
    embedding_batch_size: int = 256,
    query_fraction: float = 0.10,
    planner_clusters: int | None = None,
    out_path: Path | None = None,
    use_clean_cache: bool = True,
    use_embedding_cache: bool = True,
):
    if not (0.0 < selectivity <= 1.0):
        raise ValueError(f"selectivity must be in (0, 1], got {selectivity}")
    if not (0.0 < query_fraction < 1.0):
        raise ValueError(f"query_fraction must be in (0, 1), got {query_fraction}")

    prepared = _prepare_amazon_hybrid_data(
        data_dir=data_dir,
        cache_dir=cache_dir,
        metadata_column=metadata_column,
        max_rows=max_rows,
        embedding_batch_size=embedding_batch_size,
        use_clean_cache=use_clean_cache,
        use_embedding_cache=use_embedding_cache,
    )

    total_requested = int(n_queries)
    if total_requested <= 0:
        raise ValueError("Requested mixed batch must contain at least one Amazon query.")
    total_rows = len(prepared["metadata"])
    query_pool_size = max(total_requested, int(np.ceil(total_rows * query_fraction)))
    query_pool_size = min(query_pool_size, total_rows - 1)
    if query_pool_size < total_requested:
        raise ValueError(
            "Requested mixed batch is larger than the available held-out Amazon query pool: "
            f"{total_requested} > {query_pool_size}"
        )

    base_ids, query_pool_ids = _split_amazon_indices(total_rows, query_pool_size, seed)
    base_vecs = prepared["embeddings"][base_ids]
    base_metadata = prepared["metadata"][base_ids]
    query_pool_vecs = prepared["embeddings"][query_pool_ids]
    query_pool_titles = prepared["df"].iloc[query_pool_ids]["title"].astype(str).to_numpy()
    query_pool_metadata = prepared["metadata_display"][query_pool_ids]
    query_pool_ranges, query_pool_filter_selectivities = _generate_amazon_random_filters(
        base_metadata,
        metadata_kind=prepared["meta_info"]["metadata_kind"],
        selectivity=selectivity,
        n_queries=len(query_pool_vecs),
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    query_ids = rng.choice(len(query_pool_vecs), size=total_requested, replace=False).astype(np.int32)
    labels = np.zeros(len(query_ids), dtype=np.int32)
    ranges = query_pool_ranges[query_ids].astype(np.int32, copy=False)
    queries = query_pool_vecs[query_ids].astype(np.float32)
    query_row_ids = query_pool_ids[query_ids].astype(np.int32)
    backend_ids = np.zeros(len(query_ids), dtype=np.int32)

    cluster_count = int(
        planner_clusters
        if planner_clusters is not None
        else _default_amazon_cluster_count(len(base_vecs))
    )
    state_payload = {
        "dataset": AMAZON_DATASET,
        "prepared_key": prepared["dataset_key"],
        "metadata_column": prepared["meta_info"]["metadata_column"],
        "metadata_kind": prepared["meta_info"]["metadata_kind"],
        "data_dir": str(data_dir.resolve()),
        "max_rows": max_rows,
        "embedding_batch_size": embedding_batch_size,
        "split_seed": seed,
        "query_pool_size": int(query_pool_size),
        "planner_clusters": cluster_count,
    }
    state_key = hashlib.sha256(
        json.dumps(state_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    state_payload["state_key"] = state_key

    if out_path is None:
        out_path = _default_amazon_batch_path(
            cache_dir=cache_dir,
            metadata_column=prepared["meta_info"]["metadata_column"],
            selectivity=selectivity,
            n_queries=total_requested,
            seed=seed,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        dataset=np.array([AMAZON_DATASET]),
        query_ids=query_ids,
        query_row_ids=query_row_ids,
        queries=queries,
        ranges=ranges,
        labels=labels,
        label_names=np.array(["random"]),
        backend_ids=backend_ids,
        backend_names=np.array(["corr"]),
        selectivity=np.array([selectivity], dtype=np.float32),
        metadata_column=np.array([prepared["meta_info"]["metadata_column"]]),
        metadata_kind=np.array([prepared["meta_info"]["metadata_kind"]]),
        metadata_scale=np.array([prepared["meta_info"]["scale"]], dtype=np.float32),
        query_titles=query_pool_titles[query_ids],
        query_metadata=query_pool_metadata[query_ids],
        query_filter_selectivities=query_pool_filter_selectivities[query_ids],
        state_config_json=np.array([json.dumps(state_payload, sort_keys=True)]),
    )
    return out_path, len(query_ids)


def _load_amazon_benchmark_state(
    mixed,
    cache_dir: Path,
    use_clean_cache: bool,
    use_embedding_cache: bool,
):
    from sklearn.cluster import MiniBatchKMeans

    from src.helper_funcs import (
        ACORN_BUILD_TAG,
        build_acorn_index,
        choose_bins,
        compute_cluster_stats,
        compute_ground_truth,
        ensure_cpp_index,
        recall_at_k,
        run_acorn,
        run_post,
        run_pre,
    )

    if "state_config" not in mixed:
        raise ValueError("Amazon mixed batch is missing state_config metadata.")

    config = mixed["state_config"]
    data_dir = Path(os.path.expanduser(config["data_dir"]))
    prepared = _prepare_amazon_hybrid_data(
        data_dir=data_dir,
        cache_dir=cache_dir,
        metadata_column=config["metadata_column"],
        max_rows=config.get("max_rows"),
        embedding_batch_size=int(config.get("embedding_batch_size", 256)),
        use_clean_cache=use_clean_cache,
        use_embedding_cache=use_embedding_cache,
    )
    if prepared["dataset_key"] != config["prepared_key"]:
        raise ValueError(
            "Amazon dataset cache no longer matches the mixed batch metadata. "
            "Regenerate the mixed batch so the query split and base index stay aligned."
        )

    total_rows = len(prepared["metadata"])
    base_ids, _ = _split_amazon_indices(
        total_rows=total_rows,
        query_pool_size=int(config["query_pool_size"]),
        seed=int(config["split_seed"]),
    )
    vecs = prepared["embeddings"][base_ids].astype(np.float32, copy=False)
    metadata = prepared["metadata"][base_ids].astype(np.int32, copy=False)
    if len(vecs) < 2:
        raise ValueError("Need at least 2 Amazon base vectors after the train/query split.")

    faiss = _import_faiss()
    exact_index = faiss.IndexFlatL2(int(vecs.shape[1]))
    exact_index.add(vecs)

    cluster_count = max(1, min(int(config["planner_clusters"]), len(vecs)))
    metadata_kind = str(prepared["meta_info"].get("metadata_kind", "numeric"))
    print(
        f"Building Amazon planner state | base_vectors={len(vecs)} | "
        f"metadata={config['metadata_column']} ({metadata_kind}) | clusters={cluster_count}",
        flush=True,
    )
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=min(10000, len(vecs)),
        random_state=int(config["split_seed"]),
        n_init="auto",
    ).fit(vecs)

    threshold = 1.3
    if metadata_kind == "categorical":
        category_count = len(prepared["meta_info"].get("value_labels") or [])
        bin_edges = np.arange(category_count + 1, dtype=np.float64) - 0.5
    else:
        bin_edges = choose_bins(metadata)
    cluster_stats, global_hist, global_cdf = compute_cluster_stats(
        kmeans, vecs, metadata, threshold, bin_edges
    )

    centroid_idx = faiss.IndexHNSWFlat(int(kmeans.cluster_centers_.shape[1]), 32)
    centroid_idx.add(np.asarray(kmeans.cluster_centers_, dtype=np.float32))

    amazon_cache_dir = _resolve_amazon_cache_dir(cache_dir)
    amazon_cache_dir.mkdir(parents=True, exist_ok=True)
    state_key = config["state_key"]
    acorn_cache_path = amazon_cache_dir / f"acorn_{ACORN_BUILD_TAG}_{state_key}.faiss"
    if acorn_cache_path.exists():
        print(f"Loading cached Amazon ACORN index from {acorn_cache_path}", flush=True)
        acorn_index = faiss.read_index(str(acorn_cache_path))
    else:
        print("Building Amazon ACORN index from scratch...", flush=True)
        acorn_index, _ = build_acorn_index(vecs, metadata)
        faiss.write_index(acorn_index, str(acorn_cache_path))

    base_fvecs_path = amazon_cache_dir / f"base_{state_key}.fvecs"
    if not base_fvecs_path.exists():
        print(f"Materializing Amazon base vectors to {base_fvecs_path}", flush=True)
        _write_fvecs(base_fvecs_path, vecs)

    sorted_attr = np.sort(metadata)
    sorted_idx = np.argsort(metadata).astype(np.int64)
    cpp_index_dir = ensure_cpp_index(
        metadata,
        tag=f"amazon_{state_key}",
        base_fvecs_path=base_fvecs_path,
    )

    return {
        "vecs": vecs,
        "exact_index": exact_index,
        "compute_ground_truth": compute_ground_truth,
        "run_pre": run_pre,
        "run_post": run_post,
        "run_acorn": run_acorn,
        "corr": {
            "metadata": metadata,
            "metadata_kind": metadata_kind,
            "metadata_value_labels": (
                np.asarray(prepared["meta_info"].get("value_labels") or [], dtype=str)
                if metadata_kind == "categorical"
                else None
            ),
            "cpp_index_dir": cpp_index_dir,
            "sorted_attr": sorted_attr,
            "sorted_idx": sorted_idx,
            "acorn_index": acorn_index,
            "index": centroid_idx,
            "cluster_stats": cluster_stats,
            "global_hist": global_hist,
            "global_cdf": global_cdf,
            "bin_edges": bin_edges,
        },
    }


def _load_benchmark_state(cache_dir):
    print(f"Loading benchmark state from cache dir: {cache_dir}", flush=True)
    faiss = _import_faiss()

    from src.helper_funcs import (
        ACORN_BUILD_TAG,
        build_acorn_index,
        choose_bins,
        compute_cluster_stats,
        compute_ground_truth,
        ensure_cpp_index,
        import_dataset,
        recall_at_k,
        run_acorn,
        run_post,
        run_pre,
    )

    print("Importing dataset...", flush=True)
    vecs, _, _ = import_dataset()
    print(f"Dataset loaded: {vecs.shape[0]} base vectors, dim={vecs.shape[1]}", flush=True)
    exact_index = faiss.IndexFlatL2(int(vecs.shape[1]))
    exact_index.add(np.ascontiguousarray(vecs, dtype=np.float32))

    print("Loading cached correlated/random attributes and labels...", flush=True)
    attr = np.load(cache_dir / "cached_attr.npy")
    rand_attr = np.load(cache_dir / "rand_cached_attr.npy")
    labels = np.load(cache_dir / "cached_labels.npy")

    print("Reconstructing cluster state from cached labels...", flush=True)
    fitted_vecs, centroids = _build_cluster_state(vecs, labels)

    threshold = 1.3
    print("Computing correlated cluster statistics...", flush=True)
    bin_edges = choose_bins(attr)
    cluster_stats, global_hist, global_cdf = compute_cluster_stats(
        fitted_vecs, vecs, attr, threshold, bin_edges
    )

    print("Computing random cluster statistics...", flush=True)
    rand_bin_edges = choose_bins(rand_attr)
    rand_cluster_stats, rand_global_hist, rand_global_cdf = compute_cluster_stats(
        fitted_vecs, vecs, rand_attr, threshold, rand_bin_edges
    )

    print("Building centroid routing index...", flush=True)
    centroid_idx = faiss.IndexHNSWFlat(int(centroids.shape[1]), 32)
    centroid_idx.add(centroids)

    acorn_cache_path = cache_dir / f"acorn_correlated_{ACORN_BUILD_TAG}.faiss"
    acorn_rand_cache_path = cache_dir / f"acorn_random_{ACORN_BUILD_TAG}.faiss"
    if acorn_cache_path.exists() and acorn_rand_cache_path.exists():
        print("Loading cached ACORN indexes...", flush=True)
        acorn_index = faiss.read_index(str(acorn_cache_path))
        acorn_index_rand = faiss.read_index(str(acorn_rand_cache_path))
    else:
        print("Building ACORN indexes from scratch...", flush=True)
        acorn_index, _ = build_acorn_index(vecs, attr)
        acorn_index_rand, _ = build_acorn_index(vecs, rand_attr)
        faiss.write_index(acorn_index, str(acorn_cache_path))
        faiss.write_index(acorn_index_rand, str(acorn_rand_cache_path))

    print("Preparing sorted metadata views and C++ index directories...", flush=True)
    sorted_attr = np.sort(attr)
    sorted_attr_idx = np.argsort(attr).astype(np.int64)
    sorted_rand_attr = np.sort(rand_attr)
    sorted_rand_attr_idx = np.argsort(rand_attr).astype(np.int64)

    return {
        "vecs": vecs,
        "exact_index": exact_index,
        "recall_at_k": recall_at_k,
        "compute_ground_truth": compute_ground_truth,
        "run_pre": run_pre,
        "run_post": run_post,
        "run_acorn": run_acorn,
        "corr": {
            "metadata": attr,
            "cpp_index_dir": ensure_cpp_index(attr, tag="correlated"),
            "sorted_attr": sorted_attr,
            "sorted_idx": sorted_attr_idx,
            "acorn_index": acorn_index,
            "index": centroid_idx,
            "cluster_stats": cluster_stats,
            "global_hist": global_hist,
            "global_cdf": global_cdf,
            "bin_edges": bin_edges,
        },
        "rand": {
            "metadata": rand_attr,
            "cpp_index_dir": ensure_cpp_index(rand_attr, tag="random"),
            "sorted_attr": sorted_rand_attr,
            "sorted_idx": sorted_rand_attr_idx,
            "acorn_index": acorn_index_rand,
            "index": centroid_idx,
            "cluster_stats": rand_cluster_stats,
            "global_hist": rand_global_hist,
            "global_cdf": rand_global_cdf,
            "bin_edges": rand_bin_edges,
        },
    }


def _weighted_metrics(results, total_queries):
    total_time = sum(item["dt"] for item in results)
    weighted_recall = sum(item["recall"] * item["nq"] for item in results) / total_queries
    qps = total_queries / total_time if total_time > 0 else float("inf")
    return total_time, qps, weighted_recall


def _subset_gt(gt, qids):
    return [gt[int(i)] for i in qids]


def _run_single_strategy(name, mixed, state, k_eval):
    queries = mixed["queries"]
    ranges = mixed["ranges"]
    total_queries = len(queries)
    print(f"\nRunning strategy: {name}", flush=True)

    results = []
    for backend_name, qids, backend in _build_backend_groups(mixed, state):
        print(
            f"  Backend={backend_name} | queries={len(qids)} | computing ground truth...",
            flush=True,
        )
        q = queries[qids]
        r = ranges[qids]
        gt = _compute_ground_truth_subset(state, backend, q, r, k_eval)

        if name == PRE_FILTER:
            print(f"  Backend={backend_name} | executing PRE...", flush=True)
            dt, _, recall = state["run_pre"](
                backend["cpp_index_dir"],
                backend["sorted_attr"],
                backend["sorted_idx"],
                q,
                backend["metadata"],
                r,
                k_eval,
                gt,
            )
        elif name == POST_FILTER:
            print(f"  Backend={backend_name} | executing POST...", flush=True)
            dt, _, recall = state["run_post"](
                backend["cpp_index_dir"],
                q,
                backend["metadata"],
                backend["sorted_attr"],
                backend["sorted_idx"],
                r,
                k_eval,
                gt,
            )
        elif name == ACORN_FILTER:
            print(f"  Backend={backend_name} | executing ACORN...", flush=True)
            dt, _, recall = state["run_acorn"](
                backend["acorn_index"],
                q,
                backend["metadata"],
                r,
                k_eval,
                gt,
            )
        else:
            raise ValueError(f"Unknown strategy: {name}")

        print(
            f"  Backend={backend_name} finished | time={dt:.6f}s | recall={recall:.4f}",
            flush=True,
        )
        results.append({"dt": dt, "recall": recall, "nq": len(qids)})

    dt, qps, recall = _weighted_metrics(results, total_queries)
    print(
        f"Strategy {name} done | total_time={dt:.6f}s | qps={qps:.2f} | recall={recall:.4f}",
        flush=True,
    )
    return {"strategy": name, "dt": dt, "qps": qps, "recall": recall}


def _run_planner_strategy(mixed, state, k_eval):
    queries = mixed["queries"]
    ranges = mixed["ranges"]
    total_queries = len(queries)
    print(f"\nRunning strategy: {PLANNER_STRATEGY}", flush=True)

    results = []
    plan_counts = {PRE_FILTER: 0, POST_FILTER: 0, ACORN_FILTER: 0}

    for backend_name, qids, backend in _build_backend_groups(mixed, state):
        print(
            f"  Backend={backend_name} | queries={len(qids)} | computing ground truth...",
            flush=True,
        )
        q = queries[qids]
        r = ranges[qids]
        gt = _compute_ground_truth_subset(state, backend, q, r, k_eval)
        print(f"  Backend={backend_name} | planning queries...", flush=True)
        plans = plan_queries(
            q,
            r,
            backend["sorted_attr"],
            backend["index"],
            backend["cluster_stats"],
            backend["global_hist"],
            backend["global_cdf"],
            backend["bin_edges"],
        )

        pre_local = np.array([i for i, plan in enumerate(plans) if plan["plan"] == PRE_FILTER], dtype=np.int64)
        acorn_local = np.array([i for i, plan in enumerate(plans) if plan["plan"] == ACORN_FILTER], dtype=np.int64)

        plan_counts[PRE_FILTER] += len(pre_local)
        plan_counts[ACORN_FILTER] += len(acorn_local)
        print(
            f"  Backend={backend_name} planner routing | PRE={len(pre_local)} | ACORN={len(acorn_local)}",
            flush=True,
        )

        if len(pre_local) > 0:
            print(f"  Backend={backend_name} | executing planned PRE subset...", flush=True)
            dt, _, recall = state["run_pre"](
                backend["cpp_index_dir"],
                backend["sorted_attr"],
                backend["sorted_idx"],
                q[pre_local],
                backend["metadata"],
                r[pre_local],
                k_eval,
                _subset_gt(gt, pre_local),
            )
            results.append({"dt": dt, "recall": recall, "nq": len(pre_local)})

        if len(acorn_local) > 0:
            print(f"  Backend={backend_name} | executing planned ACORN subset...", flush=True)
            dt, _, recall = state["run_acorn"](
                backend["acorn_index"],
                q[acorn_local],
                backend["metadata"],
                r[acorn_local],
                k_eval,
                _subset_gt(gt, acorn_local),
            )
            results.append({"dt": dt, "recall": recall, "nq": len(acorn_local)})

    dt, qps, recall = _weighted_metrics(results, total_queries)
    print(
        f"Strategy {PLANNER_STRATEGY} done | "
        f"total_time={dt:.6f}s | qps={qps:.2f} | recall={recall:.4f} | "
        f"PRE={plan_counts[PRE_FILTER]} | POST={plan_counts[POST_FILTER]} | ACORN={plan_counts[ACORN_FILTER]}",
        flush=True,
    )
    return {
        "strategy": PLANNER_STRATEGY,
        "dt": dt,
        "qps": qps,
        "recall": recall,
        "plan_counts": plan_counts,
    }


def evaluate_mixed_batch(
    mixed_batch_path,
    cache_dir,
    k_eval=100,
    use_clean_cache=True,
    use_embedding_cache=True,
):
    print(
        f"Starting mixed-batch evaluation | batch={mixed_batch_path} | cache_dir={cache_dir} | k={k_eval}",
        flush=True,
    )
    mixed = load_mixed_test_set(mixed_batch_path)
    if mixed["dataset"] == AMAZON_DATASET:
        state = _load_amazon_benchmark_state(
            mixed,
            cache_dir=cache_dir,
            use_clean_cache=use_clean_cache,
            use_embedding_cache=use_embedding_cache,
        )
    else:
        state = _load_benchmark_state(cache_dir)

    results = [
        _run_single_strategy(PRE_FILTER, mixed, state, k_eval),
        _run_single_strategy(POST_FILTER, mixed, state, k_eval),
        _run_single_strategy(ACORN_FILTER, mixed, state, k_eval),
        _run_planner_strategy(mixed, state, k_eval),
    ]
    print("Finished mixed-batch evaluation.", flush=True)
    return mixed, results


def plot_mixed_results(results, out_path):
    print(f"Plotting summary metrics to: {out_path}", flush=True)
    strategies = [item["strategy"] for item in results]
    recalls = [item["recall"] for item in results]
    qps = [item["qps"] for item in results]
    times = [item["dt"] for item in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [recalls, qps, times]
    titles = ["Recall@k", "Queries per Second", "Query Time (s)"]

    for ax, values, title in zip(axes, metrics, titles):
        ax.bar(strategies, values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.4)
        if title == "Recall@k":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda y, _: f"{y:,.0f}" if y >= 1 else f"{y:.2g}")
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_results_csv(results, out_path):
    print(f"Writing summary metrics to: {out_path}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["strategy,time_s,qps,recall,planner_pre,planner_post,planner_acorn"]
    for item in results:
        counts = item.get("plan_counts", {})
        lines.append(
            ",".join(
                [
                    item["strategy"],
                    f"{item['dt']:.12f}",
                    f"{item['qps']:.12f}",
                    f"{item['recall']:.12f}",
                    str(counts.get(PRE_FILTER, "")),
                    str(counts.get(POST_FILTER, "")),
                    str(counts.get(ACORN_FILTER, "")),
                ]
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Build and evaluate a mixed query batch for PRE, POST, ACORN, and the heuristic planner."
    )
    general = parser.add_argument_group("General")
    general.add_argument(
        "--dataset",
        choices=[SIFT_DATASET, AMAZON_DATASET],
        default=SIFT_DATASET,
        help="Dataset backing the mixed query batch.",
    )
    general.add_argument("--sel", type=float, default=0.01, help="Target selectivity in (0, 1].")
    general.add_argument(
        "--n-queries",
        type=int,
        default=100,
        help="Total number of mixed-batch queries to sample.",
    )
    general.add_argument("--seed", type=int, default=42, help="Random seed.")
    general.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp/fvs_cache",
        help="Cache directory for mixed batches, indexes, and summary outputs.",
    )
    general.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output .npz path for the mixed batch.",
    )
    general.add_argument(
        "--mixed-batch",
        type=str,
        default=None,
        help="Existing mixed batch .npz to evaluate. Default: cache-derived path from --sel and --n-queries.",
    )
    general.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the mixed batch with PRE, POST, ACORN, and the planner.",
    )
    general.add_argument(
        "--k-eval",
        type=int,
        default=100,
        help="Recall/evaluation k.",
    )
    general.add_argument(
        "--plot-out",
        type=str,
        default=None,
        help="Optional output path for the summary plot.",
    )
    general.add_argument(
        "--results-out",
        type=str,
        default=None,
        help="Optional CSV path for the summary metrics.",
    )
    amazon = parser.add_argument_group("Amazon")
    amazon.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Amazon data directory when --dataset amazon is used.",
    )
    amazon.add_argument(
        "--metadata-column",
        choices=["price", "rating", "category"],
        default="price",
        help="Amazon metadata field used for filtering.",
    )
    amazon.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max rows per Amazon CSV for faster iterations.",
    )
    amazon.add_argument(
        "--embedding-batch-size",
        type=int,
        default=256,
        help="Sentence-transformers batch size for Amazon embeddings.",
    )
    amazon.add_argument(
        "--query-fraction",
        type=float,
        default=0.10,
        help="Held-out Amazon query-pool fraction used to build the mixed batch.",
    )
    amazon.add_argument(
        "--planner-clusters",
        type=int,
        default=None,
        help="Override the Amazon KMeans cluster count used by the planner.",
    )
    amazon.add_argument(
        "--no-clean-cache",
        action="store_true",
        help="Disable reuse of cached cleaned Amazon data.",
    )
    amazon.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help="Disable reuse of cached Amazon embeddings.",
    )
    args = parser.parse_args()

    cache_dir = Path(os.path.expanduser(args.cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = (
        Path(os.path.expanduser(args.data_dir))
        if args.data_dir
        else None
    )
    out = Path(args.out).expanduser() if args.out else None
    mixed_batch_path = (
        Path(args.mixed_batch).expanduser()
        if args.mixed_batch
        else (
            out
            if out is not None
            else (
                _default_amazon_batch_path(
                    cache_dir=cache_dir,
                    metadata_column=args.metadata_column,
                    selectivity=args.sel,
                    n_queries=args.n_queries,
                    seed=args.seed,
                )
                if args.dataset == AMAZON_DATASET
                else _default_mixed_batch_path(cache_dir, args.sel, args.n_queries)
            )
        )
    )

    if not mixed_batch_path.exists():
        if args.dataset == AMAZON_DATASET:
            from src.amazon_dataset import DEFAULT_DATA_DIR

            out_path, n = build_amazon_mixed_test_set(
                selectivity=args.sel,
                n_queries=args.n_queries,
                seed=args.seed,
                cache_dir=cache_dir,
                data_dir=(data_dir if data_dir is not None else DEFAULT_DATA_DIR),
                metadata_column=args.metadata_column,
                max_rows=args.max_rows,
                embedding_batch_size=args.embedding_batch_size,
                query_fraction=args.query_fraction,
                planner_clusters=args.planner_clusters,
                out_path=mixed_batch_path,
                use_clean_cache=not args.no_clean_cache,
                use_embedding_cache=not args.no_embedding_cache,
            )
        else:
            out_path, n = build_mixed_test_set(
                selectivity=args.sel,
                n_queries=args.n_queries,
                seed=args.seed,
                cache_dir=cache_dir,
                out_path=mixed_batch_path,
            )
        print(f"Saved mixed test set with {n} queries to: {out_path}")
    else:
        print(f"Using mixed test set: {mixed_batch_path}")

    if args.evaluate:
        mixed, results = evaluate_mixed_batch(
            mixed_batch_path,
            cache_dir,
            k_eval=args.k_eval,
            use_clean_cache=not args.no_clean_cache,
            use_embedding_cache=not args.no_embedding_cache,
        )
        query_count_suffix = f"_q{len(mixed['queries'])}"
        metadata_suffix = (
            f"_{mixed['metadata_column']}"
            if "metadata_column" in mixed
            else ""
        )
        print(
            f"Evaluated {len(mixed['queries'])} mixed queries on {mixed['dataset']} "
            f"at selectivity {mixed['selectivity']:.6f}"
        )
        for item in results:
            print(
                f"{item['strategy']:>7} | time={item['dt']:.6f}s | qps={item['qps']:.2f} | recall={item['recall']:.4f}"
            )
            if item["strategy"] == PLANNER_STRATEGY:
                counts = item["plan_counts"]
                print(
                    f"         planner_counts: PRE={counts[PRE_FILTER]}, POST={counts[POST_FILTER]}, ACORN={counts[ACORN_FILTER]}"
                )

        planner_output_dir = None
        if not args.plot_out or not args.results_out:
            planner_output_dir = _next_planner_experiment_dir(ROOT_DIR / "src" / "aws")
            print(f"Planner output directory created at: {planner_output_dir}")

        plot_out = (
            Path(args.plot_out).expanduser()
            if args.plot_out
            else planner_output_dir
            / f"{mixed['dataset']}{metadata_suffix}{query_count_suffix}_mixed_eval_plot_sel_{_sel_tag(mixed['selectivity'])}.png"
        )
        plot_mixed_results(results, plot_out)
        print(f"Saved summary plot to: {plot_out}")

        results_out = (
            Path(args.results_out).expanduser()
            if args.results_out
            else planner_output_dir
            / f"{mixed['dataset']}{metadata_suffix}{query_count_suffix}_mixed_eval_sel_{_sel_tag(mixed['selectivity'])}.csv"
        )
        write_results_csv(results, results_out)
        print(f"Saved summary CSV to: {results_out}")


if __name__ == "__main__":
    main()
