import argparse
import ctypes
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

PRE_FILTER = "PRE"
POST_FILTER = "POST"
ACORN_FILTER = "ACORN"


def _sel_tag(selectivity: float) -> str:
    tag = f"{selectivity:.8f}".rstrip("0").rstrip(".").replace(".", "p")
    return tag if tag else "0"


def choose_plan(
    selectivity,
    correlation,
    tiny_selectivity_threshold=0.005,
    negative_corr_selectivity_threshold=0.30,
):
    if not (0.0 <= selectivity <= 1.0):
        raise ValueError(f"selectivity must be in [0, 1], got {selectivity}")

    if selectivity <= tiny_selectivity_threshold:
        return PRE_FILTER
    if selectivity <= negative_corr_selectivity_threshold and correlation < 0:
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
    from src.helper_funcs import compute_correlation, compute_selectivity

    selectivity = compute_selectivity(sorted_values, query_range[0], query_range[1])
    correlation, js_divergence, lift = compute_correlation(
        query_vector=query_vector,
        query_range=query_range,
        index=index,
        cluster_stats=cluster_stats,
        global_hist=global_hist,
        global_cdf=global_cdf,
        bin_edges=bin_edges,
    )
    plan = choose_plan(selectivity, correlation)
    return {
        "plan": plan,
        "selectivity": float(selectivity),
        "correlation": float(correlation),
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


def build_mixed_test_set(
    selectivity: float,
    n_positive: int,
    n_negative: int,
    n_random: int,
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

    if out_path is None:
        out_path = cache_dir / f"mixed_test_set_sel_{_sel_tag(selectivity)}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        query_ids=query_ids,
        queries=xq,
        ranges=ranges,
        labels=labels,
        label_names=np.array(["positive", "negative", "random"]),
        selectivity=np.array([selectivity], dtype=np.float32),
    )
    return out_path, len(rows)


def load_mixed_test_set(path):
    print(f"Loading mixed test set from: {path}", flush=True)
    data = np.load(path, allow_pickle=True)
    mixed = {
        "query_ids": data["query_ids"].astype(np.int32),
        "queries": data["queries"].astype(np.float32),
        "ranges": data["ranges"].astype(np.int32),
        "labels": data["labels"].astype(np.int32),
        "label_names": data["label_names"],
        "selectivity": float(data["selectivity"][0]),
    }
    label_counts = np.bincount(mixed["labels"], minlength=3)
    print(
        "Loaded mixed batch: "
        f"{len(mixed['queries'])} queries | "
        f"positive={int(label_counts[0])}, negative={int(label_counts[1])}, random={int(label_counts[2])} | "
        f"selectivity={mixed['selectivity']:.6f}",
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


def _load_benchmark_state(cache_dir):
    print(f"Loading benchmark state from cache dir: {cache_dir}", flush=True)
    faiss = _import_faiss()

    from src.helper_funcs import (
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

    acorn_cache_path = cache_dir / "acorn_correlated.faiss"
    acorn_rand_cache_path = cache_dir / "acorn_random.faiss"
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
        "attr": attr,
        "rand_attr": rand_attr,
        "recall_at_k": recall_at_k,
        "compute_ground_truth": compute_ground_truth,
        "run_pre": run_pre,
        "run_post": run_post,
        "run_acorn": run_acorn,
        "corr": {
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
    labels = mixed["labels"]
    total_queries = len(queries)
    print(f"\nRunning strategy: {name}", flush=True)

    groups = [
        ("corr", np.where(labels != 2)[0], state["attr"]),
        ("rand", np.where(labels == 2)[0], state["rand_attr"]),
    ]

    results = []
    for backend_name, qids, metadata in groups:
        if len(qids) == 0:
            continue
        backend = state[backend_name]
        print(
            f"  Backend={backend_name} | queries={len(qids)} | computing ground truth...",
            flush=True,
        )
        q = queries[qids]
        r = ranges[qids]
        gt = state["compute_ground_truth"](state["vecs"], metadata, r, q, k_eval)

        if name == PRE_FILTER:
            print(f"  Backend={backend_name} | executing PRE...", flush=True)
            dt, _, recall = state["run_pre"](
                backend["cpp_index_dir"],
                backend["sorted_attr"],
                backend["sorted_idx"],
                q,
                metadata,
                r,
                k_eval,
                gt,
            )
        elif name == POST_FILTER:
            print(f"  Backend={backend_name} | executing POST...", flush=True)
            dt, _, recall = state["run_post"](
                backend["cpp_index_dir"],
                q,
                metadata,
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
                metadata,
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
    labels = mixed["labels"]
    total_queries = len(queries)
    print("\nRunning strategy: PLANNER", flush=True)

    groups = [
        ("corr", np.where(labels != 2)[0], state["attr"]),
        ("rand", np.where(labels == 2)[0], state["rand_attr"]),
    ]

    results = []
    plan_counts = {PRE_FILTER: 0, POST_FILTER: 0, ACORN_FILTER: 0}

    for backend_name, qids, metadata in groups:
        if len(qids) == 0:
            continue
        backend = state[backend_name]
        print(
            f"  Backend={backend_name} | queries={len(qids)} | computing ground truth...",
            flush=True,
        )
        q = queries[qids]
        r = ranges[qids]
        gt = state["compute_ground_truth"](state["vecs"], metadata, r, q, k_eval)
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
                metadata,
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
                metadata,
                r[acorn_local],
                k_eval,
                _subset_gt(gt, acorn_local),
            )
            results.append({"dt": dt, "recall": recall, "nq": len(acorn_local)})

    dt, qps, recall = _weighted_metrics(results, total_queries)
    print(
        "Strategy PLANNER done | "
        f"total_time={dt:.6f}s | qps={qps:.2f} | recall={recall:.4f} | "
        f"PRE={plan_counts[PRE_FILTER]} | POST={plan_counts[POST_FILTER]} | ACORN={plan_counts[ACORN_FILTER]}",
        flush=True,
    )
    return {
        "strategy": "PLANNER",
        "dt": dt,
        "qps": qps,
        "recall": recall,
        "plan_counts": plan_counts,
    }


def evaluate_mixed_batch(mixed_batch_path, cache_dir, k_eval=100):
    print(
        f"Starting mixed-batch evaluation | batch={mixed_batch_path} | cache_dir={cache_dir} | k={k_eval}",
        flush=True,
    )
    mixed = load_mixed_test_set(mixed_batch_path)
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


def _default_mixed_batch_path(cache_dir, selectivity):
    return cache_dir / f"mixed_test_set_sel_{_sel_tag(selectivity)}.npz"


def main():
    parser = argparse.ArgumentParser(
        description="Build and evaluate a mixed query batch for PRE, POST, ACORN, and the heuristic planner."
    )
    parser.add_argument("--sel", type=float, default=0.01, help="Selectivity in (0, 1].")
    parser.add_argument("--n-pos", type=int, default=34, help="Number of positive queries.")
    parser.add_argument("--n-neg", type=int, default=33, help="Number of negative queries.")
    parser.add_argument("--n-rand", type=int, default=33, help="Number of random queries.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp/fvs_cache",
        help="Cache directory used by script.py.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output .npz path for the mixed batch.",
    )
    parser.add_argument(
        "--mixed-batch",
        type=str,
        default=None,
        help="Existing mixed batch .npz to evaluate. Default: cache-derived path from --sel.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the mixed batch with PRE, POST, ACORN, and the planner.",
    )
    parser.add_argument(
        "--k-eval",
        type=int,
        default=100,
        help="Recall/evaluation k.",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default=None,
        help="Optional output path for the summary plot.",
    )
    args = parser.parse_args()

    cache_dir = Path(os.path.expanduser(args.cache_dir))
    out = Path(args.out).expanduser() if args.out else None
    mixed_batch_path = (
        Path(args.mixed_batch).expanduser()
        if args.mixed_batch
        else (out if out is not None else _default_mixed_batch_path(cache_dir, args.sel))
    )

    if not mixed_batch_path.exists():
        out_path, n = build_mixed_test_set(
            selectivity=args.sel,
            n_positive=args.n_pos,
            n_negative=args.n_neg,
            n_random=args.n_rand,
            seed=args.seed,
            cache_dir=cache_dir,
            out_path=mixed_batch_path,
        )
        print(f"Saved mixed test set with {n} queries to: {out_path}")
    else:
        print(f"Using mixed test set: {mixed_batch_path}")

    if args.evaluate:
        mixed, results = evaluate_mixed_batch(mixed_batch_path, cache_dir, k_eval=args.k_eval)
        print(f"Evaluated {len(mixed['queries'])} mixed queries at selectivity {mixed['selectivity']:.6f}")
        for item in results:
            print(
                f"{item['strategy']:>7} | time={item['dt']:.6f}s | qps={item['qps']:.2f} | recall={item['recall']:.4f}"
            )
            if item["strategy"] == "PLANNER":
                counts = item["plan_counts"]
                print(
                    f"         planner_counts: PRE={counts[PRE_FILTER]}, POST={counts[POST_FILTER]}, ACORN={counts[ACORN_FILTER]}"
                )

        plot_out = (
            Path(args.plot_out).expanduser()
            if args.plot_out
            else cache_dir / f"mixed_eval_plot_sel_{_sel_tag(mixed['selectivity'])}.png"
        )
        plot_mixed_results(results, plot_out)
        print(f"Saved summary plot to: {plot_out}")


if __name__ == "__main__":
    main()
