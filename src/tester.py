from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import ctypes
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "amazon"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out" / "amazon_embedding_analysis"
ACORN_FAISS_PYTHON = ROOT_DIR / "ACORN" / "build" / "faiss" / "python"
ACORN_FAISS_LIB = ROOT_DIR / "ACORN" / "build" / "faiss" / "libfaiss.so"
ACORN_FAISS_CALLBACKS = ACORN_FAISS_PYTHON / "libfaiss_python_callbacks.so"

sys.path.insert(0, str(ACORN_FAISS_PYTHON))

if ACORN_FAISS_LIB.exists():
    ctypes.CDLL(str(ACORN_FAISS_LIB), mode=ctypes.RTLD_GLOBAL)
if ACORN_FAISS_CALLBACKS.exists():
    ctypes.CDLL(str(ACORN_FAISS_CALLBACKS), mode=ctypes.RTLD_GLOBAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze correlation between product embeddings and structured attributes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing Amazon CSV files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for plots and optional exports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors to analyze for each item.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for sentence-transformers encoding.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows loaded from each CSV for faster experiments.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for shuffling baselines.",
    )
    parser.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help="Disable reuse of saved title embeddings.",
    )
    parser.add_argument(
        "--no-clean-cache",
        action="store_true",
        help="Disable reuse of saved cleaned data.",
    )
    parser.add_argument(
        "--analysis-sample-size",
        type=int,
        default=20000,
        help="Number of rows to use for neighbor analysis and plots (default: 20000).",
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    missing = []
    for module_name in (
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "sentence_transformers",
        "faiss",
    ):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(
            "Missing required Python packages: "
            f"{missing_list}\n"
            "Install the Python packages for this repo environment and make sure "
            "the local ACORN/FAISS build is available under ACORN/build/faiss."
        )


ensure_dependencies()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


TITLE_CANDIDATES = [
    "title",
    "name",
    "product_title",
    "product_name",
    "item_name",
]
DESCRIPTION_CANDIDATES = [
    "description",
    "product_description",
    "about_product",
    "about this item",
    "details",
]
CATEGORY_CANDIDATES = [
    "category",
    "main_category",
    "sub_category",
    "product_category",
]
PRICE_CANDIDATES = [
    "price",
    "discount_price",
    "actual_price",
    "sale_price",
    "list_price",
]
RATING_CANDIDATES = [
    "rating",
    "ratings",
    "review_rating",
    "stars",
]
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def normalize_column_lookup(columns: list[str]) -> dict[str, str]:
    return {str(col).strip().lower(): col for col in columns}


def first_present(columns: list[str], candidates: list[str]) -> str | None:
    lookup = normalize_column_lookup(columns)
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match
    return None


def load_data(data_dir: Path, max_rows: int | None) -> pd.DataFrame:
    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    frames = []
    for csv_path in csv_paths:
        try:
            frame = pd.read_csv(
                csv_path,
                nrows=max_rows,
                encoding="utf-8",
                low_memory=False,
            )
        except UnicodeDecodeError:
            frame = pd.read_csv(
                csv_path,
                nrows=max_rows,
                encoding="latin-1",
                low_memory=False,
            )

        if frame.empty:
            print(f"Skipping empty file: {csv_path.name}")
            continue

        frame["__source_file"] = csv_path.name
        frames.append(frame)

    if not frames:
        raise ValueError(f"CSV files were found in {data_dir}, but none contained rows.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    print("\n[1] LOAD DATA")
    print(f"Loaded {len(frames)} non-empty CSV files from {data_dir}")
    print(f"Number of rows: {len(merged):,}")
    print(f"Column names: {list(merged.columns)}")
    print("First few rows:")
    print(merged.head().to_string(index=False))
    return merged


def build_data_cache_key(data_dir: Path, max_rows: int | None) -> str:
    hasher = hashlib.sha256()
    csv_paths = sorted(data_dir.glob("*.csv"))
    config = {
        "data_dir": str(data_dir.resolve()),
        "max_rows": max_rows,
        "file_count": len(csv_paths),
    }
    hasher.update(json.dumps(config, sort_keys=True).encode("utf-8"))
    for csv_path in csv_paths:
        stat = csv_path.stat()
        file_info = f"{csv_path.name}|{stat.st_size}|{stat.st_mtime_ns}"
        hasher.update(file_info.encode("utf-8"))
    return hasher.hexdigest()[:16]


def parse_price_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_rating_series(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def simplify_category_value(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    parts = re.split(r"\s*(?:>|/|\\|\||,|:)\s*", text)
    parts = [part.strip() for part in parts if part.strip()]
    return parts[0] if parts else text


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    title_col = first_present(df.columns.tolist(), TITLE_CANDIDATES)
    description_col = first_present(df.columns.tolist(), DESCRIPTION_CANDIDATES)
    category_col = first_present(df.columns.tolist(), CATEGORY_CANDIDATES)
    price_col = first_present(df.columns.tolist(), PRICE_CANDIDATES)
    rating_col = first_present(df.columns.tolist(), RATING_CANDIDATES)

    if title_col is None:
        raise ValueError("Could not find a product title/name column.")

    cleaned = pd.DataFrame()
    cleaned["title"] = df[title_col].astype(str).str.strip()
    cleaned["description"] = (
        df[description_col].astype(str).str.strip() if description_col else np.nan
    )
    cleaned["category"] = df[category_col].map(simplify_category_value) if category_col else np.nan
    cleaned["price"] = parse_price_series(df[price_col]) if price_col else np.nan
    cleaned["rating"] = parse_rating_series(df[rating_col]) if rating_col else np.nan
    cleaned["source_file"] = df["__source_file"] if "__source_file" in df else "unknown"

    cleaned["title"] = cleaned["title"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    cleaned = cleaned.dropna(subset=["title"]).reset_index(drop=True)

    if description_col:
        cleaned["text_for_embedding"] = np.where(
            cleaned["description"].notna()
            & cleaned["description"].astype(str).str.strip().ne("")
            & cleaned["description"].astype(str).str.lower().ne("nan"),
            cleaned["title"] + ". " + cleaned["description"].astype(str),
            cleaned["title"],
        )
    else:
        cleaned["text_for_embedding"] = cleaned["title"]

    print("\n[2] CLEAN DATA")
    print(f"Detected columns -> title: {title_col}, description: {description_col}, category: {category_col}, price: {price_col}, rating: {rating_col}")
    print(f"Rows after dropping missing titles: {len(cleaned):,}")
    print("Cleaned preview:")
    print(cleaned[["title", "category", "price", "rating", "source_file"]].head().to_string(index=False))

    return cleaned, {
        "title": title_col,
        "description": description_col,
        "category": category_col,
        "price": price_col,
        "rating": rating_col,
    }


def save_cleaned_cache(
    cleaned_df: pd.DataFrame,
    selected_columns: dict[str, str | None],
    cache_dir: Path,
    cache_key: str,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cleaned_cache_path = cache_dir / f"cleaned_{cache_key}.csv"
    meta_cache_path = cache_dir / f"cleaned_{cache_key}.json"
    cleaned_df.to_csv(cleaned_cache_path, index=False)
    meta_cache_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "rows": int(len(cleaned_df)),
                "columns": list(cleaned_df.columns),
                "selected_columns": selected_columns,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved cleaned data cache to {cleaned_cache_path}")
    return cleaned_cache_path


def print_cleaned_data_overview(cleaned_df: pd.DataFrame) -> None:
    print("\n[2] CLEAN DATA")
    print(f"Rows after dropping missing titles: {len(cleaned_df):,}")
    print("Cleaned preview:")
    print(
        cleaned_df[["title", "category", "price", "rating", "source_file"]]
        .head()
        .to_string(index=False)
    )


def load_or_clean_data(
    data_dir: Path,
    max_rows: int | None,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[pd.DataFrame, dict[str, str | None], str]:
    data_cache_key = build_data_cache_key(data_dir, max_rows)
    cleaned_cache_path = cache_dir / f"cleaned_{data_cache_key}.csv"
    meta_cache_path = cache_dir / f"cleaned_{data_cache_key}.json"

    if use_cache and cleaned_cache_path.exists():
        print("\n[1] LOAD DATA")
        print(f"Loading cached cleaned data from {cleaned_cache_path}")
        cleaned_df = pd.read_csv(cleaned_cache_path)
        selected_columns: dict[str, str | None] = {}
        if meta_cache_path.exists():
            selected_columns = json.loads(meta_cache_path.read_text(encoding="utf-8")).get(
                "selected_columns",
                {},
            )
        if selected_columns:
            print(
                "Detected columns -> "
                f"title: {selected_columns.get('title')}, "
                f"description: {selected_columns.get('description')}, "
                f"category: {selected_columns.get('category')}, "
                f"price: {selected_columns.get('price')}, "
                f"rating: {selected_columns.get('rating')}"
            )
        print_cleaned_data_overview(cleaned_df)
        return cleaned_df, selected_columns, data_cache_key

    raw_df = load_data(data_dir, max_rows)
    cleaned_df, selected_columns = clean_data(raw_df)
    save_cleaned_cache(cleaned_df, selected_columns, cache_dir, data_cache_key)
    return cleaned_df, selected_columns, data_cache_key


def build_embedding_cache_key(
    df: pd.DataFrame,
    data_dir: Path,
    max_rows: int | None,
    model_name: str,
) -> str:
    hasher = hashlib.sha256()
    csv_paths = sorted(data_dir.glob("*.csv"))

    config = {
        "model_name": model_name,
        "max_rows": max_rows,
        "row_count": int(len(df)),
        "text_column": "text_for_embedding",
    }
    hasher.update(json.dumps(config, sort_keys=True).encode("utf-8"))

    for csv_path in csv_paths:
        stat = csv_path.stat()
        file_info = f"{csv_path.name}|{stat.st_size}|{stat.st_mtime_ns}"
        hasher.update(file_info.encode("utf-8"))

    for text in df["text_for_embedding"].astype(str):
        hasher.update(text.encode("utf-8", errors="replace"))
        hasher.update(b"\0")

    return hasher.hexdigest()[:16]


def create_embeddings(
    texts: pd.Series,
    batch_size: int,
    cache_dir: Path,
    cache_key: str,
    use_cache: bool,
) -> np.ndarray:
    print("\n[3] CREATE EMBEDDINGS")
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_path = cache_dir / f"embeddings_{cache_key}.npy"
    meta_cache_path = cache_dir / f"embeddings_{cache_key}.json"

    if use_cache and embedding_cache_path.exists():
        print(f"Loading cached embeddings from {embedding_cache_path}")
        embeddings = np.load(embedding_cache_path)
        print(f"Embedding matrix shape: {embeddings.shape}")
        return embeddings.astype(np.float32, copy=False)

    print(f"Loading sentence-transformers model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        texts.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = normalize(embeddings, norm="l2")
    np.save(embedding_cache_path, embeddings.astype(np.float32, copy=False))
    meta_cache_path.write_text(
        json.dumps(
            {
                "model_name": EMBEDDING_MODEL_NAME,
                "rows": int(len(texts)),
                "batch_size": int(batch_size),
                "cache_key": cache_key,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved embeddings cache to {embedding_cache_path}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    return embeddings.astype(np.float32, copy=False)


def build_neighbor_cache_key(embedding_cache_key: str, k: int) -> str:
    payload = {
        "embedding_cache_key": embedding_cache_key,
        "k": int(k),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def select_analysis_subset(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    if sample_size <= 0:
        raise ValueError(f"analysis sample size must be > 0, got {sample_size}")

    total_rows = len(df)
    if sample_size >= total_rows:
        analysis_key = hashlib.sha256(
            json.dumps(
                {
                    "mode": "full",
                    "rows": total_rows,
                    "seed": int(seed),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        print(f"Using full dataset for analysis: {total_rows:,} rows")
        return df.reset_index(drop=True), embeddings, analysis_key

    rng = np.random.default_rng(seed)
    sample_indices = np.sort(rng.choice(total_rows, size=sample_size, replace=False))
    analysis_df = df.iloc[sample_indices].reset_index(drop=True)
    analysis_embeddings = embeddings[sample_indices]
    analysis_key = hashlib.sha256(sample_indices.tobytes()).hexdigest()[:16]
    print(
        f"Using analysis sample of {sample_size:,} rows "
        f"from {total_rows:,} total rows (seed={seed})"
    )
    return analysis_df, analysis_embeddings, analysis_key


def compute_neighbors(
    embeddings: np.ndarray,
    k: int,
    cache_dir: Path,
    embedding_cache_key: str,
) -> np.ndarray:
    effective_k = min(k + 1, len(embeddings))
    if effective_k <= 1:
        raise ValueError("Need at least 2 rows to compute nearest neighbors.")

    print("\n[4] NEAREST NEIGHBORS")
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / f"faiss_ip_index_{embedding_cache_key}.index"
    neighbor_cache_key = build_neighbor_cache_key(embedding_cache_key, k)
    neighbor_indices_path = cache_dir / f"neighbor_indices_{neighbor_cache_key}.npy"
    neighbor_scores_path = cache_dir / f"neighbor_scores_{neighbor_cache_key}.npy"

    if neighbor_indices_path.exists() and neighbor_scores_path.exists():
        print(f"Loading cached neighbor search results from {neighbor_indices_path}")
        neighbor_indices = np.load(neighbor_indices_path)
        neighbor_scores = np.load(neighbor_scores_path)
        neighbor_distances = 1.0 - neighbor_scores
        print(f"Computed {neighbor_indices.shape[1]} neighbors per point (excluding self).")
        print(f"Mean cosine distance to neighbors: {neighbor_distances.mean():.4f}")
        return neighbor_indices

    if index_path.exists():
        print(f"Loading cached FAISS index from {index_path}")
        index = faiss.read_index(str(index_path))
    else:
        print(
            f"Building FAISS IndexFlatIP for {len(embeddings):,} normalized embeddings "
            f"with dimension {embeddings.shape[1]}..."
        )
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        faiss.write_index(index, str(index_path))
        print(f"Saved FAISS index to {index_path}")

    scores, indices = index.search(np.ascontiguousarray(embeddings, dtype=np.float32), effective_k)
    neighbor_indices = indices[:, 1:]
    neighbor_scores = scores[:, 1:]
    neighbor_distances = 1.0 - neighbor_scores
    np.save(neighbor_indices_path, neighbor_indices.astype(np.int32, copy=False))
    np.save(neighbor_scores_path, neighbor_scores.astype(np.float32, copy=False))
    print(f"Saved neighbor search cache to {neighbor_indices_path}")
    print(f"Computed {neighbor_indices.shape[1]} neighbors per point (excluding self).")
    print(f"Mean cosine distance to neighbors: {neighbor_distances.mean():.4f}")
    return neighbor_indices


def numeric_neighbor_diff(values: pd.Series, neighbor_indices: np.ndarray) -> tuple[float, int]:
    arr = values.to_numpy(dtype=float)
    neighbor_vals = arr[neighbor_indices]
    base_vals = arr[:, None]
    valid = np.isfinite(base_vals) & np.isfinite(neighbor_vals)
    if not np.any(valid):
        return float("nan"), 0
    diffs = np.abs(base_vals - neighbor_vals)
    return float(diffs[valid].mean()), int(valid.sum())


def shuffled_numeric_baseline(
    values: pd.Series,
    neighbor_indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, int]:
    arr = values.to_numpy(dtype=float).copy()
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return float("nan"), 0
    arr[valid] = rng.permutation(arr[valid])
    return numeric_neighbor_diff(pd.Series(arr), neighbor_indices)


def categorical_neighbor_match(values: pd.Series, neighbor_indices: np.ndarray) -> tuple[float, int]:
    arr = values.astype("object").to_numpy()
    neighbor_vals = arr[neighbor_indices]
    base_vals = arr[:, None]
    valid = pd.notna(base_vals) & pd.notna(neighbor_vals)
    if not np.any(valid):
        return float("nan"), 0
    matches = neighbor_vals == base_vals
    return float(matches[valid].mean()), int(valid.sum())


def shuffled_categorical_baseline(
    values: pd.Series,
    neighbor_indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, int]:
    arr = values.astype("object").to_numpy().copy()
    valid = pd.notna(arr)
    if valid.sum() == 0:
        return float("nan"), 0
    arr[valid] = rng.permutation(arr[valid])
    return categorical_neighbor_match(pd.Series(arr), neighbor_indices)


def analyze_correlations(
    df: pd.DataFrame,
    neighbor_indices: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    print("\n[5] CORRELATION ANALYSIS")
    rng = np.random.default_rng(seed)
    results: list[dict[str, object]] = []

    for attr in ("price", "rating"):
        real_value, pair_count = numeric_neighbor_diff(df[attr], neighbor_indices)
        random_value, _ = shuffled_numeric_baseline(df[attr], neighbor_indices, rng)
        available_count = int(df[attr].notna().sum())
        if pair_count == 0:
            print(f"{attr}: skipped (no usable numeric values)")
            continue
        print(
            f"{attr}: real diff = {real_value:.4f}, random diff = {random_value:.4f}, usable rows = {available_count:,}"
        )
        results.append(
            {
                "attribute": attr,
                "type": "numeric",
                "real": real_value,
                "random": random_value,
                "direction": "lower_is_more_correlated",
                "available_rows": available_count,
                "pair_count": pair_count,
            }
        )

    real_value, pair_count = categorical_neighbor_match(df["category"], neighbor_indices)
    random_value, _ = shuffled_categorical_baseline(df["category"], neighbor_indices, rng)
    available_count = int(df["category"].notna().sum())
    if pair_count == 0:
        print("category: skipped (no usable categorical values)")
    else:
        print(
            f"category: real same-category fraction = {real_value:.4f}, random fraction = {random_value:.4f}, usable rows = {available_count:,}"
        )
        results.append(
            {
                "attribute": "category",
                "type": "categorical",
                "real": real_value,
                "random": random_value,
                "direction": "higher_is_more_correlated",
                "available_rows": available_count,
                "pair_count": pair_count,
            }
        )

    return results


def category_plot_labels(series: pd.Series, top_n: int = 10) -> pd.Series:
    if series.dropna().empty:
        return series
    top_categories = series.value_counts(dropna=True).head(top_n).index
    return series.where(series.isin(top_categories), other="Other")


def save_visualizations(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    neighbor_indices: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n[6] VISUALIZATION")
    print(f"Saving plots to {output_dir}")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    print(
        f"PCA explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, "
        f"PC2={pca.explained_variance_ratio_[1]:.4f}"
    )

    if df["price"].notna().any():
        valid = df["price"].notna().to_numpy()
        valid_prices = df.loc[valid, "price"].astype(float).to_numpy()
        positive_prices = valid_prices[valid_prices > 0]
        norm = None
        if positive_prices.size:
            vmin = max(float(np.percentile(positive_prices, 5)), 1e-6)
            vmax = float(np.percentile(positive_prices, 95))
            if vmax > vmin:
                norm = LogNorm(vmin=vmin, vmax=vmax)
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(
            coords[valid, 0],
            coords[valid, 1],
            c=valid_prices,
            s=10,
            cmap="viridis",
            alpha=0.7,
            norm=norm,
        )
        ax.set_title("PCA of Product Embeddings Colored by Price (Log-Clipped)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label="Price")
        fig.tight_layout()
        fig.savefig(output_dir / "pca_price.png", dpi=200)
        plt.close(fig)

        price_vals = df["price"].to_numpy(dtype=float)
        neighbor_price = price_vals[neighbor_indices]
        base_price = price_vals[:, None]
        valid_pairs = np.isfinite(base_price) & np.isfinite(neighbor_price)
        diffs = np.abs(base_price - neighbor_price)[valid_pairs]
        if diffs.size:
            positive_diffs = diffs[diffs > 0]
            if positive_diffs.size:
                log_bins = np.geomspace(positive_diffs.min(), positive_diffs.max(), 60)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(positive_diffs, bins=log_bins, color="#2a7fff", alpha=0.85)
                ax.set_xscale("log")
                ax.set_title("Histogram of Neighbor Price Differences (Log X Scale)")
                ax.set_xlabel("Absolute Price Difference")
                ax.set_ylabel("Count")
                fig.tight_layout()
                fig.savefig(output_dir / "neighbor_price_diff_hist.png", dpi=200)
                plt.close(fig)

            zoom_cutoff = float(np.percentile(diffs, 95))
            zoomed_diffs = diffs[diffs <= zoom_cutoff]
            if zoomed_diffs.size:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(zoomed_diffs, bins=60, color="#1f9d55", alpha=0.85)
                ax.set_title("Histogram of Neighbor Price Differences (Lower 95%)")
                ax.set_xlabel("Absolute Price Difference")
                ax.set_ylabel("Count")
                fig.tight_layout()
                fig.savefig(output_dir / "neighbor_price_diff_hist_zoom.png", dpi=200)
                plt.close(fig)

    if df["category"].notna().any():
        category_labels = category_plot_labels(df["category"])
        valid = category_labels.notna().to_numpy()
        if valid.any():
            fig, ax = plt.subplots(figsize=(11, 8))
            categories = pd.Categorical(category_labels[valid])
            cmap = plt.get_cmap("tab10", len(categories.categories))
            sc = ax.scatter(
                coords[valid, 0],
                coords[valid, 1],
                c=categories.codes,
                s=10,
                cmap=cmap,
                alpha=0.7,
            )
            ax.set_title("PCA of Product Embeddings Colored by Category")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            handles, _ = sc.legend_elements()
            ax.legend(handles, list(categories.categories), title="Category", loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / "pca_category.png", dpi=200)
            plt.close(fig)


def print_summary(results: list[dict[str, object]]) -> None:
    print("\n[7] OUTPUT")
    if not results:
        print("No attributes had enough valid data to evaluate.")
        return

    correlated = []
    not_correlated = []

    for result in results:
        attr = str(result["attribute"])
        real = float(result["real"])
        random_value = float(result["random"])
        direction = str(result["direction"])

        if direction == "lower_is_more_correlated":
            shows_signal = real < random_value
            comparison = f"real diff {real:.4f} vs random diff {random_value:.4f}"
        else:
            shows_signal = real > random_value
            comparison = f"real match {real:.4f} vs random match {random_value:.4f}"

        if shows_signal:
            correlated.append(f"{attr} ({comparison})")
        else:
            not_correlated.append(f"{attr} ({comparison})")

    print("Attributes showing correlation:")
    if correlated:
        for line in correlated:
            print(f"  - {line}")
    else:
        print("  - None")

    print("Attributes not clearly showing correlation:")
    if not_correlated:
        for line in not_correlated:
            print(f"  - {line}")
    else:
        print("  - None")

    print("Short conclusion:")
    if correlated:
        print(
            "Embeddings appear to preserve some structured-attribute locality, because real neighbor statistics beat the shuffled baselines for at least one attribute."
        )
    else:
        print(
            "The evaluated embeddings do not show a strong structured-attribute signal beyond the shuffled baselines for the available attributes."
        )


def main() -> None:
    args = parse_args()
    cache_dir = args.output_dir / "cache"
    cleaned_df, _, _ = load_or_clean_data(
        args.data_dir,
        args.max_rows,
        cache_dir=cache_dir,
        use_cache=not args.no_clean_cache,
    )

    if len(cleaned_df) < 2:
        raise ValueError("Need at least 2 cleaned rows to run nearest-neighbor analysis.")

    cache_key = build_embedding_cache_key(
        cleaned_df,
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        model_name=EMBEDDING_MODEL_NAME,
    )
    embeddings = create_embeddings(
        cleaned_df["text_for_embedding"],
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        cache_key=cache_key,
        use_cache=not args.no_embedding_cache,
    )
    analysis_df, analysis_embeddings, analysis_subset_key = select_analysis_subset(
        cleaned_df,
        embeddings,
        sample_size=args.analysis_sample_size,
        seed=args.random_seed,
    )
    neighbor_indices = compute_neighbors(
        analysis_embeddings,
        k=args.k,
        cache_dir=cache_dir,
        embedding_cache_key=f"{cache_key}_{analysis_subset_key}",
    )
    results = analyze_correlations(analysis_df, neighbor_indices, seed=args.random_seed)
    save_visualizations(analysis_df, analysis_embeddings, neighbor_indices, args.output_dir)
    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.")
