# C++ Faiss Pre/Post Filter (Two-Phase)

This folder provides two separate executables so you can benchmark search independently from index build:

- `build_indexes`: offline build step.
- `search_filters`: query-time pre/post filter search benchmark.

Implemented baseline behavior:
- `pre`: build candidate ID set from attribute range, then search with `IDSelectorArray`.
- `post`: run ANN search without selector, then apply attribute filter on returned neighbors.

## Build

```bash
cmake -S src/cpp -B src/cpp/build
cmake --build src/cpp/build -j
```

## 1) Build Index Artifacts (offline)

```bash
./src/cpp/build/build_indexes \
  --base siftsmall/siftsmall_base.fvecs \
  --attr /tmp/fvs_cache/cached_attr.npy \
  --out-dir /tmp/fvs_cpp_idx
```

Outputs in `--out-dir`:

- `post_hnsw.faiss`
- `pre_hnsw.faiss`
- `attr.i32`, `sorted_attr.i32`, `sorted_ids.i64`
- `manifest.txt`

## 2) Run Search Only (evaluation step)

Prepare a ranges file with one line per query:

```text
10 35
42 79
...
```

Then run:

```bash
./src/cpp/build/search_filters \
  --index-dir /tmp/fvs_cpp_idx \
  --queries siftsmall/siftsmall_query.fvecs \
  --ranges /path/to/ranges.txt \
  --mode both \
  --k 100
```

`search_filters` prints total time and QPS for:

- `pre` (pre-filter path)
- `post` (post-filter path)

Use `--mode pre` or `--mode post` to benchmark one strategy only.
