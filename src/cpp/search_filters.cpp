#include "fvs_common.h"

#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct SearchConfig {
    std::string index_dir;
    std::string queries_path;
    std::string ranges_path;
    std::string mode = "both";
    std::string output_ids_path;
    int k = 100;
    int pre_ef_search = 4096;
    int post_ef_search = 1024;
    int post_max_fetch_k = 32768;
    int post_max_retries = 7;
    double post_fetch_mult = 2.5;
    double post_min_valid_mult = 12.0;
};

struct RunSummary {
    std::string name;
    double dt_sec = 0.0;
    double qps = 0.0;
    int k = 0;
    std::vector<int64_t> ids;
};

void usage() {
    std::cout
        << "search_filters\n"
        << "  --index-dir <directory>\n"
        << "  --queries <query.fvecs>\n"
        << "  --ranges <ranges.txt>\n"
        << "  [--mode pre|post|both] (default: both)\n"
        << "  [--output-ids <ids.i64>]\n"
        << "  [--k 100]\n"
        << "  [--pre-ef-search 4096]\n"
        << "  [--post-ef-search 1024]\n"
        << "  [--post-max-fetch-k 32768]\n"
        << "  [--post-max-retries 7]\n"
        << "  [--post-fetch-mult 2.5]\n"
        << "  [--post-min-valid-mult 12.0]\n";
}

int parse_int(const std::string& s, const std::string& name) {
    try {
        return std::stoi(s);
    } catch (...) {
        throw std::runtime_error("Invalid integer for " + name + ": " + s);
    }
}

double parse_double(const std::string& s, const std::string& name) {
    try {
        return std::stod(s);
    } catch (...) {
        throw std::runtime_error("Invalid float for " + name + ": " + s);
    }
}

SearchConfig parse_args(int argc, char** argv) {
    SearchConfig cfg;
    std::unordered_map<std::string, std::string> kv;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            usage();
            std::exit(0);
        }
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("Unexpected argument: " + key);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for argument: " + key);
        }
        kv[key] = argv[++i];
    }

    auto req = [&](const std::string& k) -> std::string {
        auto it = kv.find(k);
        if (it == kv.end() || it->second.empty()) {
            throw std::runtime_error("Missing required argument: " + k);
        }
        return it->second;
    };
    auto opt_s = [&](const std::string& k, const std::string& default_v) -> std::string {
        auto it = kv.find(k);
        return it == kv.end() ? default_v : it->second;
    };
    auto opt_i = [&](const std::string& k, int default_v) -> int {
        auto it = kv.find(k);
        return it == kv.end() ? default_v : parse_int(it->second, k);
    };
    auto opt_d = [&](const std::string& k, double default_v) -> double {
        auto it = kv.find(k);
        return it == kv.end() ? default_v : parse_double(it->second, k);
    };

    cfg.index_dir = req("--index-dir");
    cfg.queries_path = req("--queries");
    cfg.ranges_path = req("--ranges");
    cfg.mode = opt_s("--mode", cfg.mode);
    cfg.output_ids_path = opt_s("--output-ids", "");
    cfg.k = opt_i("--k", cfg.k);
    cfg.pre_ef_search = opt_i("--pre-ef-search", cfg.pre_ef_search);
    cfg.post_ef_search = opt_i("--post-ef-search", cfg.post_ef_search);
    cfg.post_max_fetch_k = opt_i("--post-max-fetch-k", cfg.post_max_fetch_k);
    cfg.post_max_retries = opt_i("--post-max-retries", cfg.post_max_retries);
    cfg.post_fetch_mult = opt_d("--post-fetch-mult", cfg.post_fetch_mult);
    cfg.post_min_valid_mult = opt_d("--post-min-valid-mult", cfg.post_min_valid_mult);

    if (cfg.mode != "pre" && cfg.mode != "post" && cfg.mode != "both") {
        throw std::runtime_error("--mode must be one of: pre, post, both");
    }
    if (cfg.mode == "both" && !cfg.output_ids_path.empty()) {
        throw std::runtime_error("--output-ids is only supported with --mode pre or --mode post");
    }
    if (cfg.k <= 0 || cfg.post_ef_search <= 0 ||
        cfg.post_max_retries < 0 ||
        cfg.post_max_fetch_k <= 0 || cfg.post_fetch_mult <= 0.0 ||
        cfg.post_min_valid_mult < 1.0) {
        throw std::runtime_error("Invalid numeric argument values");
    }

    return cfg;
}

void write_ids_binary(const std::string& path, const std::vector<int64_t>& ids) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open ids output file: " + path);
    }
    out.write(reinterpret_cast<const char*>(ids.data()),
              static_cast<std::streamsize>(ids.size() * sizeof(int64_t)));
    if (!out) {
        throw std::runtime_error("Failed to write ids output file: " + path);
    }
}

void print_metrics(const RunSummary& s, size_t nq) {
    std::cout << "[" << s.name << "] total_time_s=" << s.dt_sec
              << ", qps=" << s.qps
              << ", queries=" << nq << "\n";
}

std::vector<float> read_f32_binary(const std::string& path, size_t expected_count) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open base vectors file: " + path);
    }
    const std::streamsize size = in.tellg();
    if (size < 0 || size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("Invalid base vectors file size: " + path);
    }
    const size_t count = static_cast<size_t>(size / static_cast<std::streamsize>(sizeof(float)));
    if (expected_count > 0 && count != expected_count) {
        throw std::runtime_error("Base vectors count mismatch in " + path);
    }
    in.seekg(0);
    std::vector<float> out(count);
    in.read(reinterpret_cast<char*>(out.data()), size);
    if (!in) {
        throw std::runtime_error("Failed to read base vectors file: " + path);
    }
    return out;
}

RunSummary run_pre(
    const SearchConfig& cfg,
    const FvecsData& queries,
    const std::vector<std::pair<int32_t, int32_t>>& ranges,
    const std::vector<int32_t>& sorted_attr,
    const std::vector<int64_t>& sorted_ids,
    const std::vector<float>& base_vectors,
    int dim) {

    const size_t nq = queries.count();
    RunSummary summary;
    summary.name = "pre";
    summary.k = cfg.k;
    summary.ids.assign(nq * static_cast<size_t>(cfg.k), -1);

    const auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel
    {
        std::vector<std::pair<float, int64_t>> scored;
#pragma omp for schedule(static)
        for (int64_t qi_ll = 0; qi_ll < static_cast<int64_t>(nq); ++qi_ll) {
            const size_t qi = static_cast<size_t>(qi_ll);
            const int32_t lo = ranges[qi].first;
            const int32_t hi = ranges[qi].second;
            const auto [left, right] = range_bounds(sorted_attr, lo, hi);
            if (right <= left) {
                continue;
            }

            const float* q = queries.values.data() + qi * static_cast<size_t>(queries.dim);
            int64_t* i_out = summary.ids.data() + qi * static_cast<size_t>(cfg.k);

            scored.clear();
            scored.reserve(right - left);
            for (size_t pos = left; pos < right; ++pos) {
                const int64_t id = sorted_ids[pos];
                const float* x = base_vectors.data() + static_cast<size_t>(id) * static_cast<size_t>(dim);
                float d2 = 0.0f;
                for (int j = 0; j < dim; ++j) {
                    const float diff = q[j] - x[j];
                    d2 += diff * diff;
                }
                scored.emplace_back(d2, id);
            }

            const size_t take = std::min(static_cast<size_t>(cfg.k), scored.size());
            if (take <= 0) {
                continue;
            }
            if (scored.size() <= static_cast<size_t>(cfg.k)) {
                std::sort(
                    scored.begin(),
                    scored.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            } else {
                std::partial_sort(
                    scored.begin(),
                    scored.begin() + static_cast<std::ptrdiff_t>(take),
                    scored.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }
            for (size_t j = 0; j < take; ++j) {
                i_out[j] = scored[j].second;
            }
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    summary.dt_sec = std::chrono::duration<double>(t1 - t0).count();
    summary.qps = summary.dt_sec > 0.0 ? static_cast<double>(nq) / summary.dt_sec : 0.0;
    return summary;
}

RunSummary run_post(
    const SearchConfig& cfg,
    const FvecsData& queries,
    const std::vector<std::pair<int32_t, int32_t>>& ranges,
    const std::vector<int32_t>& attr,
    const std::vector<int32_t>& sorted_attr,
    faiss::Index* post_index) {
    auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(post_index);
    if (!hnsw) {
        throw std::runtime_error("Post index is not HNSW");
    }

    const size_t nq = queries.count();
    RunSummary summary;
    summary.name = "post";
    summary.k = cfg.k;
    summary.ids.assign(nq * static_cast<size_t>(cfg.k), -1);

    const float inf = std::numeric_limits<float>::infinity();
    const auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel
    {
        std::vector<float> d;
        std::vector<int64_t> i;
        std::vector<float> d_top(static_cast<size_t>(cfg.k), inf);
        faiss::SearchParametersHNSW params;
#pragma omp for schedule(static)
        for (int64_t qi_ll = 0; qi_ll < static_cast<int64_t>(nq); ++qi_ll) {
            const size_t qi = static_cast<size_t>(qi_ll);
            const float* q = queries.values.data() + qi * static_cast<size_t>(queries.dim);
            const int32_t lo = ranges[qi].first;
            const int32_t hi = ranges[qi].second;

            const auto [left, right] = range_bounds(sorted_attr, lo, hi);
            const int64_t match_count = static_cast<int64_t>(right - left);
            if (match_count <= 0) {
                continue;
            }
            const double sel =
                static_cast<double>(match_count) / static_cast<double>(sorted_attr.size());
            int fetch_k = static_cast<int>(
                std::ceil((static_cast<double>(cfg.k) / std::max(sel, 1e-6)) * cfg.post_fetch_mult));
            fetch_k = std::max(cfg.k, std::min(fetch_k, cfg.post_max_fetch_k));
            fetch_k = std::min(fetch_k, static_cast<int>(hnsw->ntotal));
            const int target_valid = std::max(
                cfg.k,
                std::min(
                    static_cast<int>(match_count),
                    static_cast<int>(std::ceil(static_cast<double>(cfg.k) * cfg.post_min_valid_mult))));

            int64_t* i_top = summary.ids.data() + qi * static_cast<size_t>(cfg.k);
            int attempt = 0;
            int current_fetch_k = fetch_k;
            const int fetch_cap = std::min(cfg.post_max_fetch_k, static_cast<int>(hnsw->ntotal));
            while (true) {
                d.assign(static_cast<size_t>(current_fetch_k), inf);
                i.assign(static_cast<size_t>(current_fetch_k), -1);
                std::fill(d_top.begin(), d_top.end(), inf);
                std::fill(i_top, i_top + cfg.k, -1);

                params.efSearch = std::max(cfg.post_ef_search, current_fetch_k);
                hnsw->search(1, q, current_fetch_k, d.data(), i.data(), &params);
                filter_topk(d.data(), i.data(), current_fetch_k, attr, lo, hi, cfg.k, d_top.data(), i_top);

                int found = 0;
                for (int j = 0; j < cfg.k; ++j) {
                    if (i_top[j] != -1) {
                        ++found;
                    }
                }
                int valid_seen = found;
                for (int j = cfg.k; j < current_fetch_k && valid_seen < target_valid; ++j) {
                    const int64_t id = i[j];
                    if (id < 0 || static_cast<size_t>(id) >= attr.size()) {
                        continue;
                    }
                    const int32_t v = attr[static_cast<size_t>(id)];
                    if (v >= lo && v <= hi) {
                        ++valid_seen;
                    }
                }
                if ((found >= cfg.k && valid_seen >= target_valid) ||
                    attempt >= cfg.post_max_retries ||
                    current_fetch_k >= fetch_cap) {
                    break;
                }
                current_fetch_k = std::min(fetch_cap, current_fetch_k * 2);
                ++attempt;
            }
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    summary.dt_sec = std::chrono::duration<double>(t1 - t0).count();
    summary.qps = summary.dt_sec > 0.0 ? static_cast<double>(nq) / summary.dt_sec : 0.0;
    return summary;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const SearchConfig cfg = parse_args(argc, argv);
        const fs::path index_dir(cfg.index_dir);

        Manifest manifest = load_manifest((index_dir / "manifest.txt").string());
        std::vector<int32_t> attr = read_i32_binary(
            (index_dir / manifest.attr_file).string(), static_cast<size_t>(manifest.num_vectors));
        std::vector<int32_t> sorted_attr = read_i32_binary(
            (index_dir / manifest.sorted_attr_file).string(), static_cast<size_t>(manifest.num_vectors));
        std::vector<int64_t> sorted_ids = read_i64_binary(
            (index_dir / manifest.sorted_ids_file).string(), static_cast<size_t>(manifest.num_vectors));
        std::vector<float> base_vectors = read_f32_binary(
            (index_dir / manifest.base_vectors_file).string(),
            static_cast<size_t>(manifest.num_vectors) * static_cast<size_t>(manifest.dimension));

        FvecsData queries = read_fvecs(cfg.queries_path);
        if (queries.dim != manifest.dimension) {
            throw std::runtime_error("Query dimension does not match built index dimension");
        }

        std::vector<std::pair<int32_t, int32_t>> ranges = read_ranges_file(cfg.ranges_path);
        if (ranges.size() != queries.count()) {
            throw std::runtime_error("Ranges count (" + std::to_string(ranges.size()) +
                                     ") must match query count (" + std::to_string(queries.count()) + ")");
        }

        if (cfg.mode == "pre" || cfg.mode == "both") {
            RunSummary s = run_pre(
                cfg, queries, ranges, sorted_attr, sorted_ids, base_vectors, manifest.dimension);
            print_metrics(s, queries.count());
            if (!cfg.output_ids_path.empty()) {
                write_ids_binary(cfg.output_ids_path, s.ids);
            }
        }

        if (cfg.mode == "post" || cfg.mode == "both") {
            std::unique_ptr<faiss::Index> post_index(
                faiss::read_index((index_dir / manifest.post_index_file).c_str()));
            RunSummary s = run_post(cfg, queries, ranges, attr, sorted_attr, post_index.get());
            print_metrics(s, queries.count());
            if (!cfg.output_ids_path.empty()) {
                write_ids_binary(cfg.output_ids_path, s.ids);
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
