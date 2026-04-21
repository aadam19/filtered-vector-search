#include "fvs_common.h"

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct BuildConfig {
    std::string base_path;
    std::string attr_path;
    std::string out_dir;
    int pre_m = 32;
    int post_m = 64;
    int ef_construction = 800;
    int pre_ef_search = 256;
    int post_ef_search = 256;
};

void usage() {
    std::cout
        << "build_indexes\n"
        << "  --base <base.fvecs>\n"
        << "  --attr <attr.npy|attr.i32>\n"
        << "  --out-dir <directory>\n"
        << "  [--pre-m 32]\n"
        << "  [--post-m 64]\n"
        << "  [--ef-construction 800]\n"
        << "  [--pre-ef-search 256]\n"
        << "  [--post-ef-search 256]\n";
}

int parse_int(const std::string& s, const std::string& name) {
    try {
        return std::stoi(s);
    } catch (...) {
        throw std::runtime_error("Invalid integer for " + name + ": " + s);
    }
}

BuildConfig parse_args(int argc, char** argv) {
    BuildConfig cfg;
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
    auto opt_i = [&](const std::string& k, int default_v) -> int {
        auto it = kv.find(k);
        return it == kv.end() ? default_v : parse_int(it->second, k);
    };

    cfg.base_path = req("--base");
    cfg.attr_path = req("--attr");
    cfg.out_dir = req("--out-dir");
    cfg.pre_m = opt_i("--pre-m", cfg.pre_m);
    cfg.post_m = opt_i("--post-m", cfg.post_m);
    cfg.ef_construction = opt_i("--ef-construction", cfg.ef_construction);
    cfg.pre_ef_search = opt_i("--pre-ef-search", cfg.pre_ef_search);
    cfg.post_ef_search = opt_i("--post-ef-search", cfg.post_ef_search);
    return cfg;
}

void print_stage(const std::string& name, const std::chrono::steady_clock::time_point& start) {
    const auto end = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(end - start).count();
    std::cout << name << " completed in " << sec << " s\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const BuildConfig cfg = parse_args(argc, argv);
        const auto total_start = std::chrono::steady_clock::now();

        const auto t0 = std::chrono::steady_clock::now();
        FvecsData base = read_fvecs(cfg.base_path);
        const size_t n = base.count();
        std::vector<int32_t> attr = read_attr_auto(cfg.attr_path, n);
        print_stage("Load base+attr", t0);

        std::cout << "Base vectors: n=" << n << ", d=" << base.dim << "\n";
        fs::create_directories(cfg.out_dir);

        const auto tsort = std::chrono::steady_clock::now();
        std::vector<int64_t> sorted_ids = argsort_i32(attr);
        std::vector<int32_t> sorted_attr = gather_i32(attr, sorted_ids);
        print_stage("Sort attributes", tsort);

        write_i32_binary((fs::path(cfg.out_dir) / "attr.i32").string(), attr);
        write_i32_binary((fs::path(cfg.out_dir) / "sorted_attr.i32").string(), sorted_attr);
        write_i64_binary((fs::path(cfg.out_dir) / "sorted_ids.i64").string(), sorted_ids);
        {
            std::ofstream out((fs::path(cfg.out_dir) / "base_vectors.f32").string(), std::ios::binary);
            if (!out) {
                throw std::runtime_error("Failed to write base_vectors.f32");
            }
            out.write(reinterpret_cast<const char*>(base.values.data()),
                      static_cast<std::streamsize>(base.values.size() * sizeof(float)));
            if (!out) {
                throw std::runtime_error("Failed writing base_vectors.f32");
            }
        }

        const auto tpre = std::chrono::steady_clock::now();
        auto pre = std::make_unique<faiss::IndexHNSWFlat>(base.dim, cfg.pre_m);
        pre->hnsw.efConstruction = cfg.ef_construction;
        pre->hnsw.efSearch = cfg.pre_ef_search;
        pre->add(n, base.values.data());
        faiss::write_index(pre.get(), (fs::path(cfg.out_dir) / "pre_hnsw.faiss").c_str());
        print_stage("Build pre index", tpre);

        const auto tpost = std::chrono::steady_clock::now();
        auto post = std::make_unique<faiss::IndexHNSWFlat>(base.dim, cfg.post_m);
        post->hnsw.efConstruction = cfg.ef_construction;
        post->hnsw.efSearch = cfg.post_ef_search;
        post->add(n, base.values.data());
        faiss::write_index(post.get(), (fs::path(cfg.out_dir) / "post_hnsw.faiss").c_str());
        print_stage("Build post index", tpost);

        Manifest manifest;
        manifest.dimension = base.dim;
        manifest.num_vectors = static_cast<int64_t>(n);
        manifest.post_index_file = "post_hnsw.faiss";
        manifest.pre_index_file = "pre_hnsw.faiss";
        manifest.base_vectors_file = "base_vectors.f32";
        manifest.attr_file = "attr.i32";
        manifest.sorted_attr_file = "sorted_attr.i32";
        manifest.sorted_ids_file = "sorted_ids.i64";
        save_manifest((fs::path(cfg.out_dir) / "manifest.txt").string(), manifest);

        print_stage("Total build", total_start);
        std::cout << "Wrote build artifacts to: " << cfg.out_dir << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
