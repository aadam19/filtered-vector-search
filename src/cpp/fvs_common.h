#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct FvecsData {
    int dim = 0;
    std::vector<float> values;

    size_t count() const {
        return dim > 0 ? values.size() / static_cast<size_t>(dim) : 0;
    }
};

struct Manifest {
    int dimension = 0;
    int64_t num_vectors = 0;
    std::string post_index_file;
    std::string pre_index_file;
    std::string base_vectors_file;
    std::string attr_file;
    std::string sorted_attr_file;
    std::string sorted_ids_file;
};

struct SearchResults {
    int k = 0;
    std::vector<float> distances;
    std::vector<int64_t> ids;
};

FvecsData read_fvecs(const std::string& path);
std::vector<int32_t> read_attr_auto(const std::string& path, size_t expected_count);
std::vector<int32_t> read_i32_binary(const std::string& path, size_t expected_count = 0);
std::vector<int64_t> read_i64_binary(const std::string& path, size_t expected_count = 0);
void write_i32_binary(const std::string& path, const std::vector<int32_t>& values);
void write_i64_binary(const std::string& path, const std::vector<int64_t>& values);
std::vector<std::pair<int32_t, int32_t>> read_ranges_file(const std::string& path);

std::vector<int64_t> argsort_i32(const std::vector<int32_t>& values);
std::vector<int32_t> gather_i32(const std::vector<int32_t>& values, const std::vector<int64_t>& ids);

void save_manifest(const std::string& path, const Manifest& manifest);
Manifest load_manifest(const std::string& path);

std::pair<size_t, size_t> range_bounds(
    const std::vector<int32_t>& sorted_attr,
    int32_t lo,
    int32_t hi);

void filter_topk(
    const float* dists,
    const int64_t* ids,
    int64_t n,
    const std::vector<int32_t>& attr,
    int32_t lo,
    int32_t hi,
    int k,
    float* out_dists,
    int64_t* out_ids);
