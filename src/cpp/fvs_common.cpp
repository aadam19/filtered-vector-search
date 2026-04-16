#include "fvs_common.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

std::string trim(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
    size_t j = s.size();
    while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1]))) {
        --j;
    }
    return s.substr(i, j - i);
}

bool has_suffix(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int64_t parse_shape_1d_from_npy_header(const std::string& header) {
    const size_t l = header.find('(');
    const size_t r = header.find(')', l == std::string::npos ? 0 : l + 1);
    if (l == std::string::npos || r == std::string::npos || r <= l + 1) {
        throw std::runtime_error("Invalid .npy header: missing shape");
    }
    const std::string inside = header.substr(l + 1, r - l - 1);
    std::stringstream ss(inside);
    int64_t n = 0;
    ss >> n;
    if (!ss || n < 0) {
        throw std::runtime_error("Invalid .npy shape for 1D int32 array");
    }
    return n;
}

std::vector<int32_t> read_npy_int32_1d(const std::string& path, size_t expected_count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open .npy file: " + path);
    }

    char magic[6] = {};
    in.read(magic, 6);
    if (in.gcount() != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Invalid .npy magic: " + path);
    }

    uint8_t major = 0;
    uint8_t minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);
    if (!in) {
        throw std::runtime_error("Failed reading .npy version: " + path);
    }

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t h = 0;
        in.read(reinterpret_cast<char*>(&h), sizeof(h));
        header_len = h;
    } else if (major == 2 || major == 3) {
        in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    } else {
        throw std::runtime_error("Unsupported .npy version in " + path);
    }
    if (!in) {
        throw std::runtime_error("Failed reading .npy header length: " + path);
    }

    std::string header(header_len, '\0');
    in.read(header.data(), header_len);
    if (!in) {
        throw std::runtime_error("Failed reading .npy header: " + path);
    }

    if (header.find("fortran_order") == std::string::npos ||
        header.find("False") == std::string::npos) {
        throw std::runtime_error("Only C-order .npy arrays are supported: " + path);
    }
    if (header.find("descr") == std::string::npos ||
        (header.find("<i4") == std::string::npos && header.find("|i4") == std::string::npos)) {
        throw std::runtime_error("Only int32 .npy arrays are supported: " + path);
    }

    const int64_t n64 = parse_shape_1d_from_npy_header(header);
    const size_t n = static_cast<size_t>(n64);
    if (expected_count > 0 && n != expected_count) {
        throw std::runtime_error("Attribute count mismatch in .npy file: expected " +
                                 std::to_string(expected_count) + ", got " +
                                 std::to_string(n));
    }

    std::vector<int32_t> out(n);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * sizeof(int32_t)));
    if (!in) {
        throw std::runtime_error("Failed reading .npy payload: " + path);
    }
    return out;
}

template <typename T>
void write_binary_vec(const std::string& path, const std::vector<T>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!out) {
        throw std::runtime_error("Failed to write file: " + path);
    }
}

template <typename T>
std::vector<T> read_binary_vec(const std::string& path, size_t expected_count) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open input file: " + path);
    }
    const std::streamsize size = in.tellg();
    if (size < 0 || size % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("Invalid binary file size for " + path);
    }
    const size_t count = static_cast<size_t>(size / static_cast<std::streamsize>(sizeof(T)));
    if (expected_count > 0 && count != expected_count) {
        throw std::runtime_error("Element count mismatch for " + path + ": expected " +
                                 std::to_string(expected_count) + ", got " +
                                 std::to_string(count));
    }
    in.seekg(0);
    std::vector<T> out(count);
    in.read(reinterpret_cast<char*>(out.data()), size);
    if (!in) {
        throw std::runtime_error("Failed to read binary file: " + path);
    }
    return out;
}

} // namespace

FvecsData read_fvecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open fvecs file: " + path);
    }

    FvecsData data;
    while (true) {
        int32_t d = 0;
        in.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (!in) {
            if (in.eof()) {
                break;
            }
            throw std::runtime_error("Error while reading fvecs dimension from " + path);
        }
        if (d <= 0) {
            throw std::runtime_error("Invalid vector dimension in fvecs file: " + path);
        }
        if (data.dim == 0) {
            data.dim = d;
        } else if (data.dim != d) {
            throw std::runtime_error("Inconsistent vector dimensions in fvecs file: " + path);
        }
        const size_t old_size = data.values.size();
        data.values.resize(old_size + static_cast<size_t>(d));
        in.read(reinterpret_cast<char*>(data.values.data() + old_size),
                static_cast<std::streamsize>(static_cast<size_t>(d) * sizeof(float)));
        if (!in) {
            throw std::runtime_error("Error while reading fvecs payload from " + path);
        }
    }

    if (data.dim == 0 || data.values.empty()) {
        throw std::runtime_error("No vectors read from fvecs file: " + path);
    }
    return data;
}

std::vector<int32_t> read_i32_binary(const std::string& path, size_t expected_count) {
    return read_binary_vec<int32_t>(path, expected_count);
}

std::vector<int64_t> read_i64_binary(const std::string& path, size_t expected_count) {
    return read_binary_vec<int64_t>(path, expected_count);
}

void write_i32_binary(const std::string& path, const std::vector<int32_t>& values) {
    write_binary_vec<int32_t>(path, values);
}

void write_i64_binary(const std::string& path, const std::vector<int64_t>& values) {
    write_binary_vec<int64_t>(path, values);
}

std::vector<int32_t> read_attr_auto(const std::string& path, size_t expected_count) {
    if (has_suffix(path, ".npy")) {
        return read_npy_int32_1d(path, expected_count);
    }
    return read_i32_binary(path, expected_count);
}

std::vector<std::pair<int32_t, int32_t>> read_ranges_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open ranges file: " + path);
    }

    std::vector<std::pair<int32_t, int32_t>> ranges;
    std::string line;
    int64_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        for (char& c : line) {
            if (c == ',') {
                c = ' ';
            }
        }
        std::stringstream ss(line);
        int64_t lo = 0;
        int64_t hi = 0;
        ss >> lo >> hi;
        if (!ss) {
            throw std::runtime_error("Invalid range line at " + std::to_string(line_no) +
                                     " in " + path);
        }
        if (lo > hi) {
            std::swap(lo, hi);
        }
        ranges.emplace_back(static_cast<int32_t>(lo), static_cast<int32_t>(hi));
    }
    return ranges;
}

std::vector<int64_t> argsort_i32(const std::vector<int32_t>& values) {
    std::vector<int64_t> ids(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        ids[i] = static_cast<int64_t>(i);
    }
    std::sort(ids.begin(), ids.end(), [&](int64_t a, int64_t b) {
        if (values[static_cast<size_t>(a)] == values[static_cast<size_t>(b)]) {
            return a < b;
        }
        return values[static_cast<size_t>(a)] < values[static_cast<size_t>(b)];
    });
    return ids;
}

std::vector<int32_t> gather_i32(const std::vector<int32_t>& values, const std::vector<int64_t>& ids) {
    std::vector<int32_t> out(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int64_t id = ids[i];
        if (id < 0 || static_cast<size_t>(id) >= values.size()) {
            throw std::runtime_error("Invalid gather index");
        }
        out[i] = values[static_cast<size_t>(id)];
    }
    return out;
}

void save_manifest(const std::string& path, const Manifest& manifest) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write manifest: " + path);
    }

    out << "version=1\n";
    out << "dimension=" << manifest.dimension << "\n";
    out << "num_vectors=" << manifest.num_vectors << "\n";
    out << "post_index=" << manifest.post_index_file << "\n";
    out << "pre_index=" << manifest.pre_index_file << "\n";
    out << "base_vectors_file=" << manifest.base_vectors_file << "\n";
    out << "attr_file=" << manifest.attr_file << "\n";
    out << "sorted_attr_file=" << manifest.sorted_attr_file << "\n";
    out << "sorted_ids_file=" << manifest.sorted_ids_file << "\n";
    if (!out) {
        throw std::runtime_error("Failed to flush manifest: " + path);
    }
}

Manifest load_manifest(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open manifest: " + path);
    }

    Manifest m;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        const std::string key = trim(line.substr(0, pos));
        const std::string value = trim(line.substr(pos + 1));
        if (key == "dimension") {
            m.dimension = std::stoi(value);
        } else if (key == "num_vectors") {
            m.num_vectors = std::stoll(value);
        } else if (key == "post_index") {
            m.post_index_file = value;
        } else if (key == "pre_index") {
            m.pre_index_file = value;
        } else if (key == "base_vectors_file") {
            m.base_vectors_file = value;
        } else if (key == "attr_file") {
            m.attr_file = value;
        } else if (key == "sorted_attr_file") {
            m.sorted_attr_file = value;
        } else if (key == "sorted_ids_file") {
            m.sorted_ids_file = value;
        }
    }

    if (m.dimension <= 0 || m.num_vectors <= 0 || m.base_vectors_file.empty() ||
        m.attr_file.empty() || m.sorted_attr_file.empty() ||
        m.sorted_ids_file.empty() || m.post_index_file.empty() || m.pre_index_file.empty()) {
        throw std::runtime_error("Manifest is missing required fields: " + path);
    }
    return m;
}

std::pair<size_t, size_t> range_bounds(
    const std::vector<int32_t>& sorted_attr,
    int32_t lo,
    int32_t hi) {
    const auto l = std::lower_bound(sorted_attr.begin(), sorted_attr.end(), lo);
    const auto r = std::upper_bound(sorted_attr.begin(), sorted_attr.end(), hi);
    return {static_cast<size_t>(std::distance(sorted_attr.begin(), l)),
            static_cast<size_t>(std::distance(sorted_attr.begin(), r))};
}

void filter_topk(
    const float* dists,
    const int64_t* ids,
    int64_t n,
    const std::vector<int32_t>& attr,
    int32_t lo,
    int32_t hi,
    int k,
    float* out_dists,
    int64_t* out_ids) {
    for (int i = 0; i < k; ++i) {
        out_dists[i] = std::numeric_limits<float>::infinity();
        out_ids[i] = -1;
    }
    int out = 0;
    for (int64_t i = 0; i < n && out < k; ++i) {
        const int64_t id = ids[i];
        if (id < 0 || static_cast<size_t>(id) >= attr.size()) {
            continue;
        }
        const int32_t v = attr[static_cast<size_t>(id)];
        if (v >= lo && v <= hi) {
            out_dists[out] = dists[i];
            out_ids[out] = id;
            ++out;
        }
    }
}
