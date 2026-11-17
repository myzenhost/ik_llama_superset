#include "llama.h"
#include "../src/llama-ext.h"

#include "arg.h"
#include "common.h"
#include "gguf.h"
#include "imatrix-loader.h"
#include "imatrix-hessian.h"
#include "log.h"
#include "llama.h"
#include "gguf.h"
// PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01
// SOURCE-READ: common/speculative.h:3-6 -- pulls in common_params_speculative,
// common_speculative_load_draft_model()/common_speculative_prepare_mtp_runtime(), and (transitively)
// src/llama-spec-features.h for llama_mtp_state_n_embd()/llama_set_draft_input_hidden_state_copy().
#include "speculative.h"

// PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01
// SOURCE-READ: examples/imatrix/imatrix.cpp:102 and common/speculative.cpp:46 -- this function is
// defined in src/llama.cpp (src/llama.cpp:13193) but declared in NO header; every caller in this
// tree forward-declares it locally, so this matches existing convention rather than deviating.
void llama_set_mtp_target_context(struct llama_context * ctx, struct llama_context * target_ctx);

#include <chrono>
#include <climits>
#include <clocale>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <map>
#include <regex>
#include <numeric>
#include <optional>
#include <sstream>
#include "mtmd.h"
#include "mtmd-helper.h"
#include <algorithm>
#include <filesystem>
#include "log-upstream.h" // upstream LOG() semantics -- see common/log-upstream.h

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s \\\n"
            "       -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {gguf,dat}] [--no-ppl] \\\n"
            "       [--process-output] [--nextn] [--model-draft draft.gguf] [--chunk 123] [--save-frequency 0] [--output-frequency 10] \\\n"
            "       [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] [--parse-special] \\\n"
            "       [--show-statistics] [--hessian-block 128 [--hessian-out imatrix.hessian.gguf] [--hessian-max-mib 4096]] [...]\n" , argv[0]);
    LOG("\n");
}

struct Stats {
    std::vector<float>   activations;
    std::vector<float>   values;
    std::vector<int64_t> counts;
};

struct tensor_statistics {
    std::string tensor;
    bool legacy = true;
    double sum = 0.0f;
    float mean = 0.0f;
    int64_t elements = 0;
    float std_deviation = 0.0f;
    float skewness = 0.0f;
    float kurtosis = 0.0f;
    float gain = std::numeric_limits<float>::quiet_NaN();
    float entropy = 0.0f;
    float l2_dist = std::numeric_limits<float>::quiet_NaN();
    float cossim = std::numeric_limits<float>::quiet_NaN();
    float pearson = std::numeric_limits<float>::quiet_NaN();
    float covariance = std::numeric_limits<float>::quiet_NaN();
    double cov_sum = 0.0;
    double var_c_sum = 0.0;
    double var_p_sum = 0.0;
    double dot_prod = 0.0;
    double norm1_sq = 0.0;
    double norm2_sq = 0.0;
    double l2_dist_sq = 0.0;
    double sum_prev = 0.0;
    int64_t elements_prev = 0;
    int64_t n_features = 0;
};

class IMatrixCollector {
public:
    IMatrixCollector() = default;
    void set_params(common_params params) { m_params = std::move(params); }
    bool collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data);
    void save_imatrix_legacy(int32_t ncall = -1) const;
    void save_imatrix(int32_t n_chunk = -1) const;
    bool load_imatrix(const char * file_name);
    const std::unordered_map<std::string, Stats> & get_mstats() const { return m_stats; }
    // PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01:
    // SOURCE-READ: examples/imatrix/imatrix.cpp:252,303,348-385 (fork-only -lsim output).
    void print_layer_importance() const;
    // --split-on: every piece is its own sequence, so the -lsim "previous layer input" must not carry
    // over from one piece into the next (Thireus th/imatrix_with_split_on reset_lsim_state()).
    // The accumulated similarity sums are kept.
    void reset_lsim_state() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_last_layer = 9999;
        m_last_ffn   = -1;
        m_last_input.clear();
        m_ffn_input.clear();
    }
    // upstream: NextN layers collected with --nextn / a NextN-only draft
    void set_n_layer_nextn(int32_t n_layer_nextn) { m_n_layer_nextn = n_layer_nextn; }
    int32_t get_n_layer_nextn() const { return m_n_layer_nextn; }
private:
    std::unordered_map<std::string, Stats> m_stats;
    common_params                          m_params;
    std::mutex                             m_mutex;
    std::vector<std::string>               m_datasets;
    int32_t                                m_last_chunk = 0;
    int32_t                                m_chunk_size = 0;
    int32_t                                m_n_layer_nextn = 0;
    std::vector<char>                      m_src1_data;
    std::vector<char>                      m_ids; // the expert ids from ggml_mul_mat_id
    int32_t e_chunk_size() const {
        return m_chunk_size > 0 ? m_chunk_size : m_params.n_ctx / m_params.n_parallel;
    }
    // -- lsim (layer-similarity) state, ported from examples/imatrix/imatrix.cpp:257-267 --
    int32_t                                m_last_layer = 9999;
    int32_t                                m_last_ffn   = -1;
    std::vector<float>                     m_last_input;
    std::vector<float>                     m_ffn_input;
    std::vector<std::pair<double,int>>     m_layer_sim;
    std::vector<std::pair<double,int>>     m_attn_sim;
    std::vector<std::pair<double,int>>     m_ffn_sim;

    // --hessian-block N (row S69-F1-PICK-IMATRIX-BLOCK-HESSIAN-OUTPUT-2026-10-06-01): the N x N diagonal blocks of
    // sum_t x_t x_t^T per matrix, next to the diagonal in m_stats. Empty and untouched when the flag is 0 (default),
    // so the default path and its output are unchanged. Layout and file format: common/imatrix-hessian.h.
    std::map<std::string, imatrix_hessian_entry> m_hess; // ordered: the sidecar is written in name order
    uint64_t                                m_hess_bytes = 0;
    std::vector<std::vector<const float *>> m_hess_rows; // per-matrix row pointers of the current call

    std::string hessian_out_file() const {
        return m_params.imat_hessian_out.empty() ? imatrix_hessian_default_path(m_params.out_file) : m_params.imat_hessian_out;
    }
    // find or allocate a matrix's Hessian blocks; refuses (exit) when the --hessian-max-mib budget would be exceeded
    imatrix_hessian_entry & hessian_entry(const std::string & wname, int64_t n_in, int64_t n_mat);
    void hessian_accumulate(const std::string & wname, int64_t n_in, int64_t n_mat);
    void save_hessian(int32_t n_chunk) const;

    // SOURCE-READ: examples/imatrix/imatrix.cpp:269-283.
    std::optional<int> layer_index(const std::string & name) const {
        if (name == m_params.output_tensor_name && m_last_layer < 199) {
            return m_last_layer + 1;
        }
        if (auto pos = name.find("blk."); pos == 0) {
            pos += 4;
            if (auto pos1 = name.find('.', pos); pos1 != std::string::npos) {
                auto index_str = name.substr(pos, pos1 - pos);
                std::istringstream str(index_str);
                int index; str >> index;
                if (!str.fail()) return index;
            }
        }
        return std::nullopt;
    }

    // SOURCE-READ: examples/imatrix/imatrix.cpp:285-301.
    static inline double cosine_similarity(int64_t n, const float * x, const float * y) {
        double sumxy = 0, sumx2 = 0, sumy2 = 0;
        for (int64_t j = 0; j < n; ++j) {
            sumxy += x[j]*y[j]; sumx2 += x[j]*x[j]; sumy2 += y[j]*y[j];
        }
        return sumx2 > 0 && sumy2 > 0 ? sumxy/sqrt(sumx2*sumy2) : 0;
    }

    static inline void collect_cos_similarity(int64_t nrow, int64_t n, const float * x, const float * y, std::pair<double, int> & p) {
        for (int64_t row = 0; row < nrow; ++row) {
            p.first  += cosine_similarity(n, x, y);
            p.second += 1;
            x += n;
            y += n;
        }
    }

    static void print_layer_importance(const char * msg, const std::vector<std::pair<double, int>> & sim);
};

// remove any prefix and suffixes from the name
// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
static std::string filter_tensor_name(const char * name) {
    std::string wname;
    const char * p = strchr(name, '#');
    if (p != NULL) {
        p = p + 1;
        const char * q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

// PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01:
// SOURCE-READ: examples/imatrix/imatrix.cpp:325-333 (fork-only). Replaces this file's inline
// `wname.substr(0, 4) == "blk." || (m_params.process_output && wname == "output.weight")` ask-phase
// filter, which hardcoded "output.weight" -- output_tensor_name (common.h:1088) existed on
// common_params unreachable (no CLI writer, see arg.cpp --output-tensor-name above) and mtp_pre_proj/
// mtp_post_proj (present: src/graphs/build_gemma4.cpp) were never matched at all.
static bool is_named_imatrix_tensor(const std::string & wname, const common_params & params, bool collect_lsim) {
    if (wname.rfind("blk.", 0) == 0) {
        return true;
    }
    if (wname == "mtp_pre_proj.weight" || wname == "mtp_post_proj.weight") {
        return true;
    }
    return (params.process_output || collect_lsim) && wname == params.output_tensor_name;
}

// PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01
// SOURCE-READ: examples/imatrix/imatrix.cpp:335-346, verbatim (no API dependency).
static std::string default_draft_imatrix_out_file(const std::string & target_out_file) {
    if (target_out_file.empty()) {
        return "imatrix-draft.dat";
    }

    const auto dot = target_out_file.rfind('.');
    if (dot == std::string::npos || dot == 0) {
        return target_out_file + "-draft";
    }

    return target_out_file.substr(0, dot) + "-draft" + target_out_file.substr(dot);
}

static void process_tensor_name(const std::string & input, std::string & layer, std::string & tensor) {
    layer.clear();
    tensor.clear();

    std::vector<std::string> name;
    std::istringstream stream(input);
    std::string item;

    while (std::getline(stream, item, '.')) {
        name.push_back(item);
    }
    for (size_t i = 0; i < name.size(); ++i) {
        if (name[i] == "blk" && i + 1 < name.size()) {
            layer = name[i + 1];
            break;
        }
    }
    for (size_t i = 0; i < name.size(); ++i) {
        if (name[i] == "weight" && i > 0) {
            for (size_t j = 0; j < name.size(); ++j) {
                if (name[j] == "blk") {
                    j += name.size() > 4 ? 1 : 2;
                    continue;
                }
                if (j == i) { break; }
                if (!tensor.empty()) { tensor += "."; }
                tensor += name[j];
            }
            break;
        }
    }

    if (tensor.empty()) { tensor = input; }
    if (layer.empty()) { layer = "-"; }
}

static std::vector<float> compute_tensor_averages(const Stats & tstats, bool use_activations) {
    if (tstats.counts.empty()) { return {}; }

    const size_t n_mat = tstats.counts.size();
    const size_t len = use_activations ? tstats.activations.size() : tstats.values.size();
    if (len == 0 || n_mat == 0 || len % n_mat != 0) { return {}; }

    const size_t row = len / n_mat;
    std::vector<float> vec(len, std::numeric_limits<float>::quiet_NaN());

    bool has_valid = false;
    for (size_t m = 0; m < n_mat; ++m) {
        const auto c = (float) tstats.counts[m];
        const size_t off = m * row;

        if (c <= 0.0f) { continue; }

        has_valid = true;
        const float scale = 1.0f / c;
        const float * src = use_activations ? &tstats.activations[off] : &tstats.values[off];
        float * dst = & vec[off];

        for (size_t j = 0; j < row; ++j) { dst[j] = src[j] * scale; }
    }

    if (!has_valid) { return {}; }

    return vec;
}

static bool compute_vector_statistics(std::vector<tensor_statistics> & tstats, const std::string & name, const Stats & e) {
    constexpr auto fnan = std::numeric_limits<float>::quiet_NaN();
    const bool legacy = e.activations.empty();
    const size_t n_mat = e.counts.size();
    const size_t len = legacy ? e.values.size() : e.activations.size();

    if (n_mat == 0 || len == 0 || len % n_mat != 0) {
        LOG_ERR("%s: data size mismatch or empty for tensor %s\n", __func__, name.c_str());
        return false;
    }
    if (!legacy && e.values.size() != len) {
        LOG_ERR("%s: activations/values size mismatch for %s\n", __func__, name.c_str());
        return false;
    }

    const size_t row_size = len / n_mat;
    double sum = 0.0;
    double mean = 0.0;
    double sum_sq_diff = 0.0;
    double sum_cu_diff = 0.0;
    double sum_qd_diff = 0.0;
    double sum_energy = 0.0;
    size_t valid_n = 0;

    // Mean
    for (size_t i = 0; i < n_mat; ++i) {
        const auto c = (float)e.counts[i];
        if (c <= 0.0f) { continue; }

        const double inv_c = 1.0 / (double)c;
        const size_t off = i * row_size;

        for (size_t j = 0; j < row_size; ++j) {
            const double v_act = legacy ? 0.0 : (double)e.activations[off + j] * inv_c;
            const double v_val = (double)e.values[off + j] * inv_c;
            const double v = legacy ? v_val : v_act; // Use activation average for non-legacy
            if (!std::isfinite(v) || !std::isfinite(v_val)) { continue; }

            sum += v_val;
            valid_n++;
            const double delta = v - mean;
            mean += delta / (double)valid_n;

            if (v_val > 0.0) { sum_energy += v_val; }
        }
    }

    if (valid_n == 0) { return false; }

    float std_deviation = 0.0f;
    double entropy = 0.0;

    // Std Dev, Skew, Kurtosis, Entropy
    const double inv_sum_energy = sum_energy > 0.0 ? 1.0 / sum_energy : 0.0;
    const double log2_inv = 1.0 / std::log(2.0);

    for (size_t i = 0; i < n_mat; ++i) {
        const auto c = (float)e.counts[i];
        if (c <= 0.0f) { continue; }
        const double inv_c = 1.0 / (double)c;
        const size_t off = i * row_size;

        for (size_t j = 0; j < row_size; ++j) {
            const double v_act = legacy ? 0.0 : (double)e.activations[off + j] * inv_c;
            const double v_val = (double)e.values[off + j] * inv_c;
            const double v = legacy ? v_val : v_act;
            if (!std::isfinite(v) || !std::isfinite(v_val)) { continue; }
            const double diff = v - mean;

            sum_sq_diff += diff * diff;
            sum_cu_diff += diff * diff * diff;
            sum_qd_diff += diff * diff * diff * diff;

            // Entropy (Distribution of Energy)
            if (inv_sum_energy > 0.0) {
                const double v_energy = (double)e.values[off + j] * inv_c;
                const double p = std::max(0.0, v_energy) * inv_sum_energy;
                if (p > 1e-10) { entropy -= p * std::log(p) * log2_inv; }
            }
        }
    }

    const double variance = valid_n > 1 ? sum_sq_diff / (double)valid_n : 0.0;
    std_deviation = std::sqrt((float)std::max(variance, 0.0));
    float skewness = 0.0f;
    float kurtosis = 0.0f;
    if (std_deviation > 1e-10f) {
        const double m2 = sum_sq_diff / (double)valid_n;
        skewness = (float)(sum_cu_diff / (double)valid_n / (m2 * std::sqrt(m2)));
        kurtosis = (float)(sum_qd_diff / (double)valid_n / (m2 * m2) - 3.0);
    }

    auto & ts = tstats.emplace_back();
    ts.tensor = name;
    ts.legacy = legacy;
    ts.sum = sum;
    ts.mean = (float)mean;
    ts.elements = (int64_t)valid_n;
    ts.std_deviation = std_deviation;
    ts.skewness = skewness;
    ts.kurtosis = kurtosis;
    ts.gain = fnan;
    ts.entropy = (float)entropy;
    ts.l2_dist = fnan;
    ts.cossim = fnan;
    ts.pearson = fnan;
    ts.covariance = fnan;

    return true;
}

static int nextn_layer_start(const std::vector<tensor_statistics> & tstats, int32_t n_layer_nextn) {
    int max_blk = -1;
    int first   = INT_MAX;

    for (const auto & ts : tstats) {
        std::string layer_str;
        std::string name;
        process_tensor_name(ts.tensor, layer_str, name);

        int blk = -1;
        try { blk = std::stoi(layer_str); } catch (...) { continue; }

        max_blk = std::max(max_blk, blk);
        if (name.rfind("nextn.", 0) == 0) { first = std::min(first, blk); }
    }

    if (n_layer_nextn > 0 && max_blk >= 0) { return std::max(0, max_blk + 1 - n_layer_nextn); }

    return first;
}

static std::string layer_label(int blk, int nextn_start) {
    if (blk < 0 || blk == INT_MAX) { return "-"; }
    if (blk >= nextn_start) { return "mtp" + std::to_string(blk - nextn_start); }

    return std::to_string(blk);
}

static void compute_tensor_statistics(std::vector<tensor_statistics> & tstats, const std::unordered_map<std::string, Stats> & mstats, int nextn_start) {
    constexpr auto fnan = std::numeric_limits<float>::quiet_NaN();
    std::unordered_map<std::string, size_t> tensor_map;
    tensor_map.reserve(tstats.size());
    for (size_t i = 0; i < tstats.size(); ++i) { tensor_map[tstats[i].tensor] = i; }

    for (auto & ts : tstats) {
        std::string layer_str;
        std::string dummy_tensor;
        process_tensor_name(ts.tensor, layer_str, dummy_tensor);

        int blk = -1;
        try { blk = std::stoi(layer_str); } catch (...) { continue; }
        if (blk <= 0) { continue; }
        if (blk == nextn_start) { continue; }

        const int blk_first = blk > nextn_start ? nextn_start : 0;
        const size_t blk_start_pos = ts.tensor.find("blk." + layer_str);
        if (blk_start_pos == std::string::npos) { continue; }

        std::string tname = ts.tensor;
        auto it = tensor_map.end();
        for (int prev = blk - 1; prev >= blk_first && it == tensor_map.end(); --prev) {
            tname = ts.tensor;
            tname.replace(blk_start_pos, layer_str.length() + 4, "blk." + std::to_string(prev));
            it = tensor_map.find(tname);
        }

        if (it == tensor_map.end()) {
            LOG_WRN("%s: no preceding-layer tensor for '%s'\n", __func__, ts.tensor.c_str());
            continue;
        }

        const auto & prev_ts = tstats[it->second];
        const auto curr_e = mstats.find(ts.tensor);
        const auto prev_e = mstats.find(prev_ts.tensor);
        if (curr_e == mstats.end() || prev_e == mstats.end()) { continue; }

        // one side may not have no activation sums so compare both on the energy
        const bool use_activations = !ts.legacy && !prev_ts.legacy;
        const auto curr_avg = compute_tensor_averages(curr_e->second, use_activations);
        const auto prev_avg = compute_tensor_averages(prev_e->second, use_activations);

        if (curr_avg.empty() || curr_avg.size() != prev_avg.size()) { continue; }

        double dot_prod = 0.0;
        double norm1_sq = 0.0;
        double norm2_sq = 0.0;
        double l2_dist_sq = 0.0;
        double sum_c = 0.0;
        double sum_p = 0.0;
        const size_t n = curr_avg.size();
        size_t valid_n = 0;

        // Sums for Means
        for (size_t i = 0; i < n; ++i) {
            const double c_val = curr_avg[i];
            const double p_val = prev_avg[i];
            if (std::isfinite(c_val) && std::isfinite(p_val)) {
                sum_c += c_val;
                sum_p += p_val;
                valid_n++;
            }
        }

        if (valid_n == 0) { continue; }
        const double mean_c = sum_c / valid_n;
        const double mean_p = sum_p / valid_n;

        double cov_sum = 0.0;
        double var_c_sum = 0.0;
        double var_p_sum = 0.0;

        // Metrics
        for (size_t i = 0; i < n; ++i) {
            const double c_val = curr_avg[i];
            const double p_val = prev_avg[i];
            if (!std::isfinite(c_val) || !std::isfinite(p_val)) { continue; }

            // Cosine Similarity & L2 Distance
            dot_prod += c_val * p_val;
            norm1_sq += c_val * c_val;
            norm2_sq += p_val * p_val;
            const double diff = c_val - p_val;
            l2_dist_sq += diff * diff;

            // Pearson (Centered stats)
            const double dc = c_val - mean_c;
            const double dp = p_val - mean_p;
            cov_sum += dc * dp;
            var_c_sum += dc * dc;
            var_p_sum += dp * dp;
        }

        ts.n_features = (int64_t)valid_n;
        ts.dot_prod = dot_prod;
        ts.norm1_sq = norm1_sq;
        ts.norm2_sq = norm2_sq;
        ts.cov_sum = cov_sum;
        ts.var_c_sum = var_c_sum;
        ts.var_p_sum = var_p_sum;
        ts.l2_dist_sq = l2_dist_sq;
        ts.l2_dist = (float)std::sqrt(l2_dist_sq);

        if (valid_n > 1) {
            ts.covariance = (float)(cov_sum / (double)valid_n);
        }

        if (norm1_sq > 0.0 && norm2_sq > 0.0) {
            ts.cossim = (float)(dot_prod / (std::sqrt(norm1_sq) * std::sqrt(norm2_sq)));
            ts.cossim = std::clamp(ts.cossim, -1.0f, 1.0f);
        } else {
            ts.cossim = (norm1_sq == 0.0 && norm2_sq == 0.0) ? fnan : 0.0f;
        }

        if (var_c_sum > 0.0 && var_p_sum > 0.0) {
            ts.pearson = (float)(cov_sum / (std::sqrt(var_c_sum) * std::sqrt(var_p_sum)));
            ts.pearson = std::clamp(ts.pearson, -1.0f, 1.0f);
        } else {
            ts.pearson = (var_c_sum == 0.0 && var_p_sum == 0.0) ? fnan : 0.0f;
        }

        if (prev_ts.sum > 1e-10f) {
            ts.gain = std::sqrt(ts.sum / ts.elements) / std::sqrt(prev_ts.sum / prev_ts.elements);
            ts.sum_prev = prev_ts.sum;
            ts.elements_prev = prev_ts.elements;
        } else {
            ts.gain = ts.sum <= 1e-10f ? 1.0f : fnan;
        }
    }
}

static void compute_layer_statistics(const std::vector<tensor_statistics> & tstats,
    std::map<int, float> & layer_cossim,
    std::map<int, float> & layer_l2_dist,
    std::map<int, float> & layer_pearson,
    std::map<int, float> & layer_covariance,
    std::map<int, float> & layer_gain
)
{
    struct layer_aggregation {
        double sum_dot_prod = 0.0;
        double sum_norm1_sq = 0.0;
        double sum_norm2_sq = 0.0;
        double sum_l2_dist_sq = 0.0;
        double sum_cov = 0.0;
        double sum_var_c = 0.0;
        double sum_var_p = 0.0;
        double sum_energy_curr = 0.0;
        double sum_energy_prev = 0.0;
        int64_t sum_elements_curr = 0;
        int64_t sum_elements_prev = 0;
        int64_t sum_n_features = 0;
        int n_tensors = 0;
    };

    constexpr auto fnan = std::numeric_limits<float>::quiet_NaN();
    std::map<int, layer_aggregation> laggr;

    for (const auto & ts : tstats) {
        std::string layer_str;
        std::string dummy;
        process_tensor_name(ts.tensor, layer_str, dummy);
        int blk = -1;
        try {
            blk = std::stoi(layer_str);
        } catch(...) {
            if (layer_str == "-") { blk = -1; }
        }

        if (blk <= 0) { continue; }

        if (ts.norm1_sq == 0.0 && ts.norm2_sq == 0.0 && ts.l2_dist_sq == 0.0) { continue; }
        auto & entry = laggr[blk];
        entry.sum_dot_prod += ts.dot_prod;
        entry.sum_norm1_sq += ts.norm1_sq;
        entry.sum_norm2_sq += ts.norm2_sq;
        entry.sum_l2_dist_sq += ts.l2_dist_sq;
        entry.sum_cov += ts.cov_sum;
        entry.sum_var_c += ts.var_c_sum;
        entry.sum_var_p += ts.var_p_sum;
        entry.n_tensors++;
        if (ts.n_features > 0) { entry.sum_n_features += ts.n_features; }

        // skip tensors with no match in a previous layer
        if (ts.elements_prev > 0) {
            entry.sum_energy_curr += ts.sum;
            entry.sum_energy_prev += ts.sum_prev;
            entry.sum_elements_curr += ts.elements;
            entry.sum_elements_prev += ts.elements_prev;
        }
    }

    for (const auto & [layer, agg] : laggr) {
        if (agg.n_tensors == 0) { continue; }

        float cossim = 0.0f;
        if (agg.sum_norm1_sq > 0.0 && agg.sum_norm2_sq > 0.0) {
            cossim = (float)(agg.sum_dot_prod / (std::sqrt(agg.sum_norm1_sq) * std::sqrt(agg.sum_norm2_sq)));
            cossim = std::clamp(cossim, -1.0f, 1.0f);
        } else if (agg.sum_norm1_sq == 0.0 && agg.sum_norm2_sq == 0.0) {
            cossim = fnan;
        }

        float gain = fnan;
        if (agg.sum_elements_curr > 0 && agg.sum_elements_prev > 0 && agg.sum_energy_prev > 0.0) {
            const double rms_curr = std::sqrt(agg.sum_energy_curr / (double)agg.sum_elements_curr);
            const double rms_prev = std::sqrt(agg.sum_energy_prev / (double)agg.sum_elements_prev);
            gain = (float)(rms_curr / rms_prev);
        }

        layer_cossim[layer] = cossim;
        layer_l2_dist[layer] = (float)std::sqrt(agg.sum_l2_dist_sq);
        layer_gain[layer] = gain;

        if (agg.sum_n_features > 0) { layer_covariance[layer] = (float)(agg.sum_cov / (double)agg.sum_n_features); }
        else { layer_covariance[layer] = fnan; }

        if (agg.sum_var_c > 0.0 && agg.sum_var_p > 0.0) {
            auto pearson = (float)(agg.sum_cov / (std::sqrt(agg.sum_var_c) * std::sqrt(agg.sum_var_p)));
            layer_pearson[layer] = std::clamp(pearson, -1.0f, 1.0f);
        } else if (agg.sum_var_c == 0.0 && agg.sum_var_p == 0.0) {
            layer_pearson[layer] = fnan;
        } else {
            layer_pearson[layer] = 0.0f;
        }
    }
}

static bool all_finite(const float * v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) { return false; }
    }

    return true;
}

// Round to nearest instead of truncating
static int32_t rows_to_chunks(int64_t n_rows, int32_t chunk_size) {
    return (int32_t) ((n_rows + chunk_size / 2) / chunk_size);
}

bool IMatrixCollector::collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const struct ggml_tensor * src0 = t->src[0];
    // PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01:
    // SOURCE-READ: examples/imatrix/imatrix.cpp:391 (fork-only). This fork's fused MoE up/gate ops
    // (ggml/include/ggml.h:654-655, GGML_OP_FUSED_UP_GATE / GGML_OP_MOE_FUSED_UP_GATE) carry their
    // real input one slot further right than GGML_OP_MUL_MAT/MUL_MAT_ID; reading src[1] for them
    // silently collected the wrong tensor's activations (or aliased data) for any model whose graph
    // fuses those ops, and this ask/collect path never even recognised the op to begin with.
    const struct ggml_tensor * src1 =
        (t->op == GGML_OP_FUSED_UP_GATE || t->op == GGML_OP_MOE_FUSED_UP_GATE) ? t->src[2] : t->src[1];
    std::string wname = filter_tensor_name(src0->name);

    const int32_t chunk_size = m_params.n_ctx / m_params.n_parallel;

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID ||
            t->op == GGML_OP_FUSED_UP_GATE ||
            t->op == GGML_OP_MOE_FUSED_UP_GATE) return true; // collect all indirect / fused matrix multiplications
        if (t->op != GGML_OP_MUL_MAT) return false;
        // why are small batches ignored (<16 tokens)?
        if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        // upstream: the NextN projections are collected too (post_projection as an output tensor, pre_projection with NextN)
        const bool is_nextn_out = wname == "nextn.post_projection.weight";
        const bool is_nextn_in  = wname == "nextn.pre_projection.weight";
        if (!(is_named_imatrix_tensor(wname, m_params, m_params.collect_lsim) ||
              (m_params.process_output && is_nextn_out) || (m_params.load_mtp && is_nextn_in))) return false;
        return true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(src1->buffer);

    if (!is_host) {
        const size_t src1_nbytes = ggml_nbytes(src1);
        m_src1_data.resize(src1_nbytes);
        ggml_backend_tensor_get(src1, m_src1_data.data(), 0, src1_nbytes);
    }

    const char * data = is_host ? (const char *) src1->data : m_src1_data.data();
    GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));

    // PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01:
    // SOURCE-READ: examples/imatrix/imatrix.cpp:420-445 (fork-only -lsim). Runs before the
    // MUL_MAT_ID/dense dispatch below because an ffn_up/ffn_gate tensor can arrive via either path
    // (plain MUL_MAT for dense models, MUL_MAT_ID/MOE_FUSED_UP_GATE for MoE) and the attn-vs-ffn
    // similarity needs to see it regardless of which branch stores its imatrix stats.
    if (m_params.collect_lsim && wname.find(".ffn_") != std::string::npos) {
        if (auto index = layer_index(wname); index.has_value() && *index == m_last_layer && *index != m_last_ffn) {
            const int64_t n    = src1->ne[0];
            const bool    is_routed = t->op == GGML_OP_MUL_MAT_ID || t->op == GGML_OP_MOE_FUSED_UP_GATE;
            const int64_t nrow = is_routed ? src1->ne[2] : src1->ne[1];
            if (is_routed) {
                GGML_ASSERT(src1->ne[1] == 1);
            }
            // the buffer is overwritten whole just below; a different size only means a different ubatch
            // (--split-on pieces end in a shorter one), which the old exit(1) here treated as fatal.
            // The real consistency check is the one against m_last_input (same layer, same ubatch) below.
            m_ffn_input.resize(nrow * n);
            std::memcpy(m_ffn_input.data(), data, nrow * n * sizeof(float));
            if ((int64_t) m_ffn_input.size() == (int64_t) m_last_input.size()) {
                if ((size_t) *index + 1 > m_attn_sim.size()) m_attn_sim.resize(*index + 1);
                collect_cos_similarity(nrow, n, m_ffn_input.data(), m_last_input.data(), m_attn_sim[*index]);
                m_last_ffn = *index;
            } else {
                LOG_ERR("%s: inconsistent ffn vs last_input size for %s\n", __func__, wname.c_str());
                exit(1);
            }
        }
    }

    // this has been adapted to the new format of storing merged experts in a single 3d tensor
    // ref: https://github.com/ggml-org/llama.cpp/pull/6387
    if (t->op == GGML_OP_MUL_MAT_ID || t->op == GGML_OP_MOE_FUSED_UP_GATE) {
        //   ids  -> [n_experts_used, n_tokens]
        //   src1 -> [cols, n_expert_used, n_tokens]
        // SOURCE-READ: examples/imatrix/imatrix.cpp:452 (fork-only op). MOE_FUSED_UP_GATE's ids
        // tensor sits at src[3], not src[2] (the fused op carries one more leading operand).
        const ggml_tensor * ids = (t->op == GGML_OP_MUL_MAT_ID) ? t->src[2] : t->src[3];
        const int64_t n_as = src0->ne[2];
        const int64_t n_ids = ids->ne[0];

        // the top-k selected expert ids are stored in the ids tensor
        // for simplicity, always copy ids to host, because it is small
        // take into account that ids is not contiguous!

        GGML_ASSERT(ids->ne[1] == src1->ne[2]);

        // the extra dimension would need to be stored somewhere to be reflected in the imatrix file
        if (ggml_nrows(src1) != src1->ne[1] * src1->ne[2]) {
            LOG_ERR("%s: tensor has more than 3 dimensions: %s", __func__, wname.c_str());
            GGML_ASSERT(false);
        }

        m_ids.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, m_ids.data(), 0, ggml_nbytes(ids));

        auto & e = m_stats[wname];

        if (e.counts.size() == 1 && n_as > 1) {
            // broadcast, when loading an old imatrix
            e.counts.resize(n_as, e.counts[0]);
        }
        if (e.values.empty()) {
            e.activations.resize(src1->ne[0]*n_as, 0);
            e.values.resize(src1->ne[0]*n_as, 0);
            e.counts.resize(n_as, 0);
        }
        else if (e.values.size() != (size_t)src1->ne[0]*n_as) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.values.size(), (int)(src1->ne[0]*n_as));
            exit(1); //GGML_ABORT("fatal error");
        }
        else if (e.counts.size() != (size_t)n_as) {
            LOG_ERR("%s: inconsistent expert count for %s (%d vs %d)\n", __func__, wname.c_str(), (int)e.counts.size(), (int)n_as);
            exit(1); //GGML_ABORT("fatal error");
        }
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_chunk, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[2], (int)src1->type);

        const int64_t ne0 = src1->ne[0];
        const int64_t n_tokens = src1->ne[2];
        const bool has_act = !e.activations.empty();

        const bool hess = m_params.imat_hessian_block > 0;
        if (hess) {
            m_hess_rows.assign(n_as, {});
        }

        // single pass over the routing ids
        std::vector<uint8_t> touched(n_as, 0);
        for (int64_t idx = 0; idx < n_ids; ++idx) {
            for (int64_t row = 0; row < n_tokens; ++row) {
                const int32_t ex = *(const int32_t *) (m_ids.data() + row * ids->nb[1] + idx * ids->nb[0]);
                GGML_ASSERT(ex >= 0 && ex < n_as);  // sanity check
                const int64_t i11 = idx % src1->ne[1];
                const float * x = (const float *) (data + i11 * src1->nb[1] + row * src1->nb[2]);
                float * acc = e.values.data() + ex * ne0;
                float * act = has_act ? e.activations.data() + ex * ne0 : nullptr; // legacy imatrix tensors do not have activation sums

                e.counts[ex]++;
                touched[ex] = 1;
                for (int64_t j = 0; j < ne0; ++j) { acc[j] += x[j] * x[j]; }
                if (act) {
                    for (int64_t j = 0; j < ne0; ++j) { act[j] += x[j]; }
                }
                if (hess) {
                    m_hess_rows[ex].push_back(x);
                }
            }
        }
        if (hess) {
            hessian_accumulate(wname, ne0, n_as);
        }

        // check for non-finite values, only checking experts that were routed to and touched
        for (int64_t ex = 0; ex < n_as; ++ex) {
            if (touched[ex] && !all_finite(e.values.data() + ex * ne0, ne0)) {
                LOG_ERR("%s: non-finite values detected in %s\n", __func__, wname.c_str());
                exit(1);
            }
        }

        for (int64_t ex = 0; ex < n_as; ++ex) {
            const int32_t n_chunk = rows_to_chunks(e.counts[ex], chunk_size);
            if (n_chunk > m_last_chunk) {
                const int32_t chunk_step = n_chunk - m_last_chunk;
                m_last_chunk = n_chunk;
                if ((m_last_chunk % m_params.n_out_freq) / chunk_step == 0) {
                    save_imatrix();
                }
                if (m_params.n_save_freq > 0 && (m_last_chunk % m_params.n_save_freq) / chunk_step == 0) {
                    save_imatrix(m_last_chunk);
                }
            }
        }
    } else {
        // PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01:
        // SOURCE-READ: examples/imatrix/imatrix.cpp:525-556 (fork-only -lsim). Tracks per-layer
        // activation cosine similarity across the dense (non-routed) tensor stream; the MoE branch
        // above never needs this because the first tensor in a layer is never a MoE tensor.
        if (m_params.collect_lsim) {
            if (auto index = layer_index(wname); index.has_value()) {
                if (*index != m_last_layer) {
                    if (*index > 0) {
                        if ((int64_t) m_last_input.size() != src1->ne[0] * src1->ne[1]) {
                            LOG_ERR("%s: lsim size mismatch (%d vs %d) for %s, m_last_layer=%d\n", __func__,
                                    (int)(src1->ne[0] * src1->ne[1]), (int)m_last_input.size(), wname.c_str(), m_last_layer);
                            exit(1);
                        }
                        if ((size_t) *index > m_layer_sim.size()) m_layer_sim.resize(*index);
                        collect_cos_similarity(src1->ne[1], src1->ne[0], m_last_input.data(), (const float *) data, m_layer_sim[*index - 1]);
                        if (*index == m_last_ffn + 1) {
                            if ((size_t) *index > m_ffn_sim.size()) m_ffn_sim.resize(*index);
                            collect_cos_similarity(src1->ne[1], src1->ne[0], m_ffn_input.data(), (const float *) data, m_ffn_sim[*index - 1]);
                        }
                    }
                    m_last_layer = *index;
                    // overwritten whole below: a size change here is a new (shorter) ubatch, e.g. the tail of a
                    // --split-on piece; consecutive layers of ONE ubatch are still checked above (index > 0).
                    m_last_input.resize(src1->ne[0] * src1->ne[1]);
                    std::memcpy(m_last_input.data(), data, src1->ne[0] * src1->ne[1] * sizeof(float));
                }
            }
        }

        auto & e = m_stats[wname];
        const int64_t n_mat = src0->ne[2] * src0->ne[3];

        // use a single count per dense tensor
        // (necessary when merging older GGUF-imatrix files with 3d tensors)
        if (e.counts.size() > 1) {
            bool all_equal = true;
            for (size_t i = 1; i < e.counts.size(); ++i) {
                if (e.counts[0] != e.counts[i]) {
                    all_equal = false;
                    break;
                }
            }
            if (all_equal) {
                e.counts.resize(1);
            }
        }
        if (e.values.empty()) {
            e.activations.resize(src1->ne[0] * n_mat, 0);
            e.values.resize(src1->ne[0] * n_mat, 0);
            e.counts.resize(1, 0);
        }
        else if (e.values.size() != (size_t)(src1->ne[0] * n_mat)) {
            LOG_ERR("%s: inconsistent size for %s (%d vs %d)\n",
                __func__, wname.c_str(), (int)e.values.size(), (int)(src1->ne[0] * n_mat));
            exit(1); //GGML_ABORT("fatal error");
        }
        LOG_DBGV(2, "%s[%d]: %32s, %s, %5d x %5d x %5d, %d\n",
            __func__, m_last_chunk, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[1], (int)src1->ne[2], (int)src1->type);

        const int64_t ne0 = src1->ne[0];

        const bool hess = m_params.imat_hessian_block > 0;
        if (hess) {
            m_hess_rows.assign(n_mat, {});
        }

        const bool has_act = !e.activations.empty();
        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                // handle 3D+ tensors, but flatten 3D+ activations when model tensor is 2D
                const int64_t mat_id = (i3 % src0->ne[3]) * src0->ne[2] + (i2 % src0->ne[2]);
                float * acc = e.values.data() + mat_id * ne0;
                float * act = has_act ? e.activations.data() + mat_id * ne0 : nullptr;
                for (int64_t row = 0; row < src1->ne[1]; ++row) {
                    const float * x = (const float *) (data + row * src1->nb[1] + i2 * src1->nb[2] + i3 * src1->nb[3]);
                    for (int64_t j = 0; j < ne0; ++j) { acc[j] += x[j] * x[j]; }
                    if (act) {
                        for (int64_t j = 0; j < ne0; ++j) { act[j] += x[j]; }
                    }
                    if (hess) {
                        m_hess_rows[mat_id].push_back(x);
                    }
                }
            }
        }
        if (hess) {
            hessian_accumulate(wname, ne0, n_mat);
        }

        // check for non-finite values
        if (!all_finite(e.values.data(), e.values.size())) {
            LOG_ERR("%s: non-finite values detected in %s\n", __func__, wname.c_str());
            exit(1);
        }
        // only 1 count in practice, except when a tensor is used for both MUL_MAT_ID and MUL_MAT
        for (size_t i = 0; i < e.counts.size(); ++i) {
            e.counts[i] += ggml_nrows(src1) / n_mat;
            const int32_t n_chunk = rows_to_chunks(e.counts[i], chunk_size);
            if (n_chunk > m_last_chunk) {
                const int32_t chunk_step = n_chunk - m_last_chunk;
                m_last_chunk = n_chunk;
                if ((m_last_chunk % m_params.n_out_freq) / chunk_step == 0) {
                    save_imatrix();
                }
                if (m_params.n_save_freq > 0 && (m_last_chunk % m_params.n_save_freq) / chunk_step == 0) {
                    save_imatrix(m_last_chunk);
                }
            }
        }
    }

    return true;
}

// PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01:
// SOURCE-READ: examples/imatrix/imatrix.cpp:348-385 (fork-only -lsim report), adapted to this
// file's LOG_INF (upstream logging) instead of the examples file's raw printf.
void IMatrixCollector::print_layer_importance(const char * msg, const std::vector<std::pair<double, int>> & sim) {
    if (sim.empty()) return;
    std::vector<std::pair<float, int>> layers;
    layers.reserve(sim.size());
    for (int i = 0; i < int(sim.size()); ++i) {
        if (sim[i].second > 0) layers.emplace_back(float(std::abs(sim[i].first / sim[i].second)), i);
    }
    if (layers.empty()) return;
    std::sort(layers.begin(), layers.end());
    LOG_INF("%s\n", msg);
    int j = 0;
    for (auto & p : layers) {
        int i = p.second;
        LOG_INF("%3d: Layer %3d, <cos_sim> = %g\n", j++, i, sim[i].first / sim[i].second);
    }
}

void IMatrixCollector::print_layer_importance() const {
    print_layer_importance("\n======================== sorted layer importances", m_layer_sim);
    print_layer_importance("\n======================== sorted attention importances", m_attn_sim);
    print_layer_importance("\n======================== sorted ffn importances", m_ffn_sim);
}

void IMatrixCollector::save_imatrix_legacy(int32_t ncall) const {
    auto fname = m_params.out_file;

    if (ncall > 0) {
        fname += ".at_";
        fname += std::to_string(ncall);
    }

    // warn when writing imatrix entries that do not have full data
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    int n_entries = 0;
    std::vector<std::string> to_store;

    bool is_first = true; // for printing
    for (const auto & kv : m_stats) {
        const int n_all = kv.second.counts.size();

        if (n_all == 0) {
            continue;
        }

        int n_zeros = 0;
        for (const int c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            LOG_INF("\n");
            is_first = false;
        }

        if (n_zeros == n_all) {
            LOG_WRN("%s: entry '%40s' has no data - skipping\n", __func__, kv.first.c_str());
            continue;
        }

        if (n_zeros > 0) {
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%)\n", __func__, kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
        }

        n_entries++;
        to_store.push_back(kv.first);
    }

    if (to_store.size() < m_stats.size()) {
        LOG_WRN("%s: storing only %zu out of %zu entries\n", __func__, to_store.size(), m_stats.size());
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end());

    const int32_t chunk_size = e_chunk_size();
    std::ofstream out(fname, std::ios::binary);
    out.write((const char *) &n_entries, sizeof(n_entries));
    for (const auto & name : to_store) {
        const auto & stat = m_stats.at(name);
        const int32_t len = name.size();
        out.write((const char *) &len, sizeof(len));
        out.write(name.c_str(), len);
        // ceiling division to avoid accidental zeros
        const int32_t ncall = (*std::max_element(stat.counts.begin(), stat.counts.end()) + (chunk_size - 1)) / chunk_size;
        out.write((const char *) &ncall, sizeof(ncall));
        const int32_t nval = stat.values.size();
        const int32_t nmat = stat.counts.size();
        out.write((const char *) &nval, sizeof(nval));
        if (nval > 0 && nmat > 0) {
            std::vector<float> tmp(nval);
            for (int32_t i = 0; i < nval; i++) {
                float count = static_cast<float>(stat.counts[i / (nval / nmat)]);
                float value = stat.values[i];
                if (count == 0.0f) {
                    // store 1 for partial data
                    value = 1.0f;
                    count = 1.0f;
                }
                tmp[i] = (value / count) * static_cast<float>(ncall);
            }
            out.write((const char *) tmp.data(), nval * sizeof(float));
        }
    }

    // Write the number of call the matrix was computed with
    out.write((const char *) &m_last_chunk, sizeof(m_last_chunk));

    // Write the input filename at the end of the file to later on specify it in quantize
    {
        const char * dataset_file = m_params.prompt_file.c_str();
        int32_t len = m_params.prompt_file.size();
        // When there is no prompt but there were other imatrix files loaded, use the last dataset
        if (m_params.prompt_file.empty() && !m_datasets.empty()) {
            const std::string & dataset_str = m_datasets[m_datasets.size() - 1];
            dataset_file = dataset_str.c_str();
            len = dataset_str.size();
        }
        out.write((const char *) &len, sizeof(len));
        out.write(dataset_file, len);
    }

    LOGV(1, "\n");
    LOG_DBGV(1, "%s: stored collected data after %d chunks in %s\n", __func__, m_last_chunk, fname.c_str());
}

imatrix_hessian_entry & IMatrixCollector::hessian_entry(const std::string & wname, int64_t n_in, int64_t n_mat) {
    const int64_t N = m_params.imat_hessian_block;
    auto it = m_hess.find(wname);
    if (it != m_hess.end()) {
        if (it->second.n_in != n_in || it->second.n_mat != n_mat) {
            LOG_ERR("%s: inconsistent Hessian shape for %s (K %lld x n_mat %lld vs K %lld x n_mat %lld)\n", __func__, wname.c_str(),
                    (long long) it->second.n_in, (long long) it->second.n_mat, (long long) n_in, (long long) n_mat);
            exit(1);
        }
        return it->second;
    }
    const uint64_t need = imatrix_hessian_entry_bytes(n_in, N, n_mat);
    const uint64_t cap  = (uint64_t) m_params.imat_hessian_max_mib << 20;
    if (m_hess_bytes + need > cap) {
        LOG_ERR("%s: --hessian-block %lld: %s needs %.1f MiB (K=%lld x N=%lld x n_mat=%lld floats), which brings the total to "
                "%.1f MiB, over --hessian-max-mib %d. Lower --hessian-block or raise --hessian-max-mib.\n", __func__,
                (long long) N, wname.c_str(), need / 1048576.0, (long long) n_in, (long long) N, (long long) n_mat,
                (m_hess_bytes + need) / 1048576.0, m_params.imat_hessian_max_mib);
        exit(1);
    }
    m_hess_bytes += need;
    LOG_DBGV(1, "%s: %s: K=%lld n_mat=%lld -> %.2f MiB (total %.1f MiB)\n", __func__, wname.c_str(), (long long) n_in,
             (long long) n_mat, need / 1048576.0, m_hess_bytes / 1048576.0);
    auto & e = m_hess[wname];
    imatrix_hessian_entry_init(e, n_in, N, n_mat);
    return e;
}

void IMatrixCollector::hessian_accumulate(const std::string & wname, int64_t n_in, int64_t n_mat) {
    const int64_t N = m_params.imat_hessian_block;
    auto & h = hessian_entry(wname, n_in, n_mat);
    const int n_threads = m_params.cpuparams_batch.n_threads > 0 ? m_params.cpuparams_batch.n_threads
                                                                  : std::max(1, m_params.cpuparams.n_threads);
    for (int64_t m = 0; m < n_mat && m < (int64_t) m_hess_rows.size(); ++m) {
        const auto & rows = m_hess_rows[m];
        if (rows.empty()) {
            continue;
        }
        imatrix_hessian_accumulate(h.blocks.data() + m * n_in * N, n_in, N, rows.data(), (int64_t) rows.size(), n_threads);
        h.counts[m] += (int64_t) rows.size();
    }
}

void IMatrixCollector::save_hessian(int32_t n_chunk) const {
    auto fname = hessian_out_file();
    if (n_chunk > 0) {
        fname += ".at_";
        fname += std::to_string(n_chunk);
    }
    imatrix_hessian_file f;
    f.block_size  = m_params.imat_hessian_block;
    f.chunk_count = m_last_chunk;
    f.chunk_size  = m_params.n_ctx / m_params.n_parallel;
    f.datasets    = m_datasets;
    if (!m_params.prompt_file.empty()) {
        f.datasets.push_back(m_params.prompt_file);
    }
    if (!imatrix_hessian_save(fname, f, m_hess)) {
        LOG_ERR("%s: failed to write the Hessian sidecar %s\n", __func__, fname.c_str());
        return;
    }
    LOG_DBGV(1, "%s: stored %zu block-diagonal Hessians (block %d, %.1f MiB) in %s\n", __func__, m_hess.size(),
             m_params.imat_hessian_block, m_hess_bytes / 1048576.0, fname.c_str());
}

void IMatrixCollector::save_imatrix(int32_t n_chunk) const {
    if (m_params.imat_hessian_block > 0) {
        save_hessian(n_chunk);
    }
    auto fname = m_params.out_file;
    int8_t use_legacy_format = m_params.imat_dat;

    if (use_legacy_format > 0) {
        this->save_imatrix_legacy(n_chunk);
        return;
    }
    // only warn when `--output-format gguf` is not specified
    if (use_legacy_format == 0 && !string_ends_with(fname, ".gguf")) {
        LOG_WRN("\n%s: saving imatrix using GGUF format with a different suffix than .gguf\n", __func__);
        LOG_WRN("%s: if you want the previous imatrix format, use --output-format dat\n", __func__);
    }

    if (n_chunk > 0) {
        fname += ".at_";
        fname += std::to_string(n_chunk);
    }

    // write imatrix entries even if they don't have full data. (can be corrected when reading)
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    std::vector<std::string> to_store;
    size_t data_size = 0;

    bool is_first = true; // for printing
    for (const auto & kv : m_stats) {
        const int n_all = kv.second.counts.size();

        int n_zeros = 0;
        for (const auto c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            LOG_INF("\n");
            is_first = false;
        }

        if (n_zeros > 0) {
            LOG_WRN("%s: entry '%40s' has partial data (%.2f%%)\n", __func__, kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
        }

        to_store.push_back(kv.first);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.activations.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.values.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * kv.second.counts.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * 10, GGML_MEM_ALIGN);
    }

    // deterministic tensor name order
    std::sort(to_store.begin(), to_store.end());

    // Compute per-tensor statistics
    std::vector<tensor_statistics> tstats;
    tstats.reserve(m_stats.size());
    for (const auto & kv : m_stats) { compute_vector_statistics(tstats, kv.first, kv.second); }
    if (!tstats.empty()) { compute_tensor_statistics(tstats, m_stats, nextn_layer_start(tstats, m_n_layer_nextn)); }

    // index by tensor name
    std::unordered_map<std::string, const tensor_statistics *> tstat_index;
    tstat_index.reserve(tstats.size());
    for (const auto & ts : tstats) { tstat_index[ts.tensor] = &ts; }

    struct ggml_init_params params = {
        /* .mem_size   = */ data_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct gguf_context * ctx_gguf = gguf_init_empty();

    {
        std::vector<const char *> datasets;
        datasets.reserve(m_datasets.size() + 1);
        for (size_t i = 0; i < m_datasets.size(); ++i) {
            datasets.push_back(m_datasets[i].c_str());
        }
        if (!m_params.prompt_file.empty()) {
            datasets.push_back(m_params.prompt_file.c_str());
        }

        gguf_set_val_str(ctx_gguf, "general.type", "imatrix");
        // Write the dataset paths
        gguf_set_arr_str(ctx_gguf, LLM_KV_IMATRIX_DATASETS, datasets.data(), datasets.size());
        // Write the number of chunks the matrix was computed with
        gguf_set_val_u32(ctx_gguf, LLM_KV_IMATRIX_CHUNK_COUNT, m_last_chunk);
        gguf_set_val_u32(ctx_gguf, LLM_KV_IMATRIX_CHUNK_SIZE, e_chunk_size());
        // Write how many of the top layers are NextN layers, so statistics can tell them apart
        if (m_n_layer_nextn > 0) { gguf_set_val_u32(ctx_gguf, LLM_KV_IMATRIX_N_LAYER_NEXTN, m_n_layer_nextn); }
        // Define the schema for the tensor statistics (for use in quantize.cpp)
        const char * stats_schema[] = {
            "sum_sq", "mean", "elements", "std_deviation", "skewness", "kurtosis", "gain", "h_norm", "l2_dist", "cossim", "pearson", "covariance"
        };
        gguf_set_arr_str(ctx_gguf, LLM_KV_IMATRIX_STATS_SCHEMA, stats_schema, 12);
    }

    for (const auto & name : to_store) {
        const auto & stat = m_stats.at(name);
        const int32_t nval = (int32_t) stat.values.size();
        const int32_t nmat = (int32_t) stat.counts.size();
        if (nval > 0 && nmat > 0) {
            struct ggml_tensor * in_sum2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nval / nmat, nmat);
            struct ggml_tensor * counts  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, nmat);
            ggml_format_name(in_sum2, "%s.in_sum2", name.c_str());
            ggml_format_name(counts, "%s.counts", name.c_str());

            for (int32_t j = 0; j < nval; ++j) {
                ((float *) in_sum2->data)[j] = (float) stat.values[j];
            }
            for (int32_t j = 0; j < nmat; ++j) {
                ((float *) counts->data)[j] = (float) stat.counts[j];
            }

            gguf_add_tensor(ctx_gguf, in_sum2);
            gguf_add_tensor(ctx_gguf, counts);

            if (!stat.activations.empty()) {
                const int32_t nact = (int32_t) stat.activations.size();
                struct ggml_tensor * in_sum  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nact / nmat, nmat);
                ggml_format_name(in_sum, "%s.in_sum", name.c_str());
                for (int32_t j = 0; j < nact; ++j) {
                    ((float *) in_sum->data)[j] = (float) stat.activations[j];
                }
                gguf_add_tensor(ctx_gguf, in_sum);
            }
        } else {
            LOG_WRN("%s: no data for tensor %s\n", __func__, name.c_str());
        }

        // Store per-tensor statistics as a small 1D tensor
        {
            float fnan = std::numeric_limits<float>::quiet_NaN();
            double sum_sq = 0.0f;
            float mean = 0.0f;
            float elements = 0.0f;
            float std_deviation = 0.0f;
            float skewness = 0.0f;
            float kurtosis = 0.0f;
            float gain = 0.0f;
            float h_norm = 0.0f;
            float l2_dist = 0.0f;
            float cossim = 0.0f;
            float pearson = 0.0f;
            float covariance = 0.0f;
            auto ts = tstat_index.find(name);
            if (ts != tstat_index.end() && ts->second != nullptr) {
                sum_sq = ts->second->sum;
                mean = ts->second->mean;
                elements = (float)ts->second->elements;
                std_deviation = ts->second->std_deviation;
                skewness = ts->second->skewness;
                kurtosis = ts->second->kurtosis;
                gain = ts->second->gain;
                h_norm = ts->second->elements > 1 ? 100.0f * (ts->second->entropy / std::log2f((float)ts->second->elements)) : fnan;
                l2_dist = ts->second->l2_dist;
                cossim = ts->second->cossim;
                pearson = ts->second->pearson;
                covariance = ts->second->covariance;
            }

            struct ggml_tensor * stats = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 12);
            ggml_format_name(stats, "%s.stats", name.c_str());
            // Store the statistics in the same order as defined in stats_schema[]
            ((float *)stats->data)[0] = (float)sum_sq;
            ((float *)stats->data)[1] = mean;
            ((float *)stats->data)[2] = elements;
            ((float *)stats->data)[3] = std_deviation;
            ((float *)stats->data)[4] = skewness;
            ((float *)stats->data)[5] = kurtosis;
            ((float *)stats->data)[6] = gain;
            ((float *)stats->data)[7] = h_norm;
            ((float *)stats->data)[8] = l2_dist;
            ((float *)stats->data)[9] = cossim;
            ((float *)stats->data)[10] = pearson;
            ((float *)stats->data)[11] = covariance;
            gguf_add_tensor(ctx_gguf, stats);
        }
    }

    gguf_write_to_file(ctx_gguf, fname.c_str(), false);

    LOGV(1, "\n");
    LOG_DBGV(1, "%s: stored collected data after %d chunks in %s\n", __func__, m_last_chunk, fname.c_str());

    gguf_free(ctx_gguf);
    ggml_free(ctx);
}

bool IMatrixCollector::load_imatrix(const char * file_name) {
    common_imatrix loaded;
    if (!common_imatrix_load(file_name, loaded)) {
        return false;
    }

    const bool is_legacy = loaded.is_legacy;
    if (loaded.n_layer_nextn > 0) {
        if (m_n_layer_nextn == 0) { m_n_layer_nextn = loaded.n_layer_nextn; }
        else if (m_n_layer_nextn != loaded.n_layer_nextn) {
            LOG_WRN("%s: NextN layer count mismatch in %s: %d != %d, using %d\n",
                __func__, file_name, loaded.n_layer_nextn, m_n_layer_nextn, m_n_layer_nextn);
        }
    }

    if (!is_legacy && loaded.chunk_size > 0) {
        if (m_chunk_size == 0) { m_chunk_size = loaded.chunk_size; }
        else if (m_chunk_size != loaded.chunk_size) {
            LOG_WRN("%s: chunk size mismatch in %s: %d != %d, using %d\n",
                __func__, file_name, loaded.chunk_size, m_chunk_size, m_chunk_size);
        }
    }

    const int32_t chunk_size = e_chunk_size();

    for (auto & [name, entry] : loaded.entries) {
        auto & e = m_stats[name];

        if (is_legacy) {
            if (e.values.empty()) {
                e.values.resize(entry.sums.size(), 0.0f);
                e.counts.resize(1, 0);
            }

            for (size_t j = 0; j < entry.sums.size(); ++j) { e.values[j] += entry.sums[j] * chunk_size; }
            for (size_t j = 0; j < e.counts.size(); ++j) { e.counts[j] += entry.counts[0] * chunk_size; }
            e.activations.clear();
        } else {
            // GGUF format: raw sums and counts, accumulate directly
            const int64_t nval    = entry.sums.size();
            const int64_t ncounts = entry.counts.size();
            const int64_t nact    = entry.activations.size();
            const bool first_contribution = e.values.empty();

            if (e.values.empty()) { e.values.resize(nval, 0.0f); }
            else if ((size_t)nval != e.values.size()) {
                LOG_ERR("%s: mismatched sums size for %s: %zu != %zu\n",
                    __func__, name.c_str(), (size_t) nval, e.values.size());

                return false;
            }

            if (e.counts.empty()) { e.counts.resize(ncounts, 0); }
            else if (e.counts.size() == 1 && ncounts > 1) { e.counts.resize(ncounts, e.counts[0]); }
            else if ((size_t) ncounts != e.counts.size()) {
                LOG_ERR("%s: mismatched counts size for %s: %zu != %zu\n",
                    __func__, name.c_str(), (size_t) ncounts, e.counts.size());

                return false;
            }

            for (int64_t j = 0; j < nval; ++j) { e.values[j] += entry.sums[j]; }
            for (int64_t j = 0; j < ncounts; ++j) { e.counts[j] += entry.counts[j]; }

            if (nact > 0 && (first_contribution || !e.activations.empty())) {
                if (nact != nval) {
                    LOG_ERR("%s: mismatched activations size for %s: %zu != %zu\n",
                        __func__, name.c_str(), (size_t) nact, (size_t) nval);

                    return false;
                }

                if (e.activations.empty()) { e.activations.resize(nact, 0.0f); }
                else if ((size_t) nact != e.activations.size()) {
                    LOG_ERR("%s: mismatched activations size for %s: %zu != %zu\n",
                        __func__, name.c_str(), (size_t) nact, e.activations.size());

                    return false;
                }

                for (int64_t j = 0; j < nact; ++j) { e.activations[j] += entry.activations[j]; }
            } else { e.activations.clear(); }
        }
    }

    m_datasets.insert(m_datasets.end(), loaded.datasets.begin(), loaded.datasets.end());

    // Calculate the last chunk count
    int64_t max_count = 0;
    for (const auto & stats : m_stats) {
        for (int64_t count : stats.second.counts) {
            if (count > max_count) {
                max_count = count;
            }
        }
    }

    m_last_chunk = rows_to_chunks(max_count, chunk_size);

    return true;
}

static IMatrixCollector g_collector;

// PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01
// SOURCE-READ: examples/imatrix/imatrix.cpp:789-790 -- second collector instance for the paired
// draft/MTP-assistant model, only ever populated when --output-draft/-md request paired mode.
static IMatrixCollector g_draft_collector;

// RETURN TYPE CHANGED bool -> int 2026-09-08. Not a behaviour change: this fork's
// ggml_backend_sched_eval_callback is three-valued (0 no, 1 yes+synchronize, 2 yes+no-sync)
// and a bool(*)(...) does not implicitly convert to an int(*)(...). The body below is
// untouched; its return true/false now mean 1/0, which is exactly what they meant before.
// See the contract at ggml/include/ggml-backend.h.
//
// PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01
// SOURCE-READ: examples/imatrix/imatrix.cpp:792-798 (ik_get_imatrix_collector) -- dispatch by
// cb_eval_user_data so the SAME callback pointer can feed either collector. user_data == nullptr
// keeps routing to g_collector exactly as before this port (no behaviour change for the
// non-paired path); non-null routes to whichever collector the caller passed as user_data.
static IMatrixCollector * ik_get_imatrix_collector(void * user_data) {
    return user_data != nullptr ? static_cast<IMatrixCollector *>(user_data) : &g_collector;
}

static int ik_collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    return ik_get_imatrix_collector(user_data)->collect_imatrix(t, ask, user_data);
}

struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

static std::vector<float> softmax(const std::vector<float> & logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum_exp;
    }
    return probs;
}

static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return {logits[tok] - max_logit - log(sum_exp), logits[tok], expf(logits[tok] - max_logit) / (float) sum_exp};
}

static void process_logits(
    int n_vocab, const float * logits, const int * tokens, int n_token, std::vector<std::thread> & workers,
    double & nll, double & nll2, float * logit_history, float * prob_history) {
    std::mutex mutex;
    int counter = 0;
    auto compute = [&mutex, &counter, &nll, &nll2, logit_history, prob_history, n_vocab, logits, tokens, n_token] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const results_log_softmax results = log_softmax(n_vocab, logits + i*n_vocab, tokens[i+1]);
            const double v = -results.log_softmax;
            local_nll += v;
            local_nll2 += v*v;

            logit_history[i] = results.logit;
            prob_history[i]  = results.prob;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
}

// PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01
// SOURCE-READ: examples/imatrix/imatrix.cpp:908-945 (compute_draft_imatrix_batch), adapted:
//   - fprintf(stderr, ...) -> LOG_ERR(...) to match this file's logging convention.
//   - llama_batch_get_one(tokens, n, pos0, seq0) (fork's 4-arg legacy form) no longer exists --
//     include/llama.h:1580-1590 removed the pos_0/seq_id parameters upstream (#9745) and its own
//     comment says explicit position/sequence now requires llama_batch_init + common_batch_add.
//     batch_dft is allocated once by the caller (compute_imatrix) and reused per call here,
//     mirroring how the target-side `batch` in compute_imatrix is itself cleared and refilled
//     every iteration rather than reallocated.
//   - llama_set_mtp_target_context() pairing itself is NOT done here (it is a one-time call made
//     once in main() after ctx_dft is created), matching the fork's own main() at
//     examples/imatrix/imatrix.cpp:1223 where pairing happens once, outside the per-batch path.
static bool compute_draft_imatrix_batch(
        llama_context     * ctx_tgt,
        llama_context     * ctx_dft,
        common_batch       & batch_dft,
        const llama_token  * draft_tokens,
        int                  batch_start,
        int                  batch_size,
        int                  batch_pos) {
    const float * hidden       = llama_get_embeddings(ctx_tgt);
    const int     n_embd_tgt   = llama_mtp_state_n_embd(ctx_tgt);
    const int     n_embd_dft   = llama_mtp_state_n_embd(ctx_dft);

    if (hidden == nullptr || n_embd_tgt <= 0 || n_embd_dft <= 0) {
        LOG_ERR("%s: missing target hidden state for paired draft calibration\n", __func__);
        return false;
    }

    if (n_embd_tgt != n_embd_dft) {
        LOG_ERR("%s: hidden width mismatch between target (%d) and draft (%d)\n",
                __func__, n_embd_tgt, n_embd_dft);
        return false;
    }

    llama_set_mtp_op_type(ctx_dft, MTP_OP_DRAFT_GEN);
    if (!llama_set_draft_input_hidden_state_copy(ctx_dft, hidden, (size_t) batch_size * n_embd_dft)) {
        llama_set_mtp_op_type(ctx_dft, MTP_OP_NONE);
        LOG_ERR("%s: failed to stage paired draft hidden snapshot\n", __func__);
        return false;
    }

    batch_dft.clear();
    for (int k = 0; k < batch_size; ++k) {
        batch_dft.add(draft_tokens[batch_start + k], batch_pos + k, 0, true);
    }

    const int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_dft.get());
    llama_set_mtp_op_type(ctx_dft, MTP_OP_NONE);

    if (ret != 0) {
        LOG_ERR("%s: paired draft eval failed\n", __func__);
        return false;
    }

    return true;
}

struct nextn_collector {
    llama_context_ptr  ctx;
    common_batch       batch;
    int32_t            n_embd = 0;
    bool               own_lm_head = false;
    std::vector<float> pending_h;
    void clear_memory() {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        pending_h.clear(); // a chunk's last h-row has no next token, the trunk restarts at position 0
    }
    bool decode(llama_context * ctx_trunk, const common_batch & batch_trunk);
};

struct nextn_model_info {
    bool has_layers = false;
    std::vector<bool> own_lm_head;
};

static nextn_model_info nextn_read_model_info(const std::string & model_path, int32_t n_layer, int32_t n_heads) {
    nextn_model_info info;
    info.own_lm_head.assign(n_heads, false);

    struct gguf_init_params gguf_params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), gguf_params);
    if (!ctx_gguf) {
        LOG_WRN("%s: cannot read '%s'\n", __func__, model_path.c_str());
        return info;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);

    for (int32_t head = 0; head < n_heads; ++head) {
        const std::string prefix  = "blk." + std::to_string(n_layer + head) + ".nextn.";
        const std::string lm_head = prefix + "shared_head_head.weight";
        for (int64_t i = 0; i < n_tensors; ++i) {
            const std::string name = gguf_get_tensor_name(ctx_gguf, i);
            if (name.rfind(prefix, 0) != 0) { continue; }

            info.has_layers = true;
            if (name == lm_head) { info.own_lm_head[head] = true; }
        }
    }

    gguf_free(ctx_gguf);

    return info;
}

struct model_file_shape {
    bool valid = false;
    bool is_split = false;
    std::string arch;
    uint32_t n_layer_all = 0;
    uint32_t n_embd_out = 0;
    std::set<uint32_t> nextn_idx;
    bool has_trunk = false;
    bool has_nextn = false;
    uint32_t n_nextn_layers = 0;
    std::string first_nextn;

    uint32_t n_trunk() const { return n_layer_all >= n_nextn_layers ? n_layer_all - n_nextn_layers : 0; }
};

static model_file_shape model_read_file_shape(const std::string & model_path) {
    model_file_shape shape;

    struct gguf_init_params gguf_params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), gguf_params);
    if (!ctx_gguf) { return shape; }
    shape.valid = true;

    const int64_t arch_key = gguf_find_key(ctx_gguf, "general.architecture");
    if (arch_key >= 0 && gguf_get_kv_type(ctx_gguf, arch_key) == GGUF_TYPE_STRING) {
        shape.arch = gguf_get_val_str(ctx_gguf, arch_key);
        const std::string prefix = shape.arch + ".";

        const int64_t block_key = gguf_find_key(ctx_gguf, (prefix + "block_count").c_str());
        if (block_key >= 0 && gguf_get_kv_type(ctx_gguf, block_key) == GGUF_TYPE_UINT32) {
            shape.n_layer_all = gguf_get_val_u32(ctx_gguf, block_key);
        }

        const int64_t nextn_key = gguf_find_key(ctx_gguf, (prefix + "nextn_predict_layers").c_str());
        if (nextn_key >= 0 && gguf_get_kv_type(ctx_gguf, nextn_key) == GGUF_TYPE_UINT32) {
            shape.n_nextn_layers = gguf_get_val_u32(ctx_gguf, nextn_key);
        }

        const int64_t embd_out_key = gguf_find_key(ctx_gguf, (prefix + "embedding_length_out").c_str());
        const int64_t embd_key     = gguf_find_key(ctx_gguf, (prefix + "embedding_length").c_str());
        if (embd_out_key >= 0 && gguf_get_kv_type(ctx_gguf, embd_out_key) == GGUF_TYPE_UINT32) {
            shape.n_embd_out = gguf_get_val_u32(ctx_gguf, embd_out_key);
        } else if (embd_key >= 0 && gguf_get_kv_type(ctx_gguf, embd_key) == GGUF_TYPE_UINT32) {
            shape.n_embd_out = gguf_get_val_u32(ctx_gguf, embd_key);
        }
    }

    const int64_t split_key = gguf_find_key(ctx_gguf, "split.count");
    if (split_key >= 0 && gguf_get_kv_type(ctx_gguf, split_key) == GGUF_TYPE_UINT16) {
        shape.is_split = gguf_get_val_u16(ctx_gguf, split_key) > 1;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(ctx_gguf, i);
        if (name.find(".nextn.") != std::string::npos) {
            shape.has_nextn = true;
            if (shape.first_nextn.empty()) { shape.first_nextn = name; }

            if (name.rfind("blk.", 0) == 0) {
                const size_t dot = name.find('.', 4);
                if (dot != std::string::npos && name.compare(dot + 1, 6, "nextn.") == 0) {
                    uint32_t idx = 0;
                    bool ok = dot > 4;
                    for (size_t j = 4; j < dot && ok; ++j) {
                        ok  = name[j] >= '0' && name[j] <= '9';
                        idx = 10*idx + (name[j] - '0');
                    }

                    if (ok) { shape.nextn_idx.insert(idx); }
                }
            }
        } else if (name.rfind("blk.0.", 0) == 0) {
            shape.has_trunk = true;
        }
    }

    gguf_free(ctx_gguf);

    return shape;
}

static std::unique_ptr<nextn_collector> nextn_collector_init(llama_model * model, const common_params & params, bool own_lm_head) {
    auto cparams = common_context_params_to_llama(params);
    cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cparams.n_rs_seq = 0;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create the NextN context\n", __func__);
        return nullptr;
    }

    auto res = std::make_unique<nextn_collector>();
    res->ctx.reset(ctx);
    res->n_embd = llama_model_n_embd_out(model);
    res->own_lm_head = own_lm_head;
    res->batch = common_batch(ctx);

    return res;
}

bool nextn_collector::decode(llama_context * ctx_trunk, const common_batch & batch_trunk) {
    const int32_t n_last = batch_trunk.size() - 1;
    batch.clear();
    if (!pending_h.empty()) {
        const int32_t idx = batch.add(batch_trunk.tokens[0].id, batch_trunk.tokens[0].pos[0], 0, own_lm_head);
        batch.set_embd_state(idx, { pending_h.data(), 1, (size_t) n_embd });
    }

    const float * h_last = nullptr;
    for (int32_t i = 0; i <= n_last; ++i) {
        const float * h = llama_get_embeddings_nextn_ith(ctx_trunk, i);
        if (h == nullptr) {
            LOG_ERR("%s: no NextN hidden state for row %d\n", __func__, i);

            return false;
        }

        if (i == n_last) { h_last = h; break; }

        const int32_t idx = batch.add(batch_trunk.tokens[i + 1].id, batch_trunk.tokens[i + 1].pos[0], 0, own_lm_head);
        batch.set_embd_state(idx, { h, 1, (size_t) n_embd });
    }

    if (batch.size() > 0) {
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            LOG_ERR("%s: failed to decode the NextN layer\n", __func__);

            return false;
        }
    }

    if (h_last != nullptr) { pending_h.assign(h_last, h_last + n_embd); }

    return true;
}

static bool compute_imatrix(llama_context * ctx, const common_params & params, const int32_t n_ctx, nextn_collector * nextn,
                            llama_context * ctx_dft = nullptr) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_LAST) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

    auto tim1 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);

    auto tim2 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenization took %g ms\n",__func__,1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count());

    if (params.i_chunk > 0) {
        if (size_t((params.i_chunk + 2)*n_ctx) >= tokens.size()) {
            LOG_ERR("%s: there will be not enough tokens left after removing %d chunks\n", __func__, params.i_chunk);
            return false;
        }
        LOG_INF("%s: removing initial %d chunks (%d tokens)\n", __func__, params.i_chunk, params.i_chunk*n_ctx);
        tokens.erase(tokens.begin(), tokens.begin() + params.i_chunk*n_ctx);
    }

    if (int(tokens.size()) < 2*n_ctx) {
        LOG_ERR("%s: you need at least %d tokens for a context of %d tokens\n", __func__, 2*n_ctx, n_ctx);
        LOG_ERR("%s: the data file you provided tokenizes to only %zu tokens\n", __func__, tokens.size());
        return false;
    }

    std::vector<float> logit_history;
    std::vector<float> prob_history;

    if (params.compute_ppl) {
        logit_history.resize(tokens.size());
        prob_history.resize(tokens.size());
    }

    const int n_chunk_max = tokens.size() / n_ctx;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = params.n_batch;

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    const int num_batches = (n_ctx + n_batch - 1) / n_batch;
    const int n_seq = std::max(1, n_batch / n_ctx);

    GGML_ASSERT(n_batch < n_ctx || n_batch % n_ctx == 0);
    GGML_ASSERT(params.n_ctx == n_seq * n_ctx);

    // PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01
    // The fork's paired-draft path (examples/imatrix/imatrix.cpp:947-1092) assumed a single
    // sequence per decode; this tool's compute_imatrix batches up to n_seq sequences together
    // (n_seq = n_batch/n_ctx, see above). main() clamps --batch-size <= --ctx-size whenever a
    // draft model is paired specifically so n_seq==1 here always -- see the clamp there. Scope
    // is deliberately restricted rather than generalizing the draft-eval path to n_seq>1.
    if (ctx_dft != nullptr && n_seq != 1) {
        LOG_ERR("%s: paired draft imatrix requires n_seq==1 (got %d); this is a caller bug, "
                "main() should have clamped --batch-size <= --ctx-size\n", __func__, n_seq);
        return false;
    }

    common_batch batch(ctx);
    // the paired draft's own batch (fork --model-draft with a non-NextN draft)
    common_batch batch_dft = ctx_dft != nullptr ? common_batch(ctx_dft) : common_batch();

    std::vector<float> logits;
    if (params.compute_ppl && num_batches > 1) {
        logits.reserve((size_t)n_ctx * n_vocab);
    }

    LOG_INF("%s: computing over %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d\n", __func__, n_chunk, n_ctx, n_batch, n_seq);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    for (int i = 0; i < n_chunk; i += n_seq) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        const int n_seq_batch = std::min(n_seq, n_chunk - i);

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_memory_clear(llama_get_memory(ctx), true);
        if (nextn) { nextn->clear_memory(); }

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            // clear the batch
            batch.clear();

            for (int seq = 0; seq < n_seq_batch; seq++) {
                int seq_start = batch_start + seq*n_ctx;

                // save original token and restore it after eval
                const auto token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if (add_bos && j == 0) {
                    tokens[seq_start] = llama_vocab_bos(vocab);
                }
                for (int k = 0; k < batch_size; ++k) {
                    // NOTE: specifying all logits to get activations for the output.weight tensor
                    //       and also for the perplexity calculation.
                    // TODO: only get outputs when (params.process_output || params.compute_ppl)
                    //       (not possible when this skips FFN computation of the last layer)
                    batch.add(tokens[seq_start + k], j*n_batch + k, seq, true);
                }

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("%s : failed to eval\n", __func__);

                return false;
            }

            if (nextn && !nextn->decode(ctx, batch)) {
                return false;
            }

            // PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01
            // SOURCE-READ: examples/imatrix/imatrix.cpp:1064-ish call site (inside its own
            // single-sequence batch loop) -- feed the just-computed target hidden state into the
            // paired draft/MTP-assistant model so its own activations get collected too.
            // n_seq_batch == 1 is guaranteed by the n_seq==1 guard above, so batch_start/batch_size
            // here already describe exactly one sequence's positions.
            if (ctx_dft != nullptr) {
                if (!compute_draft_imatrix_batch(ctx, ctx_dft, batch_dft, tokens.data(),
                                                  batch_start, batch_size, j*n_batch)) {
                    return false;
                }
            }

            if (params.compute_ppl && num_batches > 1) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);
            }
        }


        if (i == 0) {
            llama_synchronize(ctx);
            const auto t_end = std::chrono::high_resolution_clock::now();
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk / n_seq);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }

        if (params.compute_ppl) {
            const int first = n_ctx/2;
            for (int seq = 0; seq < n_seq_batch; seq++) {
                const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits_ith(ctx, seq*n_ctx);

                llama_token * tokens_data = tokens.data() + start + seq*n_ctx + first;

                process_logits(n_vocab, all_logits + first*n_vocab,
                        tokens_data, n_ctx - 1 - first,
                        workers, nll, nll2,
                        logit_history.data() + start + seq*n_ctx + first,
                        prob_history.data()  + start + seq*n_ctx + first);

                count += n_ctx - first - 1;

                LOG("[%d]%.4lf,", i + seq + 1, std::exp(nll / count));
            }
            fflush(stdout);

            logits.clear();
        }
    }

    LOG("\n");

    if (params.compute_ppl) {
        nll2 /= count;
        nll /= count;
        const double ppl = exp(nll);
        nll2 -= nll * nll;
        if (nll2 > 0) {
            nll2 = sqrt(nll2/(count-1));
            LOG("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
        } else {
            LOG("Unexpected negative standard deviation of log(prob)\n");
        }
    }


    return true;
}

// --split-on for imatrix: behaviour of Thireus/ik_llama.cpp th/imatrix_with_split_on (examples/imatrix/imatrix.cpp
// split_exact / compute_imatrix_tokens / compute_imatrix, MIT), re-implemented on this tool's batch API, the same way
// tools/perplexity/perplexity.cpp perplexity_split_on() does it for perplexity.
// The input is cut on an exact delimiter; every non-empty piece is its own sequence: KV cleared, BOS if the vocab wants
// it, truncated to n_ctx (a longer piece is NOT continued in a second window -- one piece, one sequence), every token
// an output (output.weight activations), -lsim state reset between pieces. --chunk N skips the first N pieces,
// --chunks N keeps the first N. With --compute-ppl (default) each piece is scored over its second half and one Final
// estimate is printed over all pieces. A paired draft model is refused in main(): the draft path
// (compute_draft_imatrix_batch) is written for compute_imatrix's fixed n_ctx windows.
static std::vector<std::string> imatrix_split_exact(const std::string & text, const std::string & delim) {
    std::vector<std::string> parts;
    if (delim.empty()) {
        parts.push_back(text);
        return parts;
    }
    size_t start = 0;
    while (true) {
        const size_t pos = text.find(delim, start);
        if (pos == std::string::npos) {
            parts.push_back(text.substr(start));
            break;
        }
        parts.push_back(text.substr(start, pos - start));
        start = pos + delim.size();
    }
    return parts;
}

static bool compute_imatrix_split_on(llama_context * ctx, const common_params & params, const int32_t n_ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<std::string> pieces;
    for (auto & p : imatrix_split_exact(params.prompt, params.ppl_split_on)) {
        if (!p.empty()) {
            pieces.push_back(std::move(p));
        }
    }
    LOG_INF("%s: split on '%s' -> %zu non-empty piece(s)\n", __func__, params.ppl_split_on.c_str(), pieces.size());
    if (params.i_chunk > 0) {
        if ((size_t) params.i_chunk >= pieces.size()) {
            LOG_ERR("%s: --chunk %d leaves no piece (only %zu)\n", __func__, params.i_chunk, pieces.size());
            return false;
        }
        pieces.erase(pieces.begin(), pieces.begin() + params.i_chunk);
        LOG_INF("%s: --chunk %d -> skipped the first %d piece(s)\n", __func__, params.i_chunk, params.i_chunk);
    }
    if (params.n_chunks >= 0 && (size_t) params.n_chunks < pieces.size()) {
        pieces.resize(params.n_chunks);
        LOG_INF("%s: --chunks %d -> the first %zu piece(s)\n", __func__, params.n_chunks, pieces.size());
    }
    if (pieces.empty()) {
        LOG_ERR("%s: nothing to process\n", __func__);
        return false;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = std::min(params.n_batch, n_ctx);

    std::vector<std::thread> workers(std::max(1u, std::thread::hardware_concurrency()) - 1);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    int    count = 0;
    double nll   = 0.0;
    double nll2  = 0.0;

    for (size_t ip = 0; ip < pieces.size(); ++ip) {
        if (params.ppl_verbose_chunks) {
            LOG("\n===== chunk %zu/%zu =====\n%s%s===== end chunk %zu/%zu =====\n", ip + 1, pieces.size(), pieces[ip].c_str(),
                (pieces[ip].back() == '\n' ? "" : "\n"), ip + 1, pieces.size());
        }

        std::vector<llama_token> tokens = common_tokenize(ctx, pieces[ip], true, params.parse_special);
        const size_t n_tok_full = tokens.size();
        if ((int) tokens.size() > n_ctx) {
            tokens.resize(n_ctx);
        }
        const int len = (int) tokens.size();
        LOG_INF("%s: piece %zu/%zu: %zu token(s)%s\n", __func__, ip + 1, pieces.size(), n_tok_full,
                (int) n_tok_full > n_ctx ? " (truncated to n_ctx)" : "");
        if (len < 1) {
            continue;
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        g_collector.reset_lsim_state();

        // targets keep the piece's own ids; only the decoded input gets BOS at position 0 (as compute_imatrix does)
        const std::vector<llama_token> targets = tokens;
        if (add_bos) {
            tokens[0] = llama_vocab_bos(vocab);
        }

        std::vector<float> logits;
        if (params.compute_ppl) {
            logits.reserve((size_t) len * n_vocab);
        }
        const int num_batches = (len + n_batch - 1) / n_batch;
        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = j * n_batch;
            const int batch_size  = std::min(len - batch_start, n_batch);
            common_batch_clear(batch);
            for (int k = 0; k < batch_size; ++k) {
                // every token is an output: activations for output.weight, and the ppl logits
                common_batch_add(batch, tokens[batch_start + k], batch_start + k, { 0 }, true);
            }
            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s: failed to decode piece %zu\n", __func__, ip + 1);
                llama_batch_free(batch);
                return false;
            }
            if (params.compute_ppl) {
                const float * bl = llama_get_logits(ctx);
                logits.insert(logits.end(), bl, bl + (size_t) batch_size * n_vocab);
            }
        }

        if (params.compute_ppl) {
            const int first  = len / 2;
            const int n_eval = len - 1 - first;
            if (n_eval > 0) {
                std::vector<float> logit_hist(n_eval), prob_hist(n_eval);
                process_logits(n_vocab, logits.data() + (size_t) first * n_vocab, targets.data() + first, n_eval,
                        workers, nll, nll2, logit_hist.data(), prob_hist.data());
                count += n_eval;
            }
            if (count > 0) {
                LOG("[%zu/%zu]%.4lf,", ip + 1, pieces.size(), std::exp(nll / count));
            }
        }
    }
    LOG("\n");
    llama_batch_free(batch);

    if (params.compute_ppl && count > 0) {
        const double mean = nll / count;
        const double var  = nll2 / count - mean * mean;
        const double ppl  = std::exp(mean);
        if (var > 0 && count > 1) {
            LOG("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, std::sqrt(var / (count - 1)) * ppl);
        } else {
            LOG("Final estimate: PPL = %.4lf\n", ppl);
        }
    }
    return true;
}

static bool show_statistics(const common_params & params) {
    constexpr auto fnan = std::numeric_limits<float>::quiet_NaN();
    g_collector.set_params(params);
    std::vector<tensor_statistics> ts;

    if (params.in_files.empty()) { return false; }

    // Load and process data
    if (g_collector.load_imatrix(params.in_files[0].c_str())) {
        ts.reserve(g_collector.get_mstats().size());
        for (const auto & [name, stats] : g_collector.get_mstats()) {
            compute_vector_statistics(ts, name, stats);
        }
    } else {
        return false;
    }

    if (ts.empty()) { return false; }

    const auto n_legacy = (size_t) std::count_if(ts.begin(), ts.end(), [](const tensor_statistics & t) { return t.legacy; });
    if (n_legacy > 0 && n_legacy < ts.size()) {
        LOG_WRN("%s: %zu of %zu tensors have no activation data, using the legacy layout\n", __func__, n_legacy, ts.size());
    }

    const bool legacy = n_legacy > 0;
    const int nextn_start = nextn_layer_start(ts, g_collector.get_n_layer_nextn());
    compute_tensor_statistics(ts, g_collector.get_mstats(), nextn_start);

    // Sorting logic (Layer index -> Tensor Name)
    struct tensor_comparer {
        bool operator()(const tensor_statistics & a, const tensor_statistics & b) const {
            std::string lay_a;
            std::string lay_b;
            std::string name_a;
            std::string name_b;
            process_tensor_name(a.tensor, lay_a, name_a);
            process_tensor_name(b.tensor, lay_b, name_b);

            // Handle non-numeric layers (e.g., "output")
            int blk_a = INT_MAX - 1;
            int blk_b = INT_MAX - 1;
            try {
                blk_a = std::stoi(lay_a);
            } catch(...) {
                if (a.tensor.find("output") != std::string::npos) { blk_a = INT_MAX; }
            }
            try {
                blk_b = std::stoi(lay_b);
            } catch(...) {
                if (b.tensor.find("output") != std::string::npos) { blk_b = INT_MAX; }
            }

            if (blk_a != blk_b) { return blk_a < blk_b; }
            return name_a < name_b;
        }
    };
    std::sort(ts.begin(), ts.end(), tensor_comparer());

    struct layer_stats {
        float layer_sum = 0.0f;
        int n = 0;
    };
    std::map<int, layer_stats> ls;

    // Shorten names for table formatting
    auto label_fmt = [](std::string s, size_t w) -> std::string {
        if (s.length() <= w) { return s; }
        return ".." + s.substr(s.length() - (w - 2));
    };

    constexpr int w_lay = 6;
    constexpr int w_nam = 40; // Should be wide enough for most tensor names
    const auto * sep = " | ";

    printf("\nComputing tensor statistics for %s (%d tensors)\n", params.in_files[0].c_str(), static_cast<int>(ts.size()));

    if (legacy) {
        printf("\n%*s%s%-*s%s%10s%10s%12s%12s%9s%s%17s%8s%s%10s%10s\n",
            w_lay, "Layer", sep,
            w_nam, "Tensor", sep,
            "Mean", "StdDev", "Skew", "Kurt", "H Norm", sep,
            "∑ E[A²]", "Gain", sep,
            "PCC", "Cov");
        printf("%s\n", std::string(153, '-').c_str());
    } else {
        printf("\n%*s%s%-*s%s%10s%10s%12s%12s%9s%s%17s%8s%s%12s%10s%10s\n",
            w_lay, "Layer", sep,
            w_nam, "Tensor", sep,
            "Mean", "StdDev", "Skew", "Kurt", "H Norm", sep,
            "∑ E[A²]", "Gain", sep,
            "L2 Dist", "PCC", "Cov");
        printf("%s\n", std::string(165, '-').c_str());
    }

    // Tensor Statistics
    for (const auto & tstat : ts) {
        std::string layer;
        std::string name;

        process_tensor_name(tstat.tensor, layer, name);
        const float h_norm = tstat.elements > 1 ? 100.0f * (tstat.entropy / std::log2f((float)tstat.elements)) : fnan;

        int blk;
        try {
            blk = std::stoi(layer);
        } catch (...) {
            if (tstat.tensor.find("output") != std::string::npos) { blk = INT_MAX; }
            else { blk = -1; }
        }

        layer = layer_label(blk, nextn_start);
        if (legacy) {
            printf("%*s%s%-*s%s%10.4f%10.4f%12.4f%12.4f%8.2f%%%s%14.4f%8.2f%s%10.4f%10.4f\n",
                w_lay, layer.c_str(), sep,
                w_nam, label_fmt(tstat.tensor, w_nam).c_str(), sep,
                tstat.mean, tstat.std_deviation, tstat.skewness, tstat.kurtosis, h_norm, sep,
                tstat.sum, tstat.gain, sep,
                tstat.pearson, tstat.covariance
            );
        } else {
            printf("%*s%s%-*s%s%10.4f%10.4f%12.4f%12.4f%8.2f%%%s%14.4f%8.2f%s%12.4f%10.4f%10.4f\n",
                w_lay, layer.c_str(), sep,
                w_nam, label_fmt(tstat.tensor, w_nam).c_str(), sep,
                tstat.mean, tstat.std_deviation, tstat.skewness, tstat.kurtosis, h_norm, sep,
                tstat.sum, tstat.gain, sep,
                tstat.l2_dist, tstat.pearson, tstat.covariance
            );
        }

        // Aggregate Layer Stats
        auto & l = ls[blk];
        l.layer_sum += tstat.sum;
        l.n += tstat.elements;
    }

    // Layer Statistics
    std::map<int, float> layer_cossim;
    std::map<int, float> layer_l2_dist;
    std::map<int, float> layer_pearson;
    std::map<int, float> layer_covariance;
    std::map<int, float> layer_gain;
    compute_layer_statistics(ts, layer_cossim, layer_l2_dist, layer_pearson, layer_covariance, layer_gain);

    size_t layers = 0;
    size_t trunk_layers = 0;
    int min = std::numeric_limits<int>::max();
    int max = -1;

    for (const auto & [layer, stats] : ls) {
        if (layer >= 0 && layer < 9999 && stats.n > 0) {
            layers++;
            if (layer < nextn_start) {
                trunk_layers++;
                min = std::min(layer, min);
                max = std::max(layer, max);
            }
        }
    }

    if (trunk_layers > 0) {
        const auto expected = (size_t)(max - min + 1);
        if (trunk_layers != expected) {
            LOG_WRN("\n%s: layer sequence gap detected (found %zu layers in range %d-%d, expected %zu); layer statistics will not be shown\n",
                __func__, trunk_layers, min, max, expected);
            return false;
        }
    }

    printf("\nComputing layer statistics for %s (%zu layers)\n\n", params.in_files[0].c_str(), layers);

    if (legacy) {
        printf("%*s%s%17s%8s%s%9s%9s%12s\n",
            w_lay, "Layer", sep,
            "∑ E[A²]", "Gain", sep,
            "CosSim", "PCC", "Cov");
        printf("%s\n", std::string(64, '-').c_str());
    } else {
        printf("%*s%s%17s%8s%s%12s%9s%9s%12s\n",
            w_lay, "Layer", sep,
            "∑ E[A²]", "Gain", sep,
            "L2 Dist", "CosSim", "PCC", "Cov");
        printf("%s\n", std::string(76, '-').c_str());
    }

    auto get_layer_stat = [&](const std::map<int, float>& map, const int layer) -> float {
        const auto it = map.find(layer);
        return it != map.end() ? it->second : fnan;
    };

    for (const auto & [layer, stats] : ls) {
        if (layer < 0 || stats.n == 0) { continue; }

        float lgn = layer == 0 || layer == INT_MAX ? fnan : get_layer_stat(layer_gain, layer);
        float ll2 = layer == 0 || layer == INT_MAX ? fnan : get_layer_stat(layer_l2_dist, layer);
        float lcs = layer == 0 || layer == INT_MAX ? fnan : get_layer_stat(layer_cossim, layer);
        float lpc = layer == 0 || layer == INT_MAX ? fnan : get_layer_stat(layer_pearson, layer);
        float lcv = layer == 0 || layer == INT_MAX ? fnan : get_layer_stat(layer_covariance, layer);
        const auto lyr = layer_label(layer, nextn_start);

        if (legacy) {
            printf("%*s%s%14.4f%8.2f%s%9.4f%9.4f%12.4f\n",
                w_lay, lyr.c_str(), sep,
                stats.layer_sum, lgn, sep,
                lcs, lpc, lcv);
        } else {
            printf("%*s%s%14.4f%8.2f%s%12.4f%9.4f%9.4f%12.4f\n",
                w_lay, lyr.c_str(), sep,
                stats.layer_sum, lgn, sep,
                ll2, lcs, lpc, lcv);
        }
    }

    printf("\n");
    return true;
}

// PORT S69-ENGREV-RKNN-IMATRIX-MM-EMBD-CALIB-2026-10-27-01 (idea from rknn-llm's multimodal embedding calibration; code is ours)
// Calibrate on real image-token activations: each image in `dir` is encoded by the mmproj and decoded as an embedding batch
// through the text model; the imatrix eval callback (set on the llama_context) sees those activations like any other decode.
static std::vector<std::string> image_calib_list(const std::string & dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto & e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) {
            continue;
        }
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".gif" || ext == ".tga") {
            out.push_back(e.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static bool compute_imatrix_images(llama_context * ctx, const common_params & params) {
    if (params.mmproj.path.empty()) {
        LOG_ERR("%s: --image-calib needs --mmproj\n", __func__);
        return false;
    }
    const std::vector<std::string> files = image_calib_list(params.image_calib_dir);
    if (files.empty()) {
        LOG_ERR("%s: no images (jpg/png/bmp/gif/tga) in '%s'\n", __func__, params.image_calib_dir.c_str());
        return false;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu         = params.mmproj_use_gpu;
    mparams.device          = params.mmproj_device;
    mparams.print_timings   = false;
    mparams.n_threads       = params.cpuparams.n_threads;
    mparams.flash_attn_type = params.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    mparams.warmup          = false;
    mparams.image_min_tokens = params.image_min_tokens;
    mparams.image_max_tokens = params.image_max_tokens;
    mtmd::context_ptr mctx(mtmd_init_from_file(params.mmproj.path.c_str(), llama_get_model(ctx), mparams));
    if (!mctx) {
        LOG_ERR("%s: cannot load the projector '%s'\n", __func__, params.mmproj.path.c_str());
        return false;
    }

    const std::string marker = mtmd_default_marker();
    const mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    int n_ok = 0;
    for (const auto & f : files) {
        auto wrapped = mtmd_helper_bitmap_init_from_file(mctx.get(), f.c_str(), false, init_opt);
        if (!wrapped.bitmap) {
            LOG_WRN("%s: skipping unreadable image '%s'\n", __func__, f.c_str());
            continue;
        }
        mtmd::bitmap bmp(wrapped.bitmap);
        const mtmd_bitmap * bmps[1] = { bmp.ptr.get() };
        mtmd_input_text text = { marker.c_str(), marker.size(), true, true };
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        if (mtmd_tokenize(mctx.get(), chunks.ptr.get(), &text, bmps, 1) != 0) {
            LOG_WRN("%s: cannot tokenize image '%s'\n", __func__, f.c_str());
            continue;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_pos new_n_past = 0;
        if (mtmd_helper_eval_chunks(mctx.get(), ctx, chunks.ptr.get(), 0, 0, params.n_batch, false, &new_n_past) != 0) {
            LOG_ERR("%s: decode of '%s' failed (does --ctx-size / --ubatch-size hold one image?)\n", __func__, f.c_str());
            return false;
        }
        LOG_INF("%s: image %d/%zu '%s': %d positions\n", __func__, n_ok + 1, files.size(), f.c_str(), (int) new_n_past);
        n_ok++;
    }
    if (n_ok == 0) {
        LOG_ERR("%s: no image could be decoded\n", __func__);
        return false;
    }
    LOG_INF("%s: calibrated on %d image(s)\n", __func__, n_ok);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    params.out_file = "imatrix.gguf";

    params.n_ctx = 512;
    params.escape = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    // --model-draft is read by two features: a NextN-only draft GGUF (upstream: NextN calibration) and a paired assistant draft
    // (fork: Gemma-4 MTP assistant fed the target's hidden state). The file decides which.
    const bool use_draft = [&]() {
        const std::string & md = params.speculative.dft_path();
        if (md.empty()) {
            return false;
        }
        const model_file_shape shape_md = model_read_file_shape(md);
        return shape_md.valid && shape_md.n_nextn_layers > 0 && !shape_md.has_trunk;
    }();
    if (use_draft) {
        params.load_mtp = true;
        params.speculative.n_max = 0; // keep trunk's context identical to a plain run
    }

    // set_params before show_statistics so load_imatrix has valid n_ctx/n_parallel
    g_collector.set_params(params);

    if (params.show_statistics) {
        if (!show_statistics(params)) {
            return 1;
        }
        return 0;
    }

    const int32_t n_ctx = params.n_ctx;

    if (n_ctx <= 0) {
        LOG_ERR("%s: imatrix tool requires '--ctx-size' > 0\n", __func__);
        return 1;
    }

    // PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01
    // The paired-draft eval path (compute_draft_imatrix_batch, see compute_imatrix below) only
    // handles a single sequence per decode; force n_seq==1 (below: n_seq = max(1, n_batch/n_ctx))
    // whenever a draft model is paired, rather than generalizing the draft path to n_seq>1.
    if (params.speculative.has_dft() && params.n_batch > n_ctx) {
        LOG_WRN("%s: paired draft imatrix requires n_seq==1; clamping --batch-size from %d to "
                "--ctx-size %d\n", __func__, params.n_batch, n_ctx);
        params.n_batch = n_ctx;
    }

    {
        const int32_t n_seq = std::max(1, params.n_batch / n_ctx);
        const int32_t n_kv = n_seq * n_ctx;

        if (params.load_mtp && n_seq > 1) {
            LOG_ERR("%s: '--nextn' and '--model-draft' need a single sequence per batch, set '--batch-size' to at most '--ctx-size' (%d)\n", __func__, n_ctx);

            return 1;
        }

        params.n_parallel = n_seq;
        params.n_ctx      = n_kv;

        params.n_batch = std::min(params.n_batch, n_kv);
    }

    g_collector.set_params(params);

    for (const auto & in_file : params.in_files) {
        LOG_INF("%s : loading imatrix from '%s'\n", __func__, in_file.c_str());
        if (!g_collector.load_imatrix(in_file.c_str())) {
            LOG_ERR("%s : failed to load %s\n", __func__, in_file.c_str());
            return 1;
        }
    }

    if (params.imat_hessian_block > 0) {
        if (params.prompt.empty()) {
            LOG_ERR("%s: --hessian-block needs calibration text (-f): the Hessian sidecar is accumulated from activations, "
                    "it cannot be combined from --in-file imatrix files\n", __func__);
            return 1;
        }
        if (!params.in_files.empty()) {
            LOG_WRN("%s: --hessian-block: the sidecar covers only the tokens of THIS run; the %zu --in-file imatrix file(s) "
                    "carry no Hessian and are not in it (its counts say how many tokens it holds)\n", __func__, params.in_files.size());
        }
        LOG_INF("%s: --hessian-block %d: block-diagonal X^T X sidecar -> '%s' (budget %d MiB)\n", __func__,
                params.imat_hessian_block,
                (params.imat_hessian_out.empty() ? imatrix_hessian_default_path(params.out_file) : params.imat_hessian_out).c_str(),
                params.imat_hessian_max_mib);
    }

    if (params.prompt.empty() && params.image_calib_dir.empty()) {
        LOG_INF("No prompt provided; combining precomputed matrices only.\n");

        if (params.in_files.empty()) {
            LOG_ERR("Error: No prompt provided and no precomputed matrices (--in-file) to combine.\n");
            return 1;
        }

        if (params.in_files.size() == 1) {
            LOG_INF("%s : saving imatrix to '%s'\n", __func__, params.out_file.c_str());
        } else if (params.in_files.size() > 1) {
            LOG_INF("%s : saving combined imatrix to '%s'\n", __func__, params.out_file.c_str());
        }

        g_collector.save_imatrix();

        return 0;
    }

    {
        const model_file_shape shape = model_read_file_shape(params.model.path);
        if (use_draft && shape.n_nextn_layers > 0) {
            LOG_ERR("%s: model already includes NextN layers\n", __func__);
            return 1;
        }
        if (use_draft && shape.has_nextn) {
            LOG_ERR("%s: the model already includes NextN layers (NextN tensor '%s')\n", __func__, shape.first_nextn.c_str());
            return 1;
        }
        if (!use_draft && !shape.has_trunk && (shape.n_nextn_layers > 0 || shape.has_nextn)) {
            LOG_ERR("%s: '%s' is a NextN draft, use with --model-draft/-md\n", __func__, params.model.path.c_str());
            return 1;
        }
        if (use_draft) {
            const std::string & path_md = params.speculative.dft_path();
            const model_file_shape shape_md = model_read_file_shape(path_md);
            if (!shape_md.valid || shape_md.is_split) {
                LOG_ERR("%s: '%s' is not a readable single-shard GGUF\n", __func__, path_md.c_str());
                return 1;
            }
            if (shape.valid && shape_md.arch != shape.arch) {
                LOG_ERR("%s: the draft architecture '%s' does not match the target architecture '%s'\n", __func__, shape_md.arch.c_str(), shape.arch.c_str());
                return 1;
            }
            if (shape_md.n_nextn_layers == 0 || shape_md.has_trunk) {
                LOG_ERR("%s: '%s' is not a draft (no NextN layers or trunk tensors present)\n", __func__, path_md.c_str());
                return 1;
            }
            if (shape_md.n_nextn_layers > 1) {
                LOG_ERR("%s: multi-layer NextN drafts are not supported (found %u)\n", __func__, shape_md.n_nextn_layers);
                return 1;
            }
            if (shape_md.n_trunk() == 0) {
                LOG_ERR("%s: NextN layers sharing the trunk's KV cache are not supported\n", __func__);
                return 1;
            }
            if (shape.valid && shape_md.nextn_idx.count(shape.n_trunk()) == 0) {
                LOG_ERR("%s: no NextN tensors at layer index %u in '%s' found\n", __func__, shape.n_trunk(), path_md.c_str());
                return 1;
            }
            if (shape.valid && shape_md.n_embd_out != shape.n_embd_out) {
                LOG_ERR("%s: the draft output width %u does not match the target's %u\n", __func__, shape_md.n_embd_out, shape.n_embd_out);
                return 1;
            }
            // the fork resolves the speculative stages (common_params_speculative::get_resolved_stages); upstream's list was
            // `types` with draft-mtp spelled COMMON_SPECULATIVE_TYPE_DRAFT_MTP (the fork's COMMON_SPECULATIVE_TYPE_MTP)
            for (const auto & stage : params.speculative.get_resolved_stages()) {
                const auto type = stage.type;
                if (type != COMMON_SPECULATIVE_TYPE_NONE && type != COMMON_SPECULATIVE_TYPE_MTP) {
                    LOG_ERR("%s: speculative type '%s' is not supported with --model-draft/-md (only 'draft-mtp')\n", __func__, common_speculative_type_to_str(type).c_str());
                    return 1;
                }
            }
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ik_collect_imatrix;
    params.cb_eval_user_data = NULL;
    params.warmup = false;

    // PORT S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01
    // SOURCE-READ: examples/imatrix/imatrix.cpp:1148-1224 (paired Gemma4-MTP-assistant draft
    // imatrix). Adapted to the CURRENT dual-model MTP load path: the fork's own
    // ik_load_model_from_params()/ik_init_from_loaded_model() helpers no longer exist in this
    // tree; common_speculative_load_draft_model() + common_speculative_prepare_mtp_runtime()
    // (common/speculative.cpp:3081-3201) are the harness's current equivalent -- reused here
    // rather than hand-rolling a parallel loader (HARNESS-PROVIDES-EQUIVALENT).
    const bool  has_draft_model = params.speculative.has_dft() && !use_draft; // a NextN-only draft is upstream's path
    std::string draft_out_file;

    if (has_draft_model) {
        params.speculative.type = COMMON_SPECULATIVE_TYPE_MTP;
        params.has_mtp = true;

        LOG_INF("%s : loading paired draft model '%s'\n", __func__, params.speculative.dft_path().c_str());
        if (!common_speculative_load_draft_model(params.speculative, params)) {
            LOG_ERR("%s : failed to load draft model '%s'\n", __func__, params.speculative.dft_path().c_str());
            return 1;
        }

        if (!llama_model_is_gemma4_mtp_assistant(params.speculative.model_dft)) {
            LOG_ERR("%s : paired imatrix mode currently supports Gemma 4 assistant draft models only\n", __func__);
            llama_model_free(params.speculative.model_dft);
            return 1;
        }

        draft_out_file = params.out_file_draft.empty()
                             ? default_draft_imatrix_out_file(params.out_file)
                             : params.out_file_draft;
        LOG_INF("%s : paired imatrix outputs: target='%s', draft='%s'\n",
                __func__, params.out_file.c_str(), draft_out_file.c_str());
    }

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        if (has_draft_model && params.speculative.model_dft != nullptr) {
            llama_model_free(params.speculative.model_dft);
        }
        return 1;
    }

    llama_context * ctx_dft = nullptr;

    if (has_draft_model) {
        if (std::string(llama_model_arch_string(model)) != "gemma4") {
            LOG_ERR("%s : paired imatrix mode currently supports Gemma 4 target models only\n", __func__);
            llama_model_free(params.speculative.model_dft);
            return 1;
        }

        if (!common_speculative_prepare_mtp_runtime(params.speculative, params, model, /*has_external_mtp=*/true)) {
            LOG_ERR("%s : failed to prepare paired MTP draft runtime\n", __func__);
            llama_model_free(params.speculative.model_dft);
            return 1;
        }

        ctx_dft = llama_init_from_model(params.speculative.model_dft, params.speculative.cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s : failed to create context for draft model\n", __func__);
            llama_model_free(params.speculative.model_dft);
            return 1;
        }

        llama_set_mtp_target_context(ctx_dft, ctx);

        common_params draft_collector_params = params;
        draft_collector_params.out_file      = draft_out_file;
        // the draft's Hessian sidecar is named after the draft imatrix, never the target's --hessian-out
        draft_collector_params.imat_hessian_out = "";
        g_draft_collector.set_params(draft_collector_params);
    }

    const int n_ctx_train = llama_model_n_ctx_train(model);
    if (params.n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    llama_model_ptr model_draft;
    if (use_draft) {
        auto mparams_sidecar = common_model_params_to_llama(params);
        mparams_sidecar.load_mtp = true;
        model_draft.reset(llama_model_load_from_file(params.speculative.dft_path().c_str(), mparams_sidecar));
        if (model_draft == nullptr) {
            LOG_ERR("%s: failed to load the draft model '%s'\n", __func__, params.speculative.dft_path().c_str());

            return 1;
        }
        if (!common_speculative_are_compatible(model, model_draft.get())) {
            LOG_ERR("%s: the target and draft vocab are not compatible\n", __func__);

            return 1;
        }
    }

    std::unique_ptr<nextn_collector> nextn;

    if (params.load_mtp) {
        llama_model * model_src = use_draft ? model_draft.get() : model;
        const std::string & path_src = use_draft ? params.speculative.dft_path() : params.model.path;
        const char * flag_src = use_draft ? "'-md'" : "'--nextn'";

        const int32_t n_heads = llama_model_n_layer_nextn(model_src);
        const int32_t n_trunk_src = llama_model_n_layer(model_src);
        const int32_t n_trunk = llama_model_n_layer(model);
        const nextn_model_info info = n_heads > 0 ? nextn_read_model_info(path_src, n_trunk, n_heads) : nextn_model_info();

        if (n_heads == 0) {
            LOG_WRN("%s: the model has no NextN layers, %s has no effect\n", __func__, flag_src);
        } else if (n_trunk_src == 0) {
            LOG_ERR("%s: NextN layers sharing the trunk's KV cache are not supported\n", __func__);
            return 1;
        } else if (n_heads > 1) {
            LOG_ERR("%s: multi-layer NextN drafts are not supported (found %u)\n", __func__, n_heads);
            return 1;
        } else if (!info.has_layers) {
            LOG_WRN("%s: no NextN tensor in '%s', %s has no effect\n", __func__, path_src.c_str(), flag_src);
        } else {
            nextn = nextn_collector_init(model_src, params, info.own_lm_head[0]);
            if (nextn == nullptr) { return 1; }
            llama_set_embeddings_nextn(ctx, true, /*masked*/ false);
            g_collector.set_n_layer_nextn(n_heads);

            LOG_INF("%s: processing %d NextN layer(s) from block %d\n", __func__, n_heads, n_trunk);
        }
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    const bool split_on = !params.ppl_split_on.empty();
    if (split_on && ctx_dft != nullptr) {
        LOG_ERR("%s: --split-on cannot be combined with a paired draft model\n", __func__);
        llama_free(ctx_dft);
        llama_model_free(params.speculative.model_dft);
        return 1;
    }
    if (split_on && nextn) {
        LOG_ERR("%s: --split-on cannot be combined with NextN collection (--nextn / a NextN draft)\n", __func__);
        return 1;
    }
    if (!split_on && params.ppl_verbose_chunks) {
        LOG_WRN("%s: --verbose-chunks has no effect without --split-on\n", __func__);
    }
    bool calib_ok = true;
    if (!params.prompt.empty()) {
        calib_ok = split_on ? compute_imatrix_split_on(ctx, params, n_ctx) : compute_imatrix(ctx, params, n_ctx, nextn.get(), ctx_dft);
    }
    if (calib_ok && !params.image_calib_dir.empty()) {
        calib_ok = compute_imatrix_images(ctx, params);
    }
    if (!calib_ok) {
        if (ctx_dft != nullptr) {
            llama_free(ctx_dft);
            llama_model_free(params.speculative.model_dft);
        }
        return 1;
    }

    g_collector.save_imatrix();
    g_collector.print_layer_importance();

    if (ctx_dft != nullptr) {
        g_draft_collector.save_imatrix();
        g_draft_collector.print_layer_importance();
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    if (ctx_dft != nullptr) {
        llama_free(ctx_dft);
        llama_model_free(params.speculative.model_dft);
    }

    llama_backend_free();

    return 0;
}
