// bench-tier-restore: what does it cost to bring a spilled prompt state back from the NVMe tier, compared with recomputing it?
//
// Row S69-TIERED-RESTORE-VS-RECOMPUTE-POLICY-2026-10-05-01 (and the I/O half of S69-TIERED-WINDOWS-STORAGE-IO-PATH-2026-10-05-01).
// It loads a real model, and for each prompt length N:
//   recompute  = decode N tokens from an empty memory                                   (what a cache miss costs)
//   serialize  = llama_state_seq_get_data_ext                                            (paid at spill time)
//   put        = common_disk_tier::put (write + checksum + size verify), then the file is flushed to the device
//   warm get   = common_disk_tier::get right after the write: file cache is hot, so this is an UPPER bound on speed
//   cold read  = the same file read with the OS cache bypassed (Windows: FILE_FLAG_NO_BUFFERING after FlushFileBuffers;
//                elsewhere: O_DIRECT where available), so the bytes really come from the device
//   checksum   = common_disk_tier::checksum over the payload (the verification get() performs)
//   deserialize= llama_state_seq_set_data_ext into an emptied memory
//   restore    = cold read + checksum + deserialize   (the number that is compared with recompute)
// and checks that the restored state is the right one: the logits of the next token decoded from the restored state must equal
// the logits decoded from the state that was recomputed (max abs difference is printed; the run fails when it is not 0).
// With --cold-file it also reads a stretch of an old, big file through the ordinary buffered path, which is the control for the
// "was the earlier 2184 MB/s a page-cache number" question.
//
// USAGE: bench-tier-restore <model.gguf> <tier dir> <N,N,N,...> [threads] [reps] [cold-file]
// Output: one "ROW ..." line per (N, kind) with every time in milliseconds, and "SUMMARY ..." lines. CPU only: n_gpu_layers = 0.
//
// GATE-CLAIM-OK: a measurement tool; it declares no resource unavailable.

#include "common.h"
#include "disk-tier.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX // <windows.h> min/max macros break std::min( / std::max( below
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static std::vector<int> parse_list(const std::string & s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            v.push_back(std::atoi(item.c_str()));
        }
    }
    return v;
}

// make sure the bytes of a freshly written file are on the device, so that a read that bypasses the cache sees them
static bool flush_file_to_device(const std::string & path) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    const bool ok = FlushFileBuffers(h) != 0;
    CloseHandle(h);
    return ok;
#else
    const int fd = open(path.c_str(), O_RDWR);
    if (fd < 0) {
        return false;
    }
    const bool ok = fsync(fd) == 0;
    close(fd);
    return ok;
#endif
}

// read the whole file bypassing the OS file cache; returns the bytes read (0 on failure) and the time in ms
static size_t read_uncached(const std::string & path, std::vector<uint8_t> & out, double & ms) {
    const auto t0 = clk::now();
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return 0;
    }
    LARGE_INTEGER sz;
    GetFileSizeEx(h, &sz);
    const size_t   file_size = (size_t) sz.QuadPart;
    const size_t   sector    = 4096;
    const size_t   chunk     = 4u << 20;
    const size_t   padded    = (file_size + sector - 1) / sector * sector;
    uint8_t *      buf       = (uint8_t *) VirtualAlloc(nullptr, padded + chunk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buf) {
        CloseHandle(h);
        return 0;
    }
    size_t got = 0;
    while (got < file_size) {
        const size_t want = std::min(chunk, padded - got);
        DWORD n = 0;
        if (!ReadFile(h, buf + got, (DWORD) want, &n, nullptr) || n == 0) {
            break;
        }
        got += n;
    }
    got = std::min(got, file_size);
    CloseHandle(h);
    ms = ms_since(t0);            // the device read only; the copy below into a vector is not part of it
    out.assign(buf, buf + got);
    VirtualFree(buf, 0, MEM_RELEASE);
    return got;
#else
    int fd = open(path.c_str(), O_RDONLY
#  ifdef O_DIRECT
                  | O_DIRECT
#  endif
    );
    if (fd < 0) {
        return 0;
    }
    void * buf = nullptr;
    const size_t file_size = fs::file_size(path);
    const size_t padded    = (file_size + 4095) / 4096 * 4096;
    if (posix_memalign(&buf, 4096, padded + 4096) != 0) {
        close(fd);
        return 0;
    }
    size_t got = 0;
    while (got < file_size) {
        const ssize_t n = read(fd, (uint8_t *) buf + got, std::min<size_t>(4u << 20, padded - got));
        if (n <= 0) {
            break;
        }
        got += (size_t) n;
    }
    got = std::min(got, file_size);
    close(fd);
    ms = ms_since(t0);
    out.assign((uint8_t *) buf, (uint8_t *) buf + got);
    free(buf);
    return got;
#endif
}

static bool decode_chunked(llama_context * ctx, const std::vector<llama_token> & toks, int n, int n_batch) {
    for (int off = 0; off < n; off += n_batch) {
        const int cnt = std::min(n_batch, n - off);
        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data()) + off, cnt);
        if (llama_decode(ctx, b) != 0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tier dir> <N,N,...> [threads] [reps] [cold-file]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string tier_dir   = argv[2];
    const auto        n_list     = parse_list(argv[3]);
    const int         n_threads  = argc > 4 ? std::atoi(argv[4]) : 8;
    const int         reps       = argc > 5 ? std::max(1, std::atoi(argv[5])) : 2;
    const std::string cold_file  = argc > 6 ? argv[6] : "";
    if (n_list.empty()) {
        std::fprintf(stderr, "empty N list\n");
        return 2;
    }
    const int n_max = *std::max_element(n_list.begin(), n_list.end());

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;   // CPU only
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "could not load %s\n", model_path.c_str());
        return 2;
    }
    auto cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) (n_max + 64);
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "could not create a context for n_ctx = %u\n", cparams.n_ctx);
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::printf("MODEL %s recurrent=%d hybrid=%d n_swa=%d n_layer=%d n_vocab=%d threads=%d\n", model_path.c_str(),
                (int) llama_model_is_recurrent(model), (int) llama_model_is_hybrid(model), llama_model_n_swa(model),
                llama_model_n_layer(model), n_vocab, n_threads);

    std::vector<llama_token> toks(n_max + 1);
    for (int i = 0; i <= n_max; i++) {
        toks[i] = (llama_token) (100 + (int64_t) i * 7919 % std::max(1, n_vocab - 200));
    }
    const llama_token next_tok = toks[n_max];

    fs::create_directories(tier_dir);
    common_disk_tier tier(tier_dir, 64ull << 30, 0);
    int n_stale = 0;
    std::string err;
    if (!tier.init(n_stale, err)) {
        std::fprintf(stderr, "tier dir: %s\n", err.c_str());
        return 2;
    }

    // warm-up: the first decode pays for lazy page-in of the weights and thread start-up; none of that is a property of the state
    llama_memory_clear(llama_get_memory(ctx), true);
    decode_chunked(ctx, toks, std::min(128, n_max), cparams.n_batch);

    bool all_exact = true;
    size_t cold_off = 0;
    for (int N : n_list) {
        for (int kind = 0; kind < 2; kind++) {
            const unsigned flags = kind == 0 ? LLAMA_STATE_SEQ_FLAGS_NONE : LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
            const char * kname   = kind == 0 ? "entry" : "ckpt";

            // recompute: a miss costs this (median of `reps`)
            std::vector<double> rc;
            for (int r = 0; r < reps; r++) {
                llama_memory_clear(llama_get_memory(ctx), true);
                const auto t0 = clk::now();
                if (!decode_chunked(ctx, toks, N, cparams.n_batch)) {
                    std::fprintf(stderr, "decode failed at N = %d\n", N);
                    return 3;
                }
                rc.push_back(ms_since(t0));
            }
            std::sort(rc.begin(), rc.end());
            const double t_recompute = rc[rc.size() / 2];

            // the state exists now (memory holds exactly N tokens)
            const size_t sz = llama_state_seq_get_size_ext(ctx, 0, flags);
            if (sz == 0) {
                std::printf("ROW kind=%s N=%d state_bytes=0 (nothing to spill for this kind)\n", kname, N);
                continue;
            }
            std::vector<uint8_t> data(sz);
            const auto ts = clk::now();
            const size_t got = llama_state_seq_get_data_ext(ctx, data.data(), sz, 0, flags);
            const double t_serialize = ms_since(ts);
            if (got != sz) {
                std::fprintf(stderr, "serialize returned %zu of %zu\n", got, sz);
                return 3;
            }

            // reference logits of the next token, decoded from the recomputed state (kind 0 only: a partial state is not a full one)
            std::vector<float> ref_logits;
            if (kind == 0) {
                llama_batch b = llama_batch_get_one(const_cast<llama_token *>(&next_tok), 1);
                if (llama_decode(ctx, b) != 0) {
                    return 3;
                }
                const float * lg = llama_get_logits_ith(ctx, -1);
                ref_logits.assign(lg, lg + n_vocab);
            }

            // put: write + checksum + size verification, then make sure the device has the bytes
            const auto tp = clk::now();
            const uint64_t id = tier.put(data, {}, N, kind == 1);
            const double t_put = ms_since(tp);
            if (id == 0) {
                std::fprintf(stderr, "tier refused the blob (%zu bytes)\n", sz);
                return 3;
            }
            const std::string path = tier.path_of(id);
            const auto tf = clk::now();
            flush_file_to_device(path);
            const double t_flush = ms_since(tf);
            const size_t file_bytes = (size_t) fs::file_size(path);

            // warm get (page cache hot)
            std::vector<uint8_t> ra, rb;
            const auto tw = clk::now();
            const bool ok_warm = tier.get(id, N, ra, rb);
            const double t_get_warm = ms_since(tw);
            if (!ok_warm || ra != data) {
                std::fprintf(stderr, "warm get did not return the bytes that were put\n");
                return 3;
            }

            // cold read: bypass the cache. Median of 3 reads of the same file (the device's own cache may still help; that is the device).
            std::vector<double> cr;
            std::vector<uint8_t> raw;
            for (int r = 0; r < 3; r++) {
                double t = 0;
                const size_t n = read_uncached(path, raw, t);
                if (n != file_bytes) {
                    std::fprintf(stderr, "uncached read returned %zu of %zu bytes\n", n, file_bytes);
                    return 3;
                }
                cr.push_back(t);
            }
            std::sort(cr.begin(), cr.end());
            const double t_cold_read = cr[1];

            const auto tc = clk::now();
            volatile uint64_t cs = common_disk_tier::checksum(data.data(), data.size(), 1);
            (void) cs;
            const double t_checksum = ms_since(tc);

            // buffered read of a stretch of an old file nobody has read lately (control for the page-cache question)
            double t_cold_file = -1;
            if (!cold_file.empty() && fs::exists(cold_file) && fs::file_size(cold_file) > cold_off + file_bytes) {
                std::vector<uint8_t> buf(file_bytes);
                std::ifstream f(cold_file, std::ios::binary);
                f.seekg((std::streamoff) cold_off);
                const auto t0 = clk::now();
                f.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) buf.size());
                t_cold_file = ms_since(t0);
                cold_off += file_bytes + (64u << 20);   // never re-read a stretch
            }

            // deserialize into an emptied memory, and prove the restored state is the right one
            double t_deser = 0, max_diff = -1;
            if (kind == 0) {
                llama_memory_clear(llama_get_memory(ctx), true);
                const auto td = clk::now();
                const size_t put_back = llama_state_seq_set_data_ext(ctx, data.data(), sz, 0, flags);
                t_deser = ms_since(td);
                if (put_back != sz) {
                    std::fprintf(stderr, "deserialize consumed %zu of %zu\n", put_back, sz);
                    return 3;
                }
                llama_batch b = llama_batch_get_one(const_cast<llama_token *>(&next_tok), 1);
                if (llama_decode(ctx, b) != 0) {
                    return 3;
                }
                const float * lg = llama_get_logits_ith(ctx, -1);
                max_diff = 0;
                for (int i = 0; i < n_vocab; i++) {
                    max_diff = std::max(max_diff, (double) std::fabs(lg[i] - ref_logits[i]));
                }
                if (max_diff != 0.0) {
                    all_exact = false;
                }
            } else {
                // a checkpoint is applied on top of a live sequence; time the set only (no logits comparison)
                const auto td = clk::now();
                llama_state_seq_set_data_ext(ctx, data.data(), sz, 0, flags);
                t_deser = ms_since(td);
            }

            tier.erase(id);

            const double t_restore_cold = t_cold_read + t_checksum + t_deser;
            const double t_restore_warm = t_get_warm + t_deser;
            std::printf("ROW kind=%s N=%d state_bytes=%zu file_bytes=%zu recompute_ms=%.2f serialize_ms=%.2f put_ms=%.2f flush_ms=%.2f "
                        "get_warm_ms=%.2f cold_read_ms=%.2f checksum_ms=%.2f deserialize_ms=%.2f restore_warm_ms=%.2f restore_cold_ms=%.2f "
                        "cold_MBps=%.0f warm_MBps=%.0f cold_file_buffered_ms=%.2f cold_file_MBps=%.0f speedup_cold=%.2f logits_max_abs_diff=%.3g\n",
                        kname, N, sz, file_bytes, t_recompute, t_serialize, t_put, t_flush, t_get_warm, t_cold_read, t_checksum, t_deser,
                        t_restore_warm, t_restore_cold, file_bytes / 1e6 / (t_cold_read / 1e3), file_bytes / 1e6 / (t_get_warm / 1e3),
                        t_cold_file, t_cold_file > 0 ? file_bytes / 1e6 / (t_cold_file / 1e3) : -1.0, t_recompute / t_restore_cold, max_diff);
            std::fflush(stdout);
        }
    }

    std::printf("SUMMARY logits_exact=%d\n", all_exact ? 1 : 0);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return all_exact ? 0 : 1;
}
