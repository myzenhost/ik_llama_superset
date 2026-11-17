// bench-tier-io: is the tier's buffered file I/O good enough, or is an unbuffered / overlapped / IoRing path worth having?
//
// Row S69-TIERED-WINDOWS-STORAGE-IO-PATH-2026-10-05-01. common_disk_tier::put/get use std::ofstream/std::ifstream (buffered I/O
// through the OS file cache, one big call each). This tool measures, on the machine it runs on, what other ways of moving the same
// bytes would give, so the decision is a measurement and not a belief about the OS.
//
//   READ (from <cold-file>, a big file nobody has read lately; every method reads a region no earlier method touched):
//     ifstream_1call      one read() of the whole region           <- what common_disk_tier::get does
//     ifstream_8MiB       the same region in 8 MiB read() calls
//     nobuf_seq_{1,4,16}M ReadFile with FILE_FLAG_NO_BUFFERING, one request in flight
//     overlapped_qd{2,4,8}  FILE_FLAG_NO_BUFFERING|FILE_FLAG_OVERLAPPED, 4 MiB requests, queue depth qd
//     ioring_qd{4,8,16}   Windows I/O ring (BuildIoRingReadFile) on a NO_BUFFERING|OVERLAPPED handle, 4 MiB requests
//     ifstream_reread     the last region read again (page-cache speed; what a "warm" read looks like)
//   WRITE (to <work-dir>):
//     ofstream_flush      what put() does: write + flush() (data may sit in the OS cache)
//     ofstream_fsync      write + flush() + FlushFileBuffers (the bytes are on the device when it returns)
//     nobuf_writethru_4M  WriteFile with NO_BUFFERING|WRITE_THROUGH, 4 MiB sequential
//     overlapped_wr_qd4   NO_BUFFERING|WRITE_THROUGH|OVERLAPPED, 4 MiB requests, queue depth 4
//
// USAGE: bench-tier-io <cold-file> <work-dir> [region MiB = 128] [rounds = 2]
// Output: "RESULT <phase> <method> round=<r> MiB=<n> ms=<t> MBps=<v> ok=<0|1>" and "MEDIAN <phase> <method> MBps=<v>".
// ok=1 means the bytes read were byte-identical to a reference read of the same region (checked through a hash after the
// timed part), so a method that returns fast garbage cannot win.
//
// GATE-CLAIM-OK: a measurement tool; it declares no resource unavailable.
// COMPATIBILITY-INVESTIGATED: the Windows-specific methods are separate code paths behind _WIN32; on other systems the tool runs the
//   ifstream methods only, and the I/O ring entry points are looked up at run time (GetProcAddress) so the binary also starts on a
//   Windows that does not export them (it then reports that method as FAILED with the reason instead of crashing).
// WEB-RESEARCH-DONE: learn.microsoft.com ioringapi.h reference (captures in tmp/research/tiered1005/win_*.txt).

#include "disk-tier.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX // <windows.h> min/max macros break std::min( / std::max( below
#  endif
#  include <windows.h>
// ioringapi.h needs NTDDI_VERSION >= NTDDI_WIN10_CO, and raising that globally breaks winuser.h in this MinGW; the handful of
// types the read path uses are declared here with the layout of ioringapi.h / ntioring_x.h (read from the mingw-w64 copy), and
// the entry points are looked up at run time, so this file builds without the SDK declarations.
namespace ior {
    typedef HANDLE HIORING_T;
    struct CreateFlags { UINT32 Required; UINT32 Advisory; };
    struct HandleRef   { int Kind; union { HANDLE Handle; UINT32 Index; } u; };          // Kind 0 = raw handle
    struct BufferRef   { int Kind; union { void * Address; struct { UINT32 Index; UINT32 Offset; } io; } u; };   // Kind 0 = raw pointer
    struct Cqe         { UINT_PTR UserData; HRESULT ResultCode; ULONG_PTR Information; };
    constexpr int VERSION_1 = 1;
    typedef HRESULT (WINAPI * create_t)(int version, CreateFlags flags, UINT32 sq, UINT32 cq, HIORING_T * h);
    typedef HRESULT (WINAPI * build_read_t)(HIORING_T, HandleRef, BufferRef, UINT32 n, UINT64 off, UINT_PTR user, int sqe_flags);
    typedef HRESULT (WINAPI * submit_t)(HIORING_T, UINT32 wait_ops, UINT32 ms, UINT32 * submitted);
    typedef HRESULT (WINAPI * pop_t)(HIORING_T, Cqe *);
    typedef HRESULT (WINAPI * close_t)(HIORING_T);
}
#endif

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

static std::map<std::string, std::vector<double>> g_results;

static void report(const char * phase, const char * method, int round, size_t bytes, double ms, bool ok) {
    const double mb = bytes / (1024.0 * 1024.0);
    const double v  = mb * 1.048576 / (ms / 1000.0);   // MB/s (10^6 bytes), like drive datasheets
    std::printf("RESULT %s %s round=%d MiB=%.0f ms=%.1f MBps=%.0f ok=%d\n", phase, method, round, mb, ms, v, ok ? 1 : 0);
    std::fflush(stdout);
    if (ok) {
        g_results[std::string(phase) + " " + method].push_back(v);
    }
}

static uint64_t hash_of(const uint8_t * p, size_t n) { return common_disk_tier::checksum(p, n, 99); }

#ifdef _WIN32
static uint8_t * aligned_alloc_win(size_t n) { return (uint8_t *) VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); }

static bool nobuf_read_seq(const std::string & path, size_t off, size_t len, size_t chunk, uint8_t * buf) {
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    LARGE_INTEGER pos; pos.QuadPart = (LONGLONG) off;
    SetFilePointerEx(h, pos, nullptr, FILE_BEGIN);
    size_t got = 0;
    while (got < len) {
        DWORD n = 0;
        if (!ReadFile(h, buf + got, (DWORD) std::min(chunk, len - got), &n, nullptr) || n == 0) {
            break;
        }
        got += n;
    }
    CloseHandle(h);
    return got == len;
}

static bool overlapped_read(const std::string & path, size_t off, size_t len, size_t chunk, int qd, uint8_t * buf) {
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    struct slot { OVERLAPPED ov; size_t at; size_t n; bool busy; };
    std::vector<slot> s(qd);
    for (auto & x : s) { std::memset(&x.ov, 0, sizeof(x.ov)); x.ov.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr); x.busy = false; }
    size_t next = 0, done = 0;
    bool ok = true;
    auto issue = [&](slot & x) {
        if (next >= len) { return; }
        x.at = next;
        x.n  = std::min(chunk, len - next);
        next += x.n;
        const uint64_t o = off + x.at;
        x.ov.Offset = (DWORD) (o & 0xffffffffu); x.ov.OffsetHigh = (DWORD) (o >> 32);
        ResetEvent(x.ov.hEvent);
        if (!ReadFile(h, buf + x.at, (DWORD) x.n, nullptr, &x.ov) && GetLastError() != ERROR_IO_PENDING) { ok = false; return; }
        x.busy = true;
    };
    for (auto & x : s) { issue(x); }
    while (ok && done < len) {
        bool any = false;
        for (auto & x : s) {
            if (!x.busy) { continue; }
            any = true;
            DWORD n = 0;
            if (!GetOverlappedResult(h, &x.ov, &n, TRUE) || n < x.n) { ok = false; break; }
            done += x.n;
            x.busy = false;
            issue(x);
        }
        if (!any) { break; }
    }
    CancelIo(h);
    for (auto & x : s) { CloseHandle(x.ov.hEvent); }
    CloseHandle(h);
    return ok && done == len;
}

static bool ioring_read(const std::string & path, size_t off, size_t len, size_t chunk, int qd, uint8_t * buf, std::string & why) {
    HMODULE kb = LoadLibraryA("kernelbase.dll");
    if (!kb) { why = "no kernelbase"; return false; }
    auto pCreate = reinterpret_cast<ior::create_t>(reinterpret_cast<void *>(GetProcAddress(kb, "CreateIoRing")));
    auto pBuild  = reinterpret_cast<ior::build_read_t>(reinterpret_cast<void *>(GetProcAddress(kb, "BuildIoRingReadFile")));
    auto pSubmit = reinterpret_cast<ior::submit_t>(reinterpret_cast<void *>(GetProcAddress(kb, "SubmitIoRing")));
    auto pPop    = reinterpret_cast<ior::pop_t>(reinterpret_cast<void *>(GetProcAddress(kb, "PopIoRingCompletion")));
    auto pClose  = reinterpret_cast<ior::close_t>(reinterpret_cast<void *>(GetProcAddress(kb, "CloseIoRing")));
    if (!pCreate || !pBuild || !pSubmit || !pPop || !pClose) { why = "I/O ring entry points not exported by this Windows"; return false; }
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (h == INVALID_HANDLE_VALUE) { why = "open failed"; return false; }
    ior::HIORING_T ring = nullptr;
    ior::CreateFlags fl = { 0, 0 };
    HRESULT hr = pCreate(ior::VERSION_1, fl, (UINT32) qd * 2, (UINT32) qd * 4, &ring);
    if (FAILED(hr)) { CloseHandle(h); char b[64]; std::snprintf(b, sizeof(b), "CreateIoRing hr=0x%08lx", (unsigned long) hr); why = b; return false; }

    size_t next = 0, done = 0, inflight = 0;
    bool ok = true;
    while (ok && done < len) {
        while (inflight < (size_t) qd && next < len) {
            const size_t n = std::min(chunk, len - next);
            ior::HandleRef hr_ref; hr_ref.Kind = 0; hr_ref.u.Handle = h;
            ior::BufferRef br; br.Kind = 0; br.u.Address = buf + next;
            hr = pBuild(ring, hr_ref, br, (UINT32) n, off + next, (UINT_PTR) n, 0);
            if (FAILED(hr)) { ok = false; char b[64]; std::snprintf(b, sizeof(b), "BuildIoRingReadFile hr=0x%08lx", (unsigned long) hr); why = b; break; }
            next += n; inflight++;
        }
        if (!ok) { break; }
        UINT32 submitted = 0;
        hr = pSubmit(ring, 1, INFINITE, &submitted);
        if (FAILED(hr)) { ok = false; why = "SubmitIoRing failed"; break; }
        ior::Cqe cqe;
        while (inflight > 0 && pPop(ring, &cqe) == S_OK) {
            if (FAILED(cqe.ResultCode) || cqe.Information != (ULONG_PTR) cqe.UserData) { ok = false; why = "completion error or short read"; break; }
            done += cqe.UserData; inflight--;
        }
    }
    pClose(ring);
    CloseHandle(h);
    return ok && done == len;
}
#endif

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <cold-file> <work-dir> [region MiB] [rounds]\n", argv[0]);
        return 2;
    }
    const std::string cold = argv[1], work = argv[2];
    const size_t region = (size_t) (argc > 3 ? std::atoi(argv[3]) : 128) << 20;
    const int rounds = argc > 4 ? std::atoi(argv[4]) : 2;
    const uint64_t cold_size = fs::file_size(cold);

    // methods in a fixed order; every (round, method) reads its own region of the cold file, 16 MiB apart so no read-ahead overlaps
    struct method { const char * name; int kind; size_t chunk; int qd; };
    std::vector<method> M = {
        {"ifstream_1call", 0, 0, 0}, {"ifstream_8MiB", 1, 8u << 20, 0},
#ifdef _WIN32
        {"nobuf_seq_1M", 2, 1u << 20, 0}, {"nobuf_seq_4M", 2, 4u << 20, 0}, {"nobuf_seq_16M", 2, 16u << 20, 0},
        {"overlapped_qd2", 3, 4u << 20, 2}, {"overlapped_qd4", 3, 4u << 20, 4}, {"overlapped_qd8", 3, 4u << 20, 8},
        {"ioring_qd4", 4, 4u << 20, 4}, {"ioring_qd8", 4, 4u << 20, 8}, {"ioring_qd16", 4, 4u << 20, 16},
#endif
    };
    const size_t need = (size_t) rounds * M.size() * (region + (16u << 20));
    if (need + region > cold_size) {
        std::fprintf(stderr, "cold file is %llu MiB, this run needs %zu MiB of distinct regions; lower the region or the rounds\n",
                     (unsigned long long) (cold_size >> 20), (need + region) >> 20);
        return 2;
    }

    std::vector<uint8_t> ref(region);
#ifdef _WIN32
    uint8_t * abuf = aligned_alloc_win(region + (16u << 20));
#else
    std::vector<uint8_t> abuf_v(region);
    uint8_t * abuf = abuf_v.data();
#endif
    (void) abuf;
    size_t cursor = 0;
    size_t last_off = 0;
    for (int r = 0; r < rounds; r++) {
        for (const auto & m : M) {
            const size_t off = cursor;   // a multiple of 16 MiB, so sector aligned
            cursor += region + (16u << 20);
            last_off = off;
            bool ok = false;
            double ms = 0;
            std::string why;
            std::vector<uint8_t> got(region);
            const auto t0 = clk::now();
            if (m.kind == 0 || m.kind == 1) {
                std::ifstream f(cold, std::ios::binary);
                f.seekg((std::streamoff) off);
                if (m.kind == 0) {
                    f.read(reinterpret_cast<char *>(got.data()), (std::streamsize) region);
                } else {
                    for (size_t d = 0; d < region; d += m.chunk) {
                        f.read(reinterpret_cast<char *>(got.data()) + d, (std::streamsize) std::min(m.chunk, region - d));
                    }
                }
                ok = (bool) f;
                ms = ms_since(t0);
            }
#ifdef _WIN32
            else {
                if (m.kind == 2)      { ok = nobuf_read_seq(cold, off, region, m.chunk, abuf); }
                else if (m.kind == 3) { ok = overlapped_read(cold, off, region, m.chunk, m.qd, abuf); }
                else                  { ok = ioring_read(cold, off, region, m.chunk, m.qd, abuf, why); }
                ms = ms_since(t0);
                if (ok) { std::memcpy(got.data(), abuf, region); }
            }
#endif
            if (!ok) {
                std::printf("RESULT read %s round=%d FAILED %s\n", m.name, r, why.c_str());
                continue;
            }
            // verification: re-read the same region with the plain buffered path (now cached) and compare hashes
            {
                std::ifstream f(cold, std::ios::binary);
                f.seekg((std::streamoff) off);
                f.read(reinterpret_cast<char *>(ref.data()), (std::streamsize) region);
            }
            report("read", m.name, r, region, ms, hash_of(ref.data(), region) == hash_of(got.data(), region));
        }
    }
    // page-cache speed: the last region again, buffered
    {
        std::vector<uint8_t> got(region);
        std::ifstream f(cold, std::ios::binary);
        f.seekg((std::streamoff) last_off);
        const auto t0 = clk::now();
        f.read(reinterpret_cast<char *>(got.data()), (std::streamsize) region);
        report("read", "ifstream_reread", 0, region, ms_since(t0), true);
    }

    // ---- writes ------------------------------------------------------------------------------------------------------
    fs::create_directories(work);
    std::vector<uint8_t> payload(region);
    for (size_t i = 0; i < payload.size(); i++) { payload[i] = (uint8_t) (i * 131 + (i >> 12) * 17); }
    const uint64_t want = hash_of(payload.data(), payload.size());
    auto verify_file = [&](const std::string & p) {
        std::vector<uint8_t> back(region);
        std::ifstream f(p, std::ios::binary);
        f.read(reinterpret_cast<char *>(back.data()), (std::streamsize) region);
        return (bool) f && hash_of(back.data(), region) == want;
    };
    for (int r = 0; r < rounds; r++) {
        const std::string p = (fs::path(work) / "bench-io-write.bin").string();
        {
            fs::remove(p);
            const auto t0 = clk::now();
            { std::ofstream f(p, std::ios::binary); f.write(reinterpret_cast<const char *>(payload.data()), (std::streamsize) region); f.flush(); }
            const double ms = ms_since(t0);
            report("write", "ofstream_flush", r, region, ms, verify_file(p));
        }
#ifdef _WIN32
        {
            fs::remove(p);
            const auto t0 = clk::now();
            { std::ofstream f(p, std::ios::binary); f.write(reinterpret_cast<const char *>(payload.data()), (std::streamsize) region); f.flush(); }
            HANDLE h = CreateFileA(p.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h != INVALID_HANDLE_VALUE) { FlushFileBuffers(h); CloseHandle(h); }
            const double ms = ms_since(t0);
            report("write", "ofstream_fsync", r, region, ms, verify_file(p));
        }
        {
            fs::remove(p);
            uint8_t * wb = aligned_alloc_win(region);
            std::memcpy(wb, payload.data(), region);
            const auto t0 = clk::now();
            HANDLE h = CreateFileA(p.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
            bool ok = h != INVALID_HANDLE_VALUE;
            for (size_t d = 0; ok && d < region; d += 4u << 20) {
                DWORD n = 0;
                ok = WriteFile(h, wb + d, (DWORD) std::min<size_t>(4u << 20, region - d), &n, nullptr) && n > 0;
            }
            if (h != INVALID_HANDLE_VALUE) { CloseHandle(h); }
            const double ms = ms_since(t0);
            report("write", "nobuf_writethru_4M", r, region, ms, ok && verify_file(p));
            VirtualFree(wb, 0, MEM_RELEASE);
        }
        {
            fs::remove(p);
            uint8_t * wb = aligned_alloc_win(region);
            std::memcpy(wb, payload.data(), region);
            const int qd = 4;
            const size_t chunk = 4u << 20;
            const auto t0 = clk::now();
            HANDLE h = CreateFileA(p.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                                   FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH | FILE_FLAG_OVERLAPPED, nullptr);
            bool ok = h != INVALID_HANDLE_VALUE;
            if (ok) {
                struct slot { OVERLAPPED ov; size_t n; bool busy; };
                std::vector<slot> s(qd);
                for (auto & x : s) { std::memset(&x.ov, 0, sizeof(x.ov)); x.ov.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr); x.busy = false; x.n = 0; }
                size_t next = 0, done = 0;
                auto issue = [&](slot & x) {
                    if (next >= region) { return; }
                    x.n = std::min(chunk, region - next);
                    x.ov.Offset = (DWORD) (next & 0xffffffffu); x.ov.OffsetHigh = (DWORD) ((uint64_t) next >> 32);
                    ResetEvent(x.ov.hEvent);
                    const size_t at = next;
                    next += x.n;
                    if (!WriteFile(h, wb + at, (DWORD) x.n, nullptr, &x.ov) && GetLastError() != ERROR_IO_PENDING) { ok = false; return; }
                    x.busy = true;
                };
                for (auto & x : s) { issue(x); }
                while (ok && done < region) {
                    bool any = false;
                    for (auto & x : s) {
                        if (!x.busy) { continue; }
                        any = true;
                        DWORD n = 0;
                        if (!GetOverlappedResult(h, &x.ov, &n, TRUE) || n < x.n) { ok = false; break; }
                        done += x.n; x.busy = false; issue(x);
                    }
                    if (!any) { break; }
                }
                for (auto & x : s) { CloseHandle(x.ov.hEvent); }
                CloseHandle(h);
            }
            const double ms = ms_since(t0);
            report("write", "overlapped_wr_qd4", r, region, ms, ok && verify_file(p));
            VirtualFree(wb, 0, MEM_RELEASE);
        }
#endif
        fs::remove(p);
    }

    for (const auto & kv : g_results) {
        std::printf("MEDIAN %s MBps=%.0f n=%zu\n", kv.first.c_str(), median(kv.second), kv.second.size());
    }
    return 0;
}
