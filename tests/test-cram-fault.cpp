// test-cram-fault: page-level fault path for the soft compressed tier. MEASURE-ONLY model of what CRAM does
// in hardware (pages absent until touched, a fault materialises them) and of what a Windows port would do
// with PAGE_NOACCESS + a vectored exception handler (no Windows code is written here).
//
// Two mechanisms, same store, same access patterns:
//   (A) mmap(PROT_NONE) + SIGSEGV handler (sigaction, SA_SIGINFO): the handler decodes the faulting 64 KiB block
//       straight into the region (mprotect RW -> decode -> mprotect R). A FIFO ring bounds residency; evicting
//       is mprotect(PROT_NONE) + madvise(MADV_DONTNEED).
//   (B) userfaultfd (SYS_userfaultfd) + UFFDIO_COPY served by a helper thread -- only if the sandbox allows it.
//
// ASYNC-SIGNAL-SAFETY, honestly: the handler calls only mprotect/madvise (raw syscall wrappers; mprotect is NOT on
// POSIX's formal async-signal-safe list, but is a plain syscall on Linux) and llama_cram_soft_store::
// decode_block_nolock(), which takes no mutex and does no heap allocation (stack buffers only, ~8 KiB Huffman
// table). It deliberately does NOT go through read()/the LRU hot window: that path holds a std::mutex and allocates
// (make_shared), neither of which is async-signal-safe -- a fault taken while the faulting thread already holds
// that mutex would deadlock. The handler state (ring, counters) is single-threaded: this test faults from ONE thread.
// Also NOT safe/handled: faults from several threads (ring races), a fault inside libc/malloc code, a write
// fault (the region is only ever read), decode failure in the handler (it aborts via _exit(3)).
#if !defined(__unix__) && !defined(__APPLE__)
#include <cstdio>
int main() { std::printf("SKIPPED: POSIX-only test\n"); return 0; }
#else

#include "llama-cram-soft.h"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__linux__)
#include <linux/userfaultfd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#endif

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000; int e = (int)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (e >= 31) return (uint16_t)(sign | 0x7bff);
    if (e <= 0) return (uint16_t) sign; // flush tiny values; irrelevant for this test
    uint32_t r = ((uint32_t) e << 10) | (m >> 13);
    return (uint16_t)(sign | r);
}

static constexpr size_t B = 64 * 1024;
static constexpr size_t REGION = 32u << 20; // 512 blocks
static constexpr size_t NBLK = REGION / B;
static constexpr int    RING_MAX = 1024;

// ---- state shared with the handler (single faulting thread) ----
static llama_cram_soft_store * g_store;
static uint8_t * g_base;
static std::atomic<uint64_t> g_faults{0};
static int  g_ring[RING_MAX]; static int g_ring_head, g_ring_n, g_ring_cap;
static bool g_resident[NBLK];

static void evict_block(int blk) {
    mprotect(g_base + (size_t) blk * B, B, PROT_NONE);
    madvise(g_base + (size_t) blk * B, B, MADV_DONTNEED);
    g_resident[blk] = false;
}

static void segv_handler(int, siginfo_t * si, void *) {
    uint8_t * a = (uint8_t *) si->si_addr;
    if (a < g_base || a >= g_base + REGION) { // not ours: restore default and let the re-executed access crash
        signal(SIGSEGV, SIG_DFL);
        return;
    }
    size_t blk = (size_t)(a - g_base) / B;
    g_faults.fetch_add(1, std::memory_order_relaxed);
    if (g_ring_n == g_ring_cap) { // FIFO eviction keeps residency bounded
        evict_block(g_ring[g_ring_head]);
        g_ring_head = (g_ring_head + 1) % RING_MAX; g_ring_n--;
    }
    uint8_t * dst = g_base + blk * B;
    mprotect(dst, B, PROT_READ | PROT_WRITE);
    if (!g_store->decode_block_nolock(blk, dst)) _exit(3);
    mprotect(dst, B, PROT_READ);
    g_resident[blk] = true;
    g_ring[(g_ring_head + g_ring_n) % RING_MAX] = (int) blk; g_ring_n++;
}

static void ring_reset(int cap) { g_ring_head = g_ring_n = 0; g_ring_cap = cap; std::memset(g_resident, 0, sizeof(g_resident)); }

// ---- access patterns ----
struct acc { size_t off, len; };
static std::vector<acc> make_random(size_t n) {
    std::mt19937_64 g(77); std::vector<acc> v(n);
    for (auto & a : v) { a.len = 256; a.off = (g() % (REGION - 256)); }
    return v;
}
static bool run_seq(const uint8_t * reg, const uint8_t * orig) { return std::memcmp(reg, orig, REGION) == 0; }
static bool run_rand(const uint8_t * reg, const uint8_t * orig, const std::vector<acc> & v) {
    bool ok = true;
    for (auto & a : v) ok = ok && std::memcmp(reg + a.off, orig + a.off, a.len) == 0;
    return ok;
}

static std::vector<uint8_t> gen(int kind) {
    std::mt19937_64 g(5 + kind); std::vector<uint8_t> v(REGION);
    if (kind == 0) { std::normal_distribution<float> d(0.f, 0.02f); for (size_t i = 0; i < REGION; i += 2) { uint16_t h = f2h(d(g)); std::memcpy(&v[i], &h, 2); } }
    else { std::memset(v.data(), 0, REGION); for (size_t i = 0; i < REGION / 64; i++) v[g() % REGION] = (uint8_t) g(); }
    return v;
}

#if defined(__linux__)
// ---- userfaultfd helper ----
static int  g_uffd = -1, g_stop_fd = -1;
static std::atomic<uint64_t> g_uffd_faults{0};
static void * uffd_thread(void *) {
    static uint8_t stage[B] __attribute__((aligned(4096)));
    struct pollfd pf[2] = {{g_uffd, POLLIN, 0}, {g_stop_fd, POLLIN, 0}};
    for (;;) {
        if (poll(pf, 2, -1) < 0) { if (errno == EINTR) continue; break; }
        if (pf[1].revents) break;
        struct uffd_msg msg;
        ssize_t r = read(g_uffd, &msg, sizeof(msg));
        if (r != (ssize_t) sizeof(msg)) continue;
        if (msg.event != UFFD_EVENT_PAGEFAULT) continue;
        size_t blk = (size_t)((uint8_t *) msg.arg.pagefault.address - g_base) / B;
        g_uffd_faults.fetch_add(1, std::memory_order_relaxed);
        if (!g_store->decode_block_nolock(blk, stage)) _exit(3);
        struct uffdio_copy c; std::memset(&c, 0, sizeof(c));
        c.dst = (unsigned long) (g_base + blk * B); c.src = (unsigned long) stage; c.len = B;
        if (ioctl(g_uffd, UFFDIO_COPY, &c) < 0 && errno != EEXIST) _exit(4);
    }
    return nullptr;
}
#endif

int main() {
    const long PS = sysconf(_SC_PAGESIZE);
    std::printf("page size %ld, block %zu KiB, region %zu MiB (%zu blocks)\n", PS, B / 1024, REGION >> 20, NBLK);
    auto rnd = make_random(20000);
    const char * kname[] = {"f16-gauss(ratio~1.18)", "zero-heavy(ratio~15)"};

    for (int kind = 0; kind < 2; kind++) {
        auto orig = gen(kind);
        llama_cram_soft_params p; p.stride = kind == 0 ? 2 : 1; p.hot_blocks = 1;
        llama_cram_soft_store store(orig.data(), orig.size(), p);
        g_store = &store;
        auto st = store.stats();
        std::printf("\n=== data: %s  stored ratio %.3fx ===\n", kname[kind], (double) st.raw_bytes / st.compressed_bytes);

        // baseline: plain memcpy'd buffer, plus decode-only cost of N blocks without any fault machinery
        std::vector<uint8_t> plain(REGION); std::memcpy(plain.data(), orig.data(), REGION);
        double t0 = now_s(); bool ok = run_seq(plain.data(), orig.data()); double base_seq = now_s() - t0; CHECK(ok);
        t0 = now_s(); ok = run_rand(plain.data(), orig.data(), rnd); double base_rnd = now_s() - t0; CHECK(ok);
        std::vector<uint8_t> tmp(B);
        t0 = now_s(); for (size_t b = 0; b < NBLK; b++) CHECK(store.decode_block_nolock(b, tmp.data())); double dec_all = now_s() - t0;
        std::printf("baseline plain buffer: seq %.4f s, random(20000x256B) %.4f s; decode-only of all %zu blocks (no faults) %.4f s = %.0f us/block\n",
                    base_seq, base_rnd, NBLK, dec_all, dec_all / NBLK * 1e6);

        // ---- (A) SIGSEGV ----
        g_base = (uint8_t *) mmap(nullptr, REGION, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (g_base == MAP_FAILED) { std::printf("mmap failed\n"); return 2; }
        struct sigaction sa; std::memset(&sa, 0, sizeof(sa));
        sa.sa_sigaction = segv_handler; sa.sa_flags = SA_SIGINFO | SA_NODEFER; sigemptyset(&sa.sa_mask);
        struct sigaction old; sigaction(SIGSEGV, &sa, &old);

        ring_reset(RING_MAX); g_faults = 0;                                  // seq, everything stays resident
        t0 = now_s(); ok = run_seq(g_base, orig.data()); double t_seq = now_s() - t0; CHECK(ok);
        uint64_t f_seq = g_faults;
        CHECK(f_seq == NBLK);                                                // exactly one fault per block
        std::printf("[SIGSEGV] seq : %llu faults, %.4f s, %.0f faults/s, extra vs plain %.4f s = %.1f us/fault (decode share %.1f us/block)\n",
                    (unsigned long long) f_seq, t_seq, f_seq / t_seq, t_seq - base_seq, (t_seq - base_seq) / f_seq * 1e6, dec_all / NBLK * 1e6);

        for (size_t b = 0; b < NBLK; b++) if (g_resident[b]) evict_block((int) b); // re-arm everything
        ring_reset(32); g_faults = 0;                                        // random, only 32 of 512 blocks resident
        t0 = now_s(); ok = run_rand(g_base, orig.data(), rnd); double t_rnd = now_s() - t0; CHECK(ok);
        uint64_t f_rnd = g_faults;
        CHECK(f_rnd > rnd.size() / 2);
        std::printf("[SIGSEGV] rand: %llu faults for %zu accesses (window 32/%zu blocks), %.4f s, %.0f faults/s, extra vs plain %.4f s = %.1f us/fault\n",
                    (unsigned long long) f_rnd, rnd.size(), NBLK, t_rnd, f_rnd / t_rnd, t_rnd - base_rnd, (t_rnd - base_rnd) / f_rnd * 1e6);
        sigaction(SIGSEGV, &old, nullptr);
        munmap(g_base, REGION);

#if defined(__linux__)
        // ---- (B) userfaultfd ----
        errno = 0;
        long fd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK);
        int e1 = errno;
        if (fd < 0) { errno = 0; fd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY); }
        if (fd < 0) {
            std::printf("[UFFD] NOT AVAILABLE: userfaultfd() errno=%d (%s) (first try errno=%d); not faked, skipped\n", errno, strerror(errno), e1);
            continue;
        }
        g_uffd = (int) fd;
        struct uffdio_api api; std::memset(&api, 0, sizeof(api)); api.api = UFFD_API;
        if (ioctl(g_uffd, UFFDIO_API, &api) < 0) { std::printf("[UFFD] UFFDIO_API failed errno=%d (%s); skipped\n", errno, strerror(errno)); close(g_uffd); continue; }
        g_base = (uint8_t *) mmap(nullptr, REGION, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        struct uffdio_register reg; std::memset(&reg, 0, sizeof(reg));
        reg.range.start = (unsigned long) g_base; reg.range.len = REGION; reg.mode = UFFDIO_REGISTER_MODE_MISSING;
        if (ioctl(g_uffd, UFFDIO_REGISTER, &reg) < 0) { std::printf("[UFFD] UFFDIO_REGISTER failed errno=%d (%s); skipped\n", errno, strerror(errno)); close(g_uffd); munmap(g_base, REGION); continue; }
        int sp[2]; if (pipe(sp) != 0) return 2; g_stop_fd = sp[0];
        pthread_t th; pthread_create(&th, nullptr, uffd_thread, nullptr);
        std::printf("[UFFD] AVAILABLE: userfaultfd opened (first try errno=%d), UFFDIO_API ok (features 0x%llx), REGISTER missing-mode ok\n", e1, (unsigned long long) api.features);

        g_uffd_faults = 0;
        t0 = now_s(); ok = run_seq(g_base, orig.data()); double u_seq = now_s() - t0; CHECK(ok);
        uint64_t uf_seq = g_uffd_faults;
        CHECK(uf_seq == NBLK);
        std::printf("[UFFD]    seq : %llu faults, %.4f s, %.0f faults/s, extra vs plain %.4f s = %.1f us/fault\n",
                    (unsigned long long) uf_seq, u_seq, uf_seq / u_seq, u_seq - base_seq, (u_seq - base_seq) / uf_seq * 1e6);
        for (size_t b = 0; b < NBLK; b++) madvise(g_base + b * B, B, MADV_DONTNEED); // pages missing again
        ring_reset(32); g_uffd_faults = 0;
        // residency managed by the reader thread: evict the oldest block before touching a new one
        t0 = now_s(); ok = true;
        for (auto & a : rnd) {
            size_t b0 = a.off / B, b1 = (a.off + a.len - 1) / B;
            for (size_t b = b0; b <= b1; b++) if (!g_resident[b]) {
                if (g_ring_n == g_ring_cap) { int v = g_ring[g_ring_head]; madvise(g_base + (size_t) v * B, B, MADV_DONTNEED); g_resident[v] = false; g_ring_head = (g_ring_head + 1) % RING_MAX; g_ring_n--; }
                g_resident[b] = true; g_ring[(g_ring_head + g_ring_n) % RING_MAX] = (int) b; g_ring_n++;
            }
            ok = ok && std::memcmp(g_base + a.off, orig.data() + a.off, a.len) == 0;
        }
        double u_rnd = now_s() - t0; CHECK(ok);
        uint64_t uf_rnd = g_uffd_faults;
        std::printf("[UFFD]    rand: %llu faults for %zu accesses (window 32/%zu), %.4f s, %.0f faults/s, extra vs plain %.4f s = %.1f us/fault\n",
                    (unsigned long long) uf_rnd, rnd.size(), NBLK, u_rnd, uf_rnd / u_rnd, u_rnd - base_rnd, (u_rnd - base_rnd) / uf_rnd * 1e6);
        char c = 1; (void) !write(sp[1], &c, 1); pthread_join(th, nullptr);
        close(sp[0]); close(sp[1]); close(g_uffd); munmap(g_base, REGION);
#else
        std::printf("[UFFD] SKIPPED: Linux only\n");
#endif
    }
    std::printf(g_fail ? "test-cram-fault: FAILED (%d)\n" : "test-cram-fault: PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
#endif
