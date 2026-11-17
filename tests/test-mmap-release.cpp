// llama_mmap::release_fragment (Windows working-set release): after every page of a read-only file view is touched, releasing a range must drop
// its COMPLETE interior pages from the process working set, leave the bytes identical on the next touch, and leave the edge pages of a ragged range alone.
//
//   whole view   256 MiB temp file, every page touched, release_fragment(0, size): resident pages before vs after (QueryWorkingSetEx), re-touch checksum equal
//   ragged range [3 pages + 100 bytes, 7 pages - 100 bytes): pages 4..5 are the only COMPLETE interior pages; page 3 and page 6 must stay resident
//   control      a page outside the released range stays resident, so "everything dropped" cannot be a measuring artefact
//
// Off Windows release_fragment does nothing and this test says so and passes.
// Row S69-STRATA-S1-WINDOWS-WORKINGSET-RELEASE-2026-10-07-01. Pattern read from Strata (MIT) expert_source.cpp release(); no code copied.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "../src/llama-mmap.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>

static bool page_resident(void * p) {
    typedef BOOL (WINAPI * query_fn)(HANDLE, PVOID, DWORD);
    static query_fn fn = (query_fn) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "K32QueryWorkingSetEx");
    PSAPI_WORKING_SET_EX_INFORMATION info = {};
    info.VirtualAddress = p;
    if (!fn || !fn(GetCurrentProcess(), &info, sizeof(info))) {
        fprintf(stderr, "QueryWorkingSetEx unavailable or failed (%lu)\n", (unsigned long) GetLastError());
        exit(2);
    }
    return info.VirtualAttributes.Valid != 0;
}

static size_t count_resident(const uint8_t * base, size_t first_page, size_t n_pages, size_t page) {
    size_t n = 0;
    for (size_t i = 0; i < n_pages; ++i) {
        n += page_resident((void *) (base + (first_page + i) * page));
    }
    return n;
}

static uint64_t touch(const volatile uint8_t * base, size_t size, size_t page) {
    uint64_t sum = 0;
    for (size_t off = 0; off < size; off += page) {
        sum += base[off];
    }
    return sum;
}

int main() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t page = si.dwPageSize;
    const size_t n_pages = (256u << 20) / page;
    const size_t size = n_pages * page;

    char tmp_dir[MAX_PATH], tmp_name[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp_dir);
    GetTempFileNameA(tmp_dir, "mmr", 0, tmp_name);
    {
        FILE * f = fopen(tmp_name, "wb");
        if (!f) { fprintf(stderr, "cannot create %s\n", tmp_name); return 1; }
        std::vector<uint8_t> buf(1u << 20);
        for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint8_t) (i * 131u + 7u);
        for (size_t w = 0; w < size; w += buf.size()) {
            buf[0] = (uint8_t) (w >> 20);
            fwrite(buf.data(), 1, buf.size(), f);
        }
        fclose(f);
    }

    int fails = 0;
    {
        llama_file file(tmp_name, "rb");
        llama_mmap map(&file, 0, false, false);
        const uint8_t * base = (const uint8_t *) map.addr();

        const uint64_t sum0 = touch(base, size, page);
        const size_t before = count_resident(base, 0, n_pages, page);
        map.release_fragment(0, size);
        const size_t after = count_resident(base, 0, n_pages, page);
        const uint64_t sum1 = touch(base, size, page);
        printf("whole view: resident pages %zu -> %zu of %zu; checksum %s\n", before, after, n_pages, sum0 == sum1 ? "identical after re-touch" : "CHANGED");
        if (before < n_pages * 9 / 10) { fprintf(stderr, "FAIL: only %zu of %zu pages resident before the release, so the measurement shows nothing\n", before, n_pages); fails++; }
        if (after > before / 10)       { fprintf(stderr, "FAIL: %zu of %zu pages still resident after the release\n", after, before); fails++; }
        if (sum0 != sum1)              { fprintf(stderr, "FAIL: bytes differ after re-touch\n"); fails++; }

        // ragged range: first = 3 pages + 100, last = 7 pages - 100 -> complete interior pages are 4 and 5
        touch(base, size, page);
        const size_t first = 3 * page + 100, last = 7 * page - 100;
        map.release_fragment(first, last);
        bool r[10];
        for (int i = 0; i < 10; ++i) r[i] = page_resident((void *) (base + (size_t) i * page));
        printf("ragged range, pages 0..9 resident: ");
        for (int i = 0; i < 10; ++i) printf("%d", r[i] ? 1 : 0);
        printf("  (expect pages 4,5 dropped, 3 and 6 kept)\n");
        if (r[4] || r[5])       { fprintf(stderr, "FAIL: a complete interior page is still resident\n"); fails++; }
        if (!r[3] || !r[6])     { fprintf(stderr, "FAIL: an edge page of the ragged range was released\n"); fails++; }
        if (!r[0] || !r[9])     { fprintf(stderr, "FAIL: a page outside the range was released (control)\n"); fails++; }
    }
    remove(tmp_name);
    if (fails) return 1;
    printf("all mmap-release checks passed\n");
    return 0;
}
#else
int main() {
    printf("release_fragment does nothing off Windows: nothing to test\n");
    return 0;
}
#endif
