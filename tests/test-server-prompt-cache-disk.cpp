// The NVMe tier of the host prompt cache (--cache-disk-dir; row S69-TIERED-KV-DISK-TIER-SLICE1-2026-10-05-01).
// GATE-CLAIM-OK: test commentary only; it declares no shared resource unavailable ("RAM" below means the in-process cache).
//
// server_prompt_cache::evict_one() writes the entry it must evict to common_disk_tier instead of dropping it; the entry keeps
// its tokens and its place in `states`, so find_best() still matches it; read_spilled() brings the bytes back verified.
// No model is needed: entries are synthetic (distinct first tokens so none is a prefix of another) of 256 KiB each in a 1 MiB
// cache, so the cache holds 4 in memory. load() itself restores into a live llama_context and is covered end to end by
// tools/server/tests/unit/test_prompt_cache_disk.py. Every expectation has a partner (the tier off, a corrupted file, a
// refused write) that shows the tier is what changed the outcome.

#include "server-task.h"
#include "server-common.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static int g_fail = 0;

#define CHECK(cond, ...) do {                                   \
    if (!(cond)) {                                              \
        printf("  FAIL: %s -- ", #cond);                        \
        printf(__VA_ARGS__);                                    \
        printf("\n");                                           \
        g_fail++;                                               \
    } else {                                                    \
        printf("  ok:   %s\n", #cond);                          \
    }                                                           \
} while (0)

static const size_t ENTRY = 256 * 1024;   // bytes per entry; the 1 MiB cache holds 4 in memory

// prompt number k with n tokens: its first token is k, so no prompt is a prefix of another
static server_prompt make_prompt(llama_token k, int n = 300) {
    llama_tokens toks = { k };
    for (int i = 1; i < n; i++) {
        toks.push_back(7 + i);
    }
    server_prompt p { server_tokens(toks, false), {} };
    p.lora_fp = 0;
    return p;
}

static std::vector<uint8_t> bytes_of(llama_token k, size_t n) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = (uint8_t) (k * 13 + i * 7 + (i >> 9));
    }
    return v;
}

// alloc + fill the way slot.prompt_save() does
static server_prompt_cache_state * store(server_prompt_cache & c, llama_token k, int n_tok = 300, size_t sz = ENTRY) {
    auto * s = c.alloc(make_prompt(k, n_tok), sz, 0);
    if (s) {
        s->data.main = bytes_of(k, sz);
    }
    return s;
}

static std::list<server_prompt_cache_state>::iterator find_it(server_prompt_cache & c, llama_token k) {
    for (auto it = c.states.begin(); it != c.states.end(); ++it) {
        if (it->prompt.tokens[0] == k) {
            return it;
        }
    }
    return c.states.end();
}

static size_t n_files(const fs::path & dir) {
    size_t n = 0;
    for (const auto & e : fs::directory_iterator(dir)) { (void) e; n++; }
    return n;
}

static fs::path fresh_dir(const char * name) {
    auto d = fs::temp_directory_path() / "test-prompt-cache-disk" / name;
    fs::remove_all(d);
    return d;
}

// enable_disk with the volume's free space pinned to 1 TiB. Without the pin every test hangs on the host's real free space:
// on 2026-10-27 the temp volume had 8.0 GB free, under the 10 GiB floor, so the guard refused every spill, test 1 found no
// spilled entry and dereferenced end() (0xC0000005). Test 14 sets its own free space after this and so still drives the guard.
template <typename... A>
static bool enable_disk_t(server_prompt_cache & c, A &&... a) {
    const bool ok = c.enable_disk(std::forward<A>(a)...);
    if (ok) {
        c.disk->set_free_space_fn([](const std::string &) { return (int64_t) (1ll << 40); });
    }
    return ok;
}

static void test_eviction_goes_to_disk() {
    printf("test 1: an eviction goes to disk and the entry stays findable\n");
    const auto dir = fresh_dir("t1");
    server_prompt_cache off(1, 0);
    for (llama_token k = 1; k <= 6; ++k) { store(off, k); }
    CHECK(off.states.size() == 4, "control (tier off): 4 entries remain, got %zu", off.states.size());
    CHECK(off.size_disk() == 0 && !off.disk_enabled(), "control (tier off): no disk bytes");

    server_prompt_cache c(1, 0);
    std::string err;
    CHECK(enable_disk_t(c, dir.string(), 8u << 20, 16, 0, err), "enable_disk: %s", err.c_str());
    for (llama_token k = 1; k <= 6; ++k) { store(c, k); }
    CHECK(c.states.size() == 6, "tier on: all 6 entries are kept, got %zu", c.states.size());
    CHECK(c.n_spilled() == 2, "two entries are on disk, got %zu", c.n_spilled());
    CHECK(c.size() <= c.limit_size, "resident bytes %zu stay within the limit %zu", c.size(), c.limit_size);
    CHECK(c.size_disk() >= 2 * ENTRY && c.size_disk() <= 2 * ENTRY + 1024, "disk holds the two payloads (+ headers): %zu", c.size_disk());
    CHECK(n_files(dir) == 2, "two files in the directory, got %zu", n_files(dir));
    CHECK(c.ctr.n_spilled == 2, "n_spilled counter");
    auto it1 = find_it(c, 1), it2 = find_it(c, 2), it6 = find_it(c, 6);
    const bool all_found = it1 != c.states.end() && it2 != c.states.end() && it6 != c.states.end();
    CHECK(all_found, "entries 1, 2 and 6 are all still in the cache");
    if (!all_found) {
        return; // the rest dereferences them
    }
    CHECK(it1->spilled() && it2->spilled() && !it6->spilled(), "the two oldest are the spilled ones");
    CHECK(it1->data.main.empty() && it1->disk_n_bytes == ENTRY, "a spilled entry holds no memory");
    CHECK(c.n_tokens() == 6 * 300 && c.n_tokens_resident() == 4 * 300, "token counts: all %zu, resident %zu", c.n_tokens(), c.n_tokens_resident());

    // find_best still sees the spilled entry: a request that extends prompt 1 selects it
    llama_tokens ext = make_prompt(1).tokens.get_tokens();
    ext.push_back(9999);
    server_tokens req(ext, false);
    server_prompt empty_slot { server_tokens(llama_tokens{}, false), {} };
    thinking_tokens no_think;
    no_think.exclude = false;
    auto best = c.find_best(empty_slot, req, nullptr, no_think, 0);
    CHECK(best != c.states.end() && best->prompt.tokens[0] == 1, "find_best picks the spilled entry for its own prefix");

    // read it back: byte-identical
    server_prompt_data back;
    CHECK(c.read_spilled(it1, back), "read_spilled succeeds");
    CHECK(back.main == bytes_of(1, ENTRY), "the bytes come back identical");
    CHECK(c.ctr.n_restored == 1 && c.disk_stats().bytes_read > 0, "restore is counted");
    CHECK(it1->spilled(), "the entry stays on disk after a read (non-consuming, like the in-memory entries)");
}

static void test_corrupt_blob_is_a_miss() {
    printf("test 2: a corrupted or missing file is a miss, never wrong bytes\n");
    const auto dir = fresh_dir("t2");
    server_prompt_cache c(1, 0);
    std::string err;
    enable_disk_t(c, dir.string(), 8u << 20, 16, 0, err);
    for (llama_token k = 1; k <= 6; ++k) { store(c, k); }

    // flip a byte in the file of entry 1
    {
        auto it = find_it(c, 1);
        const std::string path = c.disk->path_of(it->disk_id);
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(1000); char ch = 0; f.read(&ch, 1); ch ^= 0x40; f.seekp(1000); f.write(&ch, 1);
    }
    server_prompt_data back;
    const size_t n_before = c.states.size();
    CHECK(!c.read_spilled(find_it(c, 1), back), "the corrupted blob is refused");
    CHECK(back.main.empty(), "no bytes were handed out");
    CHECK(c.states.size() == n_before - 1 && find_it(c, 1) == c.states.end(), "the entry was dropped from the cache");
    CHECK(c.ctr.n_restore_failed == 1 && c.disk->stats().n_checksum_failed == 1, "counted as a restore failure and a checksum failure");
    CHECK(n_files(dir) == 1, "its file was deleted (the other spilled entry's file remains), got %zu", n_files(dir));

    // delete the file of entry 2 behind the cache's back
    fs::remove(c.disk->path_of(find_it(c, 2)->disk_id));
    CHECK(!c.read_spilled(find_it(c, 2), back), "a missing file is a miss");
    CHECK(find_it(c, 2) == c.states.end() && c.ctr.n_restore_failed == 2, "and the entry is dropped");
    CHECK(c.size_disk() == 0, "disk accounting is back to 0, got %zu", c.size_disk());

    // control: an untouched spilled entry still reads
    server_prompt_cache d(1, 0);
    enable_disk_t(d, fresh_dir("t2b").string(), 8u << 20, 16, 0, err);
    for (llama_token k = 1; k <= 6; ++k) { store(d, k); }
    CHECK(d.read_spilled(find_it(d, 1), back) && back.main == bytes_of(1, ENTRY), "control: an untouched blob reads fine");
}

static void test_small_prompts_are_not_spilled() {
    printf("test 3: a prompt below --cache-disk-min-tokens is dropped, not spilled\n");
    const auto dir = fresh_dir("t3");
    server_prompt_cache c(1, 0);
    std::string err;
    enable_disk_t(c, dir.string(), 8u << 20, 100, 0, err);
    for (llama_token k = 1; k <= 6; ++k) { store(c, k, 50); } // 50 tokens < 100
    CHECK(c.states.size() == 4 && c.n_spilled() == 0, "short prompts: 4 remain, none on disk");
    CHECK(c.ctr.n_dropped_small == 2 && n_files(dir) == 0, "counted as dropped-small, nothing written");
    for (llama_token k = 11; k <= 16; ++k) { store(c, k, 150); }
    CHECK(c.n_spilled() > 0, "control: prompts above the minimum are spilled");
}

static void test_disk_cap_evicts_oldest_spilled() {
    printf("test 4: the disk cap holds; the oldest spilled entries make room\n");
    const auto dir = fresh_dir("t4");
    server_prompt_cache c(1, 0);
    std::string err;
    enable_disk_t(c, dir.string(), 2 * ENTRY + 4096, 16, 0, err); // room for two payloads
    for (llama_token k = 1; k <= 9; ++k) { store(c, k); }
    CHECK(c.size_disk() <= 2 * ENTRY + 4096, "disk bytes %zu within the cap", c.size_disk());
    CHECK(c.n_spilled() == 2, "exactly two entries on disk, got %zu", c.n_spilled());
    CHECK(c.ctr.n_disk_evicted == 3, "three spilled entries were removed to make room, got %llu", (unsigned long long) c.ctr.n_disk_evicted);
    CHECK(find_it(c, 1) == c.states.end() && find_it(c, 3) == c.states.end(), "the oldest spilled entries (1, 3) went first");
    CHECK(find_it(c, 4) != c.states.end() && find_it(c, 4)->spilled() && find_it(c, 5)->spilled(), "the two newest spilled entries (4, 5) are kept on disk");
    CHECK(n_files(dir) == 2, "files match entries: %zu", n_files(dir));
}

static void test_write_budget() {
    printf("test 5: the SSD write budget stops spilling and the entries are dropped as before\n");
    const auto dir = fresh_dir("t5");
    double now = 0;
    server_prompt_cache c(1, 0);
    std::string err;
    // budget of 300 KiB per hour: one 256 KiB spill fits, the next does not
    enable_disk_t(c, dir.string(), 64u << 20, 16, 300 * 1024, err, [&] { return now; });
    for (llama_token k = 1; k <= 7; ++k) { store(c, k); }
    CHECK(c.n_spilled() == 1, "one spill was paid for, got %zu", c.n_spilled());
    CHECK(c.ctr.n_dropped_refused == 2 && c.disk->stats().n_refused_budget == 2, "the other two were refused by the budget");
    CHECK(c.states.size() == 5, "4 in memory + 1 spilled, got %zu", c.states.size());
    CHECK(c.disk_stats().bytes_written == ENTRY + 40, "wear counter: %llu bytes", (unsigned long long) c.disk_stats().bytes_written);
    now += 3600;
    for (llama_token k = 20; k <= 22; ++k) { store(c, k); }
    CHECK(c.n_spilled() == 2, "after an hour the budget allows another spill, got %zu", c.n_spilled());
}

static void test_erase_and_obsolete_remove_blobs() {
    printf("test 6: erase and 'obsolete prompt' removal also delete the blob\n");
    const auto dir = fresh_dir("t6");
    server_prompt_cache c(1, 0);
    std::string err;
    enable_disk_t(c, dir.string(), 8u << 20, 16, 0, err);
    for (llama_token k = 1; k <= 6; ++k) { store(c, k); }
    CHECK(n_files(dir) == 2, "two blobs");
    // slot erase of prompt 1 (remove_prefixes_of): the entry is spilled
    const size_t n = c.remove_prefixes_of(make_prompt(1).tokens);
    CHECK(n == 1 && find_it(c, 1) == c.states.end(), "erase removed the spilled entry");
    CHECK(n_files(dir) == 1 && c.size_disk() < 2 * ENTRY, "and its file: %zu left", n_files(dir));
    // a longer prompt that contains cached prompt 2 makes entry 2 obsolete
    const std::string path2 = c.disk->path_of(find_it(c, 2)->disk_id);
    CHECK(fs::exists(path2), "entry 2's blob exists before");
    llama_tokens longer = make_prompt(2).tokens.get_tokens();
    for (int i = 0; i < 20; i++) { longer.push_back(5000 + i); }
    server_prompt lp { server_tokens(longer, false), {} };
    c.alloc(lp, ENTRY, 0);
    size_t n_first_token_2 = 0;
    for (const auto & s : c.states) { n_first_token_2 += s.prompt.tokens[0] == 2 ? 1 : 0; }
    CHECK(n_first_token_2 == 1 && find_it(c, 2)->prompt.n_tokens() == 320, "only the longer prompt (320 tokens) starts with token 2 now");
    CHECK(!fs::exists(path2), "alloc of a longer prompt removed the spilled prefix's file");
}

static void test_destructor_cleans_up_and_off_is_inert() {
    printf("test 7: files vanish with the cache; a bad directory leaves the cache working\n");
    const auto dir = fresh_dir("t7");
    {
        server_prompt_cache c(1, 0);
        std::string err;
        enable_disk_t(c, dir.string(), 8u << 20, 16, 0, err);
        for (llama_token k = 1; k <= 6; ++k) { store(c, k); }
        CHECK(n_files(dir) == 2, "two blobs while alive");
    }
    CHECK(n_files(dir) == 0, "no files after the cache is destroyed");

    // a path that cannot be a directory (it is a file)
    const auto bad = fresh_dir("t7file");
    fs::create_directories(bad.parent_path());
    std::ofstream(bad.string()) << "i am a file";
    server_prompt_cache c(1, 0);
    std::string err;
    CHECK(!enable_disk_t(c, bad.string(), 8u << 20, 16, 0, err) && !err.empty(), "enable_disk reports an unusable directory: '%s'", err.c_str());
    for (llama_token k = 1; k <= 6; ++k) { store(c, k); }
    CHECK(c.states.size() == 4 && !c.disk_enabled(), "control: the cache behaves exactly as without the flag");
    CHECK(!c.disk_stats().enabled, "and exports no disk metrics");
}

static void test_token_limit_counts_resident_only() {
    printf("test 8: the token limit bounds memory, so spilled entries do not count against it\n");
    const auto dir = fresh_dir("t8");
    server_prompt_cache c(0, 1500);  // no byte limit, at most 1500 tokens in memory
    std::string err;
    enable_disk_t(c, dir.string(), 8u << 20, 16, 0, err);
    for (llama_token k = 1; k <= 6; ++k) { store(c, k, 300, 64 * 1024); } // 6 x 300 tokens = 1800 tokens
    c.update();
    CHECK(c.n_tokens_resident() <= 1500, "tokens in memory within the limit: %zu", c.n_tokens_resident());
    CHECK(c.n_spilled() >= 1, "the excess went to disk, got %zu", c.n_spilled());
    CHECK(c.states.size() == 6, "nothing was lost, got %zu entries", c.states.size());

    server_prompt_cache off(0, 1500);
    for (llama_token k = 1; k <= 6; ++k) { store(off, k, 300, 64 * 1024); }
    off.update();
    CHECK(off.n_tokens() <= 1500 && off.states.size() < 6, "control (tier off): entries are dropped, %zu remain", off.states.size());
}

// ---- context checkpoints of a spilled entry (row S69-TIERED-CKPT-BLOBS-INTO-DISK-TIER-2026-10-05-01) ----------------------------

static const size_t CKPT = 64 * 1024;    // bytes of target data per checkpoint; an entry carries three

static common_prompt_checkpoint make_ckpt(llama_token k, int idx, size_t n_tgt = CKPT) {
    common_prompt_checkpoint c;
    c.n_tokens = 100 + idx * 50;
    c.pos_min  = idx * 50;
    c.pos_max  = c.pos_min + 40;
    c.data_tgt  = bytes_of(k * 10 + idx, n_tgt);
    c.data_dft  = bytes_of(k * 10 + idx + 100, 64);
    c.data_spec = bytes_of(k * 10 + idx + 200, 16);
    return c;
}

static server_prompt_cache_state * store_ck(server_prompt_cache & c, llama_token k, int n_ck = 3, int n_tok = 300) {
    server_prompt p = make_prompt(k, n_tok);
    for (int i = 0; i < n_ck; i++) {
        p.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(make_ckpt(k, i)));
    }
    auto * s = c.alloc(p, ENTRY, 0);
    if (s) {
        s->data.main = bytes_of(k, ENTRY);
    }
    return s;
}

static size_t n_files_with(const fs::path & dir, const char * prefix) {
    size_t n = 0;
    for (const auto & e : fs::directory_iterator(dir)) {
        n += e.path().filename().string().rfind(prefix, 0) == 0 ? 1 : 0;
    }
    return n;
}

static void flip(const std::string & path, std::streamoff at) {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekg(at); char ch = 0; f.read(&ch, 1); ch ^= 0x33; f.seekp(at); f.write(&ch, 1);
}

static void test_checkpoints_follow_the_entry_to_disk() {
    printf("test 9: a spilled entry's checkpoints leave memory too, and come back verified\n");
    const auto dir = fresh_dir("t9");
    server_prompt_cache off(1, 0);
    for (llama_token k = 1; k <= 5; ++k) { store_ck(off, k); }
    size_t ckpt_off = 0;
    for (const auto & s : off.states) { for (const auto & ck : s.prompt.checkpoints) { ckpt_off += ck->size(); } }
    CHECK(off.size() <= off.limit_size, "control (tier off): the cache stays within its limit by dropping entries (%zu left)", off.states.size());
    CHECK(ckpt_off > 0 && off.size_ckpt() == ckpt_off, "control (tier off): the checkpoints of the kept entries are resident (%zu B)", ckpt_off);

    server_prompt_cache c(1, 0);
    std::string err;
    CHECK(enable_disk_t(c, dir.string(), 16u << 20, 16, 0, err), "enable_disk: %s", err.c_str());
    for (llama_token k = 1; k <= 5; ++k) { store_ck(c, k); }
    CHECK(c.states.size() == 5, "tier on: all 5 entries are kept, got %zu", c.states.size());
    const size_t n_sp = c.n_spilled();
    CHECK(n_sp >= 2, "at least two entries were spilled, got %zu", n_sp);
    bool spilled_hold_nothing = true;
    for (const auto & s : c.states) {
        if (s.spilled() && s.size() != 0) {
            spilled_hold_nothing = false;
        }
    }
    CHECK(spilled_hold_nothing, "a spilled entry holds NO resident bytes: its checkpoints are on disk too");
    CHECK(c.ctr.n_ckpt_spilled == 3 * n_sp && c.ctr.n_ckpt_refused == 0, "3 checkpoints per spilled entry were written (%llu)", (unsigned long long) c.ctr.n_ckpt_spilled);
    CHECK(n_files_with(dir, "pckpt-") == 3 * n_sp && n_files_with(dir, "pcache-") == n_sp,
          "files: %zu checkpoint blobs, %zu entry blobs", n_files_with(dir, "pckpt-"), n_files_with(dir, "pcache-"));
    CHECK(c.disk_stats().bytes_written_ckpt > 0 && c.disk_stats().ram_ckpt_bytes == c.size_ckpt(), "the metrics see the checkpoint bytes");

    // restore: the clone a slot receives reloads each checkpoint from the tier, byte-identical, and the entry stays spilled
    auto it1 = find_it(c, 1);
    CHECK(it1->spilled(), "entry 1 is on disk");
    {
        server_prompt clone = it1->prompt.clone();
        bool all_ok = true;
        int i = 0;
        for (auto & ckp : clone.checkpoints) {
            auto & ck = server_ckpt_private(ckp);   // what the slot restore does: reload into the slot's own copy
            const auto want = make_ckpt(1, i++);
            all_ok = all_ok && ck.spilled() && ck.reload() && ck.data_tgt == want.data_tgt && ck.data_dft == want.data_dft &&
                     ck.data_spec == want.data_spec && ck.pos_max == want.pos_max && ck.n_tokens == want.n_tokens;
        }
        CHECK(all_ok && i == 3, "all three checkpoints reload byte-identical (target, draft and speculative parts)");
        CHECK(it1->size() == 0 && it1->spilled(), "the cache entry itself is still spilled and still holds nothing");
        CHECK(c.disk->stats().n_get_ckpt >= 3 && c.disk_stats().bytes_read_ckpt > 0, "the reads are counted as checkpoint reads");
    }

    // a flipped byte in ONE checkpoint blob: that checkpoint is refused, its siblings and the entry's main state still read
    {
        auto it2 = find_it(c, 2);
        CHECK(it2->spilled(), "entry 2 is on disk");
        server_prompt clone = it2->prompt.clone();
        auto ckp = clone.checkpoints.begin();
        common_prompt_checkpoint * ck = &server_ckpt_private(*ckp);
        flip(c.disk->path_of(ck->disk->id()), 500);
        const auto sums = c.disk->stats().n_checksum_failed;
        CHECK(!ck->reload(), "the corrupted checkpoint is refused");
        CHECK(ck->data_tgt.empty() && !ck->spilled() && !ck->disk, "no bytes were handed out and the reference is cleared (the prompt would be recomputed)");
        CHECK(c.disk->stats().n_checksum_failed == sums + 1, "counted as a checksum failure");
        ++ckp;
        ck = &server_ckpt_private(*ckp);
        CHECK(ck->reload() && ck->data_tgt == make_ckpt(2, 1).data_tgt, "control: the next checkpoint of the same entry is fine");
        server_prompt_data back;
        CHECK(c.read_spilled(it2, back) && back.main == bytes_of(2, ENTRY), "control: and so is the entry's main state");
    }

    // erase: the files of the entry go with it, but not while a slot still holds a copy of its checkpoints
    {
        const size_t ck_files = n_files_with(dir, "pckpt-");
        auto holder = std::make_unique<server_prompt>(find_it(c, 3)->prompt.clone());   // a slot restored entry 3 and keeps its checkpoints
        const size_t n = c.remove_prefixes_of(make_prompt(3).tokens);
        CHECK(n == 1 && find_it(c, 3) == c.states.end(), "entry 3 was erased");
        CHECK(n_files_with(dir, "pckpt-") == ck_files, "its checkpoint blobs stay while the slot's copy references them (%zu)", n_files_with(dir, "pckpt-"));
        holder.reset();
        CHECK(n_files_with(dir, "pckpt-") == ck_files - 3, "and are deleted when the slot lets go (%zu)", n_files_with(dir, "pckpt-"));
    }
}

static void test_checkpoints_the_tier_refuses_stay_resident() {
    printf("test 10: a checkpoint the tier refuses stays in memory, the entry is still spilled and usable\n");
    const auto dir = fresh_dir("t10");
    double now = 0;
    server_prompt_cache c(1, 0);
    std::string err;
    // 300 KiB of write budget per hour: the entry's main state (256 KiB) fits, the 64 KiB checkpoint blobs after it do not
    enable_disk_t(c, dir.string(), 16u << 20, 16, 300 * 1024, err, [&] { return now; });
    for (llama_token k = 1; k <= 5; ++k) { store_ck(c, k); }
    CHECK(c.n_spilled() == 1, "one entry was spilled, got %zu", c.n_spilled());
    CHECK(c.ctr.n_ckpt_refused == 3 && c.ctr.n_ckpt_spilled == 0, "its three checkpoints were refused by the budget (%llu)", (unsigned long long) c.ctr.n_ckpt_refused);
    auto it1 = find_it(c, 1);
    CHECK(it1->spilled() && it1->size() >= 3 * CKPT, "the spilled entry still holds its checkpoints in memory (%zu B)", it1->size());
    CHECK(n_files_with(dir, "pckpt-") == 0, "no checkpoint blob was written");
    server_prompt_data back;
    CHECK(c.read_spilled(it1, back) && back.main == bytes_of(1, ENTRY), "control: the entry's main state is restorable");
    CHECK(it1->prompt.checkpoints.front()->data_tgt == make_ckpt(1, 0).data_tgt, "control: the resident checkpoints are intact");
}

static void test_slot_files_are_adopted_not_shared() {
    printf("test 11: a checkpoint the slot spilled to a --ctx-ckpt-spill-dir file is copied into the entry, not shared\n");
    const auto dir = fresh_dir("t11");
    const auto slot_dir = fresh_dir("t11slot");
    fs::create_directories(slot_dir);

    for (int tier_on = 0; tier_on <= 1; ++tier_on) {
        server_prompt_cache c(8, 0);
        std::string err;
        if (tier_on) {
            enable_disk_t(c, dir.string(), 16u << 20, 16, 0, err);
        }
        server_prompt p = make_prompt(1);
        p.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(make_ckpt(1, 0)));
        p.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(make_ckpt(1, 1)));
        const std::string file = (slot_dir / "ckpt-s0-1.bin").string();
        const auto want = make_ckpt(1, 0).data_tgt;
        CHECK(p.checkpoints.front()->spill(file), "the slot spilled its first checkpoint to a file");
        auto * s = c.alloc(p, ENTRY, 0);
        CHECK(s != nullptr, "alloc");
        auto & first = *s->prompt.checkpoints.front();
        if (tier_on) {
            CHECK(first.data_tgt == want && first.spill_path.empty(), "tier on: the entry's copy holds the bytes and no path");
            CHECK(c.ctr.n_ckpt_adopted == 1, "tier on: counted as adopted");
            // the slot drops its checkpoint (deleting its file): the entry is not affected
            p.checkpoints.front()->remove_spill_file();
            CHECK(!fs::exists(file) && first.data_tgt == want, "tier on: the slot deleting its file does not hurt the entry");
        } else {
            CHECK(first.data_tgt.empty() && first.spill_path == file, "control (tier off): the copy is kept as it always was, sharing the path");
            CHECK(c.ctr.n_ckpt_adopted == 0 && !c.disk_stats().enabled, "control (tier off): nothing adopted");
            p.checkpoints.front()->remove_spill_file();
        }
    }

    // an unreadable file costs the copy, not the entry
    {
        server_prompt_cache c(8, 0);
        std::string err;
        enable_disk_t(c, dir.string(), 16u << 20, 16, 0, err);
        server_prompt p = make_prompt(2);
        p.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(make_ckpt(2, 0)));
        p.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(make_ckpt(2, 1)));
        const std::string file = (slot_dir / "ckpt-s0-2.bin").string();
        p.checkpoints.front()->spill(file);
        fs::resize_file(file, fs::file_size(file) / 2);
        auto * s = c.alloc(p, ENTRY, 0);
        CHECK(s != nullptr && s->prompt.checkpoints.size() == 1 && s->prompt.checkpoints.front()->pos_min == 50,
              "an unreadable file: the entry keeps its other checkpoint and drops that copy");
        CHECK(fs::exists(file), "and the slot's file is not deleted by the entry");
        p.checkpoints.front()->remove_spill_file();
    }
}

static void test_slot_checkpoints_use_the_tier() {
    printf("test 12: a slot's live checkpoints go through the tier (cap, budget, checksum) when it is on\n");
    // tier off: nothing to do, the checkpoint stays resident (the slot code then uses its own file spill)
    {
        server_prompt_cache off(1, 0);
        auto ck = make_ckpt(1, 0);
        CHECK(!off.spill_slot_checkpoint(ck) && !ck.data_tgt.empty() && !ck.disk, "control (tier off): refused, the checkpoint stays resident");
    }
    // tier on
    {
        const auto dir = fresh_dir("t12");
        server_prompt_cache c(1, 0);
        std::string err;
        enable_disk_t(c, dir.string(), 16u << 20, 16, 0, err);
        auto ck = make_ckpt(1, 0);
        const auto want = ck.data_tgt;
        CHECK(c.spill_slot_checkpoint(ck) && ck.spilled() && ck.size() == 0, "the checkpoint is in the tier and holds no memory");
        CHECK(c.ctr.n_ckpt_slot_spilled == 1 && n_files_with(dir, "pckpt-") == 1, "counted, one checkpoint blob");
        CHECK(c.ctr.n_ckpt_spilled == 0, "control: it is not counted as an entry's checkpoint");
        CHECK(ck.reload() && ck.data_tgt == want, "and it reloads byte-identical");
        // a copy the cache entry keeps shares the blob: the slot dropping its own checkpoint must not delete it
        auto ck2 = make_ckpt(2, 0);
        c.spill_slot_checkpoint(ck2);
        auto entry_copy = ck2;
        ck2 = common_prompt_checkpoint();
        CHECK(n_files_with(dir, "pckpt-") == 2 && entry_copy.reload(), "the slot dropping its checkpoint leaves the entry's copy readable");
    }
    // the write budget applies to it
    {
        const auto dir = fresh_dir("t12b");
        double now = 0;
        server_prompt_cache c(1, 0);
        std::string err;
        enable_disk_t(c, dir.string(), 16u << 20, 16, 100 * 1024, err, [&] { return now; });   // room for one 64 KiB blob per hour
        auto a = make_ckpt(1, 0), b = make_ckpt(2, 0);
        CHECK(c.spill_slot_checkpoint(a), "first checkpoint fits the budget");
        CHECK(!c.spill_slot_checkpoint(b) && !b.data_tgt.empty() && !b.disk, "the second is refused by the budget and stays in memory");
        CHECK(c.ctr.n_ckpt_slot_refused == 1 && c.disk->stats().n_refused_budget == 1, "refusal counted by the cache and the tier");
        now += 3600;
        CHECK(c.spill_slot_checkpoint(b), "an hour later the budget allows it");
    }
    // a full tier makes room by dropping the oldest spilled CACHE entry, never the slot's checkpoint
    {
        const auto dir = fresh_dir("t12c");
        server_prompt_cache c(1, 0);
        std::string err;
        enable_disk_t(c, dir.string(), 2 * ENTRY + 4096, 16, 0, err);   // room for two entry payloads only
        for (llama_token k = 1; k <= 9; ++k) { store(c, k); }
        CHECK(c.n_spilled() == 2, "two entries on disk, got %zu", c.n_spilled());
        const auto evicted = c.ctr.n_disk_evicted;
        auto ck = make_ckpt(50, 0);
        CHECK(c.spill_slot_checkpoint(ck) && ck.spilled(), "the slot's checkpoint is accepted");
        CHECK(c.ctr.n_disk_evicted == evicted + 1 && c.n_spilled() == 1, "by dropping the oldest spilled entry (%llu evicted)", (unsigned long long) c.ctr.n_disk_evicted);
        CHECK(c.size_disk() <= 2 * ENTRY + 4096, "the cap holds: %zu", c.size_disk());
    }
}

static void test_restore_versus_recompute_rule() {
    printf("test 13: a state is spilled and restored only while reading it back beats recomputing it\n");
    {
        server_prompt_cache c(1, 0);
        CHECK(c.restore_pays(1, 1u << 30), "no rates known: the rule has no opinion");
        c.note_prefill(1000, 1.0);                 // 1000 tokens/s
        CHECK(c.restore_pays(1, 1u << 30), "control: only one rate known: still no opinion");
        c.note_restore(100u << 20, 1.0);           // 100 MiB/s
        const size_t MB10 = 10u << 20;
        // recompute t = tokens / 1000 s ; restore t = bytes / (100 MiB/s) = 0.1 s for 10 MiB: break-even at 100 tokens
        CHECK(c.restore_pays(100, MB10), "at the break-even (100 tokens, 10 MiB) the restore pays");
        CHECK(!c.restore_pays(99, MB10), "one token below it does not");
        CHECK(c.restore_pays(100000, MB10) && !c.restore_pays(1, MB10), "far above / far below");
        c.min_speedup = 2.0;
        CHECK(!c.restore_pays(150, MB10) && c.restore_pays(200, MB10), "with a margin of 2 the break-even doubles to 200 tokens");
        c.min_speedup = 0;
        CHECK(c.restore_pays(1, 1u << 30), "control: min_speedup 0 switches the rule off");
    }
    {
        server_prompt_cache c(1, 0);
        c.note_prefill(31, 0.001);                 // below 32 tokens: a fixed-cost batch, ignored
        CHECK(c.prefill_tps() == 0, "a batch of 31 tokens does not set the prefill rate");
        c.note_prefill(1000, 1.0);
        c.note_prefill(2000, 1.0);                 // EWMA: 0.7 * 1000 + 0.3 * 2000
        CHECK(std::fabs(c.prefill_tps() - 1300.0) < 1e-6, "EWMA of the prefill rate: %f", c.prefill_tps());
        c.note_prefill(0, 1.0); c.note_prefill(1000, 0.0);
        CHECK(std::fabs(c.prefill_tps() - 1300.0) < 1e-6, "control: empty or zero-time samples are ignored");
    }
    // the tier's own write speed stands in for the restore speed until a restore has been timed
    {
        const auto dir = fresh_dir("t13a");
        server_prompt_cache c(1, 0);
        std::string err;
        enable_disk_t(c, dir.string(), 16u << 20, 16, 0, err);
        CHECK(c.restore_Bps() == 0, "nothing written yet: unknown");
        for (llama_token k = 1; k <= 6; ++k) { store(c, k); }
        CHECK(c.restore_Bps() > 0, "after spills the write throughput stands in: %.0f B/s", c.restore_Bps());
        c.note_restore(1000, 1.0);
        CHECK(c.restore_Bps() == 1000.0, "and a measured restore replaces it");
    }
    // gating the spill: rates under which a restore can never pay -> evicted entries are dropped, nothing is written
    {
        const auto dir = fresh_dir("t13b");
        server_prompt_cache c(1, 0);
        std::string err;
        enable_disk_t(c, dir.string(), 16u << 20, 16, 0, err);
        c.note_prefill(1000, 1.0);                 // 1000 tok/s
        c.note_restore(1000, 1.0);                 // 1000 B/s: a 256 KiB state would take 262 s to read; 300 tokens take 0.3 s
        for (llama_token k = 1; k <= 6; ++k) { store(c, k); }
        CHECK(c.n_spilled() == 0 && n_files(dir) == 0, "nothing was spilled, got %zu", c.n_spilled());
        CHECK(c.ctr.n_dropped_slow == 2 && c.states.size() == 4, "the two evicted entries are counted as dropped-slow (%llu)", (unsigned long long) c.ctr.n_dropped_slow);
        // control: the same traffic with the rule off spills them
        server_prompt_cache d(1, 0);
        enable_disk_t(d, fresh_dir("t13c").string(), 16u << 20, 16, 0, err);
        d.min_speedup = 0;
        d.note_prefill(1000, 1.0); d.note_restore(1000, 1.0);
        for (llama_token k = 1; k <= 6; ++k) { store(d, k); }
        CHECK(d.n_spilled() == 2 && d.ctr.n_dropped_slow == 0, "control (rule off): both are spilled, got %zu", d.n_spilled());
    }
    // gating the checkpoints separately: the main state pays (300 tokens), a 10-token checkpoint of 64 KiB does not
    {
        const auto dir = fresh_dir("t13d");
        server_prompt_cache c(1, 0);
        std::string err;
        enable_disk_t(c, dir.string(), 16u << 20, 16, 0, err);
        c.note_prefill(1000, 1.0);                 // r / p = 2000 bytes per token of saved work
        c.note_restore(2000000, 1.0);
        for (llama_token k = 1; k <= 5; ++k) {
            server_prompt p = make_prompt(k);
            auto ck = make_ckpt(k, 0);
            ck.n_tokens = 10;
            p.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(ck));
            auto * s = c.alloc(p, ENTRY, 0);
            if (s) { s->data.main = bytes_of(k, ENTRY); }
        }
        CHECK(c.n_spilled() >= 1, "the main state pays and is spilled");
        CHECK(c.ctr.n_ckpt_dropped_slow >= 1 && c.ctr.n_ckpt_spilled == 0, "its short checkpoint does not pay and stays resident (%llu)", (unsigned long long) c.ctr.n_ckpt_dropped_slow);
        CHECK(find_it(c, 1)->spilled() && find_it(c, 1)->size() >= CKPT, "so the spilled entry still holds the checkpoint in memory");
    }
}

// the value /metrics prints for llamacpp:<name>, or -1 when the line is absent
static double metric_value(const std::string & text, const std::string & name) {
    const std::string key = "\nllamacpp:" + name + " ";
    const size_t pos = text.find(key);
    if (pos == std::string::npos) {
        return -1;
    }
    return std::stod(text.substr(pos + key.size()));
}

static void test_space_and_budget_refusals_reach_metrics() {
    // row S69-FORK-DISK-TIER-REFUSED-SPACE-SERVER-METRICS-2026-10-19-01: the free-space guard counted its refusals in
    // common_disk_tier_stats::n_refused_space, but /metrics only showed refused_cap and refused_budget, so an operator saw
    // prompt states being dropped with no metric saying why. The text checked here is what server_task_result_metrics::to_metrics()
    // returns for GET /metrics (server-context.cpp fills res->disk from prompt_cache->disk_stats()).
    printf("test 14: free-space and write-budget refusals of the disk tier are exported by /metrics\n");
    {
        const auto dir = fresh_dir("t14a");
        server_prompt_cache c(1, 0);
        std::string err;
        enable_disk_t(c, dir.string(), 64u << 20, 16, 0, err);
        // the volume reports 1 GiB free: under the 10 GiB floor, so the guard refuses every write (no disk is filled)
        c.disk->set_free_space_fn([](const std::string &) { return (int64_t) (1ll << 30); });
        for (llama_token k = 1; k <= 6; ++k) { store(c, k); }
        CHECK(c.disk->stats().n_refused_space == 2 && c.ctr.n_dropped_refused == 2 && c.n_spilled() == 0,
              "the two evictions were refused by the free-space guard (tier count %llu)", (unsigned long long) c.disk->stats().n_refused_space);
        server_task_result_metrics r;
        r.disk = c.disk_stats();
        const std::string m = "\n" + r.to_metrics();
        CHECK(metric_value(m, "prompt_cache_disk_refused_space_total") == 2,
              "/metrics exports the free-space refusals: got %g (-1 = line missing)", metric_value(m, "prompt_cache_disk_refused_space_total"));
        CHECK(m.find("# TYPE llamacpp:prompt_cache_disk_refused_space_total counter\n") != std::string::npos, "as a counter");
        CHECK(metric_value(m, "prompt_cache_disk_refused_budget_total") == 0 && metric_value(m, "prompt_cache_disk_refused_cap_total") == 0,
              "control: neither the budget nor the cap was the reason");
        CHECK(metric_value(m, "prompt_cache_disk_dropped_refused_total") == 2, "and the cache's own drop counter agrees");
    }
    {
        // partner: the same evictions refused by the write budget instead (300 KiB/h: one 256 KiB spill fits, the next do not)
        const auto dir = fresh_dir("t14b");
        double now = 0;
        server_prompt_cache c(1, 0);
        std::string err;
        enable_disk_t(c, dir.string(), 64u << 20, 16, 300 * 1024, err, [&] { return now; });
        c.disk->set_free_space_fn([](const std::string &) { return (int64_t) (1ll << 40); }); // 1 TiB free: the guard allows it
        for (llama_token k = 1; k <= 7; ++k) { store(c, k); }
        server_task_result_metrics r;
        r.disk = c.disk_stats();
        const std::string m = "\n" + r.to_metrics();
        CHECK(metric_value(m, "prompt_cache_disk_refused_budget_total") == 2, "/metrics exports the budget refusals: got %g",
              metric_value(m, "prompt_cache_disk_refused_budget_total"));
        CHECK(metric_value(m, "prompt_cache_disk_refused_space_total") == 0, "control: no free-space refusal: got %g (-1 = line missing)",
              metric_value(m, "prompt_cache_disk_refused_space_total"));
    }
    {
        // the tier off: the server exports none of the disk lines, the new one included
        server_prompt_cache c(1, 0);
        server_task_result_metrics r;
        r.disk = c.disk_stats();
        const std::string m = "\n" + r.to_metrics();
        CHECK(m.find("prompt_cache_disk_") == std::string::npos, "control (tier off): no disk-tier metric lines");
    }
}

static void test_checkpoints_are_shared_not_copied() {
    // llama.cpp e152332a1 (#30137): the prompt cache and the slot share ONE checkpoint object instead of deep-copying it. On the
    // branch before this change the entry held its own copy (distinct objects, no use_count), so this test cannot even compile there.
    printf("test 15: a cache entry shares the slot's checkpoint objects; the last owner deletes the spill file\n");
    const auto slot_dir = fresh_dir("t15slot");
    fs::create_directories(slot_dir);
    server_prompt_cache c(8, 0);   // tier off
    server_prompt p = make_prompt(1);
    p.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(make_ckpt(1, 0)));
    p.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(make_ckpt(1, 1)));
    auto * s = c.alloc(p, ENTRY, 0);
    CHECK(s != nullptr && s->prompt.checkpoints.size() == 2, "alloc keeps both checkpoints");
    CHECK(s != nullptr && s->prompt.checkpoints.front().get() == p.checkpoints.front().get() &&
          s->prompt.checkpoints.back().get() == p.checkpoints.back().get(), "the entry holds the slot's checkpoint objects (no deep copy)");
    CHECK(p.checkpoints.front().use_count() == 2, "two owners: the slot and the entry (%ld)", (long) p.checkpoints.front().use_count());
    server_prompt clone = s->prompt.clone();
    CHECK(clone.checkpoints.front().get() == p.checkpoints.front().get() && p.checkpoints.front().use_count() == 3,
          "a clone (child slot) shares them too (%ld owners)", (long) p.checkpoints.front().use_count());

    const std::string file = (slot_dir / "ckpt-s0-15.bin").string();
    CHECK(p.checkpoints.front()->spill(file) && fs::exists(file), "the slot spilled the first checkpoint to a file");
    server_ckpt_release(p.checkpoints.front());
    CHECK(fs::exists(file), "the slot let go but two owners remain: the file stays");
    p.checkpoints.pop_front();
    clone.clear();
    CHECK(fs::exists(file), "the clone let go, the entry still holds it: the file stays");
    c.drop_state(c.states.begin());
    CHECK(!fs::exists(file), "the last owner (the entry) deleted the file");
}

static void test_shared_checkpoints_count_once() {
    // unique-bytes accounting: two entries holding the SAME checkpoint objects cost those bytes once (--cache-ram used to count them
    // per entry and evict / refuse early). Control: entries with distinct objects count each.
    printf("test 16: checkpoint objects shared by several entries are counted once in size() and size_ckpt()\n");
    auto ck_a = std::make_shared<common_prompt_checkpoint>(make_ckpt(1, 0));
    auto ck_b = std::make_shared<common_prompt_checkpoint>(make_ckpt(1, 1));
    const size_t ab = ck_a->size() + ck_b->size();
    CHECK(ab > 0, "the fixture checkpoints have resident bytes (%zu)", ab);
    {
        server_prompt_cache c(64, 0);   // tier off
        server_prompt p1 = make_prompt(1);
        p1.checkpoints.push_back(ck_a);
        p1.checkpoints.push_back(ck_b);
        server_prompt p2 = make_prompt(2);
        p2.checkpoints.push_back(ck_a);
        p2.checkpoints.push_back(ck_b);
        CHECK(c.alloc(p1, ENTRY, 0) != nullptr && c.alloc(p2, ENTRY, 0) != nullptr, "both entries stored");
        CHECK(c.size_ckpt() == ab, "shared objects counted once in size_ckpt (%zu vs %zu)", c.size_ckpt(), ab);
        CHECK(c.size() == 2 * ENTRY + ab, "size() = 2 entries' state bytes + the shared checkpoints once (%zu vs %zu)", c.size(), 2 * ENTRY + ab);
    }
    {
        server_prompt_cache c(64, 0);
        server_prompt p1 = make_prompt(1);
        p1.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(make_ckpt(1, 0)));
        p1.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(make_ckpt(1, 1)));
        server_prompt p2 = make_prompt(2);
        p2.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(make_ckpt(1, 0)));
        p2.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(make_ckpt(1, 1)));
        CHECK(c.alloc(p1, ENTRY, 0) != nullptr && c.alloc(p2, ENTRY, 0) != nullptr, "control: both entries stored");
        CHECK(c.size_ckpt() == 2 * ab, "control: distinct objects are each counted (%zu vs %zu)", c.size_ckpt(), 2 * ab);
    }
}

int main() {
    test_shared_checkpoints_count_once();
    test_checkpoints_are_shared_not_copied();
    test_eviction_goes_to_disk();
    test_corrupt_blob_is_a_miss();
    test_small_prompts_are_not_spilled();
    test_disk_cap_evicts_oldest_spilled();
    test_write_budget();
    test_erase_and_obsolete_remove_blobs();
    test_destructor_cleans_up_and_off_is_inert();
    test_token_limit_counts_resident_only();
    test_checkpoints_follow_the_entry_to_disk();
    test_checkpoints_the_tier_refuses_stay_resident();
    test_slot_files_are_adopted_not_shared();
    test_slot_checkpoints_use_the_tier();
    test_restore_versus_recompute_rule();
    test_space_and_budget_refusals_reach_metrics();
    fs::remove_all(fs::temp_directory_path() / "test-prompt-cache-disk");
    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
