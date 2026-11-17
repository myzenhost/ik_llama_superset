// probe-ckpt-spill: common_prompt_checkpoint::spill() / reload() / remove_spill_file() (--ctx-ckpt-spill-dir).
// No model needed: the checkpoint data is fake bytes. Prints "N checks, M failed" and exits non-zero on a failure.
// Each behaviour has a negative control, so a probe that can only say "ok" would fail here.
// Row S69-FLAG-WIRING-CTX-CKPT-SPILL-NOT-PORTED-2026-10-01-01.
#include "common.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static int g_checks = 0;
static int g_fails  = 0;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: %s\n", what); } } while (0)

static common_prompt_checkpoint make_ckpt() {
    common_prompt_checkpoint c;
    c.n_tokens = 123;
    c.pos_min  = 7;
    c.pos_max  = 130;
    for (int i = 0; i < 4096; i++) { c.data_tgt.push_back((uint8_t) (i * 7 + 1)); }
    for (int i = 0; i < 100;  i++) { c.data_dft.push_back((uint8_t) (i * 3 + 2)); }
    for (int i = 0; i < 9;    i++) { c.data_spec.push_back((uint8_t) (i + 5)); }
    return c;
}

static bool exists(const std::string & p) { return std::filesystem::exists(p); }

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "probe-ckpt-spill";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "ckpt-s0-1.bin").string();

    // ---- round trip -------------------------------------------------------------------------
    {
        auto c = make_ckpt();
        const auto tgt = c.data_tgt, dft = c.data_dft, spec = c.data_spec;
        CHECK(!c.spilled(), "a fresh checkpoint is not spilled");
        CHECK(c.spill(path), "spill succeeds");
        CHECK(exists(path), "the spill file exists");
        CHECK(c.spilled(), "spilled() after spill");
        CHECK(c.data_tgt.empty() && c.data_dft.empty() && c.data_spec.empty(), "memory is freed by spill");
        CHECK(c.empty(), "empty() is the in-memory marker: true after spill");
        CHECK(c.pos_min == 7 && c.pos_max == 130 && c.n_tokens == 123, "positions stay in memory");
        CHECK(c.reload(), "reload succeeds");
        CHECK(!c.spilled(), "not spilled after reload");
        CHECK(c.data_tgt == tgt && c.data_dft == dft && c.data_spec == spec, "all three vectors come back byte-identical");
        CHECK(exists(path), "the file stays after reload until the entry is dropped");
        c.remove_spill_file();
        CHECK(!exists(path), "remove_spill_file deletes it");
    }

    // ---- control: spill with nothing resident / no path does nothing --------------------------
    {
        common_prompt_checkpoint c;
        c.pos_min = 0; c.pos_max = 0; c.n_tokens = 0;
        CHECK(!c.spill(path), "control: spilling an empty checkpoint is refused");
        auto d = make_ckpt();
        CHECK(!d.spill(""), "control: spilling to an empty path is refused");
        CHECK(!d.data_tgt.empty() && !d.spilled(), "control: a refused spill keeps the data resident");
    }

    // ---- control: an unwritable destination keeps the data and leaves no file ----------------------
    {
        auto c = make_ckpt();
        const std::string bad = (dir / "no-such-dir" / "ckpt.bin").string();
        CHECK(!c.spill(bad), "control: spill into a missing directory fails");
        CHECK(!c.data_tgt.empty() && !c.spilled() && c.spill_path.empty(), "control: failed spill leaves data resident, no spill_path");
        CHECK(!exists(bad), "control: failed spill leaves no file");
    }

    // ---- reload refuses a file that is not this entry's -----------------------------------------
    {
        auto c = make_ckpt();
        CHECK(c.spill(path), "spill (for the mismatch controls)");
        c.pos_max = 131; // the entry no longer matches what was written
        CHECK(!c.reload(), "control: positions that differ from the file are refused");
        CHECK(!exists(path), "control: a refused file is deleted");
        CHECK(c.spill_path.empty() && c.data_tgt.empty(), "control: a refused reload leaves a clean empty entry");
    }

    // ---- truncated / corrupt / missing files ------------------------------------------------------
    {
        auto c = make_ckpt();
        CHECK(c.spill(path), "spill (for the truncation control)");
        std::filesystem::resize_file(path, std::filesystem::file_size(path) / 2);
        CHECK(!c.reload(), "control: a truncated file is refused");
        CHECK(!exists(path), "control: the truncated file is deleted");

        auto d = make_ckpt();
        CHECK(d.spill(path), "spill (for the bad-magic control)");
        {
            std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(0);
            const char junk[4] = { 'X', 'X', 'X', 'X' };
            f.write(junk, 4);
        }
        CHECK(!d.reload(), "control: a wrong magic is refused");

        auto e = make_ckpt();
        CHECK(e.spill(path), "spill (for the missing-file control)");
        std::filesystem::remove(path);
        CHECK(!e.reload(), "control: a missing file is refused, no crash");
        CHECK(e.spill_path.empty(), "control: a missing file clears spill_path");

        auto f = make_ckpt();
        CHECK(!f.reload(), "control: reload of a never-spilled entry is refused");
        CHECK(!f.data_tgt.empty(), "control: reload of a never-spilled entry leaves its data alone");
    }

    // ---- a re-spill after reload rewrites the same file ---------------------------------------------
    {
        auto c = make_ckpt();
        CHECK(c.spill(path), "first spill");
        CHECK(c.reload(), "reload");
        c.data_tgt[0] ^= 0xff; // the state changed while it was resident
        const uint8_t changed = c.data_tgt[0];
        CHECK(c.spill(path), "second spill to the same path");
        CHECK(c.reload() && c.data_tgt[0] == changed, "the second spill's bytes are what comes back");
        c.remove_spill_file();
    }

    // ---- NVMe tier blobs (--cache-disk-dir): spill_tier() / reload() --------------------------------------------------------
    // Row S69-TIERED-CKPT-BLOBS-INTO-DISK-TIER-2026-10-05-01. Same shape as the file tests above, each with a control.
    const std::string tdir = (dir / "tier").string();
    auto new_tier = [&](size_t cap = 8u << 20, size_t budget = 0) {
        auto t = std::make_shared<common_disk_tier>(tdir, cap, budget);
        int n; std::string err;
        t->init(n, err);
        return t;
    };
    auto n_in = [&](const std::string & d) { size_t n = 0; for (const auto & e : std::filesystem::directory_iterator(d)) { (void) e; n++; } return n; };
    {
        auto tier = new_tier();
        auto c = make_ckpt();
        const auto tgt = c.data_tgt, dft = c.data_dft, spec = c.data_spec;
        CHECK(c.spill_tier(tier), "spill_tier succeeds");
        CHECK(c.spilled() && c.empty() && c.disk && c.spill_path.empty(), "spilled() is true through the blob; no file path involved");
        CHECK(c.data_tgt.empty() && c.data_dft.empty() && c.data_spec.empty(), "memory is freed");
        CHECK(c.size() == 0, "size() is the resident bytes: 0 once spilled (this is what --cache-ram counts)");
        CHECK(c.pos_min == 7 && c.pos_max == 130 && c.n_tokens == 123, "positions stay in memory");
        CHECK(tier->stats().n_put_ckpt == 1 && n_in(tdir) == 1, "one checkpoint blob in the tier");
        CHECK(c.reload(), "reload succeeds");
        CHECK(c.data_tgt == tgt && c.data_dft == dft && c.data_spec == spec, "all three vectors come back byte-identical");
        CHECK(!c.spilled() && c.disk, "resident again, and the blob is still referenced");

        // spilling it again costs no write: the blob that holds exactly this data is still there
        const auto written = tier->stats().bytes_written;
        CHECK(c.spill_tier(tier) && tier->stats().bytes_written == written, "a second spill_tier writes nothing (the blob is reused)");
        CHECK(c.spilled() && c.data_tgt.empty(), "and the memory is freed again");
        // control: after the data changed the old blob must not be reused
        CHECK(c.reload(), "reload for the change control");
        c.data_tgt[0] ^= 0xff;
        c.disk.reset();                                   // what update_tgt() does when the data is rewritten
        CHECK(c.spill_tier(tier) && tier->stats().bytes_written > written, "control: changed data is written as a new blob");
        CHECK(c.reload() && c.data_tgt[0] == (uint8_t) (tgt[0] ^ 0xff), "and that is what comes back");
    }

    // ---- copies share the blob; the last release erases it -------------------------------------------------------------------
    {
        auto tier = new_tier();
        auto c = make_ckpt();
        CHECK(c.spill_tier(tier), "spill_tier");
        auto copy = c;                                    // server_prompt::clone() / a cache entry's copy
        CHECK(copy.spilled() && copy.disk == c.disk, "the copy shares the reference");
        { common_prompt_checkpoint dropped = std::move(copy); }   // a slot drops its checkpoint
        CHECK(tier->has(c.disk->id()) && n_in(tdir) == 1, "control: one holder left, the blob stays");
        CHECK(c.reload(), "and the remaining holder can still read it");
        c = common_prompt_checkpoint();                   // the last holder lets go
        CHECK(n_in(tdir) == 0 && tier->used_bytes() == 0, "the last release erased the blob");
    }

    // ---- a blob that fails its checks is a failed reload, never wrong bytes --------------------------------------------------
    {
        auto tier = new_tier();
        auto c = make_ckpt();
        CHECK(c.spill_tier(tier), "spill_tier (for the corruption control)");
        const std::string file = tier->path_of(c.disk->id());
        { std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary); f.seekg(1000); char ch = 0; f.read(&ch, 1); ch ^= 0x21; f.seekp(1000); f.write(&ch, 1); }
        CHECK(!c.reload(), "control: a flipped byte in the blob makes reload fail");
        CHECK(c.data_tgt.empty() && c.data_dft.empty() && c.data_spec.empty(), "control: no bytes were handed out");
        CHECK(!c.spilled() && !c.disk, "control: the reference was cleared, so the caller reprocesses the prompt");
        CHECK(tier->stats().n_checksum_failed == 1 && !std::filesystem::exists(file), "control: counted by the tier and deleted");

        auto d = make_ckpt();
        CHECK(d.spill_tier(tier), "spill_tier (for the tag control)");
        d.pos_max = 131;                                  // this checkpoint is no longer the one the blob was written for
        CHECK(!d.reload() && n_in(tdir) == 0, "control: a checkpoint whose positions differ gets a different tag, is refused and the blob dropped");

        auto e = make_ckpt();
        CHECK(e.spill_tier(tier), "spill_tier (for the missing-file control)");
        std::filesystem::remove(tier->path_of(e.disk->id()));
        CHECK(!e.reload() && !e.disk, "control: a missing blob is a failed reload, no crash");

        auto f = make_ckpt();
        CHECK(!f.reload() && !f.data_tgt.empty(), "control: reload of a checkpoint that was never spilled leaves it alone");
    }

    // ---- the tier may refuse: the data stays resident and nothing is half-written -----------------------------------------------
    {
        auto tiny = new_tier(1024);                       // the cap is smaller than the checkpoint
        auto c = make_ckpt();
        CHECK(!c.spill_tier(tiny), "a checkpoint over the cap is refused");
        CHECK(!c.data_tgt.empty() && !c.spilled() && !c.disk, "control: it stays resident, no reference");
        CHECK(tiny->stats().n_refused_cap == 1 && n_in(tdir) == 0, "control: counted, no file left");
        common_prompt_checkpoint none;
        none.pos_min = 0; none.pos_max = 0; none.n_tokens = 0;
        CHECK(!none.spill_tier(tiny), "control: nothing resident, nothing to spill");
        CHECK(!c.spill_tier(nullptr), "control: no tier, no spill");
    }

    // ---- adopt_spill_file(): a copy takes the bytes of a slot's file and never owns the file --------------------------------------
    {
        auto c = make_ckpt();
        const auto tgt = c.data_tgt, dft = c.data_dft, spec = c.data_spec;
        CHECK(c.spill(path), "slot checkpoint spilled to a file");
        auto copy = c;                                    // the cache entry's copy: same spill_path
        CHECK(copy.adopt_spill_file(), "the copy adopts the bytes");
        CHECK(copy.data_tgt == tgt && copy.data_dft == dft && copy.data_spec == spec && copy.spill_path.empty(), "it holds the data and no path");
        CHECK(exists(path) && c.spilled(), "the slot's file and the slot's checkpoint are untouched");
        CHECK(c.reload() && c.data_tgt == tgt, "the slot can still reload its own file");
        c.remove_spill_file();

        auto d = make_ckpt();
        CHECK(d.spill(path), "spill (for the bad-file adoption control)");
        auto dcopy = d;
        std::filesystem::resize_file(path, std::filesystem::file_size(path) / 2);
        CHECK(!dcopy.adopt_spill_file() && dcopy.spill_path.empty() && dcopy.data_tgt.empty(), "control: a bad file is not adopted and the copy lets go of the path");
        CHECK(exists(path), "control: the bad file is NOT deleted by the copy (reload() would, adopt must not)");
        d.remove_spill_file();

        common_prompt_checkpoint resident = make_ckpt();
        CHECK(!resident.adopt_spill_file() && !resident.data_tgt.empty(), "control: a resident checkpoint has nothing to adopt");
    }

    std::filesystem::remove_all(dir);
    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
