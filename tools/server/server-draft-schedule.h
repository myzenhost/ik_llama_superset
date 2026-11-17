// Draft length by the number of concurrently generating slots (--draft-max-by-slots).
//
// Verifying K drafted tokens per sequence puts K extra rows per sequence in the target batch. With one generating slot the device
// has slack and a long draft wins; with many slots the same K only adds compute-bound work and can lower total throughput. The
// schedule maps "how many slots generate in this step" to the draft budget those slots may use. It only ever LOWERS the budget
// (the lookup is clamped by the budget the slot already had), so per-request speculative.n_max and the context-room limit still win
// downward, and a spec-tuner proposal is clamped by the schedule, not replaced.
//
// Spec string: comma-separated entries, each  LO:K  or  LO-HI:K  (slot counts, inclusive; K >= 0, 0 = no drafting).
//   "1:5,3:2,9:0"       1-2 slots draft 5, 3-8 slots draft 2, 9 or more draft 0
//   "1-4:5,8-16:2"      gap 5-7 carries 5 (the previous K), 17 or more carries 2 (the last K)
// Entries must be written in ascending order, must not overlap, and the first must start at 1.
//
// Header-only, dependency-free; tested by tools/server/tests_cpp/test_server_draft_schedule.cpp.
//
// SOURCE-READ: upstream/aphrodite-engine/aphrodite/v1/spec_decode/dynamic/utils.py:1-134 and
// aphrodite/v1/core/sched/scheduler.py:240-260, 1215-1222 (AGPL-3.0 repo; both files carry a vLLM Apache-2.0 header). Read for
// behaviour, spec in my own words, own structure: no code copied.
// SOURCE-NUANCE (what the row's description did not say):
//   (1) the source SORTS the entries before checking overlap, so unsorted input is accepted there; the row's test wants unsorted
//       refused, and an operator writing "3:2,1:5" has most likely made a mistake, so this port refuses it.
//   (2) the count the source looks up is the number of requests SCHEDULED in the step, prefill requests included; here the caller
//       passes the generating slots that will share the batch (the row's design), so a slot still in prompt processing does not
//       shorten anyone's draft. A caller that wants the other reading passes the larger count.
//   (3) the looked-up K is capped by the configured maximum draft length (min()); here the cap is the slot's own n_draft_max.
//   (4) the source builds a dense array sized to the largest batch and indexes it; a count above the table (more slots than were
//       listed) carries the LAST entry's K, as its tail fill does. Same here, without a size limit to configure.
//   (5) the schedule is only computed when requests were scheduled (len > 0); with none the configured K stays. Here lookup(0)
//       returns -1 ("no opinion") for the same reason.
//   (6) a gap between two listed ranges carries the previous K forward, it is not an error and not 0.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

struct server_draft_schedule {
    struct range {
        int lo = 0; // inclusive slot counts
        int hi = 0;
        int k  = 0;
    };
    std::vector<range> ranges; // validated: ascending, non-overlapping, first lo == 1

    bool empty() const { return ranges.empty(); }

    // Returns false with `err` on a malformed spec; the schedule is then left empty.
    bool parse(const std::string & spec, std::string & err) {
        ranges.clear();
        if (spec.empty()) {
            err = "empty schedule";
            return false;
        }
        std::vector<range> out;
        size_t pos = 0;
        while (pos <= spec.size()) {
            size_t end = spec.find(',', pos);
            if (end == std::string::npos) {
                end = spec.size();
            }
            const std::string item = trim(spec.substr(pos, end - pos));
            pos = end + 1;
            if (item.empty()) {
                err = "empty entry in '" + spec + "'";
                return false;
            }
            const size_t colon = item.find(':');
            if (colon == std::string::npos) {
                err = "entry '" + item + "' needs the form LO:K or LO-HI:K";
                return false;
            }
            const std::string lhs = trim(item.substr(0, colon));
            int k = 0;
            if (!to_int(trim(item.substr(colon + 1)), k) || k < 0) {
                err = "K in '" + item + "' must be an integer >= 0";
                return false;
            }
            range r;
            r.k = k;
            const size_t dash = lhs.find('-');
            if (dash == std::string::npos) {
                if (!to_int(lhs, r.lo) || r.lo < 1) {
                    err = "slot count in '" + item + "' must be an integer >= 1";
                    return false;
                }
                r.hi = -1; // open until the next entry
            } else {
                if (!to_int(trim(lhs.substr(0, dash)), r.lo) || !to_int(trim(lhs.substr(dash + 1)), r.hi) || r.lo < 1 || r.hi < 1) {
                    err = "range in '" + item + "' must be two integers >= 1";
                    return false;
                }
                if (r.lo > r.hi) {
                    err = "range start is above its end in '" + item + "'";
                    return false;
                }
            }
            out.push_back(r);
        }
        if (out.front().lo != 1) {
            err = "the first entry must start at 1 so every slot count has a defined draft length";
            return false;
        }
        // an open entry ends just before the next entry begins
        for (size_t i = 0; i + 1 < out.size(); i++) {
            if (out[i + 1].lo <= out[i].lo) {
                err = "entries must be written in ascending order without repeats (got " + std::to_string(out[i].lo) + " then " + std::to_string(out[i + 1].lo) + ")";
                return false;
            }
            if (out[i].hi < 0) {
                out[i].hi = out[i + 1].lo - 1;
            } else if (out[i + 1].lo <= out[i].hi) {
                err = "entries overlap: " + std::to_string(out[i].lo) + "-" + std::to_string(out[i].hi) + " and " + std::to_string(out[i + 1].lo);
                return false;
            }
        }
        if (out.back().hi < 0) {
            out.back().hi = 1 << 30; // open tail
        }
        ranges = std::move(out);
        return true;
    }

    // The draft budget for `n_slots` generating slots, or -1 when there is no opinion (empty schedule, or no slot generating).
    int lookup(int n_slots) const {
        if (ranges.empty() || n_slots < 1) {
            return -1;
        }
        int k = ranges.front().k;
        for (const auto & r : ranges) {
            if (n_slots < r.lo) {
                return k; // in a gap: the previous K carries
            }
            k = r.k;
            if (n_slots <= r.hi) {
                return k;
            }
        }
        return k; // past the last range: the last K carries
    }

    // Stateful lookup with hysteresis (--draft-max-by-slots-hysteresis H). A LOWER budget applies at once; a HIGHER one (the slot
    // count fell back across a boundary) applies only when the count is H below that boundary, i.e. lookup(n + H) still gives the
    // same K. H <= 0 is the plain lookup. This stops flip-flopping when the slot count oscillates around a range edge.
    // Idea from RunanywhereAI/trnrt@ef581f5 config.py:130-135 (Apache-2.0); read for behaviour, own structure, no code copied.
    int lookup_sticky(int n_slots) {
        const int raw = lookup(n_slots);
        if (raw < 0) {
            applied_k = -1;
            return raw;
        }
        if (hysteresis > 0 && applied_k >= 0 && raw > applied_k && lookup(n_slots + hysteresis) != raw) {
            return applied_k; // rising, but not yet H slots clear of the boundary: keep the lower budget
        }
        applied_k = raw;
        return raw;
    }

    int hysteresis = 0; // slots of margin before a budget may rise again; 0 = off
    int applied_k  = -1; // last budget lookup_sticky returned (-1 = none yet)

    // clamp a slot's own budget by the schedule (never raises it)
    int clamp(int n_draft_max, int n_slots) const {
        const int k = lookup(n_slots);
        return k < 0 ? n_draft_max : std::min(n_draft_max, k);
    }

  private:
    static std::string trim(const std::string & s) {
        size_t a = 0, b = s.size();
        while (a < b && (s[a] == ' ' || s[a] == '\t')) a++;
        while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) b--;
        return s.substr(a, b - a);
    }
    static bool to_int(const std::string & s, int & v) {
        if (s.empty()) {
            return false;
        }
        char * e = nullptr;
        const long x = std::strtol(s.c_str(), &e, 10);
        if (*e != '\0' || x > (1 << 30) || x < -(1 << 30)) {
            return false;
        }
        v = (int) x;
        return true;
    }
};
