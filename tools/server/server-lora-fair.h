#pragma once

// Fair decode-group rotation across LoRA configurations (row S69-LORAX-PICK-ON-DEMAND-ADAPTER-POOL-2026-09-29-01, absorbed
// S69-ENGREV-LORAX-ADAPTER-FAIR-DECODE-GROUP-ROTATION-2026-10-27-01).
//
// LoRA is context-wide, so one decode batch can only hold slots whose lora lists are equal (server_slot::can_batch_with). The group is
// led by the first generating slot, so with two groups the lower-numbered slot's group is picked EVERY step and the other group waits
// until the first one finishes (head-of-line starvation). This helper picks the leading group by round-robin with a time slice,
// using lora_fair_scheduler (common/lora-adapter-pool.h) keyed by lora_config_fingerprint.
//
// Pure policy: no llama, no slots. The caller passes one fingerprint per candidate slot, in slot order, and gets back the index of the
// candidate that should lead the batch.

#include "lora-adapter-pool.h"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

class server_lora_fair_pick {
public:
    // slice_quanta: consecutive decode steps one group may lead while another group is waiting
    explicit server_lora_fair_pick(size_t slice_quanta = 8) : sched_(slice_quanta) {}

    // returns the index (into `fps`) of the slot that leads this step's batch; 0 when there is at most one distinct group
    size_t pick(const std::vector<uint64_t> & fps) {
        if (fps.size() < 2) {
            return 0;
        }
        std::unordered_set<uint64_t> present;
        for (uint64_t f : fps) {
            present.insert(f);
        }
        if (present.size() < 2) {
            // nothing to rotate between; forget stale entries so a later second group starts fresh
            return 0;
        }
        for (uint64_t f : present) {
            if (!pending_.count(f)) {
                sched_.enqueue(key(f), 0);
                pending_.insert(f);
            }
        }
        // entries for groups that have since gone away are popped and dropped
        for (size_t guard = sched_.waiting() + 1; guard > 0; guard--) {
            auto p = sched_.pick();
            if (!p) {
                break;
            }
            const uint64_t f = (uint64_t) std::stoull(p->first);
            pending_.erase(f);
            if (present.count(f)) {
                for (size_t i = 0; i < fps.size(); i++) {
                    if (fps[i] == f) {
                        return i;
                    }
                }
            }
        }
        return 0;
    }

private:
    static std::string key(uint64_t f) { return std::to_string(f); }

    lora_fair_scheduler          sched_;
    std::unordered_set<uint64_t> pending_;
};
