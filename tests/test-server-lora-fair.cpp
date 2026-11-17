// server_lora_fair_pick: fair decode-group leader rotation across lora configs (tools/server/server-lora-fair.h).
// BEFORE (leader = first generating slot): group of slot 0 led every step, other group starved.
// AFTER: leaders alternate in slices; a lone group is unchanged; a vanished group is skipped.
#include "server-lora-fair.h"

#include <cstdio>
#include <map>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

int main() {
    // slots 0,1 -> adapter A (fp 11), slots 2,3 -> adapter B (fp 22)
    const std::vector<uint64_t> two = { 11, 11, 22, 22 };

    {   // single group / single slot: index 0, no rotation
        server_lora_fair_pick p(2);
        CHECK(p.pick({}) == 0);
        CHECK(p.pick({ 5 }) == 0);
        CHECK(p.pick({ 5, 5, 5 }) == 0);
    }
    {   // slice 2: A A B B A A B B ; B gets the lead although slot 0 is in A
        server_lora_fair_pick p(2);
        std::vector<uint64_t> lead;
        for (int i = 0; i < 8; i++) lead.push_back(two[p.pick(two)]);
        const std::vector<uint64_t> want = { 11, 11, 22, 22, 11, 11, 22, 22 };
        CHECK(lead == want);
    }
    {   // slice 1: strict alternation, and no group gets more than ceil share over 100 steps
        server_lora_fair_pick p(1);
        std::map<uint64_t, int> n;
        for (int i = 0; i < 100; i++) n[two[p.pick(two)]]++;
        CHECK(n[11] == 50 && n[22] == 50);
    }
    {   // three groups share evenly
        server_lora_fair_pick p(1);
        const std::vector<uint64_t> three = { 1, 2, 3, 3 };
        std::map<uint64_t, int> n;
        for (int i = 0; i < 90; i++) n[three[p.pick(three)]]++;
        CHECK(n[1] == 30 && n[2] == 30 && n[3] == 30);
    }
    {   // group B finishes mid-way: A leads from then on; B coming back is served again
        server_lora_fair_pick p(1);
        (void) p.pick(two);
        (void) p.pick(two);
        const std::vector<uint64_t> onlyA = { 11, 11 };
        for (int i = 0; i < 5; i++) CHECK(onlyA[p.pick(onlyA)] == 11);
        int b = 0;
        for (int i = 0; i < 6; i++) b += two[p.pick(two)] == 22;
        CHECK(b == 3);
    }
    {   // the pick is an index into the candidate list: group at the back is reachable
        server_lora_fair_pick p(1);
        bool saw_back = false;
        for (int i = 0; i < 4; i++) saw_back = saw_back || p.pick(two) >= 2;
        CHECK(saw_back);
    }

    printf(fails ? "test-server-lora-fair: %d FAILED\n" : "test-server-lora-fair: PASS\n", fails);
    return fails ? 1 : 0;
}
