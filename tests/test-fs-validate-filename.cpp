// fs_validate_filename must reject the names Windows resolves to a DEVICE (CON, PRN, AUX, NUL, COM1-9, LPT1-9, CONIN$, CONOUT$): a file name
// whose stem before the first '.' is one of them opens the device whatever its extension, so a /slots save named "NUL" or "nul.bin" opens
// the null device instead of creating a file. Reference behaviour: Strata serve/server.py slot_filename_problem (MIT, read; this is a re-implementation),
// Microsoft "Naming Files, Paths, and Namespaces". The existing rules are pinned too so a device-name edit cannot loosen them.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "common.h"

#include <cstdio>
#include <string>

static int fails = 0;

static void expect(const std::string & name, bool subdirs, bool want, const char * why) {
    const bool got = fs_validate_filename(name, subdirs);
    if (got != want) {
        fprintf(stderr, "FAIL %-6s subdirs=%d %-22s -> %d (want %d)  %s\n", want ? "accept" : "reject", (int) subdirs, name.c_str(), (int) got, (int) want, why);
        fails++;
    }
}

int main() {
    // device names, any case, any extension, with trailing spaces before the extension
    for (const char * n : { "NUL", "nul", "Nul", "CON", "prn", "AUX", "nul.txt", "NUL.tar.gz", "con.bin", "COM1", "com9", "LPT1", "lpt9.log",
                            "nul .txt", "CONIN$", "conout$", "COM\xC2\xB9", "LPT\xC2\xB2.x", "com\xC2\xB3" }) {
        expect(n, false, false, "Windows device name");
    }
    // each path component counts when sub-directories are allowed
    for (const char * n : { "dir/NUL", "NUL/x", "a\\con.txt", "ok/COM3/y" }) {
        expect(n, true, false, "device name as a path component");
    }
    // names that merely start with or contain a device name are ordinary files
    for (const char * n : { "nullable", "console.txt", "com10", "com0", "lpt", "lpt0", "auxiliary", "prn1", "a.nul", "slot_0.bin", "COM", "conin", "xcon.txt", "cache.bin" }) {
        expect(n, false, true, "not a device name");
    }
    expect("dir/slot_0.bin", true, true, "plain sub-directory path");
    // existing rules stay
    expect("", false, false, "empty");
    expect(".", false, false, "dot");
    expect("..", false, false, "dotdot");
    expect("a:b", false, false, "colon / stream");
    expect("a/b", false, false, "separator without allow_subdirs");
    expect("trailing.", false, false, "trailing dot");
    expect("trailing ", false, false, "trailing space");
    expect(" leading", false, false, "leading space");
    expect(std::string(256, 'a'), false, false, "too long");

    if (fails) {
        fprintf(stderr, "%d fs_validate_filename check(s) FAILED\n", fails);
        return 1;
    }
    printf("all fs_validate_filename checks passed\n");
    return 0;
}
