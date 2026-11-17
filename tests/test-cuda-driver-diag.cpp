// CUDA init failure diagnostic: mock an older driver, the message must name BOTH versions (driver and toolkit) and the update/PTX hint.
// Row S69-ENGREV-JAN-CUDA-DRIVER-TOOLKIT-DIAG-2026-10-27-01. CPU only (the helper header has no CUDA dependency).
// BEFORE: ggml_cuda_init logged only cudaGetErrorString(err) -- no versions. AFTER: this test passes against the helper ggml_cuda_init now calls.
#include "../ggml/src/ggml-cuda/driver-diag.h"

#include <cstdio>
#include <cstring>

static int failures = 0;
static void check(const char * name, bool ok, const char * msg) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) { printf("   msg: %s\n", msg); failures++; }
}

int main() {
    char buf[512];

    // older driver (CUDA 11.8 = 11080) under a 12.4 toolkit (12040), insufficient-driver error 35
    ggml_cuda_driver_diag(buf, sizeof buf, 35, "CUDA driver version is insufficient for CUDA runtime version", 11080, 12040);
    check("older driver names the driver version 11.8", strstr(buf, "11.8") != nullptr, buf);
    check("older driver names the toolkit version 12.4", strstr(buf, "12.4") != nullptr, buf);
    check("older driver carries the update-driver hint", strstr(buf, "update the NVIDIA driver") != nullptr, buf);
    check("older driver carries the PTX JIT hint", strstr(buf, "PTX") != nullptr, buf);
    check("older driver keeps the CUDA error text", strstr(buf, "insufficient") != nullptr, buf);

    // driver version unreadable (0), error 35: still names the toolkit
    ggml_cuda_driver_diag(buf, sizeof buf, 35, "insufficient", 0, 12040);
    check("unreadable driver still names the toolkit 12.4", strstr(buf, "12.4") != nullptr, buf);

    // CONTROL: a current driver and an unrelated error must NOT claim the driver is too old
    ggml_cuda_driver_diag(buf, sizeof buf, 2, "out of memory", 12090, 12040);
    check("CONTROL newer driver does not blame the driver", strstr(buf, "update the NVIDIA driver") == nullptr, buf);
    check("CONTROL newer driver still shows both versions", strstr(buf, "12.9") != nullptr && strstr(buf, "12.4") != nullptr, buf);

    // no device
    ggml_cuda_driver_diag(buf, sizeof buf, 100, "no CUDA-capable device is detected", 12040, 12040);
    check("no device case says so", strstr(buf, "no CUDA device") != nullptr, buf);

    // truncation safe
    char tiny[16];
    int n = ggml_cuda_driver_diag(tiny, sizeof tiny, 35, "x", 11080, 12040);
    check("tiny buffer is NUL terminated", tiny[sizeof tiny - 1] == '\0' && n > 0, tiny);

    printf(failures ? "FAILED\n" : "ALL PASS\n");
    return failures ? 1 : 0;
}
