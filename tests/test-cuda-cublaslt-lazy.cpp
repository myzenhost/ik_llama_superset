// cuBLASLt must not be a normal (load-time) import of executables that link ggml-cuda
// (row S69-B1019F-STARTUP-COMMIT-TRANSIENT-AND-FIT-CHECK4-REACH-2026-10-26-01).
//
// The NVFP4 prefill GEMM (nvfp4-prefill.cu) calls cublasLt* directly, which put cublasLt64_NN.dll (455 MB in CUDA 13.2)
// into the import table of every such executable, next to cublas64_NN.dll, which imports it too. Measured with a
// debug-event load probe on llama-completion / llama-gguf-split --version: in about half of the runs the same
// cublasLt64_13.dll path is mapped a SECOND time at a private, non-shared base, and the peak commit goes from
// 247-270 MiB to 481 MiB (4 of 4 high runs had the duplicate mapping, 4 of 4 low runs did not). Builds without the
// direct import never showed it. ggml-cuda now links cuBLASLt with /DELAYLOAD, so the executable binds to the copy
// cuBLAS already loaded, on first use.
//
// Check: walk this executable's own PE import directory (the normal, load-time imports). cublas64_* must be listed
// (control: the walk works and this binary really links cuBLAS); cublasLt64_* must not be. A deterministic check of the
// cause, where the duplicate mapping itself only shows up in about half of the runs. The address of
// ggml_backend_cuda_init keeps the whole CUDA backend -- mul_mat and the NVFP4 GEMM that names cublasLt -- in the link,
// the same way the llama-* tools pull it in. Needs no GPU: nothing is called.
//
// Fail-before arm: build without the /DELAYLOAD block in ggml/src/ggml-cuda/CMakeLists.txt -> exit 1.
// Exit: 0 pass, 1 cuBLASLt is a normal import, 2 control failed (cuBLAS not in the import table: cannot judge),
// 77 skip (not a PE image: on ELF/Mach-O builds a DT_NEEDED library maps at start either way and the delay-load switch
// has no counterpart to test).

#include "ggml-cuda.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static bool imports_dll(const char * prefix, char * found, size_t found_size) {
    const auto * base = reinterpret_cast<const unsigned char *>(GetModuleHandleA(nullptr));
    const auto * dos  = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
    const auto * nt   = reinterpret_cast<const IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY & dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (dir.VirtualAddress == 0) {
        return false;
    }
    for (const auto * d = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR *>(base + dir.VirtualAddress); d->Name; ++d) {
        const char * name = reinterpret_cast<const char *>(base + d->Name);
        if (_strnicmp(name, prefix, strlen(prefix)) == 0) {
            snprintf(found, found_size, "%s", name);
            return true;
        }
    }
    return false;
}
#endif

int main() {
#ifndef _WIN32
    printf("SKIP: the check reads the PE import directory\n");
    return 77;
#else
    volatile auto keep = &ggml_backend_cuda_init; // links the CUDA backend in; never called
    (void) keep;

    char cublas[MAX_PATH] = "";
    char cublaslt[MAX_PATH] = "";
    const bool have_cublas   = imports_dll("cublas64_",   cublas,   sizeof(cublas));
    const bool have_cublaslt = imports_dll("cublasLt64_", cublaslt, sizeof(cublaslt));

    printf("control  cublas64_*   in the load-time import table: %s %s\n", have_cublas ? "yes" : "no", cublas);
    printf("check    cublasLt64_* in the load-time import table: %s %s\n", have_cublaslt ? "yes" : "no", cublaslt);
    if (!have_cublas) {
        printf("CONTROL FAILED: cuBLAS is not a load-time import, so this binary cannot show where cuBLASLt is bound\n");
        return 2;
    }
    if (have_cublaslt) {
        printf("FAIL: cuBLASLt is a load-time import; ggml-cuda must link it with /DELAYLOAD\n");
        return 1;
    }
    printf("PASS: cuBLASLt is not a load-time import\n");
    return 0;
#endif
}
