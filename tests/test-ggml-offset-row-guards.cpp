// Guards that turn an out-of-bounds write into an abort, ported from upstream llama.cpp:
//
//   8ae38670 (#30135) ggml_acc_impl: the size_t offset is narrowed into an int32_t op param. An offset
//            >= 2^31 truncates to a negative value; the forward sign-extends it and writes BELOW dst.
//            The fork's ggml_set_impl had lost the same check (upstream has carried it for a long time).
//   d3954b93 (#29575) get_rows_back: a row index outside [0, dst->ne[1]) wrote outside dst.
//
// A guard is an abort, so each case runs in a CHILD process (this binary, "--child <case>") with a
// SIGABRT handler that exits 42. The parent checks the exit code:
//   EXPECT ABORT  : the child must exit 42 (before the fix it builds/computes and exits 0).
//   EXPECT OK     : the child must exit 0 and its numeric check must hold, with or without the fix.
//
// CONTROLS: "abort_control" calls GGML_ABORT directly and must exit 42 -- proves the harness can see an
// abort at all (otherwise every EXPECT ABORT verdict is void). "acc_ok" and "get_rows_back_ok" use valid
// offsets/rows and verify the values, so the guards are shown not to reject legal input.
#include "ggml.h"
#include "ggml-cpu.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>

#ifdef _WIN32
#include <process.h>
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

static void on_abort(int) {
    fflush(stdout);
    fflush(stderr);
    _exit(42);
}

static ggml_context * make_ctx() {
    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, false };
    return ggml_init(ip);
}

static void fill(ggml_tensor * t, float base) {
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        d[i] = base + (float) i;
    }
}

// returns 0 when the case ran to the end and its own check held, 3 when the numeric check failed
static int run_child(const std::string & name) {
    signal(SIGABRT, on_abort);
    ggml_context * ctx = make_ctx();
    int rc = 0;
    if (name == "abort_control") {
        GGML_ABORT("abort_control: deliberate abort");
    } else if (name == "acc_big_offset" || name == "set_big_offset" || name == "acc_ok") {
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        fill(a, 0.0f);
        fill(b, 100.0f);
        // 2^31 + 64 bytes: truncates to a negative int32_t op param
        const size_t big = ((size_t) 1 << 31) + 64;
        if (name == "acc_big_offset") {
            ggml_acc(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], big);
        } else if (name == "set_big_offset") {
            ggml_set(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], big);
        } else {
            ggml_tensor * r = ggml_acc(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], 4 * sizeof(float));
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, r);
            ggml_graph_compute_with_ctx(ctx, gf, 1);
            const float * o = (const float *) r->data;
            for (int i = 0; i < 16; ++i) {
                const float want = (float) i + ((i >= 4 && i < 8) ? 100.0f + (float) (i - 4) : 0.0f);
                if (std::fabs(o[i] - want) > 1e-6f) {
                    fprintf(stderr, "acc_ok: o[%d] = %g, want %g\n", i, o[i], want);
                    rc = 3;
                }
            }
        }
    } else if (name == "get_rows_back_oob" || name == "get_rows_back_ok") {
        ggml_tensor * grad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
        ggml_tensor * like = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
        fill(grad, 1.0f);
        fill(like, 0.0f);
        ((int32_t *) rows->data)[0] = 0;
        // row 3 is one past dst->ne[1] == 3
        ((int32_t *) rows->data)[1] = name == "get_rows_back_oob" ? 3 : 2;
        ggml_tensor * r = ggml_get_rows_back(ctx, grad, rows, like);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        if (name == "get_rows_back_ok") {
            const float * o = (const float *) r->data;
            const float want[12] = { 1, 2, 3, 4,  0, 0, 0, 0,  5, 6, 7, 8 };
            for (int i = 0; i < 12; ++i) {
                if (std::fabs(o[i] - want[i]) > 1e-6f) {
                    fprintf(stderr, "get_rows_back_ok: o[%d] = %g, want %g\n", i, o[i], want[i]);
                    rc = 3;
                }
            }
        }
    } else {
        fprintf(stderr, "unknown case %s\n", name.c_str());
        rc = 4;
    }
    ggml_free(ctx);
    return rc;
}

static int spawn_child(const char * self, const char * name) {
    fflush(stdout);
    fflush(stderr);
#ifdef _WIN32
    char path[MAX_PATH];
    const DWORD n = GetModuleFileNameA(NULL, path, MAX_PATH);
    const char * exe = (n > 0 && n < MAX_PATH) ? path : self;
    const intptr_t rc = _spawnl(_P_WAIT, exe, "\"child\"", "--child", name, (const char *) NULL);
    return (int) rc;
#else
    const pid_t pid = fork();
    if (pid == 0) {
        execl(self, self, "--child", name, (char *) NULL);
        _exit(127);
    }
    int st = 0;
    if (pid < 0 || waitpid(pid, &st, 0) < 0) {
        return -1;
    }
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
#endif
}

int main(int argc, char ** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--child") == 0) {
        return run_child(argv[2]);
    }

    struct tcase { const char * name; int want; const char * role; };
    const tcase cases[] = {
        { "abort_control",     42, "CONTROL (harness sees an abort)" },
        { "acc_ok",             0, "CONTROL (legal offset, values checked)" },
        { "get_rows_back_ok",   0, "CONTROL (legal rows, values checked)" },
        { "acc_big_offset",    42, "GUARD 8ae38670" },
        { "set_big_offset",    42, "GUARD ggml_set_impl offset" },
        { "get_rows_back_oob", 42, "GUARD d3954b93" },
    };

    int n_fail = 0;
    bool controls_ok = true;
    for (const tcase & c : cases) {
        const int got = spawn_child(argv[0], c.name);
        const bool ok = got == c.want;
        printf("%-18s exit %3d want %3d  %s  %s\n", c.name, got, c.want, ok ? "PASS" : "FAIL", c.role);
        if (!ok) {
            n_fail++;
            if (std::strncmp(c.role, "CONTROL", 7) == 0) {
                controls_ok = false;
            }
        }
    }
    if (!controls_ok) {
        printf("CONTROL FAILED: the guard verdicts above are void\n");
        return 2;
    }
    printf("%s: %d failure(s)\n", n_fail ? "FAIL" : "PASS", n_fail);
    return n_fail ? 1 : 0;
}
