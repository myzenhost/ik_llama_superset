#pragma once
// GGML_BACKEND_DL: give ggml-base this CPU variant's definitions of the functions in ../ggml-cpu-syms.inc.
// Called first thing in ggml_backend_cpu_reg(). Returns the number bound (0 in a non-DL build, where the
// file is not compiled and the call is preprocessed out). Aborts on a build mismatch.
#ifdef __cplusplus
extern "C" {
#endif

int ggml_cpu_syms_bind_to_base(void);

#ifdef __cplusplus
}
#endif
