// Hand-owned (generate_cu_files.py TYPES_MMQ does not name NVFP4; same as mmq-instance-nvfp4_ks.cu).
// Plain NVFP4 MMQ, upstream 84f82e846c #21074. The fork has no DECL_MMQ_CASE_W4A4 (W4A4 prefill is nvfp4-prefill.cu).

#include "../mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_NVFP4);
