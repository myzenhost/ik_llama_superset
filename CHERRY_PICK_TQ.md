# TurboQuant Cherry-Pick Guide
## Files to patch in ik_llama to add TQ1_0 and TQ2_0

### 1. ggml/include/ggml.h
Add after existing type enums (near line ~424):
```c
GGML_TYPE_TQ1_0   = 34,
GGML_TYPE_TQ2_0   = 35,
```

### 2. ggml/src/ggml-common.h
Add ternary quantization block structs (after block_q8_1):
```c
// Ternary quantization

// 1.6875 bpw
typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // 5 elements per byte (3^5 = 243 < 256)
    uint8_t qh[QK_K/64]; // 4 elements per byte
    ggml_half d;
} block_tq1_0;
static_assert(sizeof(block_tq1_0) == sizeof(ggml_half) + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5, "wrong tq1_0 block size/padding");

// 2.0625 bpw
typedef struct {
    uint8_t qs[QK_K/4]; // 2 bits per element
    ggml_half d;
} block_tq2_0;
static_assert(sizeof(block_tq2_0) == sizeof(ggml_half) + QK_K / 4, "wrong tq2_0 block size/padding");
```

### 3. ggml/src/ggml-quants.c
Copy these functions from TurboQuant:
- quantize_row_tq1_0_ref()
- quantize_row_tq2_0_ref()
- dequantize_row_tq1_0()
- dequantize_row_tq2_0()
- quantize_tq1_0()
- quantize_tq2_0()

### 4. ggml/src/ggml.c
Add type_traits entries for TQ1_0 and TQ2_0

### 5. src/llama-model-loader.cpp
Add FTYPE mappings:
```cpp
case LLAMA_FTYPE_MOSTLY_TQ1_0: return "TQ1_0 - 1.69 bpw ternary";
case LLAMA_FTYPE_MOSTLY_TQ2_0: return "TQ2_0 - 2.06 bpw ternary";
```

### 6. src/llama-quant.cpp
Add quantization tool support for TQ types

### 7. ggml/src/ggml-cpu/quants.c + quants.h
Add CPU compute paths

### 8. ggml/src/ggml-cpu/arch/x86/quants.c
Add AVX2 optimized paths

### Source files in TurboQuant repo:
upstream/llama-cpp-turboquant\
