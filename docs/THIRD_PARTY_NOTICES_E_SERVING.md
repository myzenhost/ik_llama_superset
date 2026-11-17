# Third-party attribution for the e-serving-1005 branch

Append-only. One entry per adapted source. Permissive sources only here; copyleft sources are
re-implemented from a written spec and recorded in the pick row, not adapted.

## Qualcomm GenieX / nexa-sdk llama.cpp patches (BSD-3-Clause)

Source: https://github.com/NexaAI/nexa-sdk @ 7a0ff49c4e6ce119a069128ef66d6d0ffbe33a95, directory
`sdk/patches/`. Copyright 2024-2026 Qualcomm Technologies, Inc. BSD-3-Clause.

Behaviour adapted (patch files read in full):

- `llama-opencl-adreno-cpy-constreg.patch`: the volatile local-size round trip in `add.cl` and
  `cpy.cl`, and the Adreno-643-only `ADRENO_643_Q4K_CONSTREG_WORKAROUND` define for the Q4_K
  GEMV kernel (applied to this fork's own kernel, which has a different stride expression).
- `llama-opencl-adreno-xmem-gemm-off.patch`: Adreno XMEM GEMM stays opt-in; this fork already was,
  and now also treats `GGML_OPENCL_ADRENO_XMEM_GEMM=0` as off.
- `llama-hexagon-power-mode.patch`, `llama-hexagon-power-mode-setter.patch`,
  `llama-hexagon-release-sessions.patch`: the HTP power-mode enum, the per-mode DCVS/HMX vote
  table, `GGML_HEXAGON_POWER_MODE`, the FastRPC `start()` signature change, and the
  set-power-mode / release-sessions / reacquire-sessions registry entry points (re-structured for
  this fork, where the device context is the session object itself).

Redistribution notice (BSD-3-Clause): redistributions of source code must retain the above
copyright notice, this list of conditions and the following disclaimer; neither the name of
Qualcomm Technologies nor the names of its contributors may be used to endorse or promote
products derived from this software without specific prior written permission. THIS SOFTWARE IS
PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
