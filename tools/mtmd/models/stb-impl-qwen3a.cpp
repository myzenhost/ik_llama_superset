// Single-TU home for the stb_image_resize2 implementation, for the standalone
// llama-test-qwen3a probe.
//
// clip.cpp calls stbir_resize() from its vision preprocessing path. In the normal build that
// symbol comes from mtmd-helper.cpp, which the probe does not link (it compiles clip.cpp
// directly because the clip_* symbols are internal and not exported from the mtmd library).
// The qwen3a audio path never reaches stbir_resize, but the linker still needs it defined.

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"
