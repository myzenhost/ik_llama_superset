#pragma once

// Router reload guard (--reload-allow-empty). Row S69-ENGREV-TFSERVING-RELOAD-EMPTY-GUARD-2026-10-27-01.
// Design from tensorflow/serving (Apache-2.0): file_system_storage_path_source.cc 253-262 skips emitting when the
// source path is missing, and 450-454 refuses to unload every version of a servable; behaviour only, no code copied.
//
// A reload diffs the preset sources against the running router. A preset file that reads back EMPTY (truncated
// mid-write, an editor swap file, a deploy that wrote nothing yet) parses to zero presets without an error, so the
// diff would see every running model as removed and unload all of them. When the new source set is empty while
// models are running, the reload is refused: the mapping is kept and a warning is logged. --reload-allow-empty
// restores the old behaviour for an operator who really means "remove everything".
//
// Pure (no router state) so tests/test-server-models-reload-guard.cpp can drive it.

#include <cstddef>

// true when load_models() must keep the current mapping instead of reconciling against `n_final_presets`
static inline bool server_models_reload_refuse_empty(size_t n_final_presets, size_t n_running_models, bool allow_empty) {
    return !allow_empty && n_final_presets == 0 && n_running_models > 0;
}
