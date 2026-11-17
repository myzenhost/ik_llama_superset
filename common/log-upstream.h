// log-upstream.h -- give a file ported from upstream llama.cpp upstream's meaning of LOG(...).
//
// Include it LAST (after every other header) in such a file. Deliberately no include guard: its
// whole job is to re-assert the definition after anything else may have redefined LOG.
//
// WHY. This fork keeps two logging systems in common/log.h. Upstream's LOG(...) -- "print to stdout
// at output level" -- is commented out there (log.h:119) because it clashed with the fork's legacy
// LOG(...), which is the old LOG_IMPL tee: it writes to a log FILE (llama.<n>.log) or, when that
// target is disabled, nowhere. Every upstream-ported caller of bare LOG( therefore lost its output
// silently. MEASURED 2026-09-26: llama-perplexity's per-chunk "[1]8.12,[2]..." values never
// appeared in any log (0 matches) although tools/perplexity/perplexity.cpp emits them at :434/:633,
// and the ETA line was cut off mid-sentence; llama-eval-callback printed zero tensors while gdb
// showed common_debug_cb_eval being called. tools/completion/completion.cpp:710 prints generated
// text with LOG("%s", ...). 18 files with bare LOG( exist upstream at the same path and NOT in the
// ik legacy tree (temp/_log_origin.py); those are the files that include this header. Legacy callers
// (examples/main, examples/infill, common/chat-parser*, ...) keep the file-log LOG untouched.
//
// The bodies of the ported files stay byte-identical to upstream, so future merges do not churn.

#include "log.h"

#undef LOG
#define LOG(...) LOG_TMPL(GGML_LOG_LEVEL_NONE, LOG_LEVEL_OUTPUT, __VA_ARGS__)
