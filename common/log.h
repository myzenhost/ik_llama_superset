#pragma once

#include "ggml.h" // for ggml_log_level
#include <chrono>
#include <cstring>
#include <sstream>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <cinttypes>

#define LOG_CLR_TO_EOL  "\033[K\r"
#define LOG_COL_DEFAULT "\033[0m"
#define LOG_COL_BOLD    "\033[1m"
#define LOG_COL_RED     "\033[31m"
#define LOG_COL_GREEN   "\033[32m"
#define LOG_COL_YELLOW  "\033[33m"
#define LOG_COL_BLUE    "\033[34m"
#define LOG_COL_MAGENTA "\033[35m"
#define LOG_COL_CYAN    "\033[36m"
#define LOG_COL_WHITE   "\033[37m"

#ifndef __GNUC__
#    define LOG_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__) && !defined(__clang__)
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#define LOG_LEVEL_DEBUG  5
#define LOG_LEVEL_TRACE  4
#define LOG_LEVEL_INFO   3
#define LOG_LEVEL_WARN   2
#define LOG_LEVEL_ERROR  1
#define LOG_LEVEL_OUTPUT 0 // output data from tools

#define LOG_DEFAULT_DEBUG LOG_LEVEL_DEBUG
#define LOG_DEFAULT_LLAMA LOG_LEVEL_INFO

enum log_colors {
    LOG_COLORS_AUTO     = -1,
    LOG_COLORS_DISABLED = 0,
    LOG_COLORS_ENABLED  = 1,
};

// needed by the LOG_TMPL macro to avoid computing log arguments if the verbosity lower
// set via common_log_set_verbosity()
int  common_log_get_verbosity_thold(void);

void common_log_set_verbosity_thold(int verbosity); // not thread-safe

bool common_log_get_jsonl(void);

void common_log_set_jsonl(bool jsonl); // not thread-safe

int common_log_get_verbosity(enum ggml_log_level level);

void common_log_default_callback(enum ggml_log_level level, const char * text, void * user_data);

// the common_log uses an internal worker thread to print/write log messages
// when the worker thread is paused, incoming log messages are discarded
struct common_log;

struct common_log * common_log_init();

// Singleton, intentionally leaked to avoid Windows teardown hangs.
// Its first call registers an atexit drain (common_log_drain, 1 s), so a normal exit() or a return
// from main prints what is still queued. _exit(), abort() and a killed process still lose it.
struct common_log * common_log_main();

// wait up to timeout_ms until every queued message is printed; false if the deadline passed first.
// Never joins the worker thread, so it is safe at DLL teardown, unlike pause().
bool                common_log_drain (struct common_log * log, int64_t timeout_ms);

void                common_log_pause (struct common_log * log); // pause  the worker thread, not thread-safe
void                common_log_resume(struct common_log * log); // resume the worker thread, not thread-safe
void                common_log_free  (struct common_log * log);

LOG_ATTRIBUTE_FORMAT(3, 4)
void common_log_add(struct common_log * log, enum ggml_log_level level, const char * fmt, ...);

// defaults: file = NULL, colors = false, prefix = false, timestamps = false
//
// regular log output:
//
//   ggml_backend_metal_log_allocated_size: allocated buffer, size =  6695.84 MiB, ( 6695.91 / 21845.34)
//   llm_load_tensors: ggml ctx size =    0.27 MiB
//   llm_load_tensors: offloading 32 repeating layers to GPU
//   llm_load_tensors: offloading non-repeating layers to GPU
//
// with prefix = true, timestamps = true, the log output will look like this:
//
//   0.00.035.060 D ggml_backend_metal_log_allocated_size: allocated buffer, size =  6695.84 MiB, ( 6695.91 / 21845.34)
//   0.00.035.064 I llm_load_tensors: ggml ctx size =    0.27 MiB
//   0.00.090.578 I llm_load_tensors: offloading 32 repeating layers to GPU
//   0.00.090.579 I llm_load_tensors: offloading non-repeating layers to GPU
//
// D - debug   (stderr, V = LOG_DEFAULT_DEBUG)
// I - info    (stdout, V = LOG_DEFAULT_INFO)
// W - warning (stderr, V = LOG_DEFAULT_WARN)
// E - error   (stderr, V = LOG_DEFAULT_ERROR)
// O - output  (stdout, V = LOG_DEFAULT_OUTPUT)
//

void common_log_set_file      (struct common_log * log, const char * file); // not thread-safe
void common_log_set_stdo      (struct common_log * log, bool enabled);      // whether to output to stdout/stderr
void common_log_set_colors    (struct common_log * log, log_colors colors); // not thread-safe
bool common_log_get_colors    (struct common_log * log);                    // whether colors are enabled
void common_log_set_prefix    (struct common_log * log, bool prefix);       // whether to output prefix to each log
void common_log_set_timestamps(struct common_log * log, bool timestamps);   // whether to output timestamps in the prefix
void common_log_flush         (struct common_log * log);                    // flush all pending log messages

// Ported from the pre-merge fork's examples/server/server.cpp:62,470 + server-common.cpp:26-65 @
// 35f0cac75 (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, item 4).
// The fork had its own separate server_log()/server_log_json global for server.cpp only; this split
// server's logging (SLT_TRC/SRV_TRC/LOG_INF/...) all goes through the ONE shared common_log every
// binary uses, so the setting lives here -- like its --log-colors/--log-prefix/--log-timestamps
// siblings, --log-format is registered generically (not restricted to the server binary), and every
// binary defaults to false (today's plain-text output), so this is a purely additive, opt-in change.
// Renders {"timestamp","level","msg"} as one JSON object per line; does not reproduce the fork's
// "function"/"line"/"tid" fields, which by this point in the pipeline are already folded into the
// formatted message text (or would need new common_log_entry fields to carry the calling site through
// the async queue) -- not attempted here, see the commit message.
void common_log_set_json_format(struct common_log * log, bool json_format); // whether to emit JSON lines instead of the default prefixed text

// helper macros for logging
// use these to avoid computing log arguments if the verbosity of the log is higher than the threshold
//
// for example:
//
//   LOG_DBG("this is a debug message: %d\n", expensive_function());
//
// this will avoid calling expensive_function() if LOG_DEFAULT_DEBUG > common_log_verbosity_thold
//

#define LOG_TMPL(level, verbosity, ...) \
    do { \
        if ((verbosity) <= common_log_get_verbosity_thold()) { \
            common_log_add(common_log_main(), (level), __VA_ARGS__); \
        } \
    } while (0)

//#define LOG(...)             LOG_TMPL(GGML_LOG_LEVEL_NONE, LOG_LEVEL_OUTPUT, __VA_ARGS__) // commented: clashes with the legacy LOG_IMPL-based LOG() kept below for fork backward-compat
#define LOGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_NONE, verbosity,        __VA_ARGS__)

#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_LEVEL_DEBUG,  __VA_ARGS__)
#define LOG_TRC(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_TRACE,  __VA_ARGS__)
#define LOG_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_INFO,   __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  LOG_LEVEL_WARN,   __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, LOG_LEVEL_ERROR,  __VA_ARGS__)
#define LOG_CNT(...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  LOG_LEVEL_INFO,   __VA_ARGS__) // same as INFO

#define LOG_DBGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, verbosity, __VA_ARGS__)
#define LOG_TRCV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_TRACE, verbosity, __VA_ARGS__)
#define LOG_INFV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  verbosity, __VA_ARGS__)
#define LOG_WRNV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  verbosity, __VA_ARGS__)
#define LOG_ERRV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, verbosity, __VA_ARGS__)
#define LOG_CNTV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  verbosity, __VA_ARGS__)

// --------------------------------------------------------------------------------------
// Legacy LOG()/LOG_TEE() macro-based logging API, kept for ik_llama-fork backward
// compatibility: examples/*.cpp and common/common.cpp (log_param_single_parse) still call
// into this. Deliberately kept alongside the common_log-based API above (same policy as
// include/llama.h keeping both llama_sample_* and llama_sampler_* families).
// --------------------------------------------------------------------------------------

// Specifies a log target.
//  default uses log_handler() with "llama.log" log file
//  this can be changed, by defining LOG_TARGET
//  like so:
//
//  #define LOG_TARGET (a valid FILE*)
//  #include "log.h"
//
//  or it can be simply redirected to stdout or stderr
//  like so:
//
//  #define LOG_TARGET stderr
//  #include "log.h"
//
//  The log target can also be redirected to a different function
//  like so:
//
//  #define LOG_TARGET log_handler_different()
//  #include "log.h"
//
//  FILE* log_handler_different()
//  {
//      return stderr;
//  }
//
//  or:
//
//  #define LOG_TARGET log_handler_another_one("somelog.log")
//  #include "log.h"
//
//  FILE* log_handler_another_one(char*filename)
//  {
//      static FILE* logfile = nullptr;
//      (...)
//      if( !logfile )
//      {
//          fopen(...)
//      }
//      (...)
//      return logfile
//  }
//
#ifndef LOG_TARGET
    #define LOG_TARGET log_handler()
#endif

#ifndef LOG_TEE_TARGET
    #define LOG_TEE_TARGET stderr
#endif

// Utility for synchronizing log configuration state
//  since std::optional was introduced only in c++17
enum LogTriState
{
    LogTriStateSame,
    LogTriStateFalse,
    LogTriStateTrue
};

// Utility to obtain "pid" like unique process id and use it when creating log files.
inline std::string log_get_pid()
{
   static std::string pid;
   if (pid.empty())
   {
       // std::this_thread::get_id() is the most portable way of obtaining a "process id"
       //  it's not the same as "pid" but is unique enough to solve multiple instances
       //  trying to write to the same log.
       std::stringstream ss;
       ss << std::this_thread::get_id();
       pid = ss.str();
   }

   return pid;
}

// Utility function for generating log file names with unique id based on thread id.
//  invocation with log_filename_generator( "llama", "log" ) creates a string "llama.<number>.log"
//  where the number is a runtime id of the current thread.

#define log_filename_generator(log_file_basename, log_file_extension) log_filename_generator_impl(LogTriStateSame, log_file_basename, log_file_extension)

// INTERNAL, DO NOT USE
inline std::string log_filename_generator_impl(LogTriState multilog, const std::string & log_file_basename, const std::string & log_file_extension)
{
    static bool _multilog = false;

    if (multilog != LogTriStateSame)
    {
        _multilog = multilog == LogTriStateTrue;
    }

    std::stringstream buf;

    buf << log_file_basename;
    if (_multilog)
    {
        buf << ".";
        buf << log_get_pid();
    }
    buf << ".";
    buf << log_file_extension;

    return buf.str();
}

#ifndef LOG_DEFAULT_FILE_NAME
    #define LOG_DEFAULT_FILE_NAME log_filename_generator("llama", "log")
#endif

// Utility for turning #define values into string literals
//  so we can have a define for stderr and
//  we can print "stderr" instead of literal stderr, etc.
#define LOG_STRINGIZE1(s) #s
#define LOG_STRINGIZE(s) LOG_STRINGIZE1(s)

#define LOG_TEE_TARGET_STRING LOG_STRINGIZE(LOG_TEE_TARGET)

// Allows disabling timestamps.
//  in order to disable, define LOG_NO_TIMESTAMPS
//  like so:
//
//  #define LOG_NO_TIMESTAMPS
//  #include "log.h"
//
#ifndef LOG_NO_TIMESTAMPS
    #ifndef _MSC_VER
        #define LOG_TIMESTAMP_FMT "[%" PRIu64 "] "
        #define LOG_TIMESTAMP_VAL , (std::chrono::duration_cast<std::chrono::duration<std::uint64_t>>(std::chrono::system_clock::now().time_since_epoch())).count()
    #else
        #define LOG_TIMESTAMP_FMT "[%" PRIu64 "] "
        #define LOG_TIMESTAMP_VAL , (std::chrono::duration_cast<std::chrono::duration<std::uint64_t>>(std::chrono::system_clock::now().time_since_epoch())).count()
    #endif
#else
    #define LOG_TIMESTAMP_FMT "%s"
    #define LOG_TIMESTAMP_VAL ,""
#endif

#ifdef LOG_TEE_TIMESTAMPS
    #ifndef _MSC_VER
        #define LOG_TEE_TIMESTAMP_FMT "[%" PRIu64 "] "
        #define LOG_TEE_TIMESTAMP_VAL , (std::chrono::duration_cast<std::chrono::duration<std::uint64_t>>(std::chrono::system_clock::now().time_since_epoch())).count()
    #else
        #define LOG_TEE_TIMESTAMP_FMT "[%" PRIu64 "] "
        #define LOG_TEE_TIMESTAMP_VAL , (std::chrono::duration_cast<std::chrono::duration<std::uint64_t>>(std::chrono::system_clock::now().time_since_epoch())).count()
    #endif
#else
    #define LOG_TEE_TIMESTAMP_FMT "%s"
    #define LOG_TEE_TIMESTAMP_VAL ,""
#endif

// Allows disabling file/line/function prefix
//  in order to disable, define LOG_NO_FILE_LINE_FUNCTION
//  like so:
//
//  #define LOG_NO_FILE_LINE_FUNCTION
//  #include "log.h"
//
#ifndef LOG_NO_FILE_LINE_FUNCTION
    #ifndef _MSC_VER
        #define LOG_FLF_FMT "[%24s:%5d][%24s] "
        #define LOG_FLF_VAL , __FILE__, __LINE__, __FUNCTION__
    #else
        #define LOG_FLF_FMT "[%24s:%5ld][%24s] "
        #define LOG_FLF_VAL , __FILE__, (long)__LINE__, __FUNCTION__
    #endif
#else
    #define LOG_FLF_FMT "%s"
    #define LOG_FLF_VAL ,""
#endif

#ifdef LOG_TEE_FILE_LINE_FUNCTION
    #ifndef _MSC_VER
        #define LOG_TEE_FLF_FMT "[%24s:%5d][%24s] "
        #define LOG_TEE_FLF_VAL , __FILE__, __LINE__, __FUNCTION__
    #else
        #define LOG_TEE_FLF_FMT "[%24s:%5ld][%24s] "
        #define LOG_TEE_FLF_VAL , __FILE__, (long)__LINE__, __FUNCTION__
    #endif
#else
    #define LOG_TEE_FLF_FMT "%s"
    #define LOG_TEE_FLF_VAL ,""
#endif

// INTERNAL, DO NOT USE
//  USE LOG() INSTEAD
//
#if !defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
    #define LOG_IMPL(str, ...)                                                                                      \
    do {                                                                                                            \
        if (LOG_TARGET != nullptr)                                                                                  \
        {                                                                                                           \
            fprintf(LOG_TARGET, LOG_TIMESTAMP_FMT LOG_FLF_FMT str "%s" LOG_TIMESTAMP_VAL LOG_FLF_VAL, __VA_ARGS__); \
            fflush(LOG_TARGET);                                                                                     \
        }                                                                                                           \
    } while (0)
#else
    #define LOG_IMPL(str, ...)                                                                                           \
    do {                                                                                                                 \
        if (LOG_TARGET != nullptr)                                                                                       \
        {                                                                                                                \
            fprintf(LOG_TARGET, LOG_TIMESTAMP_FMT LOG_FLF_FMT str "%s" LOG_TIMESTAMP_VAL LOG_FLF_VAL "", ##__VA_ARGS__); \
            fflush(LOG_TARGET);                                                                                          \
        }                                                                                                                \
    } while (0)
#endif

// INTERNAL, DO NOT USE
//  USE LOG_TEE() INSTEAD
//
#if !defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
    #define LOG_TEE_IMPL(str, ...)                                                                                                      \
    do {                                                                                                                                \
        if (LOG_TARGET != nullptr)                                                                                                      \
        {                                                                                                                               \
            fprintf(LOG_TARGET, LOG_TIMESTAMP_FMT LOG_FLF_FMT str "%s" LOG_TIMESTAMP_VAL LOG_FLF_VAL, __VA_ARGS__);                     \
            fflush(LOG_TARGET);                                                                                                         \
        }                                                                                                                               \
        if (LOG_TARGET != nullptr && LOG_TARGET != stdout && LOG_TARGET != stderr && LOG_TEE_TARGET != nullptr)                         \
        {                                                                                                                               \
            fprintf(LOG_TEE_TARGET, LOG_TEE_TIMESTAMP_FMT LOG_TEE_FLF_FMT str "%s" LOG_TEE_TIMESTAMP_VAL LOG_TEE_FLF_VAL, __VA_ARGS__); \
            fflush(LOG_TEE_TARGET);                                                                                                     \
        }                                                                                                                               \
    } while (0)
#else
    #define LOG_TEE_IMPL(str, ...)                                                                                                           \
    do {                                                                                                                                     \
        if (LOG_TARGET != nullptr)                                                                                                           \
        {                                                                                                                                    \
            fprintf(LOG_TARGET, LOG_TIMESTAMP_FMT LOG_FLF_FMT str "%s" LOG_TIMESTAMP_VAL LOG_FLF_VAL "", ##__VA_ARGS__);                     \
            fflush(LOG_TARGET);                                                                                                              \
        }                                                                                                                                    \
        if (LOG_TARGET != nullptr && LOG_TARGET != stdout && LOG_TARGET != stderr && LOG_TEE_TARGET != nullptr)                              \
        {                                                                                                                                    \
            fprintf(LOG_TEE_TARGET, LOG_TEE_TIMESTAMP_FMT LOG_TEE_FLF_FMT str "%s" LOG_TEE_TIMESTAMP_VAL LOG_TEE_FLF_VAL "", ##__VA_ARGS__); \
            fflush(LOG_TEE_TARGET);                                                                                                          \
        }                                                                                                                                    \
    } while (0)
#endif

// The '\0' as a last argument, is a trick to bypass the silly
//  "warning: ISO C++11 requires at least one argument for the "..." in a variadic macro"
//  so we can have a single macro which can be called just like printf.

// Main LOG macro.
//  behaves like printf, and supports arguments the exact same way.
//
#if !defined(_MSC_VER) || defined(__clang__)
    #define LOG(...) LOG_IMPL(__VA_ARGS__, "")
#else
    #define LOG(str, ...) LOG_IMPL("%s" str, "", ##__VA_ARGS__, "")
#endif

// Main TEE macro.
//  does the same as LOG
//  and
//  simultaneously writes stderr.
//
// Secondary target can be changed just like LOG_TARGET
//  by defining LOG_TEE_TARGET
//
#if !defined(_MSC_VER) || defined(__clang__)
    #define LOG_TEE(...) LOG_TEE_IMPL(__VA_ARGS__, "")
#else
    #define LOG_TEE(str, ...) LOG_TEE_IMPL("%s" str, "", ##__VA_ARGS__, "")
#endif

// LOG macro variants with auto endline.
#if !defined(_MSC_VER) || defined(__clang__)
    #define LOGLN(...) LOG_IMPL(__VA_ARGS__, "\n")
    #define LOG_TEELN(...) LOG_TEE_IMPL(__VA_ARGS__, "\n")
#else
    #define LOGLN(str, ...) LOG_IMPL("%s" str, "", ##__VA_ARGS__, "\n")
    #define LOG_TEELN(str, ...) LOG_TEE_IMPL("%s" str, "", ##__VA_ARGS__, "\n")
#endif

// INTERNAL, DO NOT USE
inline FILE *log_handler1_impl(bool change = false, LogTriState append = LogTriStateSame, LogTriState disable = LogTriStateSame, const std::string & filename = LOG_DEFAULT_FILE_NAME, FILE *target = nullptr)
{
    static bool _initialized = false;
    static bool _append = false;
    static bool _disabled = filename.empty() && target == nullptr;
    static std::string log_current_filename{filename};
    static FILE *log_current_target{target};
    static FILE *logfile = nullptr;

    if (change)
    {
        if (append != LogTriStateSame)
        {
            _append = append == LogTriStateTrue;
            return logfile;
        }

        if (disable == LogTriStateTrue)
        {
            // Disable primary target
            _disabled = true;
        }
        // If previously disabled, only enable, and keep previous target
        else if (disable == LogTriStateFalse)
        {
            _disabled = false;
        }
        // Otherwise, process the arguments
        else if (log_current_filename != filename || log_current_target != target)
        {
            _initialized = false;
        }
    }

    if (_disabled)
    {
        // Log is disabled
        return nullptr;
    }

    if (_initialized)
    {
        // with fallback in case something went wrong
        return logfile ? logfile : stderr;
    }

    // do the (re)initialization
    if (target != nullptr)
    {
        if (logfile != nullptr && logfile != stdout && logfile != stderr)
        {
            fclose(logfile);
        }

        log_current_filename = LOG_DEFAULT_FILE_NAME;
        log_current_target = target;

        logfile = target;
    } else {
        if (log_current_filename != filename)
        {
            if (logfile != nullptr && logfile != stdout && logfile != stderr)
            {
                fclose(logfile);
            }
        }

        logfile = fopen(filename.c_str(), _append ? "a" : "w");
    }

    if (!logfile)
    {
        //  Verify whether the file was opened, otherwise fallback to stderr
        logfile = stderr;

        fprintf(stderr, "Failed to open logfile '%s' with error '%s'\n", filename.c_str(), std::strerror(errno));
        fflush(stderr);

        // At this point we let the init flag be to true below, and let the target fallback to stderr
        //  otherwise we would repeatedly fopen() which was already unsuccessful
    }

    _initialized = true;

    return logfile ? logfile : stderr;
}

// INTERNAL, DO NOT USE
inline FILE *log_handler2_impl(bool change = false, LogTriState append = LogTriStateSame, LogTriState disable = LogTriStateSame, FILE *target = nullptr, const std::string & filename = LOG_DEFAULT_FILE_NAME)
{
    return log_handler1_impl(change, append, disable, filename, target);
}

// Disables logs entirely at runtime.
//  Makes LOG() and LOG_TEE() produce no output,
//  until enabled back.
#define log_disable() log_disable_impl()

// INTERNAL, DO NOT USE
inline FILE *log_disable_impl()
{
    return log_handler1_impl(true, LogTriStateSame, LogTriStateTrue);
}

// Enables logs at runtime.
#define log_enable() log_enable_impl()

// INTERNAL, DO NOT USE
inline FILE *log_enable_impl()
{
    return log_handler1_impl(true, LogTriStateSame, LogTriStateFalse);
}

// Sets target fir logs, either by a file name or FILE* pointer (stdout, stderr, or any valid FILE*)
#define log_set_target(target) log_set_target_impl(target)

// INTERNAL, DO NOT USE
inline FILE *log_set_target_impl(const std::string & filename) { return log_handler1_impl(true, LogTriStateSame, LogTriStateSame, filename); }
inline FILE *log_set_target_impl(FILE *target) { return log_handler2_impl(true, LogTriStateSame, LogTriStateSame, target); }

// INTERNAL, DO NOT USE
inline FILE *log_handler() { return log_handler1_impl(); }

// Enable or disable creating separate log files for each run.
//  can ONLY be invoked BEFORE first log use.
#define log_multilog(enable) log_filename_generator_impl((enable) ? LogTriStateTrue : LogTriStateFalse, "", "")
// Enable or disable append mode for log file.
//  can ONLY be invoked BEFORE first log use.
#define log_append(enable) log_append_impl(enable)
// INTERNAL, DO NOT USE
inline FILE *log_append_impl(bool enable)
{
    return log_handler1_impl(true, enable ? LogTriStateTrue : LogTriStateFalse, LogTriStateSame);
}

inline void log_test()
{
    log_disable();
    LOG("01 Hello World to nobody, because logs are disabled!\n");
    log_enable();
    LOG("02 Hello World to default output, which is \"%s\" ( Yaaay, arguments! )!\n", LOG_STRINGIZE(LOG_TARGET));
    LOG_TEE("03 Hello World to **both** default output and " LOG_TEE_TARGET_STRING "!\n");
    log_set_target(stderr);
    LOG("04 Hello World to stderr!\n");
    LOG_TEE("05 Hello World TEE with double printing to stderr prevented!\n");
    log_set_target(LOG_DEFAULT_FILE_NAME);
    LOG("06 Hello World to default log file!\n");
    log_set_target(stdout);
    LOG("07 Hello World to stdout!\n");
    log_set_target(LOG_DEFAULT_FILE_NAME);
    LOG("08 Hello World to default log file again!\n");
    log_disable();
    LOG("09 Hello World _1_ into the void!\n");
    log_enable();
    LOG("10 Hello World back from the void ( you should not see _1_ in the log or the output )!\n");
    log_disable();
    log_set_target("llama.anotherlog.log");
    LOG("11 Hello World _2_ to nobody, new target was selected but logs are still disabled!\n");
    log_enable();
    LOG("12 Hello World this time in a new file ( you should not see _2_ in the log or the output )?\n");
    log_set_target("llama.yetanotherlog.log");
    LOG("13 Hello World this time in yet new file?\n");
    log_set_target(log_filename_generator("llama_autonamed", "log"));
    LOG("14 Hello World in log with generated filename!\n");
#ifdef _MSC_VER
    LOG_TEE("15 Hello msvc TEE without arguments\n");
    LOG_TEE("16 Hello msvc TEE with (%d)(%s) arguments\n", 1, "test");
    LOG_TEELN("17 Hello msvc TEELN without arguments\n");
    LOG_TEELN("18 Hello msvc TEELN with (%d)(%s) arguments\n", 1, "test");
    LOG("19 Hello msvc LOG without arguments\n");
    LOG("20 Hello msvc LOG with (%d)(%s) arguments\n", 1, "test");
    LOGLN("21 Hello msvc LOGLN without arguments\n");
    LOGLN("22 Hello msvc LOGLN with (%d)(%s) arguments\n", 1, "test");
#endif
}

inline bool log_param_single_parse(const std::string & param)
{
    if ( param == "--log-test")
    {
        log_test();
        return true;
    }

    if ( param == "--log-disable")
    {
        log_disable();
        return true;
    }

    if ( param == "--log-enable")
    {
        log_enable();
        return true;
    }

    if (param == "--log-new")
    {
        log_multilog(true);
        return true;
    }

    if (param == "--log-append")
    {
        log_append(true);
        return true;
    }

    return false;
}

inline bool log_param_pair_parse(bool check_but_dont_parse, const std::string & param, const std::string & next = std::string())
{
    if ( param == "--log-file")
    {
        if (!check_but_dont_parse)
        {
            log_set_target(log_filename_generator(next.empty() ? "unnamed" : next, "log"));
        }

        return true;
    }

    return false;
}

inline void log_print_usage()
{
    printf("log options:\n");
    /* format
    printf("  -h, --help            show this help message and exit\n");*/
    /* spacing
    printf("__-param----------------Description\n");*/
    printf("  --log-test            Run simple logging test\n");
    printf("  --log-disable         Disable trace logs\n");
    printf("  --log-enable          Enable trace logs\n");
    printf("  --log-file            Specify a log filename (without extension)\n");
    printf("  --log-new             Create a separate new log file on start. "
                                   "Each log file will have unique name: \"<name>.<ID>.log\"\n");
    printf("  --log-append          Don't truncate the old log file.\n");
    printf("\n");
}

#define log_dump_cmdline(argc, argv) log_dump_cmdline_impl(argc, argv)

// INTERNAL, DO NOT USE
inline void log_dump_cmdline_impl(int argc, char **argv)
{
    std::stringstream buf;
    for (int i = 0; i < argc; ++i)
    {
        if (std::string(argv[i]).find(' ') != std::string::npos)
        {
            buf << " \"" << argv[i] <<"\"";
        } else {
            buf << " " << argv[i];
        }
    }
    LOGLN("Cmd:%s", buf.str().c_str());
}

#define log_tostr(var) log_var_to_string_impl(var).c_str()

inline std::string log_var_to_string_impl(bool var)
{
    return var ? "true" : "false";
}

inline std::string log_var_to_string_impl(std::string var)
{
    return var;
}

inline std::string log_var_to_string_impl(const std::vector<int> & var)
{
    std::stringstream buf;
    buf << "[ ";
    bool first = true;
    for (auto e : var)
    {
        if (first)
        {
            first = false;
        } else {
            buf << ", ";
        }
        buf << std::to_string(e);
    }
    buf << " ]";

    return buf.str();
}

template <typename C, typename T>
inline std::string LOG_TOKENS_TOSTR_PRETTY(const C & ctx, const T & tokens)
{
    std::stringstream buf;
    buf << "[ ";

    bool first = true;
    for (const auto & token : tokens)
    {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, token);

        detokenized.erase(
            std::remove_if(
                detokenized.begin(),
                detokenized.end(),
                [](const unsigned char c) { return !std::isprint(c); }),
            detokenized.end());

        buf
            << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }
    buf << " ]";

    return buf.str();
}

// LOG_BATCH_TOSTR_PRETTY used to live here. Removed (row S69-FORK-SERVER-FEATURES-STRANDED-IN-
// UNBUILT-EXAMPLES-SERVER-2026-09-27-01, item 4, incidental fix): it called
// `llama_token_to_piece(ctx, batch.token[i])` -- a 2-argument, string-returning form that has never
// existed anywhere in this tree (the real llama_token_to_piece takes a vocab, a token, an output
// buffer and a length, and returns int32_t; see include/llama.h). The only two call sites, in
// examples/speculative/speculative.cpp, are commented out. Being an unused template, the broken call
// was never type-checked and never caught -- until the fork-lint decl-call-orphans gate, scanning
// this header for the first time as part of an unrelated change, correctly flagged it as calling a
// symbol the header doesn't declare. Fixing the call to match the real signature would resurrect
// genuinely dead code for no user; deleting it removes the defect instead of relocating it.
#ifdef LOG_DISABLE_LOGS

#undef LOG
#define LOG(...) // dummy stub
#undef LOGLN
#define LOGLN(...) // dummy stub

#undef LOG_TEE
#define LOG_TEE(...) fprintf(stderr, __VA_ARGS__) // convert to normal fprintf

#undef LOG_TEELN
#define LOG_TEELN(...) fprintf(stderr, __VA_ARGS__) // convert to normal fprintf

#undef LOG_DISABLE
#define LOG_DISABLE() // dummy stub

#undef LOG_ENABLE
#define LOG_ENABLE() // dummy stub

#undef LOG_SET_TARGET
#define LOG_SET_TARGET(...) // dummy stub

#undef LOG_DUMP_CMDLINE
#define LOG_DUMP_CMDLINE(...) // dummy stub

#endif // LOG_DISABLE_LOGS

class common_json; // defined in common/json.h

// helper allows different types of json output
// no-op if --log-jsonl is not set
void common_log_add_json(struct common_log * log, const char * type, const common_json & data);

// will only print if --log-jsonl is set
#define LOG_JSON(type, data) \
    do { \
        if (common_log_get_jsonl()) { \
            common_log_add_json(common_log_main(), type, data); \
        } \
    } while (0)
