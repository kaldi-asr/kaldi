// base/kaldi-error.h

// Copyright 2009-2011  Microsoft Corporation;  Ondrej Glembek;  Lukas Burget;
//                      Saarland University

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_BASE_KALDI_ERROR_H_
#define KALDI_BASE_KALDI_ERROR_H_ 1

#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#include "base/kaldi-types.h"
#include "base/kaldi-utils.h"

/* Important that this file does not depend on any other kaldi headers. */

#if _MSC_VER >= 1900 || (!defined(_MSC_VER) && __cplusplus >= 201103L)
#define KALDI_NOEXCEPT(Predicate) noexcept((Predicate))
#elif defined(__GXX_EXPERIMENTAL_CXX0X__) && \
  (__GNUC__ >= 4 && __GNUC_MINOR__ >= 6)
#define KALDI_NOEXCEPT(Predicate) noexcept((Predicate))
#else
#define KALDI_NOEXCEPT(Predicate)
#endif

#ifdef _MSC_VER
#define __func__ __FUNCTION__
#endif

namespace kaldi {

/// \addtogroup error_group
/// @{

/// This is set by util/parse-options.{h, cc} if you set --verbose = ? option.
extern int32 g_kaldi_verbose_level;

/// This is set by util/parse-options.{h, cc} (from argv[0]) and used (if set)
/// in error reporting code to display the name of the program (this is because
/// in our scripts, we often mix together the stderr of many programs).  it is
/// the base-name of the program (no directory), followed by ':' We don't use
/// std::string, due to the static initialization order fiasco.
extern const char *g_program_name;

inline int32 GetVerboseLevel() { return g_kaldi_verbose_level; }

/// This should be rarely used; command-line programs set the verbose level
/// automatically from ParseOptions.
inline void SetVerboseLevel(int32 i) { g_kaldi_verbose_level = i; }

/// Log message severity and source location info.
struct LogMessageEnvelope {
  enum Severity {
    Error = -2,
    Warning = -1,
    Info = 0,
  };
  // An 'enum Severity' value, or a positive number indicating verbosity level.
  int severity;
  const char *func;
  const char *file;
  int32 line;
};

/// Type of user-provided logging function.
typedef void (*LogHandler)(const LogMessageEnvelope &envelope,
                           const char *message);

/// Set logging handler. If called with a non-NULL function pointer, the
/// function pointed by it is called to send messages to a caller-provided
/// log. If called with NULL pointer, restores default Kaldi error logging to
/// stderr.  SetLogHandler is obviously not thread safe.
LogHandler SetLogHandler(LogHandler);

// Class MessageLogger is invoked from the KALDI_ERR, KALDI_WARN, KALDI_LOG and
// KALDI_LOG macros. It formats the message, then either prints it to stderr or
// passes to the log custom handler if provided, then, in case of the error,
// throws an std::runtime_exception.
//
// Note: we avoid using std::cerr, since it does not guarantee thread safety
// in general, until C++11; even then, in "cerr << a << b", other thread's
// output is allowed to intrude between a and b. fprintf(stderr,...) is
// guaranteed thread-safe, and outputs its formatted string atomically.
class MessageLogger {
public:
  MessageLogger(LogMessageEnvelope::Severity severity, const char *func,
                  const char *file, int32 line);
  ~MessageLogger() KALDI_NOEXCEPT(false);
  inline std::ostream &stream() { return ss_; }
private:
  LogMessageEnvelope envelope_;
  std::ostringstream ss_;
};

// Note on KALDI_ASSERT and KALDI_PARANOID_ASSERT
// The original (simple) version of the code was this
//
// #define KALDI_ASSERT(cond) if (!(cond))
//              kaldi::KaldiAssertFailure_(__func__, __FILE__, __LINE__, #cond);
//
// That worked well, but we were concerned that it
// could potentially cause a performance issue due to failed branch
// prediction (best practice is to have the if branch be the commonly
// taken one).
// Therefore, we decided to move the call into the else{} branch.
// A single block {} around if /else  does not work, because it causes
// syntax error (unmatched else block) in the following code:
//
// if (condition)
//   KALDI_ASSERT(condition2);
// else
//   SomethingElse();
//
// do {} while(0)  -- note there is no semicolon at the end! --- works nicely
// and compilers will be able to optimize the loop away (as the condition
// is always false).
#ifndef NDEBUG
#define KALDI_ASSERT(cond) do { if (cond) (void)0; else \
  ::kaldi::KaldiAssertFailure_(__func__, __FILE__, __LINE__, #cond); } while(0)
#else
#define KALDI_ASSERT(cond) (void)0
#endif
// also see KALDI_COMPILE_TIME_ASSERT, defined in base/kaldi-utils.h,
// and KALDI_ASSERT_IS_INTEGER_TYPE and KALDI_ASSERT_IS_FLOATING_TYPE,
// also defined there.
// some more expensive asserts only checked if this defined
#ifdef KALDI_PARANOID
#define KALDI_PARANOID_ASSERT(cond) do { if (cond) (void)0; else \
  ::kaldi::KaldiAssertFailure_(__func__, __FILE__, __LINE__, #cond); } while(0)
#else
#define KALDI_PARANOID_ASSERT(cond) (void)0
#endif


#define KALDI_ERR \
  ::kaldi::MessageLogger(::kaldi::LogMessageEnvelope::Error, \
                         __func__, __FILE__, __LINE__).stream()
#define KALDI_WARN \
  ::kaldi::MessageLogger(::kaldi::LogMessageEnvelope::Warning, \
                         __func__, __FILE__, __LINE__).stream()
#define KALDI_LOG \
  ::kaldi::MessageLogger(::kaldi::LogMessageEnvelope::Info, \
                         __func__, __FILE__, __LINE__).stream()
#define KALDI_VLOG(v) if ((v) <= ::kaldi::g_kaldi_verbose_level)     \
  ::kaldi::MessageLogger((::kaldi::LogMessageEnvelope::Severity)(v), \
                         __func__, __FILE__, __LINE__).stream()

inline bool IsKaldiError(const std::string &str) {
  return(!strncmp(str.c_str(), "ERROR ", 6));
}

void KaldiAssertFailure_(const char *func, const char *file,
                         int32 line, const char *cond_str);

/// @} end "addtogroup error_group"

}  // namespace kaldi

#endif  // KALDI_BASE_KALDI_ERROR_H_
