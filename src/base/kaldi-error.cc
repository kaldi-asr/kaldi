// base/kaldi-error.cc

// Copyright 2009-2011  Microsoft Corporation;  Lukas Burget;  Ondrej Glembek

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

#ifdef HAVE_EXECINFO_H
#include <execinfo.h>  // To get stack trace in error messages.
// If this #include fails there is an error in the Makefile, it does not
// support your platform well. Make sure HAVE_EXECINFO_H is undefined, and the
// code will compile.
#ifdef HAVE_CXXABI_H
#include <cxxabi.h>  // For name demangling.
// Useful to decode the stack trace, but only used if we have execinfo.h
#endif  // HAVE_CXXABI_H
#endif  // HAVE_EXECINFO_H

#include "base/kaldi-common.h"
#include "base/kaldi-error.h"

namespace kaldi {
int32 g_kaldi_verbose_level = 0;
const char *g_program_name = NULL;
static LogHandler g_log_handler = NULL;


// If the program name was set (g_program_name != ""), the function
// GetProgramName returns the program name (without the path) followed by a
// colon, e.g. "gmm-align:".  Otherwise it returns the empty string "".
const char *GetProgramName() {
  return g_program_name == NULL ? "" : g_program_name;
}

// Given a filename like "/a/b/c/d/e/f.cc",  GetShortFileName
// returns "e/f.cc".  Does not currently work if backslash is
// the filename separator.
const char *GetShortFileName(const char *filename) {
  const char *last_slash = strrchr(filename, '/');
  if (!last_slash) {
    return filename;
  } else {
    while (last_slash > filename && last_slash[-1] != '/')
      last_slash--;
    return last_slash;
  }
}

#if defined(HAVE_CXXABI_H) && defined(HAVE_EXECINFO_H)
// The function name looks like a macro: it's a macro if we don't have ccxxabi.h
inline void KALDI_APPEND_POSSIBLY_DEMANGLED_STRING(const char *to_append,
                                                   std::string *ans) {
  // at input the string "to_append" looks like:
  //   ./kaldi-error-test(_ZN5kaldi13UnitTestErrorEv+0xb) [0x804965d]
  // We want to extract the name e.g. '_ZN5kaldi13UnitTestErrorEv",
  // demangle it and return it.
  int32 status;
  const char *paren = strchr(to_append, '(');
  const char *plus = (paren ? strchr(paren, '+') : NULL);
  if (!plus) {  // did not find the '(' or did not find the '+'
    // This is a soft failure in case we did not get what we expected.
    ans->append(to_append);
    return;
  }
  std::string stripped(paren+1, plus-(paren+1));  // the bit between ( and +.

  char *demangled_name = abi::__cxa_demangle(stripped.c_str(), 0, 0, &status);

  // if status != 0 it is an error (demangling failure), but not all names seem
  // to demangle, so we don't check it.

  if (demangled_name != NULL) {
    ans->append(demangled_name);
    free(demangled_name);
  } else {
    ans->append(to_append);  // add the original string.
  }
}
#else  // defined(HAVE_CXXABI_H) && defined(HAVE_EXECINFO_H)
#define KALDI_APPEND_POSSIBLY_DEMANGLED_STRING(to_append, ans) \
  ans->append(to_append)
#endif  // defined(HAVE_CXXABI_H) && defined(HAVE_EXECINFO_H)

#ifdef HAVE_EXECINFO_H
std::string KaldiGetStackTrace() {
#define KALDI_MAX_TRACE_SIZE 50
#define KALDI_MAX_TRACE_PRINT 20  // must be even.
  std::string ans;
  void *array[KALDI_MAX_TRACE_SIZE];
  size_t size = backtrace(array, KALDI_MAX_TRACE_SIZE);
  char **strings = backtrace_symbols(array, size);
  if (size <= KALDI_MAX_TRACE_PRINT) {
    for (size_t i = 0; i < size; i++) {
      KALDI_APPEND_POSSIBLY_DEMANGLED_STRING(strings[i], &ans);
      ans += "\n";
    }
  } else {  // print out first+last (e.g.) 5.
    for (size_t i = 0; i < KALDI_MAX_TRACE_PRINT/2; i++) {
      KALDI_APPEND_POSSIBLY_DEMANGLED_STRING(strings[i], &ans);
      ans += "\n";
    }
    ans += ".\n.\n.\n";
    for (size_t i = size - KALDI_MAX_TRACE_PRINT/2; i < size; i++) {
      KALDI_APPEND_POSSIBLY_DEMANGLED_STRING(strings[i], &ans);
      ans += "\n";
    }
    if (size == KALDI_MAX_TRACE_SIZE)
      ans += ".\n.\n.\n";  // stack was too long, probably a bug.
  }
  free(strings);  // it's all in one big malloc()ed block.


#ifdef HAVE_CXXABI_H  // demangle the name, if possible.
#endif  // HAVE_CXXABI_H
  return ans;
}
#endif

void KaldiAssertFailure_(const char *func, const char *file,
                         int32 line, const char *cond_str) {
  MessageLogger ml(LogMessageEnvelope::Error, func, file, line);
  ml.stream() << "Assertion failed: " << cond_str;
#ifdef HAVE_EXECINFO_H
  ml.stream() << "\nStack trace is:\n" << KaldiGetStackTrace();
#endif
}

LogHandler SetLogHandler(LogHandler new_handler) {
  LogHandler old_handler = g_log_handler;
  g_log_handler = new_handler;
  return old_handler;
}

static void SendToLog(const LogMessageEnvelope &envelope,
                      const char *message) {
  // Send to a logging handler if provided.
  if (g_log_handler != NULL) {
    g_log_handler(envelope, message);
    return;
  }

  // Otherwise, use Kaldi default logging.
  std::stringstream header;
  if (envelope.severity > LogMessageEnvelope::Info)
    header << "VLOG[" << envelope.severity << "] (";
  else if (envelope.severity == LogMessageEnvelope::Info)
    header << "LOG (";
  else if (envelope.severity == LogMessageEnvelope::Warning)
    header << "WARNING (";
  else
    header << "ERROR (";
  header << GetProgramName() << envelope.func << "():"
         << envelope.file << ':' << envelope.line << ")";

  std::string header_str = header.str();
  fprintf(stderr, "%s %s\n", header_str.c_str(), message);
}

MessageLogger::MessageLogger(LogMessageEnvelope::Severity severity,
                             const char *func, const char *file, int32 line) {
  // Obviously, we assume the strings survive the destruction of this object.
  envelope_.severity = severity;
  envelope_.func = func;
  envelope_.file = GetShortFileName(file);  // Pointer inside 'file'.
  envelope_.line = line;
}

MessageLogger::~MessageLogger() KALDI_NOEXCEPT(false) {
  std::string str = ss_.str();
  while (!str.empty() && str[str.length() - 1] == '\n')
    str.resize(str.length() - 1);
  SendToLog(envelope_, str.c_str());

  if (envelope_.severity > LogMessageEnvelope::Error)
    return;

  // On error, throw an exception with the message, plus traceback info if
  // available.
  if (!std::uncaught_exception()) {
#ifdef HAVE_EXECINFO_H
    throw std::runtime_error(str + "\n\n[stack trace: ]\n" +
                             KaldiGetStackTrace() + "\n");
#else
    throw std::runtime_error(str);
#endif
  } else {
    abort();
  }
}

}  // end namespace kaldi
