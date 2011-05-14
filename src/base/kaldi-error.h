// base/kaldi-error.h

// Copyright 2009-2011  Microsoft Corporation  Ondrej Glembek  Lukas Burget  Arnab Ghoshal

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

#include <stdexcept>
#include <string>
#include <sstream>

#include "base/kaldi-types.h"

/* Important that this file does not depend on any other kaldi headers. */


namespace kaldi {

/// \addtogroup error_group
/// @{

/// This is set by util/parse-options.{h, cc} if you set --verbose = ? option
extern int32 g_kaldi_verbose_level;

/// This is set by util/parse-options.{h, cc} (from argv[0]) and used (if set) in
/// error reporting code to display the name of the program (this is because in
/// our scripts, we often mix together the stderr of many programs).
extern std::string g_program_name;

inline int32 GetVerboseLevel() { return g_kaldi_verbose_level; }

// Class KaldiLogMessage is invoked from the  KALDI_WARN, KALDI_VLOG and
// KALDI_LOG macros. It prints the message to std::cerr.

// class KaldiWarnMessage is invoked from the KALDI_WARN macro.
class KaldiWarnMessage {
 public:
  inline std::ostream &stream() { return std::cerr; }
  KaldiWarnMessage(const char *func, const char *file, int32 line);
  ~KaldiWarnMessage() { stream() << '\n'; }
};

// class KaldiLogMessage is invoked from the KALDI_LOG macro.
class KaldiLogMessage {
 public:
  inline std::ostream &stream() { return std::cerr; }
  KaldiLogMessage(const char *func, const char *file, int32 line);
  ~KaldiLogMessage() { stream() << '\n'; }
};

// Class KaldiVlogMessage is invoked from the KALDI_VLOG macro.
class KaldiVlogMessage {
 public:
  KaldiVlogMessage(const char *func, const char *file, int32 line,
                   int32 verbose_level);
  inline std::ostream &stream() { return std::cerr; }
  ~KaldiVlogMessage() { stream() << '\n'; }
};

// class KaldiExitMessage is invoked from the KALDI_EXIT macro.
// It prints the message to std::cerr and exits.  It does not print
// line-number information etc. (the idea is that this is generally called
// from main() where this doesn't help much).
class KaldiExitMessage {
 public:
  KaldiExitMessage() {}
  inline std::ostream &stream() { return std::cerr; }
  ~KaldiExitMessage() {
    stream() << '\n';
    exit(1);
  }
};

// class KaldiErrorMessage is invoked from the KALDI_ERROR macro.
// The destructor throws an exception.
class KaldiErrorMessage {
 public:
  KaldiErrorMessage(const char *func, const char *file, int32 line);
  inline std::ostream &stream() { return ss; }
  ~KaldiErrorMessage();  // defined in kaldi-error.cc
 private:
  std::ostringstream ss;
};



#ifdef _MSC_VER
#define __func__ __FUNCTION__
#endif

#ifndef NDEBUG
#define KALDI_ASSERT(cond) \
  if (!(cond)) kaldi::KaldiAssertFailure_(__func__, __FILE__, __LINE__, #cond);
#else
#define KALDI_ASSERT(cond)
#endif

#define KALDI_ERR kaldi::KaldiErrorMessage(__func__, __FILE__, __LINE__).stream() 
#define KALDI_WARN kaldi::KaldiWarnMessage(__func__, __FILE__, __LINE__).stream() 
#define KALDI_LOG kaldi::KaldiLogMessage(__func__, __FILE__, __LINE__).stream()
#define KALDI_EXIT kaldi::KaldiExitMessage().stream()
#define KALDI_VLOG(v) if (v <= kaldi::g_kaldi_verbose_level)     \
           kaldi::KaldiVlogMessage(__func__, __FILE__, __LINE__, v).stream()

inline bool IsKaldiError(const std::string &str) {
  return(!strncmp(str.c_str(), "ERROR ", 6));
}

void KaldiAssertFailure_(const char *func, const char *file,
                         int32 line, const char *cond_str);

// Note that there is also the macro KALDI_COMPILE_TIME_ASSERT,
// defined in base/kaldi-utils.h

/// @} end "addtogroup error_group"

}  // namespace kaldi

#endif  // KALDI_BASE_KALDI_ERROR_H_
