// tensor/tensor-settings.h

// Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_TENSOR_TENSOR_SETTINGS_H_
#define KALDI_TENSOR_TENSOR_SETTINGS_H_ 1

#include <cstdint>
#include <vector>
#include <string>
#include "tensor/tensor-common.h"


/**
   This file contains certain mechanisms to set settings about default
   data types and devices within scopes, some related things like
   an equivalent of PyTorch's .no_grad().  Also the `Tick()` mechanism
   is here.
*/

namespace kaldi {
namespace tensor {



// Global variable, initialized from zero, that is used in GetTick().
// This is defined in tensor-settings.cc.
extern int64 g_tick_counter;
inline int64 NextTick() { return ++g_tick_counter; }


// debug_mode activates code that checks for invalidated data in the backprop
// pass; see "Invalidated:" in glossary in tensor.h.
// Don't access this variable directly.
extern bool g_debug_mode;     // Do not access directly!
extern int64 g_debug_start_tick;   // Do not access directly!

inline bool DebugMode() {
  return debug_mode;
}
inline void SetDebugMode(bool b) {
  if (!debug_mode)
    debug_start_tick = NextTick();
  debug_mode = b;
}
/**
   Returns the tick at which debug mode most recently changed from false to
   true.
 */
inline int64 DebugTick() {
  KALDI_PARANOID_ASSERT(debug_mode);
  return debug_start_tick;
}

class WithDebugModeAs {
 public:
  // Example:
  // {
  //   WithDebugModeAs _(true);
  //   // code in this block uses debug mode.
  //   // variable name is _ because we won't use it.
  // }
  inline WithDebugModeAs(bool b):
      prev_default_(DebugMode()) {
    SetDebugMode(b);
  }
  ~WithDebugModeAs() { SetDebugMode(prev_default_); }

 private:
  bool prev_default_;
};


inline bool DebugMode() {
  return debug_mode;
}
inline void SetDebugMode(bool b) {
  if (!debug_mode)
    debug_start_tick = NextTick();
  debug_mode = b;
}

extern bool g_reference_mode;     // Do not access directly!

// Gets 'reference mode' bool.  If true, the simple reference implementation
// will be used instead of the more optimized (e.g. BLAS-based) implementation.
// This will typically affect the Expand() call of Ops instead of their
// Do() call.
inline bool ReferenceMode() {
  return reference_mode;
}
inline void SetReferenceMode(bool b) {
  reference_mode = b;
}


}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_TENSOR_SETTINGS_H_
