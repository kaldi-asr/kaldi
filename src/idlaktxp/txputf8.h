// idlaktxp/txputf8.h

// Copyright 2012 CereProc Ltd.  (Author: Matthew Aylett)

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
//

#ifndef KALDI_IDLAKTXP_TXPUTF8_H_
#define KALDI_IDLAKTXP_TXPUTF8_H_

// This class offers some limited utf8 functionality

#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"

namespace kaldi {

/// \addtogroup idlak_utils
/// @{

/// Utility class to manage a very limited set of utf8 string
/// manipulatation functions.
class TxpUtf8 {
 public:
  /// Return the size of the next utf8 character
  int32 Clen(const char* input) {
    return trailingBytesForUTF8_[static_cast<unsigned int>(
        static_cast<unsigned const char>(input[0]))] + 1;
  }

 private:
  /// Lookup table for determining number of trailing characters
  /// given the first byte of a utf8 character
  static const char trailingBytesForUTF8_[256];
};


}  // namespace kaldi

/// @} end of \addtogroup idlak_utils


#endif  // KALDI_IDLAKTXP_TXPUTF8_H_
