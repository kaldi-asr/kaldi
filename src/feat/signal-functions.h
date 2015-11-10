// feat/signal-functions.h

// Copyright 2015 Hakan Erdogan  Jonathan Le Roux

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

#ifndef KALDI_FEAT_SIGNAL_FUNCTIONS_H_
#define KALDI_FEAT_SIGNAL_FUNCTIONS_H_

#include <string>
#include <vector>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "matrix/toeplitz.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{
  void ChannelConvert(const VectorBase<BaseFloat> &a,
		      const VectorBase<BaseFloat> &b,
		      const int32 &taps,
		      Vector<BaseFloat> *h,
		      Vector<BaseFloat> *output);

/// @} End of "addtogroup feat"
}  // namespace kaldi


#endif  // KALDI_FEAT_SIGNAL_FUNCTIONS_H_
