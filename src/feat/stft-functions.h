// feat/stft-functions.h

// Copyright 2015 Hakan Erdogan

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

#ifndef KALDI_FEAT_STFT_FUNCTIONS_H_
#define KALDI_FEAT_STFT_FUNCTIONS_H_

#include <string>
#include <vector>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/mel-computations.h"
#include "feat/feature-functions.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{

void Deemphasize(VectorBase<BaseFloat> *waveform, BaseFloat preemph_coeff);

// OverlapAdd aims to reverse ExtractWindow to reconstruct a wave signal
// OverlapAdd accumulates the waveform from a windowed frame.
// It attempts to reverse pre-emphasis but cannot reverse dither or DC removal
void OverlapAdd(const VectorBase<BaseFloat> &data, // data to be overlapped added
                int32 start,  // start sample, if negative, negative part of the window will be trimmed
                int32 wav_length,  // if exceeds, will be trimmed
                Vector<BaseFloat> *wave);


/// @} End of "addtogroup feat"
}  // namespace kaldi


#endif  // KALDI_FEAT_STFT_FUNCTIONS_H_
