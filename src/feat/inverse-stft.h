// feat/inverse-stft.h

// Copyright 2015  Hakan Erdogan

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

#ifndef KALDI_FEAT_INVERSE_STFT_H_
#define KALDI_FEAT_INVERSE_STFT_H_


#include <string>

#include "feat/feature-functions.h"
#include "feat/stft-functions.h"
#include "feat/feature-stft.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{

// WARNING: we use the same StftOptions structure from feature-stft.h
// so the naming of options are with respect to FORWARD Stft (like output_type, output_layout etc.)

/// Class for computing inverse STFT;
class Istft {
public:
    explicit Istft(const StftOptions &opts);
    ~Istft();

    /// Will throw exception on failure (e.g. if features are too short)
    void Compute(const Matrix<BaseFloat> &input,
                 Vector<BaseFloat> *wave,
                 int32 wav_length = -1);

private:
    StftOptions opts_;
    BaseFloat log_energy_floor_;
    FeatureWindowFunction feature_window_function_;
    SplitRadixRealFft<BaseFloat> *srfft_;
    KALDI_DISALLOW_COPY_AND_ASSIGN(Istft);
};


/// @} End of "addtogroup feat"
}  // namespace kaldi


#endif  // KALDI_FEAT_INVERSE_STFT_H_
