// feat/feature-common-inl.h

// Copyright       2016  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_FEAT_FEATURE_COMMON_INL_H_
#define KALDI_FEAT_FEATURE_COMMON_INL_H_

// Do not include this file directly.  It is included by feat/feature-common.h

namespace kaldi {

template <class F>
void OfflineFeatureTpl<F>::Compute(
    const VectorBase<BaseFloat> &wave,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output,
    Vector<BaseFloat> *deprecated_wave_remainder) {
  KALDI_ASSERT(output != NULL);
  int32 rows_out = NumFrames(wave.Dim(), computer_.GetFrameOptions()),
      cols_out = computer_.Dim();
  if (rows_out == 0) {
    output->Resize(0, 0);
    if (deprecated_wave_remainder != NULL)
      *deprecated_wave_remainder = wave;
    return;
  }
  output->Resize(rows_out, cols_out);
  if (deprecated_wave_remainder != NULL)
    ExtractWaveformRemainder(wave, computer_.GetFrameOptions(),
                             deprecated_wave_remainder);
  Vector<BaseFloat> window;  // windowed waveform.
  bool use_raw_log_energy = computer_.NeedRawLogEnergy();
  for (int32 r = 0; r < rows_out; r++) {  // r is frame index.
    BaseFloat raw_log_energy = 0.0;
    ExtractWindow(0, wave, r, computer_.GetFrameOptions(),
                  feature_window_function_, &window,
                  (use_raw_log_energy ? &raw_log_energy : NULL));

    SubVector<BaseFloat> output_row(*output, r);
    computer_.Compute(raw_log_energy, vtln_warp, &window, &output_row);
  }
}

template <class F>
void OfflineFeatureTpl<F>::Compute(
    const VectorBase<BaseFloat> &wave,
    BaseFloat vtln_warp,
    Matrix<BaseFloat> *output,
    Vector<BaseFloat> *deprecated_wave_remainder) const {
  OfflineFeatureTpl<F> temp(*this);
  // call the non-const version of Compute() on a temporary copy of this object.
  // This is a workaround for const-ness that may sometimes be useful in
  // multi-threaded code, although it's not optimally efficient.
  temp.Compute(wave, vtln_warp, output, deprecated_wave_remainder);
}

} // end namespace kaldi

#endif
