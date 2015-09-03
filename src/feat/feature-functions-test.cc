// feat/feature-functions-test.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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


#include <iostream>

#include "feat/feature-mfcc.h"
#include "base/kaldi-math.h"
#include "matrix/kaldi-matrix-inl.h"
#include "feat/wave-reader.h"


// TODO: some of the other functions should be tested.  
namespace kaldi {

void UnitTestOnlineCmvn() {
  for (int32 i = 0; i < 1000; i++) {
    int32 num_frames = 1 + (Rand() % 10 * 10);
    int32 dim = 1 + Rand() % 10;
    SlidingWindowCmnOptions opts;
    opts.center = (Rand() % 2 == 0);
    opts.normalize_variance = (Rand() % 2 == 0);
    opts.cmn_window = 5 + Rand() % 50;
    opts.min_window = 1 + Rand() % 100;
    if (opts.min_window > opts.cmn_window)
      opts.min_window = opts.cmn_window;

    Matrix<BaseFloat> feats(num_frames, dim),
        output_feats(num_frames, dim),
        output_feats2(num_frames, dim);
    feats.SetRandn();
    SlidingWindowCmn(opts, feats, &output_feats);

    for (int32 t = 0; t < num_frames; t++) {
      int32 window_begin, window_end;
      if (opts.center) {
        window_begin = t - (opts.cmn_window / 2),
            window_end = window_begin + opts.cmn_window;
        int32 shift = 0;
        if (window_begin < 0)
          shift = -window_begin;
        else if (window_end > num_frames)
          shift = num_frames - window_end;
        window_end += shift;
        window_begin += shift;
      } else {
        window_begin = t - opts.cmn_window;
        window_end = t + 1;
        if (window_end < opts.min_window)
            window_end = opts.min_window;
      }
      if (window_begin < 0) window_begin = 0;
      if (window_end > num_frames) window_end = num_frames;
      int32 window_size = window_end - window_begin;
      for (int32 d = 0; d < dim; d++) {
        double sum = 0.0, sumsq = 0.0;
        for (int32 t2 = window_begin; t2 < window_end; t2++) {
          sum += feats(t2, d);
          sumsq += feats(t2, d) * feats(t2, d);
        }
        double mean = sum / window_size, uncentered_covar = sumsq / window_size,
            covar = uncentered_covar - mean * mean;
        covar = std::max(covar, 1.0e-20);
        double data = feats(t, d),
            norm_data = data - mean;
        if (opts.normalize_variance) {
          if (window_size == 1) norm_data = 0.0;
          else norm_data /= sqrt(covar);
        }
        output_feats2(t, d) = norm_data;
      }
    }
    if (! output_feats.ApproxEqual(output_feats2, 0.0001)) {
      KALDI_ERR << "Features differ " << output_feats << " vs. " << output_feats2;
    }
  }
}


}



int main() {
  using namespace kaldi;
  try {
    UnitTestOnlineCmvn();
    std::cout << "Tests succeeded.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
}


