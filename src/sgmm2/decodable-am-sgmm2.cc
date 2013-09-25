// sgmm2/decodable-am-sgmm2.cc

// Copyright 2009-2012  Saarland University;  Lukas Burget;
//                      Johns Hopkins University (author: Daniel Povey)

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

#include <vector>
using std::vector;

#include "sgmm2/decodable-am-sgmm2.h"

namespace kaldi {


DecodableAmSgmm2::~DecodableAmSgmm2() {
  if (delete_vars_) {
    delete gselect_;
    delete feature_matrix_;
    delete spk_;
  }
}

BaseFloat DecodableAmSgmm2::LogLikelihoodForPdf(int32 frame, int32 pdf_id) {
  if (frame != cur_frame_) {
    cur_frame_ = frame;
    sgmm_cache_.NextFrame(); // it has a frame-index internally but it doesn't
    // have to match up with our index here, it just needs to be unique.


    SubVector<BaseFloat> data(*feature_matrix_, frame);
    
    sgmm_.ComputePerFrameVars(data, (*gselect_)[frame], *spk_,
                              &per_frame_vars_);
  }
  return sgmm_.LogLikelihood(per_frame_vars_, pdf_id, &sgmm_cache_, spk_,
                             log_prune_);  
}


}  // namespace kaldi
