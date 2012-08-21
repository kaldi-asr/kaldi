// runon/run-on-cmvn.h

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

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


#ifndef KALDI_ONLINE_CMN_H_
#define KALDI_ONLINE_CMN_H_

#include <vector>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "hmm/transition-model.h"

namespace kaldi {

//support for online cepstral mean normalization (variance normalization not 
//supported here) -- it's just doing a simple moving average over a history 
//of n preceding frames ...
class OnlineCMN {

 public:
  OnlineCMN(int32 dim, int32 history) {
    dim_ = dim;
    hist_ = history;
    oldest_row_ = 0;

    stats_.Resize(history, dim);
    norm_.Resize(0);
    features_cached_ = 0;
  }


  void ApplyCmvn(const MatrixBase<BaseFloat> &feats,
                 Matrix<BaseFloat> *norm_feats) {

    KALDI_ASSERT(feats.NumCols() == dim_ &&
                 "Inconsistent feature vectors dimensionality");

    norm_feats->Resize(0, 0);
    //first, fill up the history
    int32 i = 0;
    while (features_cached_ < hist_) {
      if (i < feats.NumRows()) {
        stats_.CopyRowFromVec(feats.Row(i), features_cached_);
        features_cached_++;
        i++;
      } else
        break;
    }

    //could we fill the history?
    if (features_cached_ != hist_) {
      //well, let's just compute the mean based on what we have so far ...
      ComputeMean(feats, norm_feats);
      return;
    }

    //now that we have our cache of features filled, we need to compute
    //the full sum only once
    if (norm_.Dim() == 0) {
      norm_.Resize(dim_);

      Vector<BaseFloat> sum;
      sum.Resize(dim_);
      for(int32 r = 0; r < hist_; r++) {
        for(int32 d = 0; d < dim_; d++) {
          sum(d) += stats_(r, d);
        }
      }

      for(int32 d = 0; d < dim_; d++) {
        norm_(d) = sum(d) / hist_;
      }
    }

    //compute the rolling mean
    for( ; i < feats.NumRows(); i++) {

      //update rolling mean
      for(int32 d = 0; d < dim_; d++) {
        norm_(d) = norm_(d) - stats_(oldest_row_, d) / hist_ +
            feats(i,d) / hist_;
      }

      //replace the oldest row with the current feature vector
      stats_.CopyRowFromVec(feats.Row(i), oldest_row_);
      //update the oldestRow_ variable
      oldest_row_ = (oldest_row_ + 1) % hist_;
    }

    //ok,  now we are ready to spit out the mean normalized features
    norm_feats->Resize(feats.NumRows(), dim_);

    for(int32 j = 0; j < feats.NumRows(); j++) {
      //apply mean
      for(int32 d = 0; d < dim_; d++) {
        (*norm_feats)(j, d) = feats(j, d) - norm_(d);
      }
    }
  } // ApplyCmvn()


 private:
  void ComputeMean(const MatrixBase<BaseFloat> &feats,
                   Matrix<BaseFloat> *norm_feats) {

    Vector<BaseFloat> sum;
    Vector<BaseFloat> norm;
    sum.Resize(dim_);
    norm.Resize(dim_);
    for(int32 r = 0; r < features_cached_; r++) {
      for(int32 d = 0; d < dim_; d++) {
        sum(d) += stats_(r, d);
      }
    }

    for(int32 d = 0; d < dim_; d++) {
      norm(d) = sum(d) / features_cached_;
    }

    norm_feats->Resize(feats.NumRows(), dim_);
    for(int32 i = 0; i < feats.NumRows(); i++) {
      for(int32 d = 0; d < dim_; d++) {
        (*norm_feats)(i, d) = feats(i, d) - norm(d);
      }
    }
  }


  int32 oldest_row_;
  int32 dim_;
  int32 hist_;
  Matrix<BaseFloat> stats_;
  Vector<double> norm_;
  int32 features_cached_;
};  // class OnlineCMN

}  // namespace kaldi

#endif // KALDI_ONLINE_CMN_H_
