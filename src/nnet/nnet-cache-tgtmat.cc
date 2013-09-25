// nnet/nnet-cache-tgtmat.cc

// Copyright 2011  Karel Vesely

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

#include "nnet/nnet-cache-tgtmat.h"

#include "cudamatrix/cu-math.h"

#include <algorithm>

namespace kaldi {
namespace nnet1 {



void CacheTgtMat::Init(int32 cachesize, int32 bunchsize) {

  KALDI_ASSERT(cachesize>0);
  if(cachesize > 8388479) {
    KALDI_ERR << "Cachesize " << cachesize << " too large, use cachesize smaller than 8388480.";
  }
  KALDI_ASSERT(bunchsize>0);
  KALDI_ASSERT(cachesize>=bunchsize);

  if ((cachesize % bunchsize) != 0) {
    KALDI_ERR << "Non divisible cachesize by bunchsize";
  }
  
  cachesize_ = cachesize;
  bunchsize_ = bunchsize;

  state_ = EMPTY;

  filling_pos_ = 0;
  emptying_pos_ = 0;

  randomized_ = false;
}



void CacheTgtMat::AddData(const CuMatrix<BaseFloat> &features, const CuMatrix<BaseFloat> &targets) {
  if (state_ == FULL) {
    KALDI_ERR << "Cannot add data, cache already full";
  }

  KALDI_ASSERT(features.NumRows() == targets.NumRows());

  int32 dim_fea = features.NumCols();
  int32 dim_tgt = targets.NumCols();
  
  // lazy buffers allocation
  if (features_.NumRows() != cachesize_) {
    features_.Resize(cachesize_, dim_fea);
    targets_.Resize(cachesize_, dim_tgt);
  }

  // warn if segment longer than half-cache 
  // (frame level shuffling accross sentences will be poor)
  if (features.NumRows() > cachesize_/4) {
    KALDI_WARN << "Too long segment or small cachesize!"
       << " (cache-size " << cachesize_ << ") < (4 x"
       << " segment-size " << features.NumRows() << ").";
  }

  // change state
  if (state_ == EMPTY) { 
    state_ = FILLING; filling_pos_ = 0;
   
    // check for leftover from previous segment
    int leftover = features_leftover_.NumRows();
    // check if leftover is not bigger than half-cachesize
    if (leftover > cachesize_/2) {
      KALDI_WARN << "Truncating "
         << leftover - cachesize_/2
         << " frames from leftover of previous segment "
         << "(max leftover " << cachesize_/2 << ").";
      leftover = cachesize_/2;
    }
    // prefill cache with leftover
    if (leftover > 0) {
      features_.RowRange(0, leftover).CopyFromMat(
        features_leftover_.RowRange(0, leftover)
      );
      targets_.RowRange(0, leftover).CopyFromMat(
        targets_leftover_.RowRange(0, leftover)
      );
      
      features_leftover_.Resize(0, 0);
      targets_leftover_.Resize(0, 0);
      filling_pos_ += leftover;
    } 
  }

  KALDI_ASSERT(state_ == FILLING);
  KALDI_ASSERT(features.NumRows() == targets.NumRows());

  int cache_space = cachesize_ - filling_pos_;
  int feature_length = features.NumRows();
  int fill_rows = (cache_space<feature_length)? cache_space : feature_length;
  int leftover = feature_length - fill_rows;

  KALDI_ASSERT(cache_space > 0);

  // copy the data to cache
  features_.RowRange(filling_pos_, fill_rows).CopyFromMat(
    features.RowRange(0, fill_rows)
  );
  targets_.RowRange(filling_pos_, fill_rows).CopyFromMat(
    targets.RowRange(0, fill_rows)
  );
      
  // copy leftovers
  if (leftover > 0) {
    features_leftover_.Resize(leftover, dim_fea);
    features_leftover_.CopyFromMat(
      features.RowRange(fill_rows, leftover)
    );
    targets_leftover_.Resize(leftover, dim_tgt);
    targets_leftover_.CopyFromMat(
      targets.RowRange(fill_rows, leftover)
    );
  }

  // update cursor
  filling_pos_ += fill_rows;
  
  // change state
  if (filling_pos_ == cachesize_) { 
    state_ = FULL;
  }
}



void CacheTgtMat::Randomize() {
  KALDI_ASSERT(state_ == FULL || state_ == FILLING);

  // lazy initialization of the output buffers
  features_random_.Resize(cachesize_, features_.NumCols());
  targets_random_.Resize(cachesize_, targets_.NumCols());

  // generate random series of integers
  randmask_.resize(filling_pos_);
  GenerateRandom randomizer;
  for(int32 i=0; i<filling_pos_; i++) { randmask_[i]=i; }
  std::random_shuffle(randmask_.begin(), randmask_.end(), randomizer);
  // get it to the gpu
  randmask_device_.CopyFromVec(randmask_);

  // randomize the features
  cu::Randomize(features_, randmask_device_, &features_random_);
  // randomize the targets
  cu::Randomize(targets_, randmask_device_, &targets_random_);
  
  randomized_ = true;
}



void CacheTgtMat::GetBunch(CuMatrix<BaseFloat> *features, CuMatrix<BaseFloat> *targets) {
  if (state_ == EMPTY) {
    KALDI_ERR << "GetBunch on empty cache!!!";
  }

  // change state if full...
  if (state_ == FULL) { 
    state_ = EMPTYING; emptying_pos_ = 0; 
  }

  // final cache is not completely filled
  if (state_ == FILLING) { 
    state_ = EMPTYING; emptying_pos_ = 0; 
  } 

  KALDI_ASSERT(state_ == EMPTYING);

  // init the output
  features->Resize(bunchsize_, features_.NumCols());
  targets->Resize(bunchsize_, targets_.NumCols());

  // copy the output
  if (randomized_) {
    features->RowRange(0, bunchsize_).CopyFromMat(
        features_random_.RowRange(emptying_pos_, bunchsize_));
    targets->RowRange(0, bunchsize_).CopyFromMat(
        targets_random_.RowRange(emptying_pos_, bunchsize_));
  } else {
    features->RowRange(0, bunchsize_).CopyFromMat(
        features_.RowRange(emptying_pos_, bunchsize_));
    targets->RowRange(0, bunchsize_).CopyFromMat(
        targets_.RowRange(emptying_pos_, bunchsize_));
  }

  // update cursor
  emptying_pos_ += bunchsize_;

  // change state to EMPTY
  if (emptying_pos_ > filling_pos_-bunchsize_) {
    // we don't have more complete bunches...
    state_ = EMPTY;
  }
}


} // namespace nnet1
} // namespace kaldi
