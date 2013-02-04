// nnet/nnet-cache.cc

// Copyright 2011  Karel Vesely

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



void CacheTgtMat::Init(int32 cachesize, int32 bunchsize) {

  KALDI_ASSERT(cachesize>0);
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

  assert(features.NumRows() == targets.NumRows());

  // lazy buffers allocation
  if (features_.NumRows() != cachesize_) {
    features_.Resize(cachesize_, features.NumCols());
    targets_.Resize(cachesize_, targets_.NumCols());
  }

  // warn if segment longer than half-cache 
  // (the frame level shuffling will have poor effect)
  if (features.NumRows() > cachesize_/2) {
    KALDI_WARN << "Too long segment and small feature cache!"
       << " cachesize: " << cachesize_
       << " segmentsize: " << features.NumRows();
  }

  // change state
  if (state_ == EMPTY) { 
    state_ = FILLING; filling_pos_ = 0;
   
    // check for leftover from previous segment 
    int leftover = features_leftover_.NumRows();
    // check if leftover is not bigger than cachesize
    if (leftover > cachesize_) {
      KALDI_WARN << "Too small feature cache: " << cachesize_
         << ", truncating: "
         << leftover - cachesize_ 
         << " frames from previous segment leftover";
      leftover = cachesize_;
    }
    // prefill cache with leftover
    if (leftover > 0) {
      features_.CopyRowsFromMat(leftover, features_leftover_, 0, 0);
      targets_.CopyRowsFromMat(leftover, targets_leftover_, 0, 0);
      
      features_leftover_.Resize(0, 0);
      targets_leftover_.Resize(0, 0);
      filling_pos_ += leftover;
    } 
  }

  assert(state_ == FILLING);
  assert(features.NumRows() == targets.NumRows());

  int cache_space = cachesize_ - filling_pos_;
  int feature_length = features.NumRows();
  int fill_rows = (cache_space<feature_length)? cache_space : feature_length;
  int leftover = feature_length - fill_rows;

  assert(cache_space > 0);

  // copy the data to cache
  features_.CopyRowsFromMat(fill_rows, features, 0, filling_pos_);
  targets_.CopyRowsFromMat(fill_rows, targets, 0, filling_pos_);

  // copy leftovers
  if (leftover > 0) {
    features_leftover_.Resize(leftover, features_.NumCols());
    features_leftover_.CopyRowsFromMat(leftover, features, fill_rows, 0);
    
    targets_leftover_.Resize(leftover, targets_.NumCols());
    targets_leftover_.CopyRowsFromMat(leftover, targets, fill_rows, 0);
  }

  // update cursor
  filling_pos_ += fill_rows;
  
  // change state
  if (filling_pos_ == cachesize_) { 
    state_ = FULL;
  }
}



void CacheTgtMat::Randomize() {
  assert(state_ == FULL || state_ == FILLING);

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

  assert(state_ == EMPTYING);

  // init the output
  features->Resize(bunchsize_, features_.NumCols());
  targets->Resize(bunchsize_, targets_.NumCols());

  // copy the output
  if (randomized_) {
    features->CopyRowsFromMat(bunchsize_, features_random_, emptying_pos_, 0);
    targets->CopyRowsFromMat(bunchsize_, targets_random_, emptying_pos_, 0);
  } else {
    features->CopyRowsFromMat(bunchsize_, features_, emptying_pos_, 0);
    targets->CopyRowsFromMat(bunchsize_, targets_, emptying_pos_, 0);
  }

  // update cursor
  emptying_pos_ += bunchsize_;

  // change state to EMPTY
  if (emptying_pos_ > filling_pos_-bunchsize_) {
    // we don't have more complete bunches...
    state_ = EMPTY;
  }
}


} // namespace
