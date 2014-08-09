// nnet/nnet-randomizer.cc

// Copyright 2013  Brno University of Technology (author: Karel Vesely)

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

#include "nnet/nnet-randomizer.h"

#include <algorithm>
#include <vector>

namespace kaldi {
namespace nnet1 {

/* RandomizerMask:: */

void RandomizerMask::Init(const NnetDataRandomizerOptions& conf) {
  KALDI_LOG << "Seeding by srand with : " << conf.randomizer_seed;
  srand(conf.randomizer_seed);
}

const std::vector<int32>& RandomizerMask::Generate(int32 mask_size) {
  mask_.resize(mask_size);
  for (int32 i=0; i<mask_size; i++) mask_[i]=i;
  std::random_shuffle(mask_.begin(), mask_.end()); //with built-in random generator
  return mask_;
}


/* MatrixRandomizer:: */

void MatrixRandomizer::AddData(const CuMatrixBase<BaseFloat>& m) {
  // pre-allocate before 1st use
  if(data_.NumCols() == 0) {
    data_.Resize(conf_.randomizer_size,m.NumCols());
  }
  // optionally put previous left-over to front
  if (data_begin_ > 0) {
    KALDI_ASSERT(data_begin_ <= data_end_); // sanity check
    int32 leftover = data_end_ - data_begin_;
    KALDI_ASSERT(leftover < data_begin_); // no overlap
    if(leftover > 0) {
      data_.RowRange(0,leftover).CopyFromMat(data_.RowRange(data_begin_,leftover));
    }
    data_begin_ = 0; data_end_ = leftover;
    data_.RowRange(leftover,data_.NumRows()-leftover).SetZero(); // zeroing the rest 
  }
  // extend the buffer if necessary
  if(data_.NumRows() < data_end_ + m.NumRows()) {
    CuMatrix<BaseFloat> data_aux(data_);
    data_.Resize(data_end_ + m.NumRows() + 1000, data_.NumCols()); // +1000 row extra
    data_.RowRange(0,data_aux.NumRows()).CopyFromMat(data_aux);
  }
  // copy the data
  data_.RowRange(data_end_,m.NumRows()).CopyFromMat(m);
  data_end_ += m.NumRows();
}

void MatrixRandomizer::Randomize(const std::vector<int32>& mask) {
  KALDI_ASSERT(data_begin_ == 0);
  KALDI_ASSERT(data_end_ > 0);
  KALDI_ASSERT(data_end_ == mask.size());
  // Copy to auxiliary buffer for unshuffled data
  data_aux_ = data_;
  // Put the mask to GPU 
  CuArray<int32> mask_in_gpu(mask.size());
  mask_in_gpu.CopyFromVec(mask);
  // Randomize the data, mask is used to index rows in source matrix:
  // (Here the vector 'mask_in_gpu' is typically shorter than number of rows in 'data_aux_',
  //  because the the buffer 'data_aux_' is larger than capacity 'randomizer_size'.
  //  The extra rows in 'data_aux_' do not contain speech frames and are not copied
  //  from 'data_aux_', the extra rows in 'data_' are unchanged by cu::Randomize.)
  cu::Randomize(data_aux_, mask_in_gpu, &data_);
}

void MatrixRandomizer::Next() {
  data_begin_ += conf_.minibatch_size;
}

const CuMatrixBase<BaseFloat>& MatrixRandomizer::Value() {
  KALDI_ASSERT(data_end_ - data_begin_ >= conf_.minibatch_size); // have data for minibatch
  minibatch_.Resize(conf_.minibatch_size, data_.NumCols(),kUndefined);
  minibatch_.CopyFromMat(data_.RowRange(data_begin_,conf_.minibatch_size));
  return minibatch_;
}


/* VectorRandomizer */

void VectorRandomizer::AddData(const Vector<BaseFloat>& v) {
  // pre-allocate before 1st use
  if(data_.Dim() == 0) {
    data_.Resize(conf_.randomizer_size);
  }
  // optionally put previous left-over to front
  if (data_begin_ > 0) {
    KALDI_ASSERT(data_begin_ <= data_end_); // sanity check
    int32 leftover = data_end_ - data_begin_;
    KALDI_ASSERT(leftover < data_begin_); // no overlap
    if(leftover > 0) {
      data_.Range(0,leftover).CopyFromVec(data_.Range(data_begin_,leftover));
    }
    data_begin_ = 0; data_end_ = leftover;
    data_.Range(leftover,data_.Dim()-leftover).SetZero(); // zeroing the rest 
  }
  // extend the buffer if necessary
  if(data_.Dim() < data_end_ + v.Dim()) {
    Vector<BaseFloat> data_aux(data_);
    data_.Resize(data_end_ + v.Dim() + 1000); // +1000 row surplus
    data_.Range(0,data_aux.Dim()).CopyFromVec(data_aux);
  }
  // copy the data
  data_.Range(data_end_,v.Dim()).CopyFromVec(v);
  data_end_ += v.Dim();
}

void VectorRandomizer::Randomize(const std::vector<int32>& mask) {
  KALDI_ASSERT(data_begin_ == 0);
  KALDI_ASSERT(data_end_ > 0);
  KALDI_ASSERT(data_end_ == mask.size());
  // Use auxiliary buffer for unshuffled data
  Vector<BaseFloat> data_aux(data_);
  // randomize the data, mask is used to index elements in source vector
  for(int32 i = 0; i<mask.size(); i++) {
    data_(i) = data_aux(mask.at(i));
  }
}

void VectorRandomizer::Next() {
  data_begin_ += conf_.minibatch_size;
}

const Vector<BaseFloat>& VectorRandomizer::Value() {
  KALDI_ASSERT(data_end_ - data_begin_ >= conf_.minibatch_size); // have data for minibatch
  minibatch_.Resize(conf_.minibatch_size,kUndefined);
  minibatch_.CopyFromVec(data_.Range(data_begin_,conf_.minibatch_size));
  return minibatch_;
}


/* StdVectorRandomizer */

template<typename T>
void StdVectorRandomizer<T>::AddData(const std::vector<T>& v) {
  // pre-allocate before 1st use
  if(data_.size() == 0) {
    data_.resize(conf_.randomizer_size);
  }
  // optionally put previous left-over to front
  if (data_begin_ > 0) {
    KALDI_ASSERT(data_begin_ <= data_end_); // sanity check
    int32 leftover = data_end_ - data_begin_;
    KALDI_ASSERT(leftover < data_begin_); // no overlap
    if(leftover > 0) {
      std::copy(data_.begin()+data_begin_, data_.begin()+data_begin_+leftover, data_.begin());
    }
    data_begin_ = 0; data_end_ = leftover;
    // cannot do this, we don't know default value of arbitrary type!
    // data_.RowRange(leftover,data_.NumRows()-leftover).SetZero(); // zeroing the rest 
  }
  // extend the buffer if necessary
  if(data_.size() < data_end_ + v.size()) {
    data_.resize(data_end_ + v.size() + 1000); // +1000 row surplus
  }
  // copy the data
  std::copy(v.begin(), v.end(), data_.begin()+data_end_);
  data_end_ += v.size();
}

template<typename T>
void StdVectorRandomizer<T>::Randomize(const std::vector<int32>& mask) {
  KALDI_ASSERT(data_begin_ == 0);
  KALDI_ASSERT(data_end_ > 0);
  KALDI_ASSERT(data_end_ == mask.size());
  // Use auxiliary buffer for unshuffled data
  std::vector<T> data_aux(data_);
  // randomize the data, mask is used to index elements in source vector
  for(int32 i = 0; i<mask.size(); i++) {
    data_.at(i) = data_aux.at(mask.at(i));
  }
}

template<typename T>
void StdVectorRandomizer<T>::Next() {
  data_begin_ += conf_.minibatch_size;
}

template<typename T>
const std::vector<T>& StdVectorRandomizer<T>::Value() {
  KALDI_ASSERT(data_end_ - data_begin_ >= conf_.minibatch_size); // have data for minibatch
  minibatch_.resize(conf_.minibatch_size);

  typename std::vector<T>::iterator first = data_.begin() + data_begin_;
  typename std::vector<T>::iterator last  = data_.begin() + data_begin_ + conf_.minibatch_size; //not-copied
  std::copy(first, last, minibatch_.begin());
  return minibatch_;
}

// Instantiate template StdVectorRandomizer with types we expect to operate on
template class StdVectorRandomizer<int32>;
template class StdVectorRandomizer<std::vector<std::pair<int32, BaseFloat> > >; //PosteriorRandomizer

}
}
