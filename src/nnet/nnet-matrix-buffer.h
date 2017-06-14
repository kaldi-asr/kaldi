// nnet/nnet-matrix-buffer.h

// Copyright 2016  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_MATRIX_BUFFER_H_
#define KALDI_NNET_NNET_MATRIX_BUFFER_H_

#include <utility>
#include <vector>
#include <list>
#include <string>

#include "itf/options-itf.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


namespace kaldi {
namespace nnet1 {

struct MatrixBufferOptions {
  int32 matrix_buffer_size;

  MatrixBufferOptions():
    matrix_buffer_size(3 * 1024)  // 3 x 1GB,
  { }

  void Register(OptionsItf *opts) {
    opts->Register("matrix-buffer-size", &matrix_buffer_size,
       "Capacity of buffer for feature matrices, in MB.");
  }
};


/**
 * A buffer for caching (utterance-key, feature-matrix) pairs.
 * Typically, it reads 'matrix_buffer_size' megabytes of data,
 * and returns records with similar number of speech frames
 * through the standard Key(), Value(), Next(), Done() interface.
 *
 * The preferred length is reset by ResetLength().
 * The buffer gets refilled after having less
 * data than 50% of 'matrix_buffer_size'.
 */
class MatrixBuffer {
 public:
  MatrixBuffer():
    reader_(NULL),
    current_(NULL),
    preferred_length_(0)
  { }

  ~MatrixBuffer()
  { }

  void Init(SequentialBaseFloatMatrixReader* reader,
            MatrixBufferOptions opts = MatrixBufferOptions()) {
    KALDI_ASSERT(SizeInBytes() == 0);
    reader_ = reader;
    opts_ = opts;

    Read();
  }

  bool Done() {
    return (reader_->Done() && NumPairs() <= 1);
  }

  void Next();

  void ResetLength() {
    preferred_length_ = 0;
  }

  std::string Key() {
    return current_->first;
  }
  Matrix<BaseFloat> Value() {
    return current_->second;
  }

  /// Total amount of features in the buffer (bytes),
  size_t SizeInBytes() const;

  /// Total amount of features in the buffer (Mega-bytes),
  size_t SizeInMegaBytes() const;

  /// Total number of (key,matrix) pairs in the buffer,
  size_t NumPairs() const;

 private:

  void Read();  ///< fills the buffer,
  void DisposeValue();  ///< removes 'current_' from data structure,

  SequentialBaseFloatMatrixReader* reader_;

  typedef std::pair<std::string,Matrix<BaseFloat> > PairType;
  typedef std::list<PairType> ListType;
  typedef std::map<size_t, ListType> BufferType;
  BufferType buffer_;  ///< Buffer indexed by 'NumRows()',

  PairType* current_;  ///< The currently active (key,value) pair,

  MatrixBufferOptions opts_;

  size_t preferred_length_;
};

void MatrixBuffer::Next() {
  KALDI_ASSERT(!buffer_.empty());

  // remove old 'Value()' matrix,
  DisposeValue();

  // start re-filling,
  if (SizeInMegaBytes() < 0.5 * opts_.matrix_buffer_size) {
    Read();
  }

  KALDI_ASSERT(!buffer_.empty());

  // randomly select 'length' present in the 'map',
  // (weighted by total #frames in the bin),
  if (preferred_length_ == 0) {
    int32 longest = (--buffer_.end())->first;
    // pre-fill the vector of 'keys',
    std::vector<int32> keys;
    BufferType::iterator it;
    for (it = buffer_.begin(); it != buffer_.end(); ++it) {
      int32 key = it->first; // i.e. NumRows() of matrices in the bin,
      int32 frames_in_bin = it->second.size() * key;
      for (int32 i = 0; i < frames_in_bin; i += longest) {
        keys.push_back(key); // keys are repeated,
      }
    }
    // choose the key,
    std::vector<int32>::iterator it2 = keys.begin();
    std::advance(it2, rand() % keys.size());
    preferred_length_ = (*it2);  // NumRows(), key of the 'map',
  }

  // select list by 'preferred_length_',
  BufferType::iterator it = buffer_.lower_bound(preferred_length_);
  if (it == buffer_.end()) { --it; } // or the last one,

  // take a front element 'ptr' from that list,
  current_ = &(it->second.front());
}

size_t MatrixBuffer::SizeInBytes() const {
  size_t ans = 0;
  for (BufferType::const_iterator it = buffer_.begin(); it != buffer_.end(); ++it) {
    for (ListType::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
      ans += it2->second.SizeInBytes();
    }
  }
  return ans;
}

size_t MatrixBuffer::SizeInMegaBytes() const {
  return (SizeInBytes() / (1024 * 1024));
}

size_t MatrixBuffer::NumPairs() const {
  size_t ans = 0;
  for (BufferType::const_iterator it = buffer_.begin(); it != buffer_.end(); ++it) {
    ans += it->second.size();
  }
  return ans;
}

void MatrixBuffer::Read() {
  if (!reader_->Done())
    KALDI_LOG << "Read() started... Buffer size in MB: "
              << SizeInMegaBytes() << ", max " << opts_.matrix_buffer_size
              << ", having " << NumPairs() << " utterances.";
  for ( ; !reader_->Done(); reader_->Next()) {
    // see if we are full,
    if (SizeInMegaBytes() > opts_.matrix_buffer_size) {
      KALDI_LOG << "Read() finished... Buffer size in MB: "
                << SizeInMegaBytes() << ", max " << opts_.matrix_buffer_size
                << ", having " << NumPairs() << " utterances.";
      break;
    }
    // get matrix,
    const std::string& key = reader_->Key();
    const Matrix<BaseFloat>& mat = reader_->Value();
    size_t num_rows = mat.NumRows();
    // see if 'num_rows' already in keys,
    if (buffer_.find(num_rows) == buffer_.end()) {
      buffer_[num_rows] = ListType();  // add empty list,
    }
    // add matrix to the buffer,
    buffer_[num_rows].push_back(PairType(key, mat));
  }
}

void MatrixBuffer::DisposeValue() {
  // remove old 'Value()' matrix,
  if (current_ != NULL) {
    size_t r = current_->second.NumRows();
    KALDI_ASSERT(current_ == &(buffer_[r].front()));
    // remove the (key,value) pair,
    buffer_[r].pop_front();
    // eventually remove the 'NumRows()' key,
    if (buffer_[r].empty()) { buffer_.erase(r); }
    current_ = NULL;
  }
}


}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_MATRIX_BUFFER_H_

