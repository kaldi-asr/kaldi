// dec-wrap/dec-wrap-audio-source.cc

/* Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
 *                     2012-2013  Vassil Panayotov
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
 * WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
 * MERCHANTABLITY OR NON-INFRINGEMENT.
 * See the Apache 2 License for the specific language governing permissions and
 * limitations under the License. */


#include <cmath>
#include <unistd.h>

#include "dec-wrap/dec-wrap-audio-source.h"

namespace kaldi {

MatrixIndexT OnlBuffSource::Read(Vector<BaseFloat> *output) {
  KALDI_ASSERT(output->Dim() > 0 && "Request at least something");

  // copy as much as possible to output
  MatrixIndexT d = std::min(output->Dim(), static_cast<MatrixIndexT>(src_.size()));
  for (MatrixIndexT i = 0; i < d ; ++i) {
    (*output)(i) = src_[i];
  }
  // remove the already read elements
  std::vector<BaseFloat>(src_.begin() + d, src_.end()).swap(src_);
  KALDI_VLOG(3) << "Data read: " <<  d << " Data requested " << output->Dim();
  return d;
}


void OnlBuffSource::Reset() {
  src_.clear();
}


void OnlBuffSource::Write(unsigned char * data, size_t num_samples) {
  // allocate the space at once -> should be faster
  KALDI_VLOG(3) << "Data inserted: " <<  num_samples << std::endl
                <<" Data already buffered " << src_.size() << std::endl;
  src_.reserve(src_.size() + num_samples);
  // copy and convert the data to the buffer
  for (size_t i = 0; i < num_samples; ++i) {
      switch (opts_.bits_per_sample) {
        case 8:
          src_.push_back(*data);
          data++;
          break;
        case 16:
          {
            int16 k = *reinterpret_cast<uint16*>(data);
#ifdef __BIG_ENDIAN__
            KALDI_SWAP2(k);
#endif
            src_.push_back(k);
            data += 2;
            break;
          }
        case 32:
          {
            int32 k = *reinterpret_cast<uint32*>(data);
#ifdef __BIG_ENDIAN__
            KALDI_SWAP4(k);
#endif
            src_.push_back(k);
            data += 4;
            break;
          }
        default:
          KALDI_ERR << "unsupported bits per sample: " << opts_.bits_per_sample;
      }
  }
}

} // namespace kaldi
