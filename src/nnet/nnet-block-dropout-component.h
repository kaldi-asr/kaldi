// nnet/nnet-block-dropout-component.h

// Copyright      2016  Sri Harish Mallidi

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


#ifndef KALDI_NNET_BLOCK_DROPOUT_H_
#define KALDI_NNET_BLOCK_DROPOUT_H_

#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class BlockDropout : public Component {
 public:
  BlockDropout(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out),
      dropout_retention_(0.5)
  { }
    ~BlockDropout()
  { }

  Component* Copy() const { return new BlockDropout(*this); }
  ComponentType GetType() const { return kBlockDropout; }

  void InitData(std::istream &is) {
    // define options
    std::string stream_indices_str;
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      if (token == "<StreamIndices>") {
	ReadToken(is, false, &stream_indices_str);
	std::vector<int32> v;
	bool tmp_ret = SplitStringToIntegers(stream_indices_str, ":", true, &v);
	if (!tmp_ret) {
	  KALDI_ERR << "Cannot parse the <StreamIndices> token. " 
		    << "It should be colon-separated list of integers";
	}
	stream_indices_ = v;
      }
      else {
	KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (StreamIndices)";
      }
    }

  }

  void ReadData(std::istream &is, bool binary) {
    stream_indices_.Read(is, binary);

    std::vector<int32> stream_indices_host;
    stream_indices_.CopyToVec(&stream_indices_host);

    KALDI_ASSERT(stream_indices_host[stream_indices_host.size()-1] == OutputDim());
  }

  void WriteData(std::ostream &os, bool binary) const {
    stream_indices_.Write(os, binary);
  }

  std::string Info() const {
    return std::string("\n  linearity");
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {

    // copy in to out
    out->CopyFromMat(in);

    // new one 
    std::vector<int> stream_indices_host;
    stream_indices_.CopyToVec(&stream_indices_host);

    int32 num_streams = stream_indices_host.size()-1;

    CuMatrix<BaseFloat> stream_block_dropout_mask;
    if ((stream_block_dropout_mask.NumRows() != in.NumRows()) || (stream_block_dropout_mask.NumCols() != num_streams)) {
      stream_block_dropout_mask.Resize(in.NumRows(), num_streams); 
    }
    stream_block_dropout_mask.Set(dropout_retention_); 
    rand_.BinarizeProbs(stream_block_dropout_mask, &stream_block_dropout_mask); 

    if ((block_dropout_mask_.NumRows() != in.NumRows()) || (block_dropout_mask_.NumCols() != in.NumCols())) {
      block_dropout_mask_.Resize(in.NumRows(), in.NumCols());
    }
    block_dropout_mask_.Set(1.0);

    if (this_stream_mask.Dim() != in.NumRows()) {
      this_stream_mask.Resize(in.NumRows());
    }

    // create block_dropout_mask
    for (int32 j = 0; j < num_streams; j++) {
      this_stream_mask.CopyColFromMat(stream_block_dropout_mask, j);
      CuSubMatrix<BaseFloat> this_stream(block_dropout_mask_.ColRange(stream_indices_host[j],
								      stream_indices_host[j+1]
								      - stream_indices_host[j] ) );
      this_stream.MulRowsVec(this_stream_mask);
    }

    // Apply masking
    out->MulElements(block_dropout_mask_);

  }
  
  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    // use same mask on the error derivatives...
    in_diff->MulElements(block_dropout_mask_);
  }
 private:

  CuArray<int32> stream_indices_;
  float dropout_retention_; // TODO: add to options

  CuRand<BaseFloat> rand_;
  CuMatrix<BaseFloat> block_dropout_mask_;
  CuVector<BaseFloat> this_stream_mask; 
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_MULTISTREAM_MASK_COMPONENT_H_
