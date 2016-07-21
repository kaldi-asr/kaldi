// nnet/nnet-affine-transform.h

// Copyright 2011-2014  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_MULTISTREAM_MASK_COMPONENT_H_
#define KALDI_NNET_MULTISTREAM_MASK_COMPONENT_H_

#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class MultiStreamMaskComponent : public Component {
 public:
  MultiStreamMaskComponent(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out),
      stream_dropout_retention_(0.5)
  { }
    ~MultiStreamMaskComponent()
  { }

  Component* Copy() const { return new MultiStreamMaskComponent(*this); }
  ComponentType GetType() const { return kMultiStreamMaskComponent; }

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

    CuMatrix<BaseFloat> block_dropout_mask;
    if ((block_dropout_mask.NumRows() != in.NumRows()) || (block_dropout_mask.NumCols() != num_streams)) {
      block_dropout_mask.Resize(in.NumRows(), num_streams);
    }
    block_dropout_mask.SetZero();

    CuMatrix<BaseFloat> this_stream_dropout_mask;
    CuVector<BaseFloat> this_stream_mask;
    CuRand<BaseFloat> rand_;

    if (this_stream_mask.Dim() != in.NumRows()) {
      this_stream_dropout_mask.Resize(in.NumRows(), 1);
      this_stream_mask.Resize(in.NumRows());
    }

    // create block_dropout_mask
    for (int32 j = 0; j < num_streams; j++) {
      this_stream_dropout_mask.Set(stream_dropout_retention_); 
      rand_.BinarizeProbs(this_stream_dropout_mask, &this_stream_dropout_mask); 

      this_stream_mask.CopyColFromMat(this_stream_dropout_mask, 0); 
      block_dropout_mask.CopyColFromVec(this_stream_mask, j);
    }

    for (int32 j = 0; j < num_streams; j++) {     
      this_stream_mask.CopyColFromMat(block_dropout_mask, j);

      CuSubMatrix<BaseFloat> this_stream(out->ColRange(stream_indices_host[j],
    						       stream_indices_host[j+1]
    						       - stream_indices_host[j] ) );
      this_stream.MulRowsVec(this_stream_mask);
    }
    
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {

    static bool warning_displayed = false;
    if (!warning_displayed) {
      KALDI_WARN << Component::TypeToMarker(GetType()) << " : "
                 << __func__ << "() Not implemented!";

      warning_displayed = true;
    }
    in_diff->SetZero();
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {

    static bool warning_displayed = false;
    if (!warning_displayed) {
      KALDI_WARN << Component::TypeToMarker(GetType()) << " : "
                 << __func__ << "() Not implemented!";

      warning_displayed = true;
    }

  }

 private:
  // std::vector<int32> stream_indices_;
  CuArray<int32> stream_indices_;
  float stream_dropout_retention_; // TODO: add to options

  /* 
  CuMatrix<BaseFloat> this_stream_dropout_mask;
  CuVector<BaseFloat> this_stream_mask;
  CuRand<BaseFloat> rand_;
  */
  
 
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_MULTISTREAM_MASK_COMPONENT_H_
