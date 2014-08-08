// nnet/nnet-average-pooling-component.h

// Copyright 2014  Brno University of Technology (author: Karel Vesely)
//                 Johns Hopkins University (author: Sri Harish Mallidi)

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


#ifndef KALDI_NNET_NNET_AVERAGE_POOLING2D_COMPONENT_H_
#define KALDI_NNET_NNET_AVERAGE_POOLING2D_COMPONENT_H_


#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

/**
 * AveragePoolingComponent :
 * The input/output matrices are split to submatrices with width 'pool_stride_'.
 * The pooling is done over 3rd axis, of the set of 2d matrices.
 * Our pooling supports overlaps, overlaps occur when (pool_step_ < pool_size_).
 */
class AveragePooling2DComponent : public Component {
 public:
  AveragePooling2DComponent(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out), 
      fmap_x_len_(0), fmap_y_len_(0), pool_x_len_(0), pool_y_len_(0), pool_x_step_(0), pool_y_step_(0)
  { }
  ~AveragePooling2DComponent()
  { }

  Component* Copy() const { return new AveragePooling2DComponent(*this); }
  ComponentType GetType() const { return kAveragePooling2DComponent; }
  
  void InitData(std::istream &is) {
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<FmapXLen>") ReadBasicType(is, false, &fmap_x_len_);
      else if (token == "<FmapYLen>") ReadBasicType(is, false, &fmap_y_len_);
      else if (token == "<PoolXLen>") ReadBasicType(is, false, &pool_x_len_);
      else if (token == "<PoolYLen>") ReadBasicType(is, false, &pool_y_len_);
      else if (token == "<PoolXStep>") ReadBasicType(is, false, &pool_x_step_);
      else if (token == "<PoolYStep>") ReadBasicType(is, false, &pool_y_step_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (FmapXLen|FmapYLen|PoolXLen|PoolYLen|PoolXStep|PoolYStep)";
      is >> std::ws; // eat-up whitespace
    }
    // check
    KALDI_ASSERT(fmap_x_len_ * fmap_y_len_ * pool_x_len_ * pool_y_len_ * pool_x_step_ * pool_y_step_  != 0 );
  }

  void ReadData(std::istream &is, bool binary) {
    // pooling hyperparameters
    ExpectToken(is, binary, "<FmapXLen>");
    ReadBasicType(is, binary, &fmap_x_len_);
    ExpectToken(is, binary, "<FmapYLen>");
    ReadBasicType(is, binary, &fmap_y_len_);
    ExpectToken(is, binary, "<PoolXLen>");
    ReadBasicType(is, binary, &pool_x_len_);
    ExpectToken(is, binary, "<PoolYLen>");
    ReadBasicType(is, binary, &pool_y_len_);
    ExpectToken(is, binary, "<PoolXStep>");
    ReadBasicType(is, binary, &pool_x_step_);
    ExpectToken(is, binary, "<PoolYStep>");
    ReadBasicType(is, binary, &pool_y_step_);

    // 
    // Sanity checks:
    //
    // input sanity checks
    // input_dim_ should be multiple of (fmap_x_len_ * fmap_y_len_)
    KALDI_ASSERT(input_dim_ % (fmap_x_len_ * fmap_y_len_) == 0);
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);
    KALDI_LOG << "num_fmaps " << num_input_fmaps;
    // check if step is in sync with fmap_len and filt_len
    KALDI_ASSERT((fmap_x_len_ - pool_x_len_) % (pool_x_step_) == 0);
    KALDI_ASSERT((fmap_y_len_ - pool_y_len_) % (pool_y_step_) == 0);
    int32 out_fmap_x_len = (fmap_x_len_ - pool_x_len_)/pool_x_step_ + 1;
    int32 out_fmap_y_len = (fmap_y_len_ - pool_y_len_)/pool_y_step_ + 1;
    //    int32 out_fmap_size = out_fmap_x_len*out_fmap_y_len;
    // output sanity checks
    KALDI_ASSERT(output_dim_ % (out_fmap_x_len * out_fmap_y_len)  == 0);
    int32 num_output_fmaps = output_dim_ / (out_fmap_x_len * out_fmap_y_len);
    KALDI_ASSERT(num_input_fmaps == num_output_fmaps);

  }

  void WriteData(std::ostream &os, bool binary) const {
    // pooling hyperparameters
    WriteToken(os, binary, "<FmapXLen>");
    WriteBasicType(os, binary, fmap_x_len_);
    WriteToken(os, binary, "<FmapYLen>");
    WriteBasicType(os, binary, fmap_y_len_);
    WriteToken(os, binary, "<PoolXLen>");
    WriteBasicType(os, binary, pool_x_len_);
    WriteToken(os, binary, "<PoolYLen>");
    WriteBasicType(os, binary, pool_y_len_);
    WriteToken(os, binary, "<PoolXStep>");
    WriteBasicType(os, binary, pool_x_step_);
    WriteToken(os, binary, "<PoolYStep>");
    WriteBasicType(os, binary, pool_y_step_);

  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    
        // useful dims
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);
    // int32 inp_fmap_size = fmap_x_len_ * fmap_y_len_;
    // int32 out_fmap_x_len = (fmap_x_len_ - pool_x_len_)/pool_x_step_ + 1;
    // int32 out_fmap_y_len = (fmap_y_len_ - pool_y_len_)/pool_y_step_ + 1;
    // int32 out_fmap_size = out_fmap_x_len*out_fmap_y_len;
    // int32 num_output_fmaps = num_input_fmaps;
    // int32 num_frames = in.NumRows();
    
    int out_fmap_cnt=0;
    for (int32 m=0; m < fmap_x_len_-pool_x_len_+1;m=m+pool_x_step_){
      for (int32 n=0; n< fmap_y_len_-pool_y_len_+1; n=n+pool_y_step_){
	int32 st=0;
	st=(m*fmap_y_len_+n)*num_input_fmaps;	  
	CuSubMatrix<BaseFloat> pool(out->ColRange(out_fmap_cnt*num_input_fmaps, num_input_fmaps));
	pool.SetZero(); // reset
	for (int32 i=0; i< pool_x_len_; i++){
	  for (int32 j=0; j< pool_y_len_; j++){
	    int32 c=0;
	    c=st+i*(num_input_fmaps*fmap_y_len_)+j*num_input_fmaps;
	    pool.AddMat(1.0, in.ColRange(c, num_input_fmaps));
	    }
	  }
	pool.Scale(1.0/(pool_x_len_*pool_y_len_));
	out_fmap_cnt++;
	}
      }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {

    // useful dims
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);
    int32 inp_fmap_size = fmap_x_len_ * fmap_y_len_;
    // int32 out_fmap_x_len = (fmap_x_len_ - pool_x_len_)/pool_x_step_ + 1;
    // int32 out_fmap_y_len = (fmap_y_len_ - pool_y_len_)/pool_y_step_ + 1;
    // int32 out_fmap_size = out_fmap_x_len*out_fmap_y_len;
    // int32 num_output_fmaps = output_dim_ / (out_fmap_x_len * out_fmap_y_len);
    // int32 num_frames = in.NumRows();

    //
    // here we note how many diff matrices are summed for each input patch,
    std::vector<int32> patch_summands(inp_fmap_size, 0);
    // this metainfo will be used to divide diff of patches 
    // used in more than one pool.
    //

    in_diff->SetZero(); // reset

    int out_fmap_cnt=0;
    for (int32 m=0; m < fmap_x_len_-pool_x_len_+1;m=m+pool_x_step_){
      for (int32 n=0; n< fmap_y_len_-pool_y_len_+1; n=n+pool_y_step_){
	int32 st=0;
	st=(m*fmap_y_len_+n)*num_input_fmaps;	  
	CuSubMatrix<BaseFloat> src(out_diff.ColRange(out_fmap_cnt*num_input_fmaps, num_input_fmaps));
	for (int32 i=0; i< pool_x_len_; i++){
	  for (int32 j=0; j< pool_y_len_; j++){
	    int32 c=0;
	    c=st+i*(num_input_fmaps*fmap_y_len_)+j*num_input_fmaps;
	    CuSubMatrix<BaseFloat> tgt(in_diff->ColRange(c, num_input_fmaps));
	    tgt.AddMat(1.0, src);
	    patch_summands[c/num_input_fmaps] += 1;
	    }
	  }
	out_fmap_cnt++;
	}
      }
    
    // divide diff by average-pooling-dim (derivative of averaging)
    in_diff->Scale(1.0/(pool_x_len_*pool_y_len_));

    // divide diff by #summands (compensate for patches used in more pools)
    for (int i=0; i<fmap_x_len_; i++){
      for (int32 j=0; j<fmap_y_len_; j++){
	int32 c=i*fmap_y_len_+j;
	CuSubMatrix<BaseFloat> tgt(in_diff->ColRange(c*num_input_fmaps, num_input_fmaps));
	KALDI_ASSERT(patch_summands[c] > 0); // patch at least in one pool
	tgt.Scale(1.0/patch_summands[c]);
      }
    }
  }

 private:
  int32 fmap_x_len_, fmap_y_len_,
    pool_x_len_, pool_y_len_,
    pool_x_step_, pool_y_step_;

};

} // namespace nnet1
} // namespace kaldi

#endif
