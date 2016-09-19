// nnet/nnet-max-pooling-component.h

// Copyright 2014  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_MAX_POOLING_COMPONENT_H_
#define KALDI_NNET_NNET_MAX_POOLING_COMPONENT_H_

#include <string>
#include <vector>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

/**
 * MaxPoolingComponent :
 * The input/output matrices are split to submatrices with width 'pool_stride_'.
 * The pooling is done over 3rd axis, of the set of 2d matrices.
 * Our pooling supports overlaps, overlaps occur when (pool_step_ < pool_size_).
 */
class MaxPoolingComponent : public Component {
 public:
  MaxPoolingComponent(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out),
    pool_size_(0),
    pool_step_(0),
    pool_stride_(0)
  { }

  ~MaxPoolingComponent()
  { }

  Component* Copy() const { return new MaxPoolingComponent(*this); }
  ComponentType GetType() const { return kMaxPoolingComponent; }

  void InitData(std::istream &is) {
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<PoolSize>") ReadBasicType(is, false, &pool_size_);
      else if (token == "<PoolStep>") ReadBasicType(is, false, &pool_step_);
      else if (token == "<PoolStride>") ReadBasicType(is, false, &pool_stride_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (PoolSize|PoolStep|PoolStride)";
    }
    // check
    KALDI_ASSERT(pool_size_ != 0 && pool_step_ != 0 && pool_stride_ != 0);
  }

  void ReadData(std::istream &is, bool binary) {
    // pooling hyperparameters
    ExpectToken(is, binary, "<PoolSize>");
    ReadBasicType(is, binary, &pool_size_);
    ExpectToken(is, binary, "<PoolStep>");
    ReadBasicType(is, binary, &pool_step_);
    ExpectToken(is, binary, "<PoolStride>");
    ReadBasicType(is, binary, &pool_stride_);

    //
    // Sanity checks:
    //
    // number of patches:
    KALDI_ASSERT(input_dim_ % pool_stride_ == 0);
    int32 num_patches = input_dim_ / pool_stride_;
    // number of pools:
    KALDI_ASSERT((num_patches - pool_size_) % pool_step_ == 0);
    int32 num_pools = 1 + (num_patches - pool_size_) / pool_step_;
    // check output dim:
    KALDI_ASSERT(output_dim_ == num_pools * pool_stride_);
    //
  }

  void WriteData(std::ostream &os, bool binary) const {
    // pooling hyperparameters
    WriteToken(os, binary, "<PoolSize>");
    WriteBasicType(os, binary, pool_size_);
    WriteToken(os, binary, "<PoolStep>");
    WriteBasicType(os, binary, pool_step_);
    WriteToken(os, binary, "<PoolStride>");
    WriteBasicType(os, binary, pool_stride_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // useful dims
    int32 num_patches = input_dim_ / pool_stride_;
    int32 num_pools = 1 + (num_patches - pool_size_) / pool_step_;

    // do the max-pooling (pools indexed by q)
    for (int32 q = 0; q < num_pools; q++) {
      // get output buffer of the pool
      CuSubMatrix<BaseFloat> pool(out->ColRange(q*pool_stride_, pool_stride_));
      pool.Set(-1e20);  // reset (large negative value)
      for (int32 r = 0; r < pool_size_; r++) {  // max
        int32 p = r + q * pool_step_;  // p = input patch
        pool.Max(in.ColRange(p*pool_stride_, pool_stride_));
      }
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // useful dims
    int32 num_patches = input_dim_ / pool_stride_;
    int32 num_pools = 1 + (num_patches - pool_size_) / pool_step_;

    //
    // here we note how many diff matrices are summed for each input patch,
    std::vector<int32> patch_summands(num_patches, 0);
    // this metainfo will be used to divide diff of patches
    // used in more than one pool.
    //

    in_diff->SetZero();  // reset

    for (int32 q = 0; q<num_pools; q++) {  // sum
      for (int32 r = 0; r<pool_size_; r++) {
        int32 p = r + q * pool_step_;  // patch number
        //
        CuSubMatrix<BaseFloat> in_p(in.ColRange(p*pool_stride_, pool_stride_));
        CuSubMatrix<BaseFloat> out_q(out.ColRange(q*pool_stride_, pool_stride_));
        //
        CuSubMatrix<BaseFloat> tgt(in_diff->ColRange(p*pool_stride_, pool_stride_));
        CuMatrix<BaseFloat> src(out_diff.ColRange(q*pool_stride_, pool_stride_));

        // Only the pool-inputs with 'max-values' are used to back-propagate into,
        // the rest of derivatives is zeroed-out by a mask.
        CuMatrix<BaseFloat> mask;
        in_p.EqualElementMask(out_q, &mask);
        src.MulElements(mask);
        tgt.AddMat(1.0, src);

        patch_summands[p] += 1;
      }
    }

    // divide diff by #summands (compensate for patches used in more pools)
    for (int32 p = 0; p < num_patches; p++) {
      CuSubMatrix<BaseFloat> tgt(in_diff->ColRange(p*pool_stride_, pool_stride_));
      KALDI_ASSERT(patch_summands[p] > 0);  // patch at least in one pool
      tgt.Scale(1.0/patch_summands[p]);
    }
  }

 private:
  int32 pool_size_,    // input patches used for pooling
        pool_step_,    // shift used for pooling (allow overlapping pools)
        pool_stride_;  // stride used to slice input to a vector of matrices
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_MAX_POOLING_COMPONENT_H_
