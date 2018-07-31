// nnet/nnet-frame-pooling-component.h

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


#ifndef KALDI_NNET_NNET_FRAME_POOLING_COMPONENT_H_
#define KALDI_NNET_NNET_FRAME_POOLING_COMPONENT_H_

#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

/**
 * FramePoolingComponent :
 * The input/output matrices are split to frames of width 'feature_dim_'.
 * Here we do weighted pooling of frames along the temporal axis,
 * given a frame-offset of leftmost frame, the pool-size is defined
 * by weight-vector size.
 */
class FramePoolingComponent : public UpdatableComponent {
 public:
  FramePoolingComponent(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out),
    feature_dim_(0),
    normalize_(false)
  { }

  ~FramePoolingComponent()
  { }

  Component* Copy() const { return new FramePoolingComponent(*this); }
  ComponentType GetType() const { return kFramePoolingComponent; }

  /**
   * Here the offsets are w.r.t. central frames, which has offset 0.
   * Note.: both the offsets and pool sizes can be negative.
   */
  void InitData(std::istream &is) {
    // temporary, for initialization,
    std::vector<int32> pool_size;
    std::vector<int32> central_offset;
    Vector<BaseFloat> pool_weight;
    float learn_rate_coef = 0.01;
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<FeatureDim>") ReadBasicType(is, false, &feature_dim_);
      else if (token == "<CentralOffset>") ReadIntegerVector(is, false, &central_offset);
      else if (token == "<PoolSize>") ReadIntegerVector(is, false, &pool_size);
      else if (token == "<PoolWeight>") pool_weight.Read(is, false);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
      else if (token == "<Normalize>") ReadBasicType(is, false, &normalize_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (FeatureDim|CentralOffset <vec>|PoolSize <vec>|LearnRateCoef|Normalize)";
    }
    // check inputs:
    KALDI_ASSERT(feature_dim_ > 0);
    KALDI_ASSERT(central_offset.size() > 0);
    KALDI_ASSERT(central_offset.size() == pool_size.size());
    // initialize:
    int32 num_frames = InputDim() / feature_dim_;
    int32 central_frame = (num_frames -1) / 2;
    int32 num_pools = central_offset.size();
    offset_.resize(num_pools);
    weight_.resize(num_pools);
    for (int32 p = 0; p < num_pools; p++) {
      offset_[p] = central_frame + central_offset[p] + std::min(0, pool_size[p]+1);
      weight_[p].Resize(std::abs(pool_size[p]));
      weight_[p].Set(1.0/std::abs(pool_size[p]));
    }
    learn_rate_coef_ = learn_rate_coef;
    if (pool_weight.Dim() != 0) {
      KALDI_LOG << "Initializing from pool-weight vector";
      int32 num_weights = 0;
      for (int32 p = 0; p < num_pools; p++) {
        weight_[p].CopyFromVec(pool_weight.Range(num_weights, weight_[p].Dim()));
        num_weights += weight_[p].Dim();
      }
      KALDI_ASSERT(num_weights == pool_weight.Dim());
    }
    // check that offsets are within the splice we had,
    for (int32 p = 0; p < num_pools; p++) {
      KALDI_ASSERT(offset_[p] >= 0);
      KALDI_ASSERT(offset_[p] + weight_[p].Dim() <= num_frames);
    }
  }

  /**
   * Here the offsets are w.r.t. leftmost frame from splice, its offset is 0.
   * If we spliced +/- 15 frames, the central frames has index '15'.
   */
  void ReadData(std::istream &is, bool binary) {
    // get the input dimension before splicing
    ExpectToken(is, binary, "<FeatureDim>");
    ReadBasicType(is, binary, &feature_dim_);
    ExpectToken(is, binary, "<LearnRateCoef>");
    ReadBasicType(is, binary, &learn_rate_coef_);
    ExpectToken(is, binary, "<Normalize>");
    ReadBasicType(is, binary, &normalize_);
    // read the offsets w.r.t. central frame
    ExpectToken(is, binary, "<FrameOffset>");
    ReadIntegerVector(is, binary, &offset_);
    // read the frame-weights
    ExpectToken(is, binary, "<FrameWeight>");
    int32 num_pools = offset_.size();
    weight_.resize(num_pools);
    for (int32 p = 0; p < num_pools; p++) {
      weight_[p].Read(is, binary);
    }
    //
    // Sanity checks:
    //
    KALDI_ASSERT(input_dim_ % feature_dim_ == 0);
    KALDI_ASSERT(output_dim_ % feature_dim_ == 0);
    KALDI_ASSERT(output_dim_ / feature_dim_ == num_pools);
    KALDI_ASSERT(offset_.size() == weight_.size());
    // check the shifts don't exceed the splicing
    int32 total_frame = InputDim() / feature_dim_;
    for (int32 p = 0; p < num_pools; p++) {
      KALDI_ASSERT(offset_[p] >= 0);
      KALDI_ASSERT(offset_[p] + (weight_[p].Dim()-1) < total_frame);
    }
    //
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<FeatureDim>");
    WriteBasicType(os, binary, feature_dim_);
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<Normalize>");
    WriteBasicType(os, binary, normalize_);
    WriteToken(os, binary, "<FrameOffset>");
    WriteIntegerVector(os, binary, offset_);
    // write pooling weights of individual frames
    WriteToken(os, binary, "<FrameWeight>");
    int32 num_pools = offset_.size();
    for (int32 p = 0; p < num_pools; p++) {
      weight_[p].Write(os, binary);
    }
  }

  int32 NumParams() const {
    int32 ans = 0;
    for (int32 p = 0; p < weight_.size(); p++) {
      ans += weight_[p].Dim();
    }
    return ans;
  }

  void GetGradient(VectorBase<BaseFloat> *gradient) const {
    KALDI_ERR << "Unimplemented.";
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 offset = 0;
    for (int32 p = 0; p < weight_.size(); p++) {
      params->Range(offset, weight_[p].Dim()).CopyFromVec(weight_[p]);
      offset += weight_[p].Dim();
    }
    KALDI_ASSERT(offset == params->Dim());
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ERR << "Unimplemented.";
  }

  std::string Info() const {
    std::ostringstream oss;
    oss << "\n  (offset,weights) : ";
    for (int32 p = 0; p < weight_.size(); p++) {
      oss << "(" << offset_[p] << "," << weight_[p] << "), ";
    }
    return oss.str();
  }

  std::string InfoGradient() const {
    std::ostringstream oss;
    oss << "\n  lr-coef " << ToString(learn_rate_coef_);
    oss << "\n  (offset,weights_grad) : ";
    for (int32 p = 0; p < weight_diff_.size(); p++) {
      oss << "(" << offset_[p] << ",";
      // pass the weight vector, remove '\n' as last char
      oss << weight_diff_[p];
      oss.seekp(-1, std::ios_base::cur);
      oss << "), ";
    }
    return oss.str();
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // check dims
    KALDI_ASSERT(in.NumCols() % feature_dim_ == 0);
    KALDI_ASSERT(out->NumCols() % feature_dim_ == 0);
    // useful dims
    int32 num_pools = offset_.size();
    // compute the output pools
    for (int32 p = 0; p < num_pools; p++) {
      CuSubMatrix<BaseFloat> tgt(out->ColRange(p*feature_dim_, feature_dim_));
      tgt.SetZero();  // reset
      for (int32 i = 0; i < weight_[p].Dim(); i++) {
        tgt.AddMat(weight_[p](i), in.ColRange((offset_[p]+i) * feature_dim_, feature_dim_));
      }
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    KALDI_ERR << "Unimplemented.";
  }


  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    // useful dims
    int32 num_pools = offset_.size();
    // lazy init
    if (weight_diff_.size() != num_pools) weight_diff_.resize(num_pools);
    // get the derivatives
    for (int32 p = 0; p < num_pools; p++) {
      weight_diff_[p].Resize(weight_[p].Dim(), kSetZero);  // reset
      for (int32 i = 0; i < weight_[p].Dim(); i++) {
        // multiply matrices element-wise, and sum to get the derivative
        CuSubMatrix<BaseFloat> in_frame(
          input.ColRange((offset_[p]+i) * feature_dim_, feature_dim_)
        );
        CuSubMatrix<BaseFloat> diff_frame(
          diff.ColRange(p * feature_dim_, feature_dim_)
        );
        CuMatrix<BaseFloat> mul_elems(in_frame);
        mul_elems.MulElements(diff_frame);
        weight_diff_[p](i) = mul_elems.Sum();
      }
    }
    // update
    for (int32 p = 0; p < num_pools; p++) {
      weight_[p].AddVec(- learn_rate_coef_ * opts_.learn_rate, weight_diff_[p]);
    }
    // force to be positive, re-normalize the sum
    if (normalize_) {
      for (int32 p = 0; p < num_pools; p++) {
        weight_[p].ApplyFloor(0.0);
        weight_[p].Scale(1.0/weight_[p].Sum());
      }
    }
  }

 private:
  int32 feature_dim_;  // feature dimension before splicing
  std::vector<int32> offset_;  // vector of pooling offsets
  /// Vector of pooling weight vectors,
  std::vector<Vector<BaseFloat> > weight_;
  /// detivatives of weight vectors,
  std::vector<Vector<BaseFloat> > weight_diff_;

  bool normalize_;  // apply normalization after each update
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_FRAME_POOLING_COMPONENT_H_
