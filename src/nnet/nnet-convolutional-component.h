// nnet/nnet-convolutional-component.h

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


#ifndef KALDI_NNET_NNET_CONVOLUTIONAL_COMPONENT_H_
#define KALDI_NNET_NNET_CONVOLUTIONAL_COMPONENT_H_

#include <string>
#include <vector>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

/**
 * ConvolutionalComponent implements convolution over single axis
 * (i.e. frequency axis in case we are the 1st component in NN).
 * We don't do convolution along temporal axis, which simplifies the
 * implementation (and was not helpful for Tara).
 *
 * We assume the input featrues are spliced, i.e. each frame
 * is in fact a set of stacked frames, where we can form patches
 * which span over several frequency bands and whole time axis.
 *
 * The convolution is done over whole axis with same filters,
 * i.e. we don't use separate filters for different 'regions'
 * of frequency axis.
 *
 * In order to have a fast implementations, the filters
 * are represented in vectorized form, where each rectangular
 * filter corresponds to a row in a matrix, where all the filters
 * are stored. The features are then re-shaped to a set of matrices,
 * where one matrix corresponds to single patch-position,
 * where all the filters get applied.
 *
 * The type of convolution is controled by hyperparameters:
 * patch_dim_     ... frequency axis size of the patch
 * patch_step_    ... size of shift in the convolution
 * patch_stride_  ... shift for 2nd dim of a patch
 *                    (i.e. frame length before splicing)
 *
 * Due to convolution same weights are used repeateadly,
 * the final gradient is a sum of all position-specific
 * gradients (the sum was found better than averaging).
 *
 */
class ConvolutionalComponent : public UpdatableComponent {
 public:
  ConvolutionalComponent(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out),
    patch_dim_(0),
    patch_step_(0),
    patch_stride_(0),
    max_norm_(0.0)
  { }

  ~ConvolutionalComponent()
  { }

  Component* Copy() const { return new ConvolutionalComponent(*this); }
  ComponentType GetType() const { return kConvolutionalComponent; }

  void InitData(std::istream &is) {
    // define options
    BaseFloat bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1;
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<BiasMean>")    ReadBasicType(is, false, &bias_mean);
      else if (token == "<BiasRange>")   ReadBasicType(is, false, &bias_range);
      else if (token == "<PatchDim>")    ReadBasicType(is, false, &patch_dim_);
      else if (token == "<PatchStep>")   ReadBasicType(is, false, &patch_step_);
      else if (token == "<PatchStride>") ReadBasicType(is, false, &patch_stride_);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef_);
      else if (token == "<MaxNorm>") ReadBasicType(is, false, &max_norm_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|PatchDim|PatchStep|PatchStride)";
    }

    //
    // Sanity checks:
    //
    // splice (input are spliced frames):
    KALDI_ASSERT(input_dim_ % patch_stride_ == 0);
    int32 num_splice = input_dim_ / patch_stride_;
    KALDI_LOG << "num_splice " << num_splice;
    // number of patches:
    KALDI_ASSERT((patch_stride_ - patch_dim_) % patch_step_ == 0);
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    KALDI_LOG << "num_patches " << num_patches;
    // filter dim:
    int32 filter_dim = num_splice * patch_dim_;
    KALDI_LOG << "filter_dim " << filter_dim;
    // num filters:
    KALDI_ASSERT(output_dim_ % num_patches == 0);
    int32 num_filters = output_dim_ / num_patches;
    KALDI_LOG << "num_filters " << num_filters;
    //

    //
    // Initialize trainable parameters,
    //
    // Gaussian with given std_dev (mean = 0),
    filters_.Resize(num_filters, filter_dim);
    RandGauss(0.0, param_stddev, &filters_);
    // Uniform,
    bias_.Resize(num_filters);
    RandUniform(bias_mean, bias_range, &bias_);
  }

  void ReadData(std::istream &is, bool binary) {
    // convolution hyperparameters,
    ExpectToken(is, binary, "<PatchDim>");
    ReadBasicType(is, binary, &patch_dim_);
    ExpectToken(is, binary, "<PatchStep>");
    ReadBasicType(is, binary, &patch_step_);
    ExpectToken(is, binary, "<PatchStride>");
    ReadBasicType(is, binary, &patch_stride_);

    // variant-length list of parameters,
    bool end_loop = false;
    while (!end_loop) {
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'L': ExpectToken(is, binary, "<LearnRateCoef>");
          ReadBasicType(is, binary, &learn_rate_coef_);
          break;
        case 'B': ExpectToken(is, binary, "<BiasLearnRateCoef>");
          ReadBasicType(is, binary, &bias_learn_rate_coef_);
          break;
        case 'M': ExpectToken(is, binary, "<MaxNorm>");
          ReadBasicType(is, binary, &max_norm_);
          break;
        case '!': ExpectToken(is, binary, "<!EndOfComponent>");
        default: end_loop = true;
      }
    }

    // trainable parameters
    ExpectToken(is, binary, "<Filters>");
    filters_.Read(is, binary);
    ExpectToken(is, binary, "<Bias>");
    bias_.Read(is, binary);

    //
    // Sanity checks:
    //
    // splice (input are spliced frames):
    KALDI_ASSERT(input_dim_ % patch_stride_ == 0);
    int32 num_splice = input_dim_ / patch_stride_;
    // number of patches:
    KALDI_ASSERT((patch_stride_ - patch_dim_) % patch_step_ == 0);
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    // filter dim:
    int32 filter_dim = num_splice * patch_dim_;
    // num filters:
    KALDI_ASSERT(output_dim_ % num_patches == 0);
    int32 num_filters = output_dim_ / num_patches;
    // check parameter dims:
    KALDI_ASSERT(num_filters == filters_.NumRows());
    KALDI_ASSERT(num_filters == bias_.Dim());
    KALDI_ASSERT(filter_dim == filters_.NumCols());
    //
  }

  void WriteData(std::ostream &os, bool binary) const {
    // convolution hyperparameters
    WriteToken(os, binary, "<PatchDim>");
    WriteBasicType(os, binary, patch_dim_);
    WriteToken(os, binary, "<PatchStep>");
    WriteBasicType(os, binary, patch_step_);
    WriteToken(os, binary, "<PatchStride>");
    WriteBasicType(os, binary, patch_stride_);
    if (!binary) os << "\n";

    // re-scale learn rate
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    // max-norm regularization
    WriteToken(os, binary, "<MaxNorm>");
    WriteBasicType(os, binary, max_norm_);
    if (!binary) os << "\n";

    // trainable parameters
    WriteToken(os, binary, "<Filters>");
    if (!binary) os << "\n";
    filters_.Write(os, binary);
    WriteToken(os, binary, "<Bias>");
    if (!binary) os << "\n";
    bias_.Write(os, binary);
  }

  int32 NumParams() const {
    return filters_.NumRows()*filters_.NumCols() + bias_.Dim();
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 filters_num_elem = filters_.NumRows() * filters_.NumCols();
    gradient->Range(0, filters_num_elem).CopyRowsFromMat(filters_);
    gradient->Range(filters_num_elem, bias_.Dim()).CopyFromVec(bias_);
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 filters_num_elem = filters_.NumRows() * filters_.NumCols();
    params->Range(0, filters_num_elem).CopyRowsFromMat(filters_);
    params->Range(filters_num_elem, bias_.Dim()).CopyFromVec(bias_);
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 filters_num_elem = filters_.NumRows() * filters_.NumCols();
    filters_.CopyRowsFromVec(params.Range(0, filters_num_elem));
    bias_.CopyFromVec(params.Range(filters_num_elem, bias_.Dim()));
  }

  std::string Info() const {
    return std::string("\n  filters") + MomentStatistics(filters_) +
      ", lr-coef " + ToString(learn_rate_coef_) +
      ", max-norm " + ToString(max_norm_) +
      "\n  bias" + MomentStatistics(bias_) +
      ", lr-coef " + ToString(bias_learn_rate_coef_);
  }

  std::string InfoGradient() const {
    return std::string("\n  filters_grad") + MomentStatistics(filters_grad_) +
      ", lr-coef " + ToString(learn_rate_coef_) +
      ", max-norm " + ToString(max_norm_) +
      "\n  bias_grad" + MomentStatistics(bias_grad_) +
      ", lr-coef " + ToString(bias_learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // useful dims
    int32 num_splice = input_dim_ / patch_stride_;
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    int32 num_filters = filters_.NumRows();
    int32 num_frames = in.NumRows();
    int32 filter_dim = filters_.NumCols();

    // we will need the buffers
    if (vectorized_feature_patches_.NumRows() != num_frames) {
      vectorized_feature_patches_.Resize(num_frames, filter_dim * num_patches, kUndefined);
      feature_patch_diffs_.Resize(num_frames, filter_dim * num_patches, kSetZero);
    }

    /* Prepare feature patches, the layout is:
     * |----------|----------|----------|---------| (in = spliced frames)
     *   xxx        xxx        xxx        xxx       (x = selected elements)
     *
     *   xxx : patch dim
     *    xxx
     *   ^---: patch step
     * |----------| : patch stride
     *
     *   xxx-xxx-xxx-xxx : filter dim
     *
     */
    // build-up a column selection map:
    int32 index = 0;
    column_map_.resize(filter_dim * num_patches);
    for (int32 p = 0; p < num_patches; p++) {
      for (int32 s = 0; s < num_splice; s++) {
        for (int32 d = 0; d < patch_dim_; d++) {
          column_map_[index] = p * patch_step_ + s * patch_stride_ + d;
          index++;
        }
      }
    }
    // select the columns
    CuArray<int32> cu_column_map(column_map_);
    vectorized_feature_patches_.CopyCols(in, cu_column_map);

    // compute filter activations
    for (int32 p = 0; p < num_patches; p++) {
      CuSubMatrix<BaseFloat> tgt(out->ColRange(p * num_filters, num_filters));
      CuSubMatrix<BaseFloat> patch(vectorized_feature_patches_.ColRange(
                                   p * filter_dim, filter_dim));
      tgt.AddVecToRows(1.0, bias_, 0.0);  // add bias
      // apply all filters
      tgt.AddMatMat(1.0, patch, kNoTrans, filters_, kTrans, 1.0);
    }
  }

  /*
   This function does an operation similar to reversing a map,
   except it handles maps that are not one-to-one by outputting
   the reversed map as a vector of lists.
   @param[in] forward_indexes is a vector of int32, each of whose
              elements is between 0 and input_dim - 1.
   @param[in] input_dim. See definitions of forward_indexes and
              backward_indexes.
   @param[out] backward_indexes is a vector of dimension input_dim
              of lists, The list at (backward_indexes[i]) is a list
              of all indexes j such that forward_indexes[j] = i.
  */
  void ReverseIndexes(const std::vector<int32> &forward_indexes,
                      std::vector<std::vector<int32> > *backward_indexes) {
    int32 i;
    int32 size = forward_indexes.size();
    backward_indexes->resize(input_dim_);
    int32 reserve_size = 2+ forward_indexes.size() / input_dim_;
    std::vector<std::vector<int32> >::iterator iter = backward_indexes->begin(),
      end = backward_indexes->end();
    for (; iter != end; ++iter)
      iter->reserve(reserve_size);
    for (int32 j = 0; j < size; j++) {
      i = forward_indexes[j];
      KALDI_ASSERT(i < input_dim_);
      (*backward_indexes)[i].push_back(j);
    }
  }

  /*
   This function transforms a vector of lists into a list of vectors,
   padded with -1.
   @param[in] The input vector of lists. Let in.size() be D, and let
              the longest list length (i.e. the max of in[i].size()) be L.
   @param[out] The output list of vectors. The length of the list will
              be L, each vector-dimension will be D (i.e. out[i].size() == D),
              and if in[i] == j, then for some k we will have that
              out[k][j] = i. The output vectors are padded with -1
              where necessary if not all the input lists have the same side.
  */
  void RearrangeIndexes(const std::vector<std::vector<int32> > &in,
                        std::vector<std::vector<int32> > *out) {
    int32 D = in.size();
    int32 L = 0;
    for (int32 i = 0; i < D; i++)
      if (in[i].size() > L)
        L = in[i].size();
    out->resize(L);
    for (int32 i = 0; i < L; i++)
      (*out)[i].resize(D, -1);
    for (int32 i = 0; i < D; i++) {
      for (int32 j = 0; j < in[i].size(); j++) {
        (*out)[j][i] = in[i][j];
      }
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // useful dims
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    int32 num_filters = filters_.NumRows();
    int32 filter_dim = filters_.NumCols();

    // backpropagate to vector of matrices
    // (corresponding to position of a filter)
    for (int32 p = 0; p < num_patches; p++) {
      CuSubMatrix<BaseFloat> patch_diff(feature_patch_diffs_.ColRange(
                                        p * filter_dim, filter_dim));
      CuSubMatrix<BaseFloat> out_diff_patch(out_diff.ColRange(
                                            p * num_filters, num_filters));
      patch_diff.AddMatMat(1.0, out_diff_patch, kNoTrans,
                           filters_, kNoTrans, 0.0);
    }

    // sum the derivatives into in_diff, we will compensate #summands
    std::vector<std::vector<int32> > reversed_column_map;
    ReverseIndexes(column_map_, &reversed_column_map);
    std::vector<std::vector<int32> > rearranged_column_map;
    RearrangeIndexes(reversed_column_map, &rearranged_column_map);
    for (int32 p = 0; p < rearranged_column_map.size(); p++) {
      CuArray<int32> cu_cols(rearranged_column_map[p]);
      in_diff->AddCols(feature_patch_diffs_, cu_cols);
    }
  }


  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    // useful dims
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    int32 num_filters = filters_.NumRows();
    int32 filter_dim = filters_.NumCols();

    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate;

    //
    // calculate the gradient
    //
    filters_grad_.Resize(num_filters, filter_dim, kSetZero);  // reset
    bias_grad_.Resize(num_filters, kSetZero);  // reset
    // use all the patches
    for (int32 p = 0; p < num_patches; p++) {  // sum
      CuSubMatrix<BaseFloat> diff_patch(diff.ColRange(p * num_filters,
                                                      num_filters));
      CuSubMatrix<BaseFloat> patch(vectorized_feature_patches_.ColRange(
                                   p * filter_dim, filter_dim));
      filters_grad_.AddMatMat(1.0, diff_patch, kTrans, patch, kNoTrans, 1.0);
      bias_grad_.AddRowSumMat(1.0, diff_patch, 1.0);
    }

    //
    // update
    //
    filters_.AddMat(-lr*learn_rate_coef_, filters_grad_);
    bias_.AddVec(-lr*bias_learn_rate_coef_, bias_grad_);
    //

    // max-norm
    if (max_norm_ > 0.0) {
      CuMatrix<BaseFloat> lin_sqr(filters_);
      lin_sqr.MulElements(filters_);
      CuVector<BaseFloat> l2(filters_.NumRows());
      l2.AddColSumMat(1.0, lin_sqr, 0.0);
      l2.ApplyPow(0.5);  // we have per-neuron L2 norms
      CuVector<BaseFloat> scl(l2);
      scl.Scale(1.0/max_norm_);
      scl.ApplyFloor(1.0);
      scl.InvertElements();
      filters_.MulRowsVec(scl);  // shink to sphere!
    }
  }

 private:
  int32 patch_dim_,    ///< number of consecutive inputs, 1st dim of patch
        patch_step_,   ///< step of the convolution
                       ///  (i.e. shift between 2 patches)
        patch_stride_;  ///< shift for 2nd dim of a patch
                       ///  (i.e. frame length before splicing)

  CuMatrix<BaseFloat> filters_;  ///< row = vectorized rectangular filter
  CuVector<BaseFloat> bias_;  ///< bias for each filter

  CuMatrix<BaseFloat> filters_grad_;  ///< gradient of filters
  CuVector<BaseFloat> bias_grad_;  ///< gradient of biases

  BaseFloat max_norm_;  ///< limit L2 norm of a neuron weights to positive value

  /** Buffer of reshaped inputs:
   *  1row = vectorized rectangular feature patches,
   *  1col = dim over speech frames
   *  Map of input features:
   *  std::vector-dim = patch-position
   */
  CuMatrix<BaseFloat> vectorized_feature_patches_;
  std::vector<int32> column_map_;

  /** Buffer for backpropagation:
   *  derivatives in the domain of 'vectorized_feature_patches_',
   *  1row = vectorized rectangular feature patches,
   *  1col = dim over speech frames,
   */
  CuMatrix<BaseFloat> feature_patch_diffs_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_CONVOLUTIONAL_COMPONENT_H_
