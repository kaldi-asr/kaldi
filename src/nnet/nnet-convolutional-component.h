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


#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
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
 * filter corresponds to a row in a matrix, where all filters 
 * are stored. The features are then re-shaped to a set of matrices, 
 * where one matrix corresponds to single patch-position, 
 * where the filters get applied.
 * 
 * The type of convolution is controled by hyperparameters:
 * patch_dim_     ... frequency axis size of the patch
 * patch_step_    ... size of shift in the convolution
 * patch_stride_  ... shift for 2nd dim of a patch 
 *                    (i.e. frame length before splicing)
 *
 * Due to convolution same weights are used repeateadly, 
 * the final gradient is average of all position-specific 
 * gradients.
 *
 */
class ConvolutionalComponent : public UpdatableComponent {
 public:
  ConvolutionalComponent(int32 dim_in, int32 dim_out) 
    : UpdatableComponent(dim_in, dim_out),
      patch_dim_(0), patch_step_(0), patch_stride_(0)  
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
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<BiasMean>")    ReadBasicType(is, false, &bias_mean);
      else if (token == "<BiasRange>")   ReadBasicType(is, false, &bias_range);
      else if (token == "<PatchDim>")    ReadBasicType(is, false, &patch_dim_);
      else if (token == "<PatchStep>")   ReadBasicType(is, false, &patch_step_);
      else if (token == "<PatchStride>") ReadBasicType(is, false, &patch_stride_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|PatchDim|PatchStep|PatchStride)";
      is >> std::ws; // eat-up whitespace
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
    // Initialize parameters
    //
    Matrix<BaseFloat> mat(num_filters, filter_dim);
    for(int32 r=0; r<num_filters; r++) {
      for(int32 c=0; c<filter_dim; c++) {
        mat(r,c) = param_stddev * RandGauss(); // 0-mean Gauss with given std_dev
      }
    }
    filters_ = mat;
    //
    Vector<BaseFloat> vec(num_filters);
    for(int32 i=0; i<num_filters; i++) {
      // +/- 1/2*bias_range from bias_mean:
      vec(i) = bias_mean + (RandUniform() - 0.5) * bias_range; 
    }
    bias_ = vec;
    //
  }

  void ReadData(std::istream &is, bool binary) {
    // convolution hyperparameters
    ExpectToken(is, binary, "<PatchDim>");
    ReadBasicType(is, binary, &patch_dim_);
    ExpectToken(is, binary, "<PatchStep>");
    ReadBasicType(is, binary, &patch_step_);
    ExpectToken(is, binary, "<PatchStride>");
    ReadBasicType(is, binary, &patch_stride_);
 
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
 
    // trainable parameters
    WriteToken(os, binary, "<Filters>");
    filters_.Write(os, binary);
    WriteToken(os, binary, "<Bias>");
    bias_.Write(os, binary);
  }

  int32 NumParams() const { 
    return filters_.NumRows()*filters_.NumCols() + bias_.Dim(); 
  }
  
  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());
    int32 filters_num_elem = filters_.NumRows() * filters_.NumCols();
    wei_copy->Range(0,filters_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(filters_));
    wei_copy->Range(filters_num_elem, bias_.Dim()).CopyFromVec(Vector<BaseFloat>(bias_));
  }

  std::string Info() const {
    return std::string("\n  filters") + MomentStatistics(filters_) +
           "\n  bias" + MomentStatistics(bias_);
  }
  std::string InfoGradient() const {
    return std::string("\n  filters_grad") + MomentStatistics(filters_grad_) +
           "\n  bias_grad" + MomentStatistics(bias_grad_);
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // useful dims
    int32 num_splice = input_dim_ / patch_stride_;
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    int32 num_filters = filters_.NumRows();
    int32 num_frames = in.NumRows();
    int32 filter_dim = filters_.NumCols();

    // we will need the buffers 
    if (vectorized_feature_patches_.size() == 0) {
      vectorized_feature_patches_.resize(num_patches);
      feature_patch_diffs_.resize(num_patches);
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
    for (int32 p=0; p<num_patches; p++) {
      vectorized_feature_patches_[p].Resize(num_frames, filter_dim, kSetZero);
      // build-up a column selection mask:
      std::vector<int32> column_mask;
      for (int32 s=0; s<num_splice; s++) {
        for (int32 d=0; d<patch_dim_; d++) {
          column_mask.push_back(p * patch_step_ + s * patch_stride_ + d);
        }
      }
      KALDI_ASSERT(column_mask.size() == filter_dim);
      // select the columns
      vectorized_feature_patches_[p].CopyCols(in, column_mask);
    }

    // compute filter activations
    for (int32 p=0; p<num_patches; p++) {
      CuSubMatrix<BaseFloat> tgt(out->ColRange(p * num_filters, num_filters));
      tgt.AddVecToRows(1.0, bias_, 0.0); // add bias
      // apply all filters
      tgt.AddMatMat(1.0, vectorized_feature_patches_[p], kNoTrans, filters_, kTrans, 1.0);
    }
  }


  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
    // useful dims
    int32 num_splice = input_dim_ / patch_stride_;
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    int32 num_filters = filters_.NumRows();
    int32 num_frames = in.NumRows();
    int32 filter_dim = filters_.NumCols();

    // backpropagate to vector of matrices (corresponding to position of a filter)
    for (int32 p=0; p<num_patches; p++) {
      feature_patch_diffs_[p].Resize(num_frames, filter_dim, kSetZero); // reset
      CuSubMatrix<BaseFloat> out_diff_patch(out_diff.ColRange(p * num_filters, num_filters));
      feature_patch_diffs_[p].AddMatMat(1.0, out_diff_patch, kNoTrans, filters_, kNoTrans, 0.0);
    }

    // sum the derivatives into in_diff, we will compensate #summands
    // TODO: rewrite to use : std::vector<int32>
    in_diff_summands_.Resize(in_diff->NumCols(), kSetZero); // reset
    for (int32 p=0; p<num_patches; p++) {
      for (int32 s=0; s<num_splice; s++) {
        CuSubMatrix<BaseFloat> src(feature_patch_diffs_[p].ColRange(s * patch_dim_, patch_dim_));
        CuSubMatrix<BaseFloat> tgt(in_diff->ColRange(p * patch_step_ + s * patch_stride_, patch_dim_));
        tgt.AddMat(1.0, src); // sum
        // add 1.0 to keep track of target columns in the sum
        in_diff_summands_.Range(p * patch_step_ + s * patch_stride_, patch_dim_).Add(1.0);
      }
    }
    // compensate #summands
    in_diff_summands_.InvertElements();
    in_diff->MulColsVec(in_diff_summands_);
  }


  void Update(const CuMatrix<BaseFloat> &input, const CuMatrix<BaseFloat> &diff) {
    // useful dims
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    int32 num_filters = filters_.NumRows();
    int32 filter_dim = filters_.NumCols();

    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate;
    /* NOT NOW:
    const BaseFloat mmt = opts_.momentum;
    const BaseFloat l2 = opts_.l2_penalty;
    const BaseFloat l1 = opts_.l1_penalty;
    */

    //
    // calculate the gradient
    //
    filters_grad_.Resize(num_filters, filter_dim, kSetZero); // reset
    bias_grad_.Resize(num_filters, kSetZero); // reset
    // use all the patches
    for (int32 p=0; p<num_patches; p++) { // sum
      CuSubMatrix<BaseFloat> diff_patch(diff.ColRange(p * num_filters, num_filters));
      filters_grad_.AddMatMat(1.0, diff_patch, kTrans, vectorized_feature_patches_[p], kNoTrans, 1.0);
      bias_grad_.AddRowSumMat(1.0, diff_patch, 1.0);
    }
    // scale
    filters_grad_.Scale(1.0/num_patches);
    bias_grad_.Scale(1.0/num_patches);
    //

    //
    // update
    // 
    filters_.AddMat(-lr, filters_grad_);
    bias_.AddVec(-lr, bias_grad_);
    //
  }

 private:
  int32 patch_dim_,    ///< number of consecutive inputs, 1st dim of patch
        patch_step_,   ///< step of the convolution (i.e. shift between 2 patches)
        patch_stride_; ///< shift for 2nd dim of a patch (i.e. frame length before splicing)

  CuMatrix<BaseFloat> filters_; ///< row = vectorized rectangular filter
  CuVector<BaseFloat> bias_; ///< bias for each filter

  CuMatrix<BaseFloat> filters_grad_; ///< gradient of filters
  CuVector<BaseFloat> bias_grad_; ///< gradient of biases

  /** Buffer of reshaped inputs:
   *  1row = vectorized rectangular feature patch,
   *  1col = dim over speech frames,
   *  std::vector-dim = patch-position
   */
  std::vector<CuMatrix<BaseFloat> > vectorized_feature_patches_; 

  /** Buffer for backpropagation:
   *  derivatives in the domain of 'vectorized_feature_patches_',
   *  1row = vectorized rectangular feature patch,
   *  1col = dim over speech frames,
   *  std::vector-dim = patch-position
   */
  std::vector<CuMatrix<BaseFloat> > feature_patch_diffs_;

  /// Auxiliary vector for compensating #summands when backpropagating
  CuVector<BaseFloat> in_diff_summands_;
};

} // namespace nnet1
} // namespace kaldi

#endif
