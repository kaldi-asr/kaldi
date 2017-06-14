// nnet/nnet-convolutional-2d-component.h

// Copyright 2014-2015  Johns Hopkins University (author: Sri Harish Mallidi)
//                      Brno University of Technology (author: Karel Vesely),
//

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


#ifndef KALDI_NNET_NNET_CONVOLUTIONAL_2D_COMPONENT_H_
#define KALDI_NNET_NNET_CONVOLUTIONAL_2D_COMPONENT_H_

#include <string>
#include <vector>

#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

/**
 * Convolutional2DComponent implements convolution over 2-axis (frequency and temporal)
 * (i.e. frequency axis in case we are the 1st component in NN).
 * // We don't do convolution along temporal axis, which simplifies the
 * // implementation (and was not helpful for Tara).
 *
 * We assume the input featrues are spliced, i.e. each frame
 * is in fact a set of stacked frames, where we can form patches
 * which span over several frequency bands and time axes.
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
 * x_patch_dim_,y_patch_dim_     ... temporal and frequency axes sizes of the patch (e.g. (9,9) for 9x9 2D filter)
 * x_patch_step_,y_patch_step_    ... temporal and frequencey sizes of shifts in the convolution (e.g. (1,1) 2D filter with 1 step shift in both axes)
 * x_patch_stride_,y_patch_stride_  ... dimension of the feature (maps if inside convolutional layer) (e.g. (11,32) for 32-band 11 frame spliced spectrogram patch)
 * The type of convolution is controlled by hyperparameters:
 * fmap_x_len_, fmap_y_len_ ... dimension of the feature (maps if inside convolutional layer) (e.g. (11,32) for 32-band 11 frame spliced spectrogram patch)
 * filt_x_len_, filt_y_len_ ... temporal and frequency sizes of the filters (e.g. (9,9) for 9x9 2D filter)
 * filt_x_step_, filt_y_step_ ... temporal and frequency sizes of the filters (e.g. (1,1) for 2D-filter, with 1 step shift in both axes)
 *
 *
 * Due to convolution same weights are used repeateadly,
 * the final gradient is average of all position-specific
 * gradients.
 *
 */
class Convolutional2DComponent : public UpdatableComponent {
 public:
  Convolutional2DComponent(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out),
    fmap_x_len_(0), fmap_y_len_(0),
    filt_x_len_(0), filt_y_len_(0),
    filt_x_step_(0), filt_y_step_(0),
    connect_fmap_(0)
  { }

  ~Convolutional2DComponent()
  { }

  Component* Copy() const { return new Convolutional2DComponent(*this); }
  ComponentType GetType() const { return kConvolutional2DComponent; }

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
      else if (token == "<FmapXLen>")    ReadBasicType(is, false, &fmap_x_len_);
      else if (token == "<FmapYLen>")    ReadBasicType(is, false, &fmap_y_len_);
      else if (token == "<FiltXLen>")    ReadBasicType(is, false, &filt_x_len_);
      else if (token == "<FiltYLen>")    ReadBasicType(is, false, &filt_y_len_);
      else if (token == "<FiltXStep>")   ReadBasicType(is, false, &filt_x_step_);
      else if (token == "<FiltYStep>")   ReadBasicType(is, false, &filt_y_step_);
      else if (token == "<ConnectFmap>") ReadBasicType(is, false, &connect_fmap_);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config? "
                     << "(ParamStddev|BiasMean|BiasRange|FmapXLen|FmapYLen|"
                        "FiltXLen|FiltYLen|FiltXStep|FiltYStep|ConnectFmap|"
                        "LearnRateCoef|BiasLearnRateCoef)";
    }

    //
    // Sanity checks:
    //
    // input sanity checks
    // input_dim_ should be multiple of (fmap_x_len_ * fmap_y_len_)
    KALDI_ASSERT(input_dim_ % (fmap_x_len_ * fmap_y_len_) == 0);
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);
    KALDI_LOG << "num_input_fmaps " << num_input_fmaps;
    // check if step is in sync with fmap_len and filt_len
    KALDI_ASSERT((fmap_x_len_ - filt_x_len_) % (filt_x_step_) == 0);
    KALDI_ASSERT((fmap_y_len_ - filt_y_len_) % (filt_y_step_) == 0);
    int32 out_fmap_x_len = (fmap_x_len_ - filt_x_len_)/filt_x_step_ + 1;
    int32 out_fmap_y_len = (fmap_y_len_ - filt_y_len_)/filt_y_step_ + 1;
    // output sanity checks
    KALDI_ASSERT(output_dim_ % (out_fmap_x_len * out_fmap_y_len)  == 0);
    int32 num_output_fmaps = output_dim_ / (out_fmap_x_len * out_fmap_y_len);
    KALDI_LOG << "num_output_fmaps " << num_output_fmaps;
    int32 num_filters = output_dim_/(out_fmap_x_len*out_fmap_y_len);
    KALDI_LOG << "num_filters " << num_filters;

    //
    // Initialize trainable parameters,
    //
    filters_.Resize(num_filters, num_input_fmaps*filt_x_len_*filt_y_len_);
    RandGauss(0.0, param_stddev, &filters_);
    //
    bias_.Resize(num_filters);
    RandUniform(bias_mean, bias_range, &bias_);
  }

  void ReadData(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<LearnRateCoef>");
    ReadBasicType(is, binary, &learn_rate_coef_);
    ExpectToken(is, binary, "<BiasLearnRateCoef>");
    ReadBasicType(is, binary, &bias_learn_rate_coef_);
    // convolution hyperparameters
    ExpectToken(is, binary, "<FmapXLen>");
    ReadBasicType(is, binary, &fmap_x_len_);
    ExpectToken(is, binary, "<FmapYLen>");
    ReadBasicType(is, binary, &fmap_y_len_);
    ExpectToken(is, binary, "<FiltXLen>");
    ReadBasicType(is, binary, &filt_x_len_);
    ExpectToken(is, binary, "<FiltYLen>");
    ReadBasicType(is, binary, &filt_y_len_);
    ExpectToken(is, binary, "<FiltXStep>");
    ReadBasicType(is, binary, &filt_x_step_);
    ExpectToken(is, binary, "<FiltYStep>");
    ReadBasicType(is, binary, &filt_y_step_);
    ExpectToken(is, binary, "<ConnectFmap>");
    ReadBasicType(is, binary, &connect_fmap_);

    // trainable parameters
    ExpectToken(is, binary, "<Filters>");
    filters_.Read(is, binary);
    ExpectToken(is, binary, "<Bias>");
    bias_.Read(is, binary);

    //
    // Sanity checks:
    //
    // input sanity checks
    // input_dim_ should be multiple of (fmap_x_len_ * fmap_y_len_)
    KALDI_ASSERT(input_dim_ % (fmap_x_len_ * fmap_y_len_) == 0);
    // int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);
    // KALDI_LOG << "num_input_fmaps " << num_input_fmaps;
    // check if step is in sync with fmap_len and filt_len
    KALDI_ASSERT((fmap_x_len_ - filt_x_len_) % (filt_x_step_) == 0);
    KALDI_ASSERT((fmap_y_len_ - filt_y_len_) % (filt_y_step_) == 0);
    int32 out_fmap_x_len = (fmap_x_len_ - filt_x_len_)/filt_x_step_ + 1;
    int32 out_fmap_y_len = (fmap_y_len_ - filt_y_len_)/filt_y_step_ + 1;

    // output sanity checks
    KALDI_ASSERT(output_dim_ % (out_fmap_x_len * out_fmap_y_len)  == 0);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    if (!binary) os << "\n";

    // convolution hyperparameters
    WriteToken(os, binary, "<FmapXLen>");
    WriteBasicType(os, binary, fmap_x_len_);
    WriteToken(os, binary, "<FmapYLen>");
    WriteBasicType(os, binary, fmap_y_len_);
    WriteToken(os, binary, "<FiltXLen>");
    WriteBasicType(os, binary, filt_x_len_);
    WriteToken(os, binary, "<FiltYLen>");
    WriteBasicType(os, binary, filt_y_len_);
    WriteToken(os, binary, "<FiltXStep>");
    WriteBasicType(os, binary, filt_x_step_);
    WriteToken(os, binary, "<FiltYStep>");
    WriteBasicType(os, binary, filt_y_step_);
    WriteToken(os, binary, "<ConnectFmap>");
    WriteBasicType(os, binary, connect_fmap_);
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
           "\n  bias" + MomentStatistics(bias_) +
           ", lr-coef " + ToString(bias_learn_rate_coef_);
  }
  std::string InfoGradient() const {
    return std::string("\n  filters_grad") + MomentStatistics(filters_grad_) +
           ", lr-coef " + ToString(learn_rate_coef_) +
           "\n  bias_grad" + MomentStatistics(bias_grad_) +
           ", lr-coef " + ToString(bias_learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // useful dims
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);
    // int32 inp_fmap_size = fmap_x_len_ * fmap_y_len_;
    int32 out_fmap_x_len = (fmap_x_len_ - filt_x_len_)/filt_x_step_ + 1;
    int32 out_fmap_y_len = (fmap_y_len_ - filt_y_len_)/filt_y_step_ + 1;
    int32 out_fmap_size = out_fmap_x_len*out_fmap_y_len;
    int32 num_output_fmaps = output_dim_ / (out_fmap_x_len * out_fmap_y_len);
    // this is total num_filters,
    // so each input_fmap has size num_filters/num_input_fmaps
    int32 num_filters = filters_.NumRows();
    KALDI_ASSERT(num_filters == num_output_fmaps);
    // int32 filter_size = filt_x_len_*filt_y_len_;
    int32 num_frames = in.NumRows();

    // we will need the buffers
    if (vectorized_feature_patches_.size() == 0) {
      vectorized_feature_patches_.resize(out_fmap_size);
      feature_patch_diffs_.resize(out_fmap_size);
    }

    for (int32 p = 0; p < out_fmap_size; p++) {
      vectorized_feature_patches_[p].Resize(num_frames, filters_.NumCols());
    }

    // Checked for num_input_fmaps=1, check for num_inp_fmaps>1
    int32 out_fmap_cnt = 0;
    for (int32 m = 0; m < fmap_x_len_-filt_x_len_+1; m = m+filt_x_step_) {
      for (int32 n = 0; n < fmap_y_len_-filt_y_len_+1; n = n+filt_y_step_) {
    std::vector<int32> column_mask;
    int32 st = 0;
    if (connect_fmap_ == 1) {
      st = (m * fmap_y_len_ + n) * num_input_fmaps;
    } else {
      st = m * fmap_y_len_ * num_input_fmaps + n;
    }

    for (int32 i = 0; i < filt_x_len_; i++) {
      for (int32 j = 0; j < filt_y_len_*num_input_fmaps; j++) {
        int32 c = 0;
        if (connect_fmap_ == 1) {
          c = st + i * (num_input_fmaps*fmap_y_len_) + j;
        } else {
          c = st + i * (num_input_fmaps * fmap_y_len_)
                     + (j / num_input_fmaps)
                     + (j % num_input_fmaps) * fmap_y_len_;
        }
        column_mask.push_back(c);
      }
    }
    CuArray<int32> cu_column_mask(column_mask);
    vectorized_feature_patches_[out_fmap_cnt].CopyCols(in, cu_column_mask);
    out_fmap_cnt++;
      }
    }

    for (int32 p = 0; p < out_fmap_size; p++) {
      CuSubMatrix<BaseFloat> tgt(out->ColRange(p*num_filters, num_filters));
      tgt.AddVecToRows(1.0, bias_, 0.0);
      tgt.AddMatMat(1.0, vectorized_feature_patches_[p], kNoTrans, filters_, kTrans, 1.0);
    }
  }


  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // useful dims
    int32 num_input_fmaps = input_dim_ / (fmap_x_len_ * fmap_y_len_);

    int32 out_fmap_x_len = (fmap_x_len_ - filt_x_len_)/filt_x_step_ + 1;
    int32 out_fmap_y_len = (fmap_y_len_ - filt_y_len_)/filt_y_step_ + 1;
    int32 out_fmap_size = out_fmap_x_len * out_fmap_y_len;
    int32 num_output_fmaps = output_dim_ / (out_fmap_x_len * out_fmap_y_len);
    // this is total num_filters,
    // so each input_fmap has num_filters/num_input_fmaps
    int32 num_filters = filters_.NumRows();
    KALDI_ASSERT(num_filters == num_output_fmaps);
    // int32 filter_size = filt_x_len_*filt_y_len_;
    int32 num_frames = in.NumRows();

    for (int32 p = 0; p < out_fmap_size; p++) {
      feature_patch_diffs_[p].Resize(num_frames, filters_.NumCols(), kSetZero);
      CuSubMatrix<BaseFloat> out_diff_patch(out_diff.ColRange(p*num_filters, num_filters));
      feature_patch_diffs_[p].AddMatMat(1.0, out_diff_patch, kNoTrans, filters_, kNoTrans, 0.0);
    }

    // compute in_diff_summands_ once
    if (in_diff_summands_.Dim() == 0) {
      in_diff_summands_.Resize(in_diff->NumCols(), kSetZero);
      for (int32 m = 0; m < fmap_x_len_-filt_x_len_+1; m = m+filt_x_step_) {
        for (int32 n = 0; n < fmap_y_len_-filt_y_len_+1; n = n+filt_y_step_) {
          int32 st = 0;
          if (connect_fmap_ == 1) {
            st = (m * fmap_y_len_ + n) * num_input_fmaps;
          } else {
            st = m * fmap_y_len_ * num_input_fmaps + n;
          }
          for (int32 i = 0; i < filt_x_len_; i++) {
            for (int32 j = 0; j < filt_y_len_*num_input_fmaps; j++) {
              int32 c = 0;
              if (connect_fmap_ == 1) {
                c = st + i * (num_input_fmaps * fmap_y_len_) + j;
              } else {
                c = st + i * (num_input_fmaps * fmap_y_len_)
                       + (j / num_input_fmaps)
                       + (j % num_input_fmaps) * fmap_y_len_;
              }
              // add 1.0
              in_diff_summands_.Range(c, 1).Add(1.0);
            }
          }
        }
      }
      in_diff_summands_.InvertElements();
    }

    int32 out_fmap_cnt = 0;

    for (int32 m = 0; m < fmap_x_len_-filt_x_len_+1; m = m+filt_x_step_) {
      for (int32 n = 0; n< fmap_y_len_-filt_y_len_+1; n = n+filt_y_step_) {
        int32 st = 0;
        if (connect_fmap_ == 1) {
          st = (m * fmap_y_len_ + n) * num_input_fmaps;
        } else {
          st = m * fmap_y_len_ * num_input_fmaps + n;
        }

        for (int32 i = 0; i < filt_x_len_; i++) {
          for (int32 j = 0; j < filt_y_len_*num_input_fmaps; j++) {
            int32 c = 0;
            if (connect_fmap_ == 1) {
              c = st + i *(num_input_fmaps*fmap_y_len_)+j;
            } else {
              c = st + i * (num_input_fmaps * fmap_y_len_)
                     + (j / num_input_fmaps)
                     + (j % num_input_fmaps) * fmap_y_len_;
            }
            // from which col?
            CuMatrix<BaseFloat>& diff_mat = feature_patch_diffs_[out_fmap_cnt];
            CuSubMatrix<BaseFloat> src(diff_mat.ColRange(i*filt_y_len_*num_input_fmaps+j, 1));
            // to which col?
            CuSubMatrix<BaseFloat> tgt(in_diff->ColRange(c, 1));
            tgt.AddMat(1.0, src);
          }
        }
        out_fmap_cnt++;
      }
    }
    // compensate for summands
    in_diff->MulColsVec(in_diff_summands_);
  }


  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    // useful dims,
    int32 out_fmap_x_len = (fmap_x_len_ - filt_x_len_)/filt_x_step_ + 1;
    int32 out_fmap_y_len = (fmap_y_len_ - filt_y_len_)/filt_y_step_ + 1;
    int32 num_output_fmaps = output_dim_ / (out_fmap_x_len * out_fmap_y_len);

    // This is total num_filters,
    // each input_fmap has num_filters / num_input_fmaps:
    int32 num_filters = filters_.NumRows();
    KALDI_ASSERT(num_filters == num_output_fmaps);

    // we use following hyperparameters from the option class,
    const BaseFloat lr = opts_.learn_rate;

    //
    // calculate the gradient
    //
    filters_grad_.Resize(filters_.NumRows(), filters_.NumCols(), kSetZero);
    bias_grad_.Resize(filters_.NumRows(), kSetZero);
    //
    for (int32 p = 0; p < num_output_fmaps; p++) {
      CuSubMatrix<BaseFloat> diff_patch(diff.ColRange(p * num_filters, num_filters));
      filters_grad_.AddMatMat(1.0, diff_patch, kTrans, vectorized_feature_patches_[p], kNoTrans, 1.0);
      bias_grad_.AddRowSumMat(1.0, diff_patch, 1.0);
    }
    // scale
    filters_grad_.Scale(1.0/num_output_fmaps);
    bias_grad_.Scale(1.0/num_output_fmaps);

    //
    // update
    //
    filters_.AddMat(-lr * learn_rate_coef_, filters_grad_);
    bias_.AddVec(-lr * bias_learn_rate_coef_, bias_grad_);
  }

 private:
  /// feature maps dimensions (for input x_ is usually splice
  /// and y_ is num of fbanks) shift for 2nd dim of a patch
  /// (i.e. frame length before splicing),
  int32 fmap_x_len_, fmap_y_len_;

  /// 2D filter dimensions, x_ temporal, y_ spectral,
  int32 filt_x_len_, filt_y_len_;

  /// 2D shifts along temporal and spectral axis,
  int32 filt_x_step_, filt_y_step_;

  int32 connect_fmap_;  ///< if connect_fmap_ = 1, then each fmap has num_filt

  CuMatrix<BaseFloat> filters_;  ///< row = vectorized rectangular filter
  CuVector<BaseFloat> bias_;  ///< bias for each filter

  CuMatrix<BaseFloat> filters_grad_;  ///< gradient of filters
  CuVector<BaseFloat> bias_grad_;  ///< gradient of biases

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

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_CONVOLUTIONAL_2D_COMPONENT_H_
