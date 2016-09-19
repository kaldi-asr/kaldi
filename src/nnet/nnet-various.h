// nnet/nnet-various.h

// Copyright 2012-2016  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_VARIOUS_H_
#define KALDI_NNET_NNET_VARIOUS_H_

#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "util/text-utils.h"

namespace kaldi {
namespace nnet1 {

/**
 * Splices the time context of the input features
 * in N, out k*N, FrameOffset o_1,o_2,...,o_k
 * FrameOffset example 11frames: -5 -4 -3 -2 -1 0 1 2 3 4 5
 */
class Splice: public Component {
 public:
  Splice(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out)
  { }

  ~Splice()
  { }

  Component* Copy() const { return new Splice(*this); }
  ComponentType GetType() const { return kSplice; }

  void InitData(std::istream &is) {
    // define options,
    std::vector<std::vector<int32> > build_vector;
    // parse config,
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<ReadVector>") {
        frame_offsets_.Read(is, false);
      } else if (token == "<BuildVector>") {
        // Parse the list of 'matlab-like' indices:
        // <BuildVector> 1:1:1000 1 2 3 1:10 </BuildVector>
        while (is >> std::ws, !is.eof()) {
          std::string colon_sep_list_or_end;
          ReadToken(is, false, &colon_sep_list_or_end);
          if (colon_sep_list_or_end == "</BuildVector>") break;
          std::vector<int32> v;
          SplitStringToIntegers(colon_sep_list_or_end, ":", false, &v);
          build_vector.push_back(v);
        }
      } else {
        KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                  << " (ReadVector|BuildVector)";
      }
    }

    if (build_vector.size() > 0) {
      // build the vector, using <BuildVector> ... </BuildVector> inputs,
      BuildIntegerVector(build_vector, &frame_offsets_);
    }

    // check dim
    KALDI_ASSERT(frame_offsets_.Dim()*InputDim() == OutputDim());
  }

  void ReadData(std::istream &is, bool binary) {
    frame_offsets_.Read(is, binary);
    KALDI_ASSERT(frame_offsets_.Dim() * InputDim() == OutputDim());
  }

  void WriteData(std::ostream &os, bool binary) const {
    frame_offsets_.Write(os, binary);
  }

  std::string Info() const {
    std::ostringstream ostr;
    ostr << "\n  frame_offsets " << frame_offsets_;
    std::string str = ostr.str();
    str.erase(str.end()-1);
    return str;
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    cu::Splice(in, frame_offsets_, out);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // WARNING!!! WARNING!!! WARNING!!!
    // THIS BACKPROPAGATION CAN BE USED ONLY WITH 'PER-UTTERANCE' TRAINING!
    // IN MINI-BATCH TRAINING, THIS <Splice> COMPONENT HAS TO BE PART OF THE
    // 'feature_transform' SO WE DON'T BACKPROPAGATE THROUGH IT...

    // dims,
    int32 input_dim = in.NumCols(),
          num_frames = out_diff.NumRows();
    // Copy offsets to 'host',
    std::vector<int32> offsets(frame_offsets_.Dim());
    frame_offsets_.CopyToVec(&offsets);
    // loop over the offsets,
    for (int32 i = 0; i < offsets.size(); i++) {
      int32 o_i = offsets.at(i);
      int32 n_rows = num_frames - abs(o_i),
            src_row = std::max(-o_i, 0),
            tgt_row = std::max(o_i, 0);
      const CuSubMatrix<BaseFloat> src = out_diff.Range(src_row, n_rows, i*input_dim, input_dim);
      CuSubMatrix<BaseFloat> tgt = in_diff->RowRange(tgt_row, n_rows);
      tgt.AddMat(1.0, src, kNoTrans);
    }
  }

 protected:
  CuArray<int32> frame_offsets_;
};


/**
 * Rearrange the matrix columns according to the indices in copy_from_indices_
 */
class CopyComponent: public Component {
 public:
  CopyComponent(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out)
  { }

  ~CopyComponent()
  { }

  Component* Copy() const { return new CopyComponent(*this); }
  ComponentType GetType() const { return kCopy; }

  void InitData(std::istream &is) {
    // define options,
    std::vector<std::vector<int32> > build_vector;
    // parse config,
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<ReadVector>") {
        copy_from_indices_.Read(is, false);
      } else if (token == "<BuildVector>") {
        // <BuildVector> 1:1:1000 1:1:1000 1 2 3 1:10 </BuildVector>
        // 'matlab-line' indexing, read the colon-separated-lists:
        while (is >> std::ws, !is.eof()) {
          std::string colon_sep_list_or_end;
          ReadToken(is, false, &colon_sep_list_or_end);
          if (colon_sep_list_or_end == "</BuildVector>") break;
          std::vector<int32> v;
          SplitStringToIntegers(colon_sep_list_or_end, ":", false, &v);
          build_vector.push_back(v);
        }
      } else {
        KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                  << " (ReadVector|BuildVector)";
      }
    }

    if (build_vector.size() > 0) {
      // build the vector, using <BuildVector> ... </BuildVector> inputs,
      BuildIntegerVector(build_vector, &copy_from_indices_);
    }

    // decrease by 1,
    copy_from_indices_.Add(-1);

    // check range,
    KALDI_ASSERT(copy_from_indices_.Min() >= 0);
    KALDI_ASSERT(copy_from_indices_.Max() < InputDim());
    // check dim,
    KALDI_ASSERT(copy_from_indices_.Dim() == OutputDim());
  }

  void ReadData(std::istream &is, bool binary) {
    copy_from_indices_.Read(is, binary);
    KALDI_ASSERT(copy_from_indices_.Dim() == OutputDim());
    copy_from_indices_.Add(-1);  // -1 from each element,
  }

  void WriteData(std::ostream &os, bool binary) const {
    CuArray<int32> tmp(copy_from_indices_);
    tmp.Add(1);  // +1 to each element,
    tmp.Write(os, binary);
  }

  std::string Info() const {
    return std::string("\n  min ") + ToString(copy_from_indices_.Min()) +
                         ", max "  + ToString(copy_from_indices_.Max());
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    cu::Copy(in, copy_from_indices_,out);
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

 protected:
  CuArray<int32> copy_from_indices_;
};



/**
 * Rescale the matrix-rows to have unit length (L2-norm).
 */
class LengthNormComponent: public Component {
 public:
  LengthNormComponent(int32 dim_in, int32 dim_out):
    Component(dim_in, dim_out)
  { }

  ~LengthNormComponent()
  { }

  Component* Copy() const { return new LengthNormComponent(*this); }
  ComponentType GetType() const { return kLengthNormComponent; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // resize vector when needed,
    if (row_scales_.Dim() != in.NumRows()) {
      row_scales_.Resize(in.NumRows());
    }
    // get the normalization scalars,
    l2_aux_ = in;
    l2_aux_.MulElements(l2_aux_);  // x^2,
    row_scales_.AddColSumMat(1.0, l2_aux_, 0.0);  // sum_of_cols(x^2),
    row_scales_.ApplyPow(0.5);  // L2norm = sqrt(sum_of_cols(x^2)),
    row_scales_.InvertElements();  // 1/L2norm,
    // compute the output,
    out->CopyFromMat(in);
    out->MulRowsVec(row_scales_);  // re-normalize,
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    in_diff->MulRowsVec(row_scales_);  // diff_by_x(s * x) = s,
  }

 private:
  CuMatrix<BaseFloat> l2_aux_;  ///< auxiliary matrix for L2 norm computation,
  CuVector<BaseFloat> row_scales_;  ///< normalization scale of each row,
};


/**
 * Adds shift to all the lines of the matrix
 * (can be used for global mean normalization)
 */
class AddShift : public UpdatableComponent {
 public:
  AddShift(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out),
    shift_data_(dim_in)
  { }

  ~AddShift()
  { }

  Component* Copy() const { return new AddShift(*this); }
  ComponentType GetType() const { return kAddShift; }

  void InitData(std::istream &is) {
    // define options
    float init_param = 0.0;
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<InitParam>") ReadBasicType(is, false, &init_param);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (InitParam)";
    }
    // initialize
    shift_data_.Resize(InputDim(), kSetZero);  // set to zero
    shift_data_.Set(init_param);
  }

  void ReadData(std::istream &is, bool binary) {
    // optional learning-rate coef,
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
    }
    // read the shift data
    shift_data_.Read(is, binary);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    shift_data_.Write(os, binary);
  }

  int32 NumParams() const { return shift_data_.Dim(); }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    shift_data_grad_.CopyToVec(gradient);
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    shift_data_.CopyToVec(params);
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    shift_data_.CopyFromVec(params);
  }

  std::string Info() const {
    return std::string("\n  shift_data") +
      MomentStatistics(shift_data_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }

  std::string InfoGradient() const {
    return std::string("\n  shift_data_grad") +
      MomentStatistics(shift_data_grad_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // copy, add the shift,
    out->CopyFromMat(in);
    out->AddVecToRows(1.0, shift_data_, 1.0);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // the derivative of additive constant is zero...
    in_diff->CopyFromMat(out_diff);
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class,
    const BaseFloat lr = opts_.learn_rate;
    // gradient,
    shift_data_grad_.Resize(InputDim(), kSetZero);  // reset to zero,
    shift_data_grad_.AddRowSumMat(1.0, diff, 0.0);
    // update,
    shift_data_.AddVec(-lr * learn_rate_coef_, shift_data_grad_);
  }

  void SetLearnRateCoef(float c) { learn_rate_coef_ = c; }

 protected:
  CuVector<BaseFloat> shift_data_;
  CuVector<BaseFloat> shift_data_grad_;
};


/**
 * Rescale the data column-wise by a vector
 * (can be used for global variance normalization)
 */
class Rescale : public UpdatableComponent {
 public:
  Rescale(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out),
    scale_data_(dim_in)
  { }

  ~Rescale()
  { }

  Component* Copy() const { return new Rescale(*this); }
  ComponentType GetType() const { return kRescale; }

  void InitData(std::istream &is) {
    // define options
    float init_param = 0.0;
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<InitParam>") ReadBasicType(is, false, &init_param);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (InitParam)";
    }
    // initialize
    scale_data_.Resize(InputDim(), kSetZero);
    scale_data_.Set(init_param);
  }

  void ReadData(std::istream &is, bool binary) {
    // optional learning-rate coef,
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
    }
    // read the shift data
    scale_data_.Read(is, binary);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    scale_data_.Write(os, binary);
  }

  int32 NumParams() const { return scale_data_.Dim(); }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    scale_data_grad_.CopyToVec(gradient);
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    scale_data_.CopyToVec(params);
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    scale_data_.CopyFromVec(params);
  }

  std::string Info() const {
    return std::string("\n  scale_data") +
      MomentStatistics(scale_data_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }

  std::string InfoGradient() const {
    return std::string("\n  scale_data_grad") +
      MomentStatistics(scale_data_grad_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // copy, rescale the data,
    out->CopyFromMat(in);
    out->MulColsVec(scale_data_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // derivatives are scaled with the scale_data_,
    in_diff->CopyFromMat(out_diff);
    in_diff->MulColsVec(scale_data_);
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class,
    const BaseFloat lr = opts_.learn_rate;
    // gradient,
    scale_data_grad_.Resize(InputDim(), kSetZero);  // reset,
    CuMatrix<BaseFloat> gradient_aux(diff);
    gradient_aux.MulElements(input);
    scale_data_grad_.AddRowSumMat(1.0, gradient_aux, 0.0);
    // update,
    scale_data_.AddVec(-lr * learn_rate_coef_, scale_data_grad_);
  }

  void SetLearnRateCoef(float c) { learn_rate_coef_ = c; }

 protected:
  CuVector<BaseFloat> scale_data_;
  CuVector<BaseFloat> scale_data_grad_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_VARIOUS_H_
