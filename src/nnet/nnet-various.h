// nnet/nnet-various.h

// Copyright 2012-2015  Brno University of Technology (author: Karel Vesely)

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

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "util/text-utils.h"

#include <algorithm>
#include <sstream>

namespace kaldi {
namespace nnet1 {


/**
 * Splices the time context of the input features
 * in N, out k*N, FrameOffset o_1,o_2,...,o_k
 * FrameOffset example 11frames: -5 -4 -3 -2 -1 0 1 2 3 4 5
 */
class Splice: public Component {
 public:
  Splice(int32 dim_in, int32 dim_out)
    : Component(dim_in, dim_out)
  { }
  ~Splice()
  { }

  Component* Copy() const { return new Splice(*this); }
  ComponentType GetType() const { return kSplice; }

  void InitData(std::istream &is) {
    // define options
    std::vector<int32> frame_offsets;
    std::vector<std::vector<int32> > build_vector;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<ReadVector>") {
        ReadIntegerVector(is, false, &frame_offsets);
      } else if (token == "<BuildVector>") { 
        // <BuildVector> 1:1:1000 1:1:1000 1 2 3 1:10 </BuildVector> [matlab indexing]
        // read the colon-separated-lists:
        while (!is.eof()) { 
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
      is >> std::ws; // eat-up whitespace
    }

    // build the vector, using <BuildVector> ... </BuildVector> inputs
    if (build_vector.size() > 0) {
      for (int32 i=0; i<build_vector.size(); i++) {
        switch (build_vector[i].size()) {
          case 1:
            frame_offsets.push_back(build_vector[i][0]);
            break;
          case 2: { // assuming step 1
            int32 min=build_vector[i][0], max=build_vector[i][1];
            KALDI_ASSERT(min <= max);
            for (int32 j=min; j<=max; j++) {
              frame_offsets.push_back(j);
            }}
            break;
          case 3: { // step can be negative -> flipped min/max
            int32 min=build_vector[i][0], step=build_vector[i][1], max=build_vector[i][2];
            KALDI_ASSERT((min <= max && step > 0) || (min >= max && step < 0));
            for (int32 j=min; j<=max; j += step) {
              frame_offsets.push_back(j);
            }}
            break;
          case 0:
          default: 
            KALDI_ERR << "Error parsing <BuildVector>";
        }
      }
    }
    
    // copy to GPU
    frame_offsets_ = frame_offsets;

    // check dim
    KALDI_ASSERT(frame_offsets_.Dim()*InputDim() == OutputDim());
  }


  void ReadData(std::istream &is, bool binary) {
    std::vector<int32> frame_offsets;
    ReadIntegerVector(is, binary, &frame_offsets);
    frame_offsets_ = frame_offsets; // to GPU
    KALDI_ASSERT(frame_offsets_.Dim() * InputDim() == OutputDim());
  }

  void WriteData(std::ostream &os, bool binary) const {
    std::vector<int32> frame_offsets(frame_offsets_.Dim());
    frame_offsets_.CopyToVec(&frame_offsets);
    WriteIntegerVector(os, binary, frame_offsets);
  }
  
  std::string Info() const {
    std::ostringstream ostr;
    ostr << "\n  frame_offsets " << frame_offsets_;
    std::string str = ostr.str();
    str.erase(str.end()-1);
    return str;
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    cu::Splice(in, frame_offsets_, out); 
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    KALDI_ERR << __func__ << "Not implemented!";
  }

 protected:
  CuArray<int32> frame_offsets_;
};



/**
 * Rearrange the matrix columns according to the indices in copy_from_indices_
 */
class CopyComponent: public Component {
 public:
  CopyComponent(int32 dim_in, int32 dim_out)
    : Component(dim_in, dim_out)
  { }
  ~CopyComponent()
  { }

  Component* Copy() const { return new CopyComponent(*this); }
  ComponentType GetType() const { return kCopy; }

  void InitData(std::istream &is) {
    // define options
    std::vector<int32> copy_from_indices;
    std::vector<std::vector<int32> > build_vector;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<ReadVector>") {
        ReadIntegerVector(is, false, &copy_from_indices);
      } else if (token == "<BuildVector>") { 
        // <BuildVector> 1:1:1000 1:1:1000 1 2 3 1:10 </BuildVector> [matlab indexing]
        // read the colon-separated-lists:
        while (!is.eof()) { 
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
      is >> std::ws; // eat-up whitespace
    }

    // build the vector, using <BuildVector> ... </BuildVector> inputs
    if (build_vector.size() > 0) {
      for (int32 i=0; i<build_vector.size(); i++) {
        switch (build_vector[i].size()) {
          case 1:
            copy_from_indices.push_back(build_vector[i][0]);
            break;
          case 2: { // assuming step 1
            int32 min=build_vector[i][0], max=build_vector[i][1];
            KALDI_ASSERT(min <= max);
            for (int32 j=min; j<=max; j++) {
              copy_from_indices.push_back(j);
            }}
            break;
          case 3: { // step can be negative -> flipped min/max
            int32 min=build_vector[i][0], step=build_vector[i][1], max=build_vector[i][2];
            KALDI_ASSERT((min <= max && step > 0) || (min >= max && step < 0));
            for (int32 j=min; j<=max; j += step) {
              copy_from_indices.push_back(j);
            }}
            break;
          case 0:
          default: 
            KALDI_ERR << "Error parsing <BuildVector>";
        }
      }
    }
    
    // decrease by 1
    std::vector<int32>& v = copy_from_indices;
    std::transform(v.begin(), v.end(), v.begin(), op_decrease);
    // copy to GPU
    copy_from_indices_ = copy_from_indices;

    // check range
    for (int32 i=0; i<copy_from_indices.size(); i++) {
      KALDI_ASSERT(copy_from_indices[i] >= 0);
      KALDI_ASSERT(copy_from_indices[i] < InputDim());
    }
    // check dim
    KALDI_ASSERT(copy_from_indices_.Dim() == OutputDim());
  }

  void ReadData(std::istream &is, bool binary) { 
    std::vector<int32> copy_from_indices;
    ReadIntegerVector(is, binary, &copy_from_indices);
    // -1 from each element 
    std::vector<int32>& v = copy_from_indices;
    std::transform(v.begin(), v.end(), v.begin(), op_decrease);
    // 
    copy_from_indices_ = copy_from_indices;
    KALDI_ASSERT(copy_from_indices_.Dim() == OutputDim());
  }

  void WriteData(std::ostream &os, bool binary) const {
    std::vector<int32> copy_from_indices(copy_from_indices_.Dim());
    copy_from_indices_.CopyToVec(&copy_from_indices);
    // +1 to each element 
    std::vector<int32>& v = copy_from_indices;
    std::transform(v.begin(), v.end(), v.begin(), op_increase);
    // 
    WriteIntegerVector(os, binary, copy_from_indices); 
  }
 
  std::string Info() const {
    /*
    std::ostringstream ostr;
    ostr << "\n  copy_from_indices " << copy_from_indices_;
    std::string str = ostr.str();
    str.erase(str.end()-1);
    return str;
    */
    return "";
  }
  
  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) { 
    cu::Copy(in,copy_from_indices_,out); 
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    static bool warning_displayed = false;
    if (!warning_displayed) {
      KALDI_WARN << __func__ << "Not implemented!";
      warning_displayed = true;
    }
    in_diff->SetZero();
  }

 protected:
  CuArray<int32> copy_from_indices_;

 private:
  static int32 op_increase (int32 i) { return ++i; }
  static int32 op_decrease (int32 i) { return --i; }
};



/**
 * Adds shift to all the lines of the matrix
 * (can be used for global mean normalization)
 */
class AddShift : public UpdatableComponent {
 public:
  AddShift(int32 dim_in, int32 dim_out)
    : UpdatableComponent(dim_in, dim_out), shift_data_(dim_in), learn_rate_coef_(1.0)
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
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<InitParam>") ReadBasicType(is, false, &init_param);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (InitParam)";
      is >> std::ws; // eat-up whitespace
    }
    // initialize
    shift_data_.Resize(InputDim(), kSetZero); // set to zero
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

  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(InputDim());
    shift_data_.CopyToVec(wei_copy);
  }
   
  std::string Info() const {
    return std::string("\n  shift_data") + MomentStatistics(shift_data_);
  }

  std::string InfoGradient() const {
    return std::string("\n  shift_data_grad") + MomentStatistics(shift_data_grad_) + 
           ", lr-coef " + ToString(learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) { 
    out->CopyFromMat(in);
    //add the shift
    out->AddVecToRows(1.0, shift_data_, 1.0);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    //derivative of additive constant is zero...
    in_diff->CopyFromMat(out_diff);
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate;
    // gradient
    shift_data_grad_.Resize(InputDim(), kSetZero); // reset
    shift_data_grad_.AddRowSumMat(1.0, diff, 0.0);
    // update
    shift_data_.AddVec(-lr*learn_rate_coef_, shift_data_grad_);
  }

  // Data accessors
  const CuVectorBase<BaseFloat>& GetShiftVec() {
    return shift_data_;
  }
  void SetShiftVec(const CuVectorBase<BaseFloat>& shift_data) {
    KALDI_ASSERT(shift_data.Dim() == shift_data_.Dim());
    shift_data_.CopyFromVec(shift_data);
  }
  void SetLearnRateCoef(float c) {
    learn_rate_coef_ = c;
  }

 protected:
  CuVector<BaseFloat> shift_data_;
  CuVector<BaseFloat> shift_data_grad_;
  BaseFloat learn_rate_coef_;
};



/**
 * Rescale the data column-wise by a vector
 * (can be used for global variance normalization)
 */
class Rescale : public UpdatableComponent {
 public:
  Rescale(int32 dim_in, int32 dim_out)
    : UpdatableComponent(dim_in, dim_out), scale_data_(dim_in), learn_rate_coef_(1.0)
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
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<InitParam>") ReadBasicType(is, false, &init_param);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (InitParam)";
      is >> std::ws; // eat-up whitespace
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

  void GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(InputDim());
    scale_data_.CopyToVec(wei_copy);
  }
 
  std::string Info() const {
    return std::string("\n  scale_data") + MomentStatistics(scale_data_);
  }
  
  std::string InfoGradient() const {
    return std::string("\n  scale_data_grad") + MomentStatistics(scale_data_grad_) +
           ", lr-coef " + ToString(learn_rate_coef_);
  }
  
  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) { 
    out->CopyFromMat(in);
    // rescale the data
    out->MulColsVec(scale_data_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    // derivative gets also scaled by the scale_data_
    in_diff->MulColsVec(scale_data_);
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate;
    // gradient
    scale_data_grad_.Resize(InputDim(), kSetZero); // reset
    CuMatrix<BaseFloat> gradient_aux(diff);
    gradient_aux.MulElements(input);
    scale_data_grad_.AddRowSumMat(1.0, gradient_aux, 0.0);
    // update
    scale_data_.AddVec(-lr*learn_rate_coef_, scale_data_grad_);
  }

  // Data accessors
  const CuVectorBase<BaseFloat>& GetScaleVec() {
    return scale_data_;
  }
  void SetScaleVec(const CuVectorBase<BaseFloat>& scale_data) {
    KALDI_ASSERT(scale_data.Dim() == scale_data_.Dim());
    scale_data_.CopyFromVec(scale_data);
  }
  void SetLearnRateCoef(float c) {
    learn_rate_coef_ = c;
  }

 protected:
  CuVector<BaseFloat> scale_data_;
  CuVector<BaseFloat> scale_data_grad_;
  BaseFloat learn_rate_coef_;
};



} // namespace nnet1
} // namespace kaldi

#endif // KALDI_NNET_NNET_VARIOUS_H_
