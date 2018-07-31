// nnet/nnet-multibasis-component.h

// Copyright 2016  Brno University of Technology (Author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_MULTIBASIS_COMPONENT_H_
#define KALDI_NNET_NNET_MULTIBASIS_COMPONENT_H_

#include <sstream>
#include <vector>
#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-affine-transform.h"

namespace kaldi {
namespace nnet1 {

class MultiBasisComponent : public UpdatableComponent {
 public:
  MultiBasisComponent(int32 dim_in, int32 dim_out) :
    UpdatableComponent(dim_in, dim_out),
    selector_lr_coef_(1.0),
    threshold_(0.1)
  { }

  ~MultiBasisComponent()
  { }

  Component* Copy() const { return new MultiBasisComponent(*this); }
  ComponentType GetType() const { return kMultiBasisComponent; }

  void InitData(std::istream &is) {
    // define options,
    std::string selector_proto;
    std::string selector_filename;
    std::string basis_proto;
    std::string basis_filename;
    std::vector<std::string> basis_filename_vector;

    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<SelectorProto>") ReadToken(is, false, &selector_proto);
      else if (token == "<SelectorFilename>") ReadToken(is, false, &selector_filename);
      else if (token == "<SelectorLearnRateCoef>") ReadBasicType(is, false, &selector_lr_coef_);
      else if (token == "<BasisProto>") ReadToken(is, false, &basis_proto);
      else if (token == "<BasisFilename>") ReadToken(is, false, &basis_filename);
      else if (token == "<BasisFilenameVector>") {
        while(is >> std::ws, !is.eof()) {
          std::string file_or_end;
          ReadToken(is, false, &file_or_end);
          if (file_or_end == "</BasisFilenameVector>") break;
          basis_filename_vector.push_back(file_or_end);
        }
      } else KALDI_ERR << "Unknown token " << token << ", typo in config?"
               << " (SelectorProto|SelectorFilename|BasisProto|BasisFilename|BasisFilenameVector)";
    }

    //// INITIALIZE

    // selector,
    if (selector_proto != "") {
      KALDI_LOG << "Initializing 'selector' from : " << selector_proto;
      selector_.Init(selector_proto);
    }
    if (selector_filename != "") {
      KALDI_LOG << "Reading 'selector' from : " << selector_filename;
      selector_.Read(selector_filename);
    }

    // as many empty basis as outputs of the selector,
    nnet_basis_.resize(selector_.OutputDim());
    // fill the basis,
    if (basis_proto != "") {
      // Initialized from prototype,
      KALDI_LOG << "Initializing 'basis' from : " << basis_proto;
      for (int32 i = 0; i < nnet_basis_.size(); i++) {
        nnet_basis_[i].Init(basis_proto);
      }
    } else if (basis_filename != "") {
      // Load 1 initial basis repeateadly,
      KALDI_LOG << "Reading 'basis' from : " << basis_filename;
      for (int32 i = 0; i < nnet_basis_.size(); i++) {
        nnet_basis_[i].Read(basis_filename);
      }
    } else if (basis_filename_vector.size() > 0) {
      // Read a list of basis functions,
      if (basis_filename_vector.size() != nnet_basis_.size()) {
        KALDI_ERR << "We need " << nnet_basis_.size() << " filenames. "
                  << "We got " << basis_filename_vector.size();
      }
      for (int32 i = 0; i < nnet_basis_.size(); i++) {
        KALDI_LOG << "Reading 'basis' from : "
                  << basis_filename_vector[i];
        nnet_basis_[i].Read(basis_filename_vector[i]);
      }
    } else {
      // Initialize basis by square identity matrix,
      int32 basis_input_dim = InputDim() - selector_.InputDim();
      KALDI_LOG << "Initializing 'basis' to Identity <AffineTransform> "
                << OutputDim() << "x" << basis_input_dim;
      KALDI_ASSERT(OutputDim() == basis_input_dim);  // has to be square!
      Matrix<BaseFloat> m(OutputDim(), basis_input_dim);
      m.SetUnit();
      // wrap identity into AffineTransform,
      // (bias is vector of zeros),
      AffineTransform identity_comp(basis_input_dim, OutputDim());
      identity_comp.SetLinearity(CuMatrix<BaseFloat>(m));
      //
      for (int32 i = 0; i < nnet_basis_.size(); i++) {
        nnet_basis_[i].AppendComponent(identity_comp);
      }
    }

    // check,
    KALDI_ASSERT(InputDim() == selector_.InputDim() + nnet_basis_[0].InputDim());
    KALDI_ASSERT(OutputDim() == nnet_basis_[0].OutputDim());
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    bool end_loop = false;
    while (!end_loop && '<' == Peek(is, binary)) {
      std::string token;
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'S': ReadToken(is, false, &token);
          /**/ if (token == "<SelectorLearnRateCoef>") ReadBasicType(is, binary, &selector_lr_coef_);
          else if (token == "<Selector>") selector_.Read(is, binary);
          else KALDI_ERR << "Unknown token: " << token;
          break;
        case 'N': ExpectToken(is, binary, "<NumBasis>");
          int32 num_basis;
          ReadBasicType(is, binary, &num_basis);
          nnet_basis_.resize(num_basis);
          for (int32 i = 0; i < num_basis; i++) {
            int32 dummy;
            ExpectToken(is, binary, "<Basis>");
            ReadBasicType(is, binary, &dummy);
            nnet_basis_[i].Read(is, binary);
          }
          break;
        case '!':
          ExpectToken(is, binary, "<!EndOfComponent>");
          end_loop=true;
          break;
        default:
          ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }

    // check,
    KALDI_ASSERT(nnet_basis_.size() == selector_.OutputDim());
    KALDI_ASSERT(InputDim() == selector_.InputDim() + nnet_basis_[0].InputDim());
    KALDI_ASSERT(OutputDim() == nnet_basis_[0].OutputDim());
  }

  void WriteData(std::ostream &os, bool binary) const {
    int32 num_basis = nnet_basis_.size();
    WriteToken(os, binary, "<SelectorLearnRateCoef>");
    WriteBasicType(os, binary, selector_lr_coef_);
    if (!binary) os << "\n\n";
    WriteToken(os, binary, "<Selector>");
    if (!binary) os << "\n";
    selector_.Write(os, binary);
    if (!binary) os << "\n";
    WriteToken(os, binary, "<NumBasis>");
    WriteBasicType(os, binary, num_basis);
    if (!binary) os << "\n";
    for (int32 i = 0; i < num_basis; i++) {
      WriteToken(os, binary, "<Basis>");
      WriteBasicType(os, binary, i+1);
      if (!binary) os << "\n";
      nnet_basis_.at(i).Write(os, binary);
    }
  }

  Nnet& GetBasis(int32 id) { return nnet_basis_.at(id); }
  const Nnet& GetBasis(int32 id) const { return nnet_basis_.at(id); }

  int32 NumParams() const {
    int32 num_params_sum = selector_.NumParams();
    for (int32 i = 0; i < nnet_basis_.size(); i++) {
      num_params_sum += nnet_basis_[i].NumParams();
    }
    return num_params_sum;
  }

  void GetGradient(VectorBase<BaseFloat> *gradient) const {
    KALDI_ERR << "TODO, not yet implemented!";
  }

  void GetParams(VectorBase<BaseFloat> *params) const {
    int32 offset = 0;
    Vector<BaseFloat> params_tmp;
    // selector,
    selector_.GetParams(&params_tmp);
    params->Range(offset, params_tmp.Dim()).CopyFromVec(params_tmp);
    offset += params_tmp.Dim();
    // basis,
    for (int32 i = 0; i < nnet_basis_.size(); i++) {
      nnet_basis_[i].GetParams(&params_tmp);
      params->Range(offset, params_tmp.Dim()).CopyFromVec(params_tmp);
      offset += params_tmp.Dim();
    }
    KALDI_ASSERT(offset == NumParams());
  }

  void SetParams(const VectorBase<BaseFloat> &params) {
    int32 offset = 0;
    // selector,
    selector_.SetParams(params.Range(offset, selector_.NumParams()));
    offset += selector_.NumParams();
    // basis,
    for (int32 i = 0; i < nnet_basis_.size(); i++) {
      nnet_basis_[i].SetParams(params.Range(offset, nnet_basis_[i].NumParams()));
      offset += nnet_basis_[i].NumParams();
    }
    KALDI_ASSERT(offset == NumParams());
  }

  std::string Info() const {
    std::ostringstream os;
    for (int32 i = 0; i < nnet_basis_.size(); i++) {
      os << "basis_network #" << i+1 << " {\n"
         << nnet_basis_[i].Info()
         << "}\n";
    }
    os << "\nselector {\n"
       << selector_.Info()
       << "}";
    return os.str();
  }

  std::string InfoGradient() const {
    std::ostringstream os;
    for (int32 i = 0; i < nnet_basis_.size(); i++) {
      if (posterior_sum_(i) > threshold_) {
        os << "basis_gradient #" << i+1 << " {\n"
           << nnet_basis_[i].InfoGradient(false)
           << "}\n";
      }
    }
    os << "selector_gradient {\n"
       << selector_.InfoGradient(false)
       << "}";
    return os.str();
  }

  std::string InfoPropagate() const {
    std::ostringstream os;
    for (int32 i = 0; i < nnet_basis_.size(); i++) {
      if (posterior_sum_(i) > threshold_) {
        os << "basis_propagate #" << i+1 << " {\n"
           << nnet_basis_[i].InfoPropagate(false)
           << "}\n";
      }
    }
    os << "selector_propagate {\n"
       << selector_.InfoPropagate(false)
       << "}\n";
    return os.str();
  }

  std::string InfoBackPropagate() const {
    std::ostringstream os;
    for (int32 i = 0; i < nnet_basis_.size(); i++) {
      if (posterior_sum_(i) > threshold_) {
        os << "basis_backpropagate #" << i+1 << "{\n"
           << nnet_basis_[i].InfoBackPropagate(false)
           << "}\n";
      }
    }
    os << "selector_backpropagate {\n"
       << selector_.InfoBackPropagate(false)
       << "}\n";
    return os.str();
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // dimensions,
    int32 num_basis = nnet_basis_.size();

    // make sure we have all the buffers,
    if (basis_out_.size() != num_basis) {
      basis_out_.resize(num_basis);
    }

    // split the input,
    const CuSubMatrix<BaseFloat> in_basis(
        in.ColRange(0, nnet_basis_[0].InputDim())
    );
    const CuSubMatrix<BaseFloat> in_selector(
        in.ColRange(nnet_basis_[0].InputDim(), selector_.InputDim())
    );

    // get the 'selector_' posteriors,
    selector_.Propagate(in_selector, &posterior_);
    KALDI_ASSERT(posterior_.Row(0).Min() >= 0.0);
    KALDI_ASSERT(posterior_.Row(0).Max() <= 1.0);
    KALDI_ASSERT(ApproxEqual(posterior_.Row(0).Sum(), 1.0));
    posterior_.Transpose();  // trans,

    // sum 'selector_' posteriors over time,
    CuVector<BaseFloat> posterior_sum(num_basis);
    posterior_sum.AddColSumMat(1.0, posterior_, 0.0);
    posterior_sum_ = Vector<BaseFloat>(posterior_sum);

    // combine the 'basis' outputs,
    for (int32 i = 0; i < nnet_basis_.size(); i++) {
      if (posterior_sum_(i) > threshold_) {
        // use only basis with occupancy >0.1,
        nnet_basis_[i].Propagate(in_basis, &basis_out_[i]);
        out->AddDiagVecMat(1.0, posterior_.Row(i), basis_out_[i], kNoTrans);
      }
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // dimensions,
    int32 num_basis = nnet_basis_.size(),
          num_frames = in.NumRows();

    // split the in_diff,
    CuSubMatrix<BaseFloat> in_diff_basis(
        in_diff->ColRange(0, nnet_basis_[0].InputDim())
    );
    CuSubMatrix<BaseFloat> in_diff_selector(
        in_diff->ColRange(nnet_basis_[0].InputDim(), selector_.InputDim())
    );

    // backprop through 'selector',
    CuMatrix<BaseFloat> selector_out_diff(num_basis, num_frames);
    for (int32 i = 0; i < num_basis; i++) {
      if (posterior_sum_(i) > threshold_) {
        selector_out_diff.Row(i).AddDiagMatMat(1.0, out_diff, kNoTrans, basis_out_[i], kTrans, 0.0);
      }
    }
    selector_out_diff.Transpose();
    selector_out_diff.Scale(selector_lr_coef_);
    CuMatrix<BaseFloat> in_diff_selector_tmp;
    selector_.Backpropagate(selector_out_diff, &in_diff_selector_tmp);
    in_diff_selector.CopyFromMat(in_diff_selector_tmp);

    // backprop through 'basis',
    CuMatrix<BaseFloat> out_diff_scaled(num_frames, OutputDim()),
                        in_diff_basis_tmp;
    for (int32 i = 0; i < num_basis; i++) {
      // use only basis with occupancy >0.1,
      if (posterior_sum_(i) > threshold_) {
        out_diff_scaled.AddDiagVecMat(1.0, posterior_.Row(i), out_diff, kNoTrans, 0.0);
        nnet_basis_[i].Backpropagate(out_diff_scaled, &in_diff_basis_tmp);
        in_diff_basis.AddMat(1.0, in_diff_basis_tmp);
      }
    }
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    { }  // do nothing
  }

  /**
   * Overriding the default,
   * which was UpdatableComponent::SetTrainOptions(...)
   */
  void SetTrainOptions(const NnetTrainOptions &opts) {
    selector_.SetTrainOptions(opts);
    for (int32 i=0; i<nnet_basis_.size(); i++) {
      nnet_basis_[i].SetTrainOptions(opts);
    }
  }

  /**
   * Overriding the default,
   * which was UpdatableComponent::SetLearnRateCoef(...)
   */
  void SetLearnRateCoef(BaseFloat val) {
    // loop over nnets,
    for (int32 i = 0; i < nnet_basis_.size(); i++) {
      // loop over components,
      for (int32 j = 0; j < nnet_basis_[i].NumComponents(); j++) {
        if (nnet_basis_[i].GetComponent(j).IsUpdatable()) {
          UpdatableComponent& comp =
            dynamic_cast<UpdatableComponent&>(nnet_basis_[i].GetComponent(j));
          // set the value,
          comp.SetLearnRateCoef(val);
        }
      }
    }
  }

  /**
   * Overriding the default,
   * which was UpdatableComponent::SetBiasLearnRateCoef(...)
   */
  void SetBiasLearnRateCoef(BaseFloat val) {
    // loop over nnets,
    for (int32 i = 0; i < nnet_basis_.size(); i++) {
      // loop over components,
      for (int32 j = 0; j < nnet_basis_[i].NumComponents(); j++) {
        if (nnet_basis_[i].GetComponent(j).IsUpdatable()) {
          UpdatableComponent& comp =
            dynamic_cast<UpdatableComponent&>(nnet_basis_[i].GetComponent(j));
          // set the value,
          comp.SetBiasLearnRateCoef(val);
        }
      }
    }
  }

 private:
  /// The vector of 'basis' networks (output of basis is combined
  /// according to the posterior_ from the selector_)
  std::vector<Nnet> nnet_basis_;
  std::vector<CuMatrix<BaseFloat> > basis_out_;

  /// Selector network,
  Nnet selector_;
  BaseFloat selector_lr_coef_;

  /// The output of 'selector_',
  CuMatrix<BaseFloat> posterior_;
  Vector<BaseFloat> posterior_sum_;

  /// Threshold, applied to posterior_sum_, disables the unused basis,
  BaseFloat threshold_;

};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_MULTIBASIS_COMPONENT_H_
