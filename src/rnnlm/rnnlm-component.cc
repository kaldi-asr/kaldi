
#include <iterator>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include "rnnlm/nnet-parse.h"
#include "rnnlm/rnnlm-component.h"

namespace kaldi {
namespace rnnlm {

void AffineSampleLogSoftmaxComponent::Scale(BaseFloat scale) {
  linear_params_.Scale(scale);
  bias_params_.Scale(scale);
}

void AffineSampleLogSoftmaxComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  bias_params_.Resize(1, output_dim);
  linear_params_.Resize(output_dim, input_dim);
}

void AffineSampleLogSoftmaxComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const AffineSampleLogSoftmaxComponent *other =
             dynamic_cast<const AffineSampleLogSoftmaxComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.Row(0).AddVec(alpha, other->bias_params_.Row(0));
}

AffineSampleLogSoftmaxComponent::AffineSampleLogSoftmaxComponent(
                            const AffineSampleLogSoftmaxComponent &component):
    LmOutputComponent(component),
    linear_params_(component.linear_params_),
    bias_params_(component.bias_params_) { }

AffineSampleLogSoftmaxComponent::AffineSampleLogSoftmaxComponent(
                                   const CuMatrixBase<BaseFloat> &linear_params,
                                   const CuMatrixBase<BaseFloat> &bias_params,
                                            BaseFloat learning_rate):
                                            linear_params_(linear_params),
                                            bias_params_(bias_params) {
  SetUnderlyingLearningRate(learning_rate);
  KALDI_ASSERT(linear_params.NumRows() == bias_params.NumCols() &&
               bias_params.NumCols() != 0);
}

void AffineSampleLogSoftmaxComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
}

void AffineSampleLogSoftmaxComponent::SetParams(const CuMatrixBase<BaseFloat> &bias,
                                const CuMatrixBase<BaseFloat> &linear) {
  bias_params_ = bias;
  linear_params_ = linear;
  KALDI_ASSERT(bias_params_.NumCols() == linear_params_.NumRows());
}

void AffineSampleLogSoftmaxComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);

  CuMatrix<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddMat(stddev, temp_bias_params);
}

std::string AffineSampleLogSoftmaxComponent::Info() const {
  std::ostringstream stream;
  stream << LmComponent::Info();
  nnet3::PrintParameterStats(stream, "linear-params", linear_params_);
  nnet3::PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

LmComponent* AffineSampleLogSoftmaxComponent::Copy() const {
  AffineSampleLogSoftmaxComponent *ans = new AffineSampleLogSoftmaxComponent(*this);
  return ans;
}

BaseFloat AffineSampleLogSoftmaxComponent::DotProduct(const LmComponent &other_in) const {
  const AffineSampleLogSoftmaxComponent *other =
      dynamic_cast<const AffineSampleLogSoftmaxComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_.Row(0), other->bias_params_.Row(0));
}

void AffineSampleLogSoftmaxComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev, BaseFloat bias_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(1, output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);

  bias_params_.Set(bias_stddev);

//  bias_params_.SetRandn();
//  bias_params_.Scale(bias_stddev);
}

void AffineSampleLogSoftmaxComponent::Init(std::string matrix_filename) {
  CuMatrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(1, output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.Row(0).CopyColFromMat(mat, input_dim);
}

void AffineSampleLogSoftmaxComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
//    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
//        bias_stddev = 1.0;
    BaseFloat param_stddev = 0.0, /// log(1.0 / output_dim),
        bias_stddev = log(1.0 / output_dim);
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    Init(input_dim, output_dim, param_stddev, bias_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void AffineSampleLogSoftmaxComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                const vector<int> &indexes,
                                                CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(out->NumRows() == in.NumRows());
  CuMatrix<BaseFloat> new_linear(indexes.size(), linear_params_.NumCols());
  CuArray<int> idx(indexes);
  new_linear.CopyRows(linear_params_, idx);

  out->RowRange(0, 1).AddCols(bias_params_, idx);
  out->CopyRowsFromVec(out->Row(0));
  out->AddMatMat(1.0, in, kNoTrans, new_linear, kTrans, 1.0); 
}

void AffineSampleLogSoftmaxComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {
  KALDI_ASSERT (input_deriv != NULL);

  CuMatrix<BaseFloat> new_linear;
  CuArray<int> idx(indexes);
  new_linear.CopyRows(linear_params_, idx);

  input_deriv->AddMatMat(1.0, output_deriv, kNoTrans, new_linear, kNoTrans, 1.0);

  AffineSampleLogSoftmaxComponent* to_update
             = dynamic_cast<AffineSampleLogSoftmaxComponent*>(to_update_0);

  if (to_update != NULL) {
    new_linear.SetZero();  // clear the contents
    new_linear.AddMatMat(learning_rate_, output_deriv, kTrans,
                         in_value, kNoTrans, 1.0);
    CuMatrix<BaseFloat> delta_bias(1, output_deriv.NumCols(), kSetZero);
    delta_bias.Row(0).AddRowSumMat(learning_rate_, output_deriv, kTrans);

    vector<int> indexes_2(bias_params_.NumCols(), -1);
    for (int i = 0; i < indexes.size(); i++) {
      indexes_2[indexes[i]] = i;
    }

    CuArray<int> idx2(indexes_2);
    to_update->linear_params_.AddRows(1.0, new_linear, idx2);

    to_update->bias_params_.AddCols(delta_bias, idx2);
  }
}

void AffineSampleLogSoftmaxComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</AffineSampleLogSoftmaxComponent>");
}

void AffineSampleLogSoftmaxComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</AffineSampleLogSoftmaxComponent>");
}

int32 AffineSampleLogSoftmaxComponent::NumParameters() const {
  return (InputDim() + 1) * OutputDim();
}

void AffineSampleLogSoftmaxComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
  params->Range(InputDim() * OutputDim(),
                OutputDim()).CopyFromVec(bias_params_.Row(0));
}
void AffineSampleLogSoftmaxComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  bias_params_.Row(0).CopyFromVec(params.Range(InputDim() * OutputDim(),
                                        OutputDim()));
}

void LinearNormalizedLogSoftmaxComponent::Normalize() {
  KALDI_ASSERT(!normalized_);

  if (actual_params_.NumRows() != linear_params_.NumRows() ||
      actual_params_.NumCols() != linear_params_.NumCols()) {
    actual_params_.Resize(linear_params_.NumRows(), linear_params_.NumCols());
//    normalizer_.Resize(linear_params_.NumCols());
  }

  CuMatrix<BaseFloat> ht(linear_params_.NumCols(), linear_params_.NumRows());

  ht.CopyFromMat(linear_params_, kTrans);
//  ht.AddVecToCols(-1.0, linear_params_.Row(0));

//  linear_params_.CopyFromMat(ht, kTrans);

  ht.ApplySoftMaxPerRow(ht);
//  for (int i = 0; i < ht.NumRows(); i++) {
//    normalizer_(i) = ht.Row(i).ApplySoftMax();
//  }

  actual_params_.CopyFromMat(ht, kTrans);
//  KALDI_ASSERT(ApproxEqual(actual_params_.Sum(), actual_params_.NumCols()));
  normalized_ = true;
}

void LinearNormalizedLogSoftmaxComponent::Scale(BaseFloat scale) {
//  KALDI_ASSERT(is_gradient_);
  linear_params_.Scale(scale);
  normalized_ = false;
  Normalize();
}

void LinearNormalizedLogSoftmaxComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  linear_params_.Resize(output_dim, input_dim);
  normalized_ = false;
}

void LinearNormalizedLogSoftmaxComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const LinearNormalizedLogSoftmaxComponent *other =
      dynamic_cast<const LinearNormalizedLogSoftmaxComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  normalized_ = false;
  Normalize();
}

LinearNormalizedLogSoftmaxComponent::LinearNormalizedLogSoftmaxComponent(const LinearNormalizedLogSoftmaxComponent &component):
    LmOutputComponent(component),
    linear_params_(component.linear_params_),
//    normalizer_(component.normalizer_),
    actual_params_(component.actual_params_),
    normalized_(component.normalized_) { }

LinearNormalizedLogSoftmaxComponent::LinearNormalizedLogSoftmaxComponent(const CuMatrixBase<BaseFloat> &linear_params,
                                 BaseFloat learning_rate):
    linear_params_(linear_params) {
  SetUnderlyingLearningRate(learning_rate);
  normalized_ = false;
  Normalize();
}

void LinearNormalizedLogSoftmaxComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
  normalized_ = false;
  Normalize();
}

void LinearNormalizedLogSoftmaxComponent::SetParams(
                                const CuMatrixBase<BaseFloat> &linear) {
  linear_params_ = linear;
  normalized_ = false;
  Normalize();
}

void LinearNormalizedLogSoftmaxComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  temp_linear_params.Row(0).SetZero();

  linear_params_.AddMat(stddev, temp_linear_params);
  normalized_ = false;
  Normalize();
}

std::string LinearNormalizedLogSoftmaxComponent::Info() const {
  std::ostringstream stream;
  stream << LmComponent::Info();
  Matrix<BaseFloat> l(linear_params_);
  PrintParameterStats(stream, "linear-params", l);
  return stream.str();
}

LmComponent* LinearNormalizedLogSoftmaxComponent::Copy() const {
  LinearNormalizedLogSoftmaxComponent *ans = new LinearNormalizedLogSoftmaxComponent(*this);
  return ans;
}

BaseFloat LinearNormalizedLogSoftmaxComponent::DotProduct(const LmComponent &other_in) const {
//  KALDI_ASSERT(is_gradient_); // actually there are more problems here ...
  const LinearNormalizedLogSoftmaxComponent *other =
      dynamic_cast<const LinearNormalizedLogSoftmaxComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans);
}

void LinearNormalizedLogSoftmaxComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Row(0).SetZero();
  linear_params_.Scale(param_stddev);
  normalized_ = false;
  Normalize();
}

void LinearNormalizedLogSoftmaxComponent::Init(std::string matrix_filename) {
  ReadKaldiObject(matrix_filename, &linear_params_); // will abort on failure.
  normalized_ = false;
  Normalize();
}

void LinearNormalizedLogSoftmaxComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim);
    cfl->GetValue("param-stddev", &param_stddev);
    Init(input_dim, output_dim, param_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void LinearNormalizedLogSoftmaxComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                const vector<int> &indexes,
                                                CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(false);
}

void LinearNormalizedLogSoftmaxComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  KALDI_ASSERT(false);
}


//void LinearNormalizedLogSoftmaxComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
//                                                const vector<vector<int> > &indexes,
//                                                vector<vector<BaseFloat> > *out) const {
//  KALDI_ASSERT(in.NumRows() == indexes.size());
//  KALDI_ASSERT(normalized_);
//  out->resize(indexes.size());
//
//  for (int i = 0; i < indexes.size(); i++) {
//    KALDI_ASSERT(indexes[i].size() == 1);
//    int w = indexes[i][0];
//    BaseFloat res = VecVec(in.Row(i), actual_params_.Row(w));
////    KALDI_ASSERT(res >= 0 && res <= 1);
//    (*out)[i].push_back(res);
//  }
//}
//
//void LinearNormalizedLogSoftmaxComponent::Backprop(
//                               const vector<vector<int> > &indexes,
//                               const CuMatrixBase<BaseFloat> &in_value,
//                               const CuMatrixBase<BaseFloat> &, // out_value
//                               const vector<vector<BaseFloat> > &output_deriv,
//                               LmOutputComponent *to_update_0,
//                               CuMatrixBase<BaseFloat> *input_deriv) const {
//
//  int k = indexes.size();
//
//  if (input_deriv != NULL) {
//    for (int i = 0; i < k; i++) {
//      KALDI_ASSERT(indexes[i].size() == 1);
//      KALDI_ASSERT(output_deriv[i][0] == 1);
//      int index = indexes[i][0];
//      input_deriv->Row(i).AddVec(1.0, actual_params_.Row(index));
//    }
//  }
//
//  LinearNormalizedLogSoftmaxComponent* to_update
//             = dynamic_cast<LinearNormalizedLogSoftmaxComponent*>(to_update_0);
//
//  if (to_update != NULL) {
//    CuMatrix<BaseFloat> aT(actual_params_, kTrans);
//    CuMatrix<BaseFloat> dapT(actual_params_, kTrans);
//
//    CuMatrix<BaseFloat> daT(actual_params_, kTrans);
//    daT.SetZero();
//    dapT.SetZero();
//    for (int i = 0; i < k; i++) {
//      int index = indexes[i][0];
//      daT.ColRange(index, 1).AddVecToCols(1.0, in_value.Row(i), 1.0);
//    }
//    dapT.DiffSoftmaxPerRow(aT, daT);
//    to_update->linear_params_.AddMat(learning_rate_, dapT, kTrans);
//  }
//
////  if (to_update != NULL) {
////    for (int m = 0; m < k; m++) {
////      // index'th row of linear_params
////      for (int i = 1; i < linear_params_.NumRows(); i++) { // first row is all 0's by definition
////        for (int k = 0; k < linear_params_.NumCols(); k++) {
////          BaseFloat deriv = 0.0;
////          if (k == indexes[m][0]) {
////            // correct label
////            for (int j = 0; j < linear_params_.NumCols(); j++) {
////              deriv += in_value(m, j) +
////                actual_params_(i, j) * (1 - actual_params_(i, j));
////            }
////          } else {
////            int j = indexes[m][0];
////            deriv = - in_value(m, k) * actual_params_(i, k) * actual_params_(j, k);
////          }
////
////          to_update->linear_params_(i, k) +=
////            learning_rate_ * deriv;
////        
//////        to_update->linear_params_(i, ) +=
//////          learning_rate_ * output_deriv[i][j] * in_value(i, index) *
//////           (1 - actual_params_
//////         ;
////
//////          (*input_deriv)(i, m) += output_deriv[i][j] * linear_params_(index, m);
////        }
////      }
////    }
////  }
//}

void LinearNormalizedLogSoftmaxComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</LinearNormalizedLogSoftmaxComponent>");
  normalized_ = false;
  Normalize();
}

void LinearNormalizedLogSoftmaxComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</LinearNormalizedLogSoftmaxComponent>");
}

int32 LinearNormalizedLogSoftmaxComponent::NumParameters() const {
  return InputDim() * OutputDim(); // actually should be (InputDim() - 1 ) * OutputDim()
}

void LinearNormalizedLogSoftmaxComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
}

void LinearNormalizedLogSoftmaxComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  normalized_ = false;
  Normalize();
}


void LmLinearComponent::Scale(BaseFloat scale) {
  linear_params_.Scale(scale);
}

void LmLinearComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  linear_params_.Resize(output_dim, input_dim);
}

void LmLinearComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const LmLinearComponent *other =
      dynamic_cast<const LmLinearComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
//  bias_params_.AddVec(alpha, other->bias_params_);
}

LmLinearComponent::LmLinearComponent(const LmLinearComponent &component):
    LmInputComponent(component),
    linear_params_(component.linear_params_) {}

LmLinearComponent::LmLinearComponent(const MatrixBase<BaseFloat> &linear_params,
                                 BaseFloat learning_rate):
    linear_params_(linear_params) {
  SetUnderlyingLearningRate(learning_rate);
}

void LmLinearComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
}

void LmLinearComponent::SetParams(//const VectorBase<BaseFloat> &bias,
                                const MatrixBase<BaseFloat> &linear) {
  linear_params_ = linear;
}

void LmLinearComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
}

std::string LmLinearComponent::Info() const {
  std::ostringstream stream;
  stream << LmInputComponent::Info();
  nnet3::PrintParameterStats(stream, "linear-params", linear_params_);
  return stream.str();
}

LmComponent* LmLinearComponent::Copy() const {
  LmLinearComponent *ans = new LmLinearComponent(*this);
  return ans;
}

BaseFloat LmLinearComponent::DotProduct(const LmComponent &other_in) const {
  const LmLinearComponent *other =
      dynamic_cast<const LmLinearComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans);
}

void LmLinearComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev) {//, BaseFloat bias_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
}

void LmLinearComponent::Init(std::string matrix_filename) {
  Matrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
}

void LmLinearComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  int32 input_dim = -1, output_dim = -1;
  InitLearningRatesFromConfig(cfl);
  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim);
//        bias_stddev = 1.0;
    cfl->GetValue("param-stddev", &param_stddev);
//    cfl->GetValue("bias-stddev", &bias_stddev);
    Init(input_dim, output_dim,
         param_stddev);//, bias_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void LmLinearComponent::Propagate(const SparseMatrix<BaseFloat> &sp,
                                  CuMatrixBase<BaseFloat> *out) const {

  std::vector<MatrixIndexT> vis;

  CuMatrix<BaseFloat> cpu_out_transpose(out->NumCols(), out->NumRows());

  for (size_t i = 0; i < sp.NumRows(); i++) {
    const SparseVector<BaseFloat> &sv = sp.Row(i);
    int non_zero_index = -1;
    sv.Max(&non_zero_index);
    vis.push_back(non_zero_index);
  }

  cpu_out_transpose.AddCols(linear_params_, CuArray<int>(vis));
  out->CopyFromMat(cpu_out_transpose, kTrans);
}

void LmLinearComponent::UpdateSimple(const SparseMatrix<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv) {
  std::vector<MatrixIndexT> vis;
  const SparseMatrix<BaseFloat> &sp = in_value;

  for (size_t i = 0; i < sp.NumRows(); i++) {
    const SparseVector<BaseFloat> &sv = sp.Row(i);
    int non_zero_index = -1;
    ApproxEqual(sv.Max(&non_zero_index), 1.0);
    vis.push_back(non_zero_index);
  }
  KALDI_ASSERT(vis.size() == sp.NumRows());

  // TODO(hxu)
  for (int i = 0; i < vis.size(); i++) {
    MatrixIndexT j = vis[i];
    // i.e. in_value (i, j) = 1

    for (int k = 0; k < out_deriv.NumCols(); k++) {
      linear_params_(k, j) += learning_rate_ * out_deriv(i, k);
//      KALDI_LOG << k << ", " << j << " added " << out_deriv(k, i);
    }
  }
}

void LmLinearComponent::UpdateSimple(const CuMatrixBase<BaseFloat> &in_value,
                                   const CuMatrixBase<BaseFloat> &out_deriv) {
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);
}

void LmLinearComponent::Backprop(
                               const SparseMatrix<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &out_deriv,
                               LmComponent *to_update_in,
                               CuMatrixBase<BaseFloat> *in_deriv) const {
  LmLinearComponent *to_update = dynamic_cast<LmLinearComponent*>(to_update_in);

  // Propagate the derivative back to the input.
  // add with coefficient 1.0 since property kBackpropAdds is true.
  // If we wanted to add with coefficient 0.0 we'd need to zero the
  // in_deriv, in case of infinities.
  if (in_deriv)
    in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                        1.0);

  if (to_update != NULL) {
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    if (to_update->is_gradient_)
      to_update->UpdateSimple(in_value, out_deriv);
    else  // the call below is to a virtual function that may be re-implemented
      to_update->Update(in_value, out_deriv);  // by child classes.
  }
}

void LmLinearComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
//  ExpectToken(is, binary, "<BiasParams>");
//  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</LmLinearComponent>");
}

void LmLinearComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
//  WriteToken(os, binary, "<BiasParams>");
//  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</LmLinearComponent>");
}

int32 LmLinearComponent::NumParameters() const {
  return InputDim() * OutputDim();
}
void LmLinearComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
}
void LmLinearComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
}

} // namespace rnnlm
} // namespace kaldi
