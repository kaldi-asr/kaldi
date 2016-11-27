
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
  bias_params_.Resize(output_dim);
  linear_params_.Resize(output_dim, input_dim);
}

void AffineSampleLogSoftmaxComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const AffineSampleLogSoftmaxComponent *other =
      dynamic_cast<const AffineSampleLogSoftmaxComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}

AffineSampleLogSoftmaxComponent::AffineSampleLogSoftmaxComponent(const AffineSampleLogSoftmaxComponent &component):
    LmUpdatableComponent(component),
    linear_params_(component.linear_params_),
    bias_params_(component.bias_params_) { }

AffineSampleLogSoftmaxComponent::AffineSampleLogSoftmaxComponent(const MatrixBase<BaseFloat> &linear_params,
                                 const VectorBase<BaseFloat> &bias_params,
                                 BaseFloat learning_rate):
    linear_params_(linear_params),
    bias_params_(bias_params) {
  SetUnderlyingLearningRate(learning_rate);
  KALDI_ASSERT(linear_params.NumRows() == bias_params.Dim()&&
               bias_params.Dim() != 0);
}



void AffineSampleLogSoftmaxComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
}

void AffineSampleLogSoftmaxComponent::SetParams(const VectorBase<BaseFloat> &bias,
                                const MatrixBase<BaseFloat> &linear) {
  bias_params_ = bias;
  linear_params_ = linear;
  KALDI_ASSERT(bias_params_.Dim() == linear_params_.NumRows());
}

void AffineSampleLogSoftmaxComponent::PerturbParams(BaseFloat stddev) {
  Matrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);

  Vector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

std::string AffineSampleLogSoftmaxComponent::Info() const {
  std::ostringstream stream;
  stream << LmUpdatableComponent::Info();
  PrintParameterStats(stream, "linear-params", linear_params_);
  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

LmComponent* AffineSampleLogSoftmaxComponent::Copy() const {
  AffineSampleLogSoftmaxComponent *ans = new AffineSampleLogSoftmaxComponent(*this);
  return ans;
}

BaseFloat AffineSampleLogSoftmaxComponent::DotProduct(const LmUpdatableComponent &other_in) const {
  const AffineSampleLogSoftmaxComponent *other =
      dynamic_cast<const AffineSampleLogSoftmaxComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

void AffineSampleLogSoftmaxComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev, BaseFloat bias_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
}

void AffineSampleLogSoftmaxComponent::Init(std::string matrix_filename) {
  Matrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
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
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0;
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    Init(input_dim, output_dim,
         param_stddev, bias_stddev);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

// this function will most likely not be used
void AffineSampleLogSoftmaxComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                const MatrixBase<BaseFloat> &in,
                                 MatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(false);

  // No need for asserts as they'll happen within the matrix operations.
  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);

  for(MatrixIndexT r = 0; r < out->NumRows(); r++) {                           
    out->Row(r).ApplyLogSoftMax();                                                
  }    
}

void AffineSampleLogSoftmaxComponent::UpdateSimple(const MatrixBase<BaseFloat> &in_value,
                                   const MatrixBase<BaseFloat> &out_deriv) {
  KALDI_ASSERT(false);
  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);
}

void AffineSampleLogSoftmaxComponent::Backprop(const std::string &debug_info,
                               const ComponentPrecomputedIndexes *indexes,
                               const MatrixBase<BaseFloat> &in_value,
                               const MatrixBase<BaseFloat> &, // out_value
                               const MatrixBase<BaseFloat> &out_deriv,
                               LmComponent *to_update_in,
                               MatrixBase<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(false);
  AffineSampleLogSoftmaxComponent *to_update = dynamic_cast<AffineSampleLogSoftmaxComponent*>(to_update_in);

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
      to_update->Update(debug_info, in_value, out_deriv);  // by child classes.
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
                OutputDim()).CopyFromVec(bias_params_);
}
void AffineSampleLogSoftmaxComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  bias_params_.CopyFromVec(params.Range(InputDim() * OutputDim(),
                                        OutputDim()));
}

void LmLinearComponent::Scale(BaseFloat scale) {
  linear_params_.Scale(scale);
//  bias_params_.Scale(scale);
}

void LmLinearComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
//  bias_params_.Resize(output_dim);
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
    LmUpdatableComponent(component),
    linear_params_(component.linear_params_) {}
//    bias_params_(component.bias_params_) { }

LmLinearComponent::LmLinearComponent(const MatrixBase<BaseFloat> &linear_params,
//                                 const VectorBase<BaseFloat> &bias_params,
                                 BaseFloat learning_rate):
    linear_params_(linear_params) {
//    bias_params_(bias_params) {
  SetUnderlyingLearningRate(learning_rate);
//  KALDI_ASSERT(linear_params.NumRows() == bias_params.Dim()&&
//               bias_params.Dim() != 0);
}



void LmLinearComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
//  bias_params_.SetZero();
}

void LmLinearComponent::SetParams(//const VectorBase<BaseFloat> &bias,
                                const MatrixBase<BaseFloat> &linear) {
//  bias_params_ = bias;
  linear_params_ = linear;
//  KALDI_ASSERT(bias_params_.Dim() == linear_params_.NumRows());
}

void LmLinearComponent::PerturbParams(BaseFloat stddev) {
  Matrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);

//  Vector<BaseFloat> temp_bias_params(bias_params_);
//  temp_bias_params.SetRandn();
//  bias_params_.AddVec(stddev, temp_bias_params);
}

std::string LmLinearComponent::Info() const {
  std::ostringstream stream;
  stream << LmUpdatableComponent::Info();
  PrintParameterStats(stream, "linear-params", linear_params_);
//  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

LmComponent* LmLinearComponent::Copy() const {
  LmLinearComponent *ans = new LmLinearComponent(*this);
  return ans;
}

BaseFloat LmLinearComponent::DotProduct(const LmUpdatableComponent &other_in) const {
  const LmLinearComponent *other =
      dynamic_cast<const LmLinearComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans);
//      + VecVec(bias_params_, other->bias_params_);
}

void LmLinearComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev) {//, BaseFloat bias_stddev) {
  linear_params_.Resize(output_dim, input_dim);
//  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
//  bias_params_.SetRandn();
//  bias_params_.Scale(bias_stddev);
}

void LmLinearComponent::Init(std::string matrix_filename) {
  Matrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
//  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
//  bias_params_.CopyColFromMat(mat, input_dim);
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

void LmLinearComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                const MatrixBase<BaseFloat> &in,
                                 MatrixBase<BaseFloat> *out) const {

  // No need for asserts as they'll happen within the matrix operations.
//  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
}

void LmLinearComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                const SparseMatrix<BaseFloat> &sp,
                                MatrixBase<BaseFloat> *out) const {

  std::vector<MatrixIndexT> vis;

  Matrix<BaseFloat> cpu_out_transpose(out->NumCols(), out->NumRows());

  for (size_t i = 0; i < sp.NumRows(); i++) {
    const SparseVector<BaseFloat> &sv = sp.Row(i);
    int non_zero_index = -1;
    sv.Max(&non_zero_index);
    vis.push_back(non_zero_index);
  }

  cpu_out_transpose.AddCols(linear_params_, vis.data());
  out->CopyFromMat(cpu_out_transpose, kTrans);
//  out->AddVecToRows(1.0, bias_params_);
}

void LmLinearComponent::UpdateSimple(const SparseMatrix<BaseFloat> &in_value,
                                   const MatrixBase<BaseFloat> &out_deriv) {
  std::vector<MatrixIndexT> vis;

//  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
//  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
//                           in_value, kNoTrans, 1.0);
  const SparseMatrix<BaseFloat> &sp = in_value;

//  KALDI_LOG << "Number of elements in sparse matrix is " << sp.NumElements();
  for (size_t i = 0; i < sp.NumRows(); i++) {
    const SparseVector<BaseFloat> &sv = sp.Row(i);
    int non_zero_index = -1;
    ApproxEqual(sv.Max(&non_zero_index), 1.0);
    vis.push_back(non_zero_index);
  }
  KALDI_ASSERT(vis.size() == sp.NumRows());

  for (int i = 0; i < vis.size(); i++) {
    MatrixIndexT j = vis[i];
    // i.e. in_value (i, j) = 1

    for (int k = 0; k < out_deriv.NumCols(); k++) {
      linear_params_(k, j) += learning_rate_ * out_deriv(i, k);
//      KALDI_LOG << k << ", " << j << " added " << out_deriv(k, i);
    }
  }
}

void LmLinearComponent::UpdateSimple(const MatrixBase<BaseFloat> &in_value,
                                   const MatrixBase<BaseFloat> &out_deriv) {
//  bias_params_.AddRowSumMat(learning_rate_, out_deriv, 1.0);
  linear_params_.AddMatMat(learning_rate_, out_deriv, kTrans,
                           in_value, kNoTrans, 1.0);
}

void LmLinearComponent::Backprop(const std::string &debug_info,
                               const ComponentPrecomputedIndexes *indexes,
                               const MatrixBase<BaseFloat> &in_value,
                               const MatrixBase<BaseFloat> &, // out_value
                               const MatrixBase<BaseFloat> &out_deriv,
                               LmComponent *to_update_in,
                               MatrixBase<BaseFloat> *in_deriv) const {
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
      to_update->Update(debug_info, in_value, out_deriv);  // by child classes.
  }
}

void LmLinearComponent::Backprop(const std::string &debug_info,
                               const ComponentPrecomputedIndexes *indexes,
                               const SparseMatrix<BaseFloat> &in_value,
                               const MatrixBase<BaseFloat> &, // out_value
                               const MatrixBase<BaseFloat> &out_deriv,
                               LmComponent *to_update_in,
                               MatrixBase<BaseFloat> *in_deriv) const {
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
      to_update->Update(debug_info, in_value, out_deriv);  // by child classes.
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
//  params->Range(InputDim() * OutputDim(),
//                OutputDim()).CopyFromVec(bias_params_);
}
void LmLinearComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
//  bias_params_.CopyFromVec(params.Range(InputDim() * OutputDim(),
//                                        OutputDim()));
}

//LmComponent *LmLinearComponent::CollapseWithNext( const LmLinearComponent &next_component) const {
//  LmLinearComponent *ans = dynamic_cast<LmLinearComponent*>(this->Copy());
//  KALDI_ASSERT(ans != NULL);
//  // Note: it's possible that "ans" is really of a derived type such
//  // as LmLinearComponentPreconditioned, but this will still work.
//  // the "copy" call will copy things like learning rates, "alpha" value
//  // for preconditioned component, etc.
//  ans->linear_params_.Resize(next_component.OutputDim(), InputDim());
//  ans->bias_params_ = next_component.bias_params_;
//
//  ans->linear_params_.AddMatMat(1.0, next_component.linear_params_, kNoTrans,
//                                this->linear_params_, kNoTrans, 0.0);
//  ans->bias_params_.AddMatVec(1.0, next_component.linear_params_, kNoTrans,
//                              this->bias_params_, 1.0);
//  return ans;
//}
//
//Component *LmLinearComponent::CollapseWithNext(
//    const LmFixedAffineSampleLogSoftmaxComponent &next_component) const {
//  // If at least one was non-updatable, make the whole non-updatable.
//  LmFixedAffineSampleLogSoftmaxComponent *ans =
//      dynamic_cast<LmFixedAffineSampleLogSoftmaxComponent*>(next_component.Copy());
//  KALDI_ASSERT(ans != NULL);
//  ans->linear_params_.Resize(next_component.OutputDim(), InputDim());
//  ans->bias_params_ = next_component.bias_params_;
//
//  ans->linear_params_.AddMatMat(1.0, next_component.linear_params_, kNoTrans,
//                                this->linear_params_, kNoTrans, 0.0);
//  ans->bias_params_.AddMatVec(1.0, next_component.linear_params_, kNoTrans,
//                              this->bias_params_, 1.0);
//  return ans;
//}

//Component *LmLinearComponent::CollapseWithNext(
//    const FixedScaleComponent &next_component) const {
//  KALDI_ASSERT(this->OutputDim() == next_component.InputDim());
//  LmLinearComponent *ans =
//      dynamic_cast<LmLinearComponent*>(this->Copy());
//  KALDI_ASSERT(ans != NULL);
//  ans->linear_params_.MulRowsVec(next_component.scales_);
//  ans->bias_params_.MulElements(next_component.scales_);
//
//  return ans;
//}

//Component *LmLinearComponent::CollapseWithPrevious(
//    const LmFixedAffineSampleLogSoftmaxComponent &prev_component) const {
//  // If at least one was non-updatable, make the whole non-updatable.
//  LmFixedAffineSampleLogSoftmaxComponent *ans =
//      dynamic_cast<LmFixedAffineSampleLogSoftmaxComponent*>(prev_component.Copy());
//  KALDI_ASSERT(ans != NULL);
//
//  ans->linear_params_.Resize(this->OutputDim(), prev_component.InputDim());
//  ans->bias_params_ = this->bias_params_;
//
//  ans->linear_params_.AddMatMat(1.0, this->linear_params_, kNoTrans,
//                                prev_component.linear_params_, kNoTrans, 0.0);
//  ans->bias_params_.AddMatVec(1.0, this->linear_params_, kNoTrans,
//                              prev_component.bias_params_, 1.0);
//  return ans;
//}

/*
NaturalGradientLmLinearComponent::NaturalGradientLmLinearComponent():
    max_change_per_sample_(0.0),
    update_count_(0.0), active_scaling_count_(0.0),
    max_change_scale_stats_(0.0) { }

// virtual
void NaturalGradientLmLinearComponent::Resize(
    int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 1 && output_dim > 1);
  if (rank_in_ >= input_dim) rank_in_ = input_dim - 1;
  if (rank_out_ >= output_dim) rank_out_ = output_dim - 1;
  bias_params_.Resize(output_dim);
  linear_params_.Resize(output_dim, input_dim);
  OnlineNaturalGradient temp;
  preconditioner_in_ = temp;
  preconditioner_out_ = temp;
  SetNaturalGradientConfigs();
}


void NaturalGradientLmLinearComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read the opening tag and learning rate
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "<RankIn>");
  ReadBasicType(is, binary, &rank_in_);
  ExpectToken(is, binary, "<RankOut>");
  ReadBasicType(is, binary, &rank_out_);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period_);
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history_);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha_);
  ExpectToken(is, binary, "<MaxChangePerSample>");
  ReadBasicType(is, binary, &max_change_per_sample_);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<UpdateCount>") {
    ReadBasicType(is, binary, &update_count_);
    ExpectToken(is, binary, "<ActiveScalingCount>");
    ReadBasicType(is, binary, &active_scaling_count_);
    ExpectToken(is, binary, "<MaxChangeScaleStats>");
    ReadBasicType(is, binary, &max_change_scale_stats_);
    ReadToken(is, binary, &token);
  }
  if (token != "<NaturalGradientLmLinearComponent>" &&
      token != "</NaturalGradientLmLinearComponent>")
    KALDI_ERR << "Expected <NaturalGradientLmLinearComponent> or "
              << "</NaturalGradientLmLinearComponent>, got " << token;
  SetNaturalGradientConfigs();
}

void NaturalGradientLmLinearComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  std::string matrix_filename;
  BaseFloat num_samples_history = 2000.0, alpha = 4.0,
      max_change_per_sample = 0.0;
  int32 input_dim = -1, output_dim = -1, rank_in = 20, rank_out = 80,
      update_period = 4;
  InitLearningRatesFromConfig(cfl);
  cfl->GetValue("num-samples-history", &num_samples_history);
  cfl->GetValue("alpha", &alpha);
  cfl->GetValue("max-change-per-sample", &max_change_per_sample);
  cfl->GetValue("rank-in", &rank_in);
  cfl->GetValue("rank-out", &rank_out);
  cfl->GetValue("update-period", &update_period);

  if (cfl->GetValue("matrix", &matrix_filename)) {
    Init(rank_in, rank_out, update_period,
         num_samples_history, alpha, max_change_per_sample,
         matrix_filename);
    if (cfl->GetValue("input-dim", &input_dim))
      KALDI_ASSERT(input_dim == InputDim() &&
                   "input-dim mismatch vs. matrix.");
    if (cfl->GetValue("output-dim", &output_dim))
      KALDI_ASSERT(output_dim == OutputDim() &&
                   "output-dim mismatch vs. matrix.");
  } else {
    ok = ok && cfl->GetValue("input-dim", &input_dim);
    ok = ok && cfl->GetValue("output-dim", &output_dim);
    if (!ok)
      KALDI_ERR << "Bad initializer " << cfl->WholeLine();
    BaseFloat param_stddev = 1.0 / std::sqrt(input_dim),
        bias_stddev = 1.0, bias_mean = 0.0;
    cfl->GetValue("param-stddev", &param_stddev);
    cfl->GetValue("bias-stddev", &bias_stddev);
    cfl->GetValue("bias-mean", &bias_mean);
    Init(input_dim, output_dim, param_stddev,
         bias_stddev, bias_mean, rank_in, rank_out, update_period,
         num_samples_history, alpha, max_change_per_sample);
  }
  if (cfl->HasUnusedValues())
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  if (!ok)
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
}

void NaturalGradientLmLinearComponent::SetNaturalGradientConfigs() {
  preconditioner_in_.SetRank(rank_in_);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_in_.SetAlpha(alpha_);
  preconditioner_in_.SetUpdatePeriod(update_period_);
  preconditioner_out_.SetRank(rank_out_);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history_);
  preconditioner_out_.SetAlpha(alpha_);
  preconditioner_out_.SetUpdatePeriod(update_period_);
}

void NaturalGradientLmLinearComponent::Init(
    int32 rank_in, int32 rank_out,
    int32 update_period, BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample,
    std::string matrix_filename) {
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetNaturalGradientConfigs();
  KALDI_ASSERT(max_change_per_sample >= 0.0);
  max_change_per_sample_ = max_change_per_sample;
  Matrix<BaseFloat> mat;
  ReadKaldiObject(matrix_filename, &mat); // will abort on failure.
  KALDI_ASSERT(mat.NumCols() >= 2);
  int32 input_dim = mat.NumCols() - 1, output_dim = mat.NumRows();
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  linear_params_.CopyFromMat(mat.Range(0, output_dim, 0, input_dim));
  bias_params_.CopyColFromMat(mat, input_dim);
  is_gradient_ = false;  // not configurable; there's no reason you'd want this
  update_count_ = 0.0;
  active_scaling_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
}

void NaturalGradientLmLinearComponent::Init(
    int32 input_dim, int32 output_dim,
    BaseFloat param_stddev, BaseFloat bias_stddev, BaseFloat bias_mean,
    int32 rank_in, int32 rank_out, int32 update_period,
    BaseFloat num_samples_history, BaseFloat alpha,
    BaseFloat max_change_per_sample) {
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0 &&
               bias_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);
  bias_params_.Add(bias_mean);
  rank_in_ = rank_in;
  rank_out_ = rank_out;
  update_period_ = update_period;
  num_samples_history_ = num_samples_history;
  alpha_ = alpha;
  SetNaturalGradientConfigs();
  if (max_change_per_sample > 0.0)
    KALDI_WARN << "You are setting a positive max_change_per_sample for "
               << "NaturalGradientLmLinearComponent. But the per-component "
               << "gradient clipping mechansim has been removed. Instead it's currently "
               << "done at the whole model level.";
  max_change_per_sample_ = max_change_per_sample;
  is_gradient_ = false;  // not configurable; there's no reason you'd want this
  update_count_ = 0.0;
  active_scaling_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
}

void NaturalGradientLmLinearComponent::Write(std::ostream &os,
                                           bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<RankIn>");
  WriteBasicType(os, binary, rank_in_);
  WriteToken(os, binary, "<RankOut>");
  WriteBasicType(os, binary, rank_out_);
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, update_period_);
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, num_samples_history_);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha_);
  WriteToken(os, binary, "<MaxChangePerSample>");
  WriteBasicType(os, binary, max_change_per_sample_);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "<UpdateCount>");
  WriteBasicType(os, binary, update_count_);
  WriteToken(os, binary, "<ActiveScalingCount>");
  WriteBasicType(os, binary, active_scaling_count_);
  WriteToken(os, binary, "<MaxChangeScaleStats>");
  WriteBasicType(os, binary, max_change_scale_stats_);
  WriteToken(os, binary, "</NaturalGradientLmLinearComponent>");
}

std::string NaturalGradientLmLinearComponent::Info() const {
  std::ostringstream stream;
  stream << UpdatableComponent::Info();
  PrintParameterStats(stream, "linear-params", linear_params_);
  PrintParameterStats(stream, "bias", bias_params_, true);
  stream << ", rank-in=" << rank_in_
         << ", rank-out=" << rank_out_
         << ", num_samples_history=" << num_samples_history_
         << ", update_period=" << update_period_
         << ", alpha=" << alpha_
         << ", max-change-per-sample=" << max_change_per_sample_;
  if (update_count_ > 0.0 && max_change_per_sample_ > 0.0) {
    stream << ", avg-scaling-factor=" << max_change_scale_stats_ / update_count_
           << ", active-scaling-portion="
           << active_scaling_count_ / update_count_;
  }
  return stream.str();
}

Component* NaturalGradientLmLinearComponent::Copy() const {
  return new NaturalGradientLmLinearComponent(*this);
}

NaturalGradientLmLinearComponent::NaturalGradientLmLinearComponent(
    const NaturalGradientLmLinearComponent &other):
    LmLinearComponent(other),
    rank_in_(other.rank_in_),
    rank_out_(other.rank_out_),
    update_period_(other.update_period_),
    num_samples_history_(other.num_samples_history_),
    alpha_(other.alpha_),
    preconditioner_in_(other.preconditioner_in_),
    preconditioner_out_(other.preconditioner_out_),
    max_change_per_sample_(other.max_change_per_sample_),
    update_count_(other.update_count_),
    active_scaling_count_(other.active_scaling_count_),
    max_change_scale_stats_(other.max_change_scale_stats_) {
  SetNaturalGradientConfigs();
}

void NaturalGradientLmLinearComponent::Update(
    const std::string &debug_info,
    const MatrixBase<BaseFloat> &in_value,
    const MatrixBase<BaseFloat> &out_deriv) {
  Matrix<BaseFloat> in_value_temp;

  in_value_temp.Resize(in_value.NumRows(),
                       in_value.NumCols() + 1, kUndefined);
  in_value_temp.Range(0, in_value.NumRows(),
                      0, in_value.NumCols()).CopyFromMat(in_value);

  // Add the 1.0 at the end of each row "in_value_temp"
  in_value_temp.Range(0, in_value.NumRows(),
                      in_value.NumCols(), 1).Set(1.0);

  Matrix<BaseFloat> out_deriv_temp(out_deriv);

  Matrix<BaseFloat> row_products(2,
                                   in_value.NumRows());
  SubVector<BaseFloat> in_row_products(row_products, 0),
      out_row_products(row_products, 1);

  // These "scale" values get will get multiplied into the learning rate (faster
  // than having the matrices scaled inside the preconditioning code).
  BaseFloat in_scale, out_scale;

  preconditioner_in_.PreconditionDirections(&in_value_temp, &in_row_products,
                                            &in_scale);
  preconditioner_out_.PreconditionDirections(&out_deriv_temp, &out_row_products,
                                             &out_scale);

  // "scale" is a scaling factor coming from the PreconditionDirections calls
  // (it's faster to have them output a scaling factor than to have them scale
  // their outputs).
  BaseFloat scale = in_scale * out_scale;

  SubMatrix<BaseFloat> in_value_precon_part(in_value_temp,
                                              0, in_value_temp.NumRows(),
                                              0, in_value_temp.NumCols() - 1);
  // this "precon_ones" is what happens to the vector of 1's representing
  // offsets, after multiplication by the preconditioner.
  Vector<BaseFloat> precon_ones(in_value_temp.NumRows());

  precon_ones.CopyColFromMat(in_value_temp, in_value_temp.NumCols() - 1);

  BaseFloat local_lrate = scale * learning_rate_;
  update_count_ += 1.0;
  bias_params_.AddMatVec(local_lrate, out_deriv_temp, kTrans,
                         precon_ones, 1.0);
  linear_params_.AddMatMat(local_lrate, out_deriv_temp, kTrans,
                           in_value_precon_part, kNoTrans, 1.0);
}

void NaturalGradientLmLinearComponent::ZeroStats()  {
  update_count_ = 0.0;
  max_change_scale_stats_ = 0.0;
  active_scaling_count_ = 0.0;
}

void NaturalGradientLmLinearComponent::Scale(BaseFloat scale) {
  update_count_ *= scale;
  max_change_scale_stats_ *= scale;
  active_scaling_count_ *= scale;
  linear_params_.Scale(scale);
  bias_params_.Scale(scale);
}

void NaturalGradientLmLinearComponent::Add(BaseFloat alpha, const Component &other_in) {
  const NaturalGradientLmLinearComponent *other =
      dynamic_cast<const NaturalGradientLmLinearComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  update_count_ += alpha * other->update_count_;
  max_change_scale_stats_ += alpha * other->max_change_scale_stats_;
  active_scaling_count_ += alpha * other->active_scaling_count_;
  linear_params_.AddMat(alpha, other->linear_params_);
  bias_params_.AddVec(alpha, other->bias_params_);
}
*/

std::string LmFixedAffineSampleLogSoftmaxComponent::Info() const {
  std::ostringstream stream;
  stream << LmComponent::Info();
  PrintParameterStats(stream, "linear-params", linear_params_);
  PrintParameterStats(stream, "bias", bias_params_, true);
  return stream.str();
}

void LmFixedAffineSampleLogSoftmaxComponent::Init(const MatrixBase<BaseFloat> &mat) {
  KALDI_ASSERT(mat.NumCols() > 1);
  linear_params_ = mat.Range(0, mat.NumRows(), 0, mat.NumCols() - 1);
  bias_params_.Resize(mat.NumRows());
  bias_params_.CopyColFromMat(mat, mat.NumCols() - 1);
}

void LmFixedAffineSampleLogSoftmaxComponent::InitFromConfig(ConfigLine *cfl) {
  std::string filename;
  // Two forms allowed: "matrix=<rxfilename>", or "input-dim=x output-dim=y"
  // (for testing purposes only).
  if (cfl->GetValue("matrix", &filename)) {
    if (cfl->HasUnusedValues())
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";

    bool binary;
    Input ki(filename, &binary);
    Matrix<BaseFloat> mat;
    mat.Read(ki.Stream(), binary);
    KALDI_ASSERT(mat.NumRows() != 0);
    Init(mat);
  } else {
    int32 input_dim = -1, output_dim = -1;
    if (!cfl->GetValue("input-dim", &input_dim) ||
        !cfl->GetValue("output-dim", &output_dim) || cfl->HasUnusedValues()) {
      KALDI_ERR << "Invalid initializer for layer of type "
                << Type() << ": \"" << cfl->WholeLine() << "\"";
    }
    Matrix<BaseFloat> mat(output_dim, input_dim + 1);
    mat.SetRandn();
    Init(mat);
  }
}


void LmFixedAffineSampleLogSoftmaxComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                     const MatrixBase<BaseFloat> &in,
                                     MatrixBase<BaseFloat> *out) const  {
  out->CopyRowsFromVec(bias_params_); // Adds the bias term first.
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
}

void LmFixedAffineSampleLogSoftmaxComponent::Backprop(const std::string &debug_info,
                                    const ComponentPrecomputedIndexes *indexes,
                                    const MatrixBase<BaseFloat> &, //in_value
                                    const MatrixBase<BaseFloat> &, //out_value
                                    const MatrixBase<BaseFloat> &out_deriv,
                                    LmComponent *, //to_update
                                    MatrixBase<BaseFloat> *in_deriv) const {
  // kBackpropAdds is true. It's the user's responsibility to zero out
  // <in_deriv> if they need it to be so.
  if (in_deriv)
    in_deriv->AddMatMat(1.0, out_deriv, kNoTrans,
                        linear_params_, kNoTrans, 1.0);
}

LmComponent* LmFixedAffineSampleLogSoftmaxComponent::Copy() const {
  LmFixedAffineSampleLogSoftmaxComponent *ans = new LmFixedAffineSampleLogSoftmaxComponent();
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  return ans;
}

void LmFixedAffineSampleLogSoftmaxComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<LmFixedAffineSampleLogSoftmaxComponent>");
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "</LmFixedAffineSampleLogSoftmaxComponent>");
}

void LmFixedAffineSampleLogSoftmaxComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<LmFixedAffineSampleLogSoftmaxComponent>", "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "</LmFixedAffineSampleLogSoftmaxComponent>");
}

void LmSoftmaxComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const MatrixBase<BaseFloat> &in,
                                 MatrixBase<BaseFloat> *out) const {
  // Apply softmax function to each row of the output...
  // for that row, we do
  // x_i = exp(x_i) / sum_j exp(x_j).
//  out->ApplySoftMaxPerRow(in);
  out->CopyFromMat(in);
  for(MatrixIndexT r = 0; r < out->NumRows(); r++) {                           
    out->Row(r).ApplySoftMax();                                                
  }    

  // This floor on the output helps us deal with
  // almost-zeros in a way that doesn't lead to overflow.
  out->ApplyFloor(1.0e-20);
}

void LmSoftmaxComponent::Backprop(const std::string &debug_info,
                                const ComponentPrecomputedIndexes *indexes,
                                const MatrixBase<BaseFloat> &, // in_value,
                                const MatrixBase<BaseFloat> &out_value,
                                const MatrixBase<BaseFloat> &out_deriv,
                                LmComponent *to_update_in,
                                MatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv == NULL)
    return;
  /*
    Note on the derivative of the softmax function: let it be
    p_i = exp(x_i) / sum_i exp_i
    The [matrix-valued] Jacobian of this function is
    diag(p) - p p^T
    Let the derivative vector at the output be e, and at the input be
    d.  We have
    d = diag(p) e - p (p^T e).
    d_i = p_i e_i - p_i (p^T e).
  */
//  in_deriv->DiffSoftmaxPerRow(out_value, out_deriv);

  const MatrixBase<BaseFloat> &P(out_value), &E(out_deriv);                               
  MatrixBase<BaseFloat> &D(*in_deriv);                                               
                                                                               
  D.CopyFromMat(P);                                                           
  D.MulElements(E);                                                           
  // At this point, D = P .* E (in matlab notation)                           
  Vector<BaseFloat> pe_vec(D.NumRows()); // For each row i, the dot product (p_t . e_t).
  pe_vec.AddDiagMatMat(1.0, P, kNoTrans, E, kTrans, 0.0);                     
                                                                               
  D.AddDiagVecMat(-1.0, pe_vec, P, kNoTrans, 1.0); // does D -= diag(pe_vec) * P.
}

void LmSoftmaxComponent::StoreStats(const MatrixBase<BaseFloat> &out_value) {
  // We don't store derivative stats for this component type, just activation
  // stats.
  StoreStatsInternal(out_value, NULL);
}


void LmLogSoftmaxComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                    const MatrixBase<BaseFloat> &in,
                                    MatrixBase<BaseFloat> *out) const {
  // Applies log softmax function to each row of the output. For each row, we do
  // x_i = x_i - log(sum_j exp(x_j))
//  out->ApplyLogSoftMaxPerRow(in);
  out->CopyFromMat(in);
  for(MatrixIndexT r = 0; r < out->NumRows(); r++) {                           
    out->Row(r).ApplyLogSoftMax();                                                
  }    
}

void LmLogSoftmaxComponent::Backprop(const std::string &debug_info,
                                   const ComponentPrecomputedIndexes *indexes,
                                   const MatrixBase<BaseFloat> &, // in_value
                                   const MatrixBase<BaseFloat> &out_value,
                                   const MatrixBase<BaseFloat> &out_deriv,
                                   LmComponent *, // to_update
                                   MatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv == NULL)
    return;
//  in_deriv->DiffLogSoftmaxPerRow(out_value, out_deriv);

  const MatrixBase<BaseFloat> &Y(out_value), &E(out_deriv);                      
  MatrixBase<BaseFloat> &D(*in_deriv);                                               
                                                                              
  D.CopyFromMat(Y);                                                           
  D.ApplyExp();                           // exp(y)                           
  Vector<BaseFloat> E_sum(D.NumRows()); // Initializes to zero                   
  E_sum.AddColSumMat(1.0, E);             // Sum(e)                           
  D.MulRowsVec(E_sum);                    // exp(y) Sum(e)                    
  D.Scale(-1.0);                          // - exp(y) Sum(e)                  
  D.AddMat(1.0, E, kNoTrans);             // e - exp(y_i) Sum(e) 
}



} // namespace rnnlm
} // namespace kaldi
