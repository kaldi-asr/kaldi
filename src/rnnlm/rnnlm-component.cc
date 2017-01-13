
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
    bias_params_(component.bias_params_)
    { }

AffineSampleLogSoftmaxComponent::AffineSampleLogSoftmaxComponent(
                                   const CuMatrixBase<BaseFloat> &linear_params,
                                   const CuMatrixBase<BaseFloat> &bias_params,
                                            BaseFloat learning_rate):
                                            linear_params_(linear_params),
                                            bias_params_(bias_params)
                                             {
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
  temp_bias_params.SetRandn();  // TODO(hxu)
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
//  bias_params_.SetZero();  // TODO(hxu)

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
    // TODO(hxu)

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

const BaseFloat kCutoff = 1.0;

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

  // map from (-inf, inf) to (-inf, 0)
  // y = x - 1, when x <= 0
  // y = x / (1 + x) - 1, otherwise

/* The Hainan trick
  CuMatrix<BaseFloat> out2(*out);
  out2.ApplyFloor(0);
  out2.Scale(1.0 / kCutoff);
  out2.Add(1);
  out->DivElements(out2);
  out->Add(-1.0 * kCutoff);
// */

//* Dan trick: if x<=0 then y = x; 
//             if x >0 then y = log(1 + x)
  CuMatrix<BaseFloat> out2(*out);
  out2.ApplyFloor(0.0);
  out2.Add(1);
  out2.ApplyLog();
  out->ApplyCeiling(0.0);
  out->AddMat(1.0, out2);
// */


//  out->ApplyCeiling(0);  // TODO(hxu), neg-relu
}

void AffineSampleLogSoftmaxComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                bool normalize,
                                                CuMatrixBase<BaseFloat> *out) const {
  out->CopyRowsFromVec(bias_params_.Row(0));
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
  if (normalize) {
    out->ApplyLogSoftMaxPerRow(*out);
  }
}

void AffineSampleLogSoftmaxComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &out_value, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

/* The Hainan Trick
  CuMatrix<BaseFloat> new_out_deriv(out_value);
  new_out_deriv.ApplyFloor(-kCutoff);
  new_out_deriv.Scale(1.0 / kCutoff);
  new_out_deriv.MulElements(new_out_deriv);
  new_out_deriv.InvertElements();
  new_out_deriv.MulElements(output_deriv);
// */

//* The dan Trick
  CuMatrix<BaseFloat> new_out_deriv(out_value);
  new_out_deriv.ApplyExp();
  new_out_deriv.ApplyFloor(1);
  new_out_deriv.InvertElements();
  new_out_deriv.MulElements(output_deriv);
// */

//  const CuMatrixBase<BaseFloat> &new_out_deriv = output_deriv;

  CuMatrix<BaseFloat> new_linear(indexes.size(), linear_params_.NumCols());
  CuArray<int> idx(indexes);
  new_linear.CopyRows(linear_params_, idx);

//  input_deriv->AddMatMat(1.0, output_deriv, kNoTrans, new_linear, kNoTrans, 1.0);
  input_deriv->AddMatMat(1.0, new_out_deriv, kNoTrans, new_linear, kNoTrans, 1.0);

  AffineSampleLogSoftmaxComponent* to_update
             = dynamic_cast<AffineSampleLogSoftmaxComponent*>(to_update_0);

  if (to_update != NULL) {
    new_linear.SetZero();  // clear the contents
    new_linear.AddMatMat(learning_rate_, new_out_deriv, kTrans,
                         in_value, kNoTrans, 1.0);
    CuMatrix<BaseFloat> delta_bias(1, output_deriv.NumCols(), kSetZero);
    delta_bias.Row(0).AddRowSumMat(learning_rate_, new_out_deriv, kTrans);

    vector<int> indexes_2(bias_params_.NumCols(), -1);
    for (int i = 0; i < indexes.size(); i++) {
      indexes_2[indexes[i]] = i;
    }

    CuArray<int> idx2(indexes_2);
    to_update->linear_params_.AddRows(1.0, new_linear, idx2);

    to_update->bias_params_.AddCols(delta_bias, idx2);  // TODO(hxu)
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

void LinearSigmoidNormalizedComponent::Normalize() {

  if (actual_params_.NumRows() != linear_params_.NumRows() ||
      actual_params_.NumCols() != linear_params_.NumCols()) {
    actual_params_.Resize(linear_params_.NumRows(), linear_params_.NumCols());
//    normalizer_.Resize(linear_params_.NumCols());
  }

  CuMatrix<BaseFloat> ht(linear_params_.NumCols(), linear_params_.NumRows());

  ht.CopyFromMat(linear_params_, kTrans);
  ht.Sigmoid(ht);

  // normalize ht s.t. every row adds to 1
  CuMatrix<BaseFloat> ones(ht.NumCols(), 1);
  CuMatrix<BaseFloat> row_sum(1, ht.NumRows(), kSetZero);
  ones.Set(1.0);
  row_sum.AddMatMat(1.0, ones, kTrans, ht, kTrans, 0.0);
  ht.DivRowsVec(row_sum.Row(0));

  actual_params_.CopyFromMat(ht, kTrans);
//  KALDI_ASSERT(ApproxEqual(actual_params_.Sum(), actual_params_.NumCols()));
}

void LinearSigmoidNormalizedComponent::Scale(BaseFloat scale) {
//  KALDI_ASSERT(is_gradient_);
  linear_params_.Scale(scale);
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  linear_params_.Resize(output_dim, input_dim);
//  normalized_ = false;
}

void LinearSigmoidNormalizedComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const LinearSigmoidNormalizedComponent *other =
      dynamic_cast<const LinearSigmoidNormalizedComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_); //  TODO(hxu)
//  KALDI_LOG << "sum is " << other->linear_params_.Sum();
//  normalized_ = false;
  Normalize();
}

LinearSigmoidNormalizedComponent::LinearSigmoidNormalizedComponent(const LinearSigmoidNormalizedComponent &component):
    LmOutputComponent(component),
    linear_params_(component.linear_params_),
//    normalizer_(component.normalizer_),
    actual_params_(component.actual_params_) {}
//    normalized_(component.normalized_) { }

LinearSigmoidNormalizedComponent::LinearSigmoidNormalizedComponent(const CuMatrixBase<BaseFloat> &linear_params,
                                 BaseFloat learning_rate):
    linear_params_(linear_params) {
  SetUnderlyingLearningRate(learning_rate);
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::SetParams(
                                const CuMatrixBase<BaseFloat> &linear) {
  linear_params_ = linear;
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
//  temp_linear_params.Row(0).SetZero();

  linear_params_.AddMat(stddev, temp_linear_params);
//  normalized_ = false;
  Normalize();
}

std::string LinearSigmoidNormalizedComponent::Info() const {
  std::ostringstream stream;
  stream << LmComponent::Info();
  Matrix<BaseFloat> l(linear_params_);
  PrintParameterStats(stream, "linear-params", l);
  return stream.str();
}

LmComponent* LinearSigmoidNormalizedComponent::Copy() const {
  LinearSigmoidNormalizedComponent *ans = new LinearSigmoidNormalizedComponent(*this);
  return ans;
}

BaseFloat LinearSigmoidNormalizedComponent::DotProduct(const LmComponent &other_in) const {
//  KALDI_ASSERT(is_gradient_); // actually there are more problems here ...
  const LinearSigmoidNormalizedComponent *other =
      dynamic_cast<const LinearSigmoidNormalizedComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans);
}

void LinearSigmoidNormalizedComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
//  linear_params_.Row(0).SetZero();
  linear_params_.Scale(param_stddev);
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::Init(std::string matrix_filename) {
  ReadKaldiObject(matrix_filename, &linear_params_); // will abort on failure.
//  normalized_ = false;
  Normalize();
}

void LinearSigmoidNormalizedComponent::InitFromConfig(ConfigLine *cfl) {
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

void LinearSigmoidNormalizedComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                const vector<int> &indexes,
                                                CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(false);
}

void LinearSigmoidNormalizedComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  KALDI_ASSERT(false);
}


void LinearSigmoidNormalizedComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                const vector<int> &indexes,
                                                vector<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() == indexes.size());
  out->resize(indexes.size());

  for (int i = 0; i < indexes.size(); i++) {
    int w = indexes[i];
//    KALDI_LOG << in.Row(i).Sum() << " should be close to 1";
    KALDI_ASSERT(ApproxEqual(in.Row(i).Sum(), 1.0));  // TODO(hxu)
    BaseFloat res = VecVec(in.Row(i), actual_params_.Row(w));
//    KALDI_ASSERT(res >= 0 && res <= 1);
    (*out)[i] = res;
  }
}

void LinearSigmoidNormalizedComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const vector<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  int k = indexes.size();

  KALDI_ASSERT(input_deriv != NULL);

  for (int i = 0; i < k; i++) {
    int index = indexes[i];
    input_deriv->Row(i).AddVec(output_deriv[i], actual_params_.Row(index));
  }

  LinearSigmoidNormalizedComponent* to_update
             = dynamic_cast<LinearSigmoidNormalizedComponent*>(to_update_0);

  KALDI_ASSERT(to_update != NULL);

  CuMatrix<BaseFloat> aT(actual_params_, kTrans);
  CuMatrix<BaseFloat> dapT(actual_params_, kTrans);
  CuMatrix<BaseFloat> daT(actual_params_, kTrans);
//  aT.SetZero();
  dapT.SetZero();
  daT.SetZero();
  for (int i = 0; i < k; i++) {
    int index = indexes[i];
    daT.ColRange(index, 1).AddVecToCols(output_deriv[i], in_value.Row(i), 1.0);
  }

  // first compute derivative of the normalization
  // daT: div on the normalized matrix
  // dapT: div on the matrix before normalization
  // aT: the "in_value"
  // use the back-prop code of NormalizeOneComponent
  {
    aT.Tanh(aT);
    const CuMatrixBase<BaseFloat> &in_value = aT;
    const CuMatrixBase<BaseFloat> &out_deriv = daT;
    CuMatrixBase<BaseFloat> *in_deriv = &dapT;

    CuMatrix<BaseFloat> ones(in_value.NumCols(), 1);
    ones.Set(1.0);

    CuMatrix<BaseFloat> in_row_sum(1, in_value.NumRows(), kSetZero);
    in_row_sum.AddMatMat(1.0, ones, kTrans, in_value, kTrans, 0.0);

  //  KALDI_ASSERT(ApproxEqual(in_row_sum.Sum(), in_value.Sum()));

    CuMatrix<BaseFloat> t(out_deriv);
    t.MulElements(in_value);
    CuMatrix<BaseFloat> row_sum2(1, in_value.NumRows(), kSetZero);

  //  row_sum2.AddMatMat(1.0, in_value, kNoTrans, ones, kNoTrans, 0.0);
    row_sum2.AddMatMat(1.0, ones, kTrans, t, kTrans, 0.0);

    row_sum2.DivElements(in_row_sum);
    row_sum2.DivElements(in_row_sum);
    row_sum2.Scale(-1);

    in_deriv->AddMatMat(1.0, row_sum2, kTrans, ones, kTrans, 1.0);

  //  KALDI_LOG << "d sum here is " << out_deriv.Sum();
  //  KALDI_LOG << "in sum here is " << in_value.Sum();
  //
  //
  //  KALDI_LOG << "a sum here is " << in_deriv->Sum();

    t.CopyFromMat(out_deriv);
    t.DivRowsVec(in_row_sum.Row(0));
    in_deriv->AddMat(1.0, t);
  }

  // now dapT is the derivative of the tanh'd matrix
  // aT is the tanh/d matrix
  dapT.DiffSigmoid(aT, dapT);

//  dapT.DiffSoftmaxPerRow(aT, daT);
//  KALDI_LOG << aT.Sum() << " and " << daT.Sum() << " and " <<dapT.Sum();

  to_update->linear_params_.AddMat(learning_rate_, dapT, kTrans);  // TODO(hxu)
//  to_update->linear_params_.Row(0).SetZero();
}

void LinearSigmoidNormalizedComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</LinearSigmoidNormalizedComponent>");
  Normalize();
}

void LinearSigmoidNormalizedComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</LinearSigmoidNormalizedComponent>");
}

int32 LinearSigmoidNormalizedComponent::NumParameters() const {
  return InputDim() * OutputDim(); // actually should be (InputDim() - 1 ) * OutputDim()
}

void LinearSigmoidNormalizedComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
}

void LinearSigmoidNormalizedComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
  Normalize();
}

void LinearSoftmaxNormalizedComponent::Normalize() {

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
}

void LinearSoftmaxNormalizedComponent::Scale(BaseFloat scale) {
//  KALDI_ASSERT(is_gradient_);
  linear_params_.Scale(scale);
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::Resize(int32 input_dim, int32 output_dim) {
  KALDI_ASSERT(input_dim > 0 && output_dim > 0);
  linear_params_.Resize(output_dim, input_dim);
//  normalized_ = false;
}

void LinearSoftmaxNormalizedComponent::Add(BaseFloat alpha, const LmComponent &other_in) {
  const LinearSoftmaxNormalizedComponent *other =
      dynamic_cast<const LinearSoftmaxNormalizedComponent*>(&other_in);
  KALDI_ASSERT(other != NULL);
  linear_params_.AddMat(alpha, other->linear_params_); //  TODO(hxu)
//  KALDI_LOG << "sum is " << other->linear_params_.Sum();
//  normalized_ = false;
  Normalize();
}

LinearSoftmaxNormalizedComponent::LinearSoftmaxNormalizedComponent(const LinearSoftmaxNormalizedComponent &component):
    LmOutputComponent(component),
    linear_params_(component.linear_params_),
//    normalizer_(component.normalizer_),
    actual_params_(component.actual_params_) {}
//    normalized_(component.normalized_) { }

LinearSoftmaxNormalizedComponent::LinearSoftmaxNormalizedComponent(const CuMatrixBase<BaseFloat> &linear_params,
                                 BaseFloat learning_rate):
    linear_params_(linear_params) {
  SetUnderlyingLearningRate(learning_rate);
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetActualLearningRate(1.0);
    is_gradient_ = true;
  }
  linear_params_.SetZero();
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::SetParams(
                                const CuMatrixBase<BaseFloat> &linear) {
  linear_params_ = linear;
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::PerturbParams(BaseFloat stddev) {
  CuMatrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  temp_linear_params.Row(0).SetZero();

  linear_params_.AddMat(stddev, temp_linear_params);
//  normalized_ = false;
  Normalize();
}

std::string LinearSoftmaxNormalizedComponent::Info() const {
  std::ostringstream stream;
  stream << LmComponent::Info();
  Matrix<BaseFloat> l(linear_params_);
  PrintParameterStats(stream, "linear-params", l);
  return stream.str();
}

LmComponent* LinearSoftmaxNormalizedComponent::Copy() const {
  LinearSoftmaxNormalizedComponent *ans = new LinearSoftmaxNormalizedComponent(*this);
  return ans;
}

BaseFloat LinearSoftmaxNormalizedComponent::DotProduct(const LmComponent &other_in) const {
//  KALDI_ASSERT(is_gradient_); // actually there are more problems here ...
  const LinearSoftmaxNormalizedComponent *other =
      dynamic_cast<const LinearSoftmaxNormalizedComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans);
}

void LinearSoftmaxNormalizedComponent::Init(int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev) {
  linear_params_.Resize(output_dim, input_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Row(0).SetZero();
  linear_params_.Scale(param_stddev);
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::Init(std::string matrix_filename) {
  ReadKaldiObject(matrix_filename, &linear_params_); // will abort on failure.
//  normalized_ = false;
  Normalize();
}

void LinearSoftmaxNormalizedComponent::InitFromConfig(ConfigLine *cfl) {
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

void LinearSoftmaxNormalizedComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                const vector<int> &indexes,
                                                CuMatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(false);
}

void LinearSoftmaxNormalizedComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const CuMatrixBase<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  KALDI_ASSERT(false);
}


void LinearSoftmaxNormalizedComponent::Propagate(const CuMatrixBase<BaseFloat> &in,
                                                const vector<int> &indexes,
                                                vector<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() == indexes.size());
  out->resize(indexes.size());

  for (int i = 0; i < indexes.size(); i++) {
    int w = indexes[i];
//    KALDI_LOG << in.Row(i).Sum() << " should be close to 1";
    KALDI_ASSERT(ApproxEqual(in.Row(i).Sum(), 1.0));
    BaseFloat res = VecVec(in.Row(i), actual_params_.Row(w));
//    KALDI_ASSERT(res >= 0 && res <= 1);
    (*out)[i] = res;
  }
}

void LinearSoftmaxNormalizedComponent::Backprop(
                               const vector<int> &indexes,
                               const CuMatrixBase<BaseFloat> &in_value,
                               const CuMatrixBase<BaseFloat> &, // out_value
                               const vector<BaseFloat> &output_deriv,
                               LmOutputComponent *to_update_0,
                               CuMatrixBase<BaseFloat> *input_deriv) const {

  int k = indexes.size();

  KALDI_ASSERT(input_deriv != NULL);

  for (int i = 0; i < k; i++) {
    int index = indexes[i];
    input_deriv->Row(i).AddVec(output_deriv[i], actual_params_.Row(index));
  }

  LinearSoftmaxNormalizedComponent* to_update
             = dynamic_cast<LinearSoftmaxNormalizedComponent*>(to_update_0);

  KALDI_ASSERT(to_update != NULL);

  CuMatrix<BaseFloat> aT(actual_params_, kTrans);
  CuMatrix<BaseFloat> dapT(actual_params_, kTrans);
  CuMatrix<BaseFloat> daT(actual_params_, kTrans);
//  aT.SetZero();
  dapT.SetZero();
  daT.SetZero();
  for (int i = 0; i < k; i++) {
    int index = indexes[i];
    daT.ColRange(index, 1).AddVecToCols(output_deriv[i], in_value.Row(i), 1.0);
  }
  dapT.DiffSoftmaxPerRow(aT, daT);
//  KALDI_LOG << aT.Sum() << " and " << daT.Sum() << " and " <<dapT.Sum();
  to_update->linear_params_.AddMat(learning_rate_, dapT, kTrans);  // TODO(hxu)
  to_update->linear_params_.Row(0).SetZero();
}

void LinearSoftmaxNormalizedComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</LinearSoftmaxNormalizedComponent>");
  Normalize();
}

void LinearSoftmaxNormalizedComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</LinearSoftmaxNormalizedComponent>");
}

int32 LinearSoftmaxNormalizedComponent::NumParameters() const {
  return InputDim() * OutputDim(); // actually should be (InputDim() - 1 ) * OutputDim()
}

void LinearSoftmaxNormalizedComponent::Vectorize(VectorBase<BaseFloat> *params) const {
  KALDI_ASSERT(params->Dim() == this->NumParameters());
  params->Range(0, InputDim() * OutputDim()).CopyRowsFromMat(linear_params_);
}

void LinearSoftmaxNormalizedComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
  KALDI_ASSERT(params.Dim() == this->NumParameters());
  linear_params_.CopyRowsFromVec(params.Range(0, InputDim() * OutputDim()));
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
