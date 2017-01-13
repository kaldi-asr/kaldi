// nnet3/nnet-simple-component.h


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

#ifndef KALDI_RNNLM_RNNLM_COMPONENT_H_
#define KALDI_RNNLM_RNNLM_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "rnnlm/rnnlm-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include "rnnlm/nnet-parse.h"
#include <iostream>

namespace kaldi {
namespace rnnlm {

class LmFixedAffineSampleLogSoftmaxComponent;
class LmSoftmaxComponent;
class LmLogSoftmaxComponent;

using std::vector;

// Affine means a linear function plus an offset.
// Note: although this class can be instantiated, it also
// functions as a base-class for more specialized versions of
// AffineSampleLogSoftmaxComponent.
class LmLinearComponent: public LmInputComponent {
  friend class LmSoftmaxComponent; // Friend declaration relates to mixing up.
 public:

  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  LmLinearComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "LmLinearComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kLinearInParameters|
        kBackpropNeedsInput|kBackpropAdds;
  }

  virtual void Backprop(const SparseMatrix<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv = NULL) const;

  virtual void Propagate(const SparseMatrix<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual LmComponent* Copy() const;

  // Some functions from base-class LmUpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const LmComponent &other);
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const LmComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.

  // This new function is used when mixing up:
  virtual void SetParams(//const VectorBase<BaseFloat> &bias,
                         const MatrixBase<BaseFloat> &linear);
//  const Vector<BaseFloat> &BiasParams() { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() { return linear_params_; }
  explicit LmLinearComponent(const LmLinearComponent &other);
  // The next constructor is used in converting from nnet1.
  LmLinearComponent(const MatrixBase<BaseFloat> &linear_params,
//                  const VectorBase<BaseFloat> &bias_params,
                  BaseFloat learning_rate);
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev);//, BaseFloat bias_stddev);
  void Init(std::string matrix_filename);

  // This function resizes the dimensions of the component, setting the
  // parameters to zero, while leaving any other configuration values the same.
  virtual void Resize(int32 input_dim, int32 output_dim);

  LmComponent *CollapseWithNext(const LmLinearComponent &next) const ;

 protected:
  virtual void Update(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }

  virtual void Update(
      const SparseMatrix<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }

  virtual void UpdateSimple(
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  virtual void UpdateSimple(
      const SparseMatrix<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv);

  const LmLinearComponent &operator = (const LmLinearComponent &other); // Disallow.
  CuMatrix<BaseFloat> linear_params_;
};

class LinearSigmoidNormalizedComponent: public LmOutputComponent {
 public:
  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  LinearSigmoidNormalizedComponent() {} // use Init to really initialize.
  virtual std::string Type() const { return "LinearSigmoidNormalizedComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kLinearInParameters|
        kBackpropNeedsInput|kBackpropAdds;
  }

  void Propagate(const CuMatrixBase<BaseFloat> &in,
                 const vector<int> &indexes,
                 vector<BaseFloat> *out) const;

  void Backprop(
         const vector<int> &indexes,
         const CuMatrixBase<BaseFloat> &in_value,
         const CuMatrixBase<BaseFloat> &, // out_value
         const vector<BaseFloat> &output_deriv,
         LmOutputComponent *to_update_0,
         CuMatrixBase<BaseFloat> *input_deriv) const;

  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                 const vector<int> &indexes,
                 CuMatrixBase<BaseFloat> *out) const;

//  virtual void Backprop(
//             const vector<vector<int> > &indexes,
//             const MatrixBase<BaseFloat> &in_value,
//             const MatrixBase<BaseFloat> &, // out_value
//             const vector<vector<BaseFloat> > &out_deriv,
//             LmOutputComponent *to_update_in,
//             MatrixBase<BaseFloat> *in_deriv) const;

  virtual void Backprop(
             const vector<int> &indexes,
             const CuMatrixBase<BaseFloat> &in_value,
             const CuMatrixBase<BaseFloat> &, // out_value
             const CuMatrixBase<BaseFloat> &out_deriv,
             LmOutputComponent *to_update_in,
             CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual LmComponent* Copy() const;


  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const LmComponent &other);
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const LmComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.

  // This new function is used when mixing up:
  virtual void SetParams(const CuMatrixBase<BaseFloat> &linear);
  const CuMatrix<BaseFloat> &LinearParams() { return linear_params_; }
  explicit LinearSigmoidNormalizedComponent(const LinearSigmoidNormalizedComponent &other);
  // The next constructor is used in converting from nnet1.
  LinearSigmoidNormalizedComponent(const CuMatrixBase<BaseFloat> &linear_params,
                  BaseFloat learning_rate);
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev);
  void Init(std::string matrix_filename);

  // This function resizes the dimensions of the component, setting the
  // parameters to zero, while leaving any other configuration values the same.
  virtual void Resize(int32 input_dim, int32 output_dim);

  // The following functions are used for collapsing multiple layers
  // together.  They return a pointer to a new Component equivalent to
  // the sequence of two components.  We haven't implemented this for
  // FixedLinearComponent yet.

 protected:

  void Normalize();

  const LinearSigmoidNormalizedComponent &operator =
     (const LinearSigmoidNormalizedComponent &other); // Disallow.
  CuMatrix<BaseFloat> linear_params_;
//  CuVector<BaseFloat> normalizer_;
  CuMatrix<BaseFloat> actual_params_;
//  bool normalized_;
};

class LinearSoftmaxNormalizedComponent: public LmOutputComponent {
 public:
  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  LinearSoftmaxNormalizedComponent() {} // use Init to really initialize.
  virtual std::string Type() const { return "LinearSoftmaxNormalizedComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kLinearInParameters|
        kBackpropNeedsInput|kBackpropAdds;
  }

  void Propagate(const CuMatrixBase<BaseFloat> &in,
                 const vector<int> &indexes,
                 vector<BaseFloat> *out) const;

  void Backprop(
         const vector<int> &indexes,
         const CuMatrixBase<BaseFloat> &in_value,
         const CuMatrixBase<BaseFloat> &, // out_value
         const vector<BaseFloat> &output_deriv,
         LmOutputComponent *to_update_0,
         CuMatrixBase<BaseFloat> *input_deriv) const;

  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                 const vector<int> &indexes,
                 CuMatrixBase<BaseFloat> *out) const;

//  virtual void Backprop(
//             const vector<vector<int> > &indexes,
//             const MatrixBase<BaseFloat> &in_value,
//             const MatrixBase<BaseFloat> &, // out_value
//             const vector<vector<BaseFloat> > &out_deriv,
//             LmOutputComponent *to_update_in,
//             MatrixBase<BaseFloat> *in_deriv) const;

  virtual void Backprop(
             const vector<int> &indexes,
             const CuMatrixBase<BaseFloat> &in_value,
             const CuMatrixBase<BaseFloat> &, // out_value
             const CuMatrixBase<BaseFloat> &out_deriv,
             LmOutputComponent *to_update_in,
             CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual LmComponent* Copy() const;


  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const LmComponent &other);
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const LmComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.

  // This new function is used when mixing up:
  virtual void SetParams(const CuMatrixBase<BaseFloat> &linear);
  const CuMatrix<BaseFloat> &LinearParams() { return linear_params_; }
  explicit LinearSoftmaxNormalizedComponent(const LinearSoftmaxNormalizedComponent &other);
  // The next constructor is used in converting from nnet1.
  LinearSoftmaxNormalizedComponent(const CuMatrixBase<BaseFloat> &linear_params,
                  BaseFloat learning_rate);
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev);
  void Init(std::string matrix_filename);

  // This function resizes the dimensions of the component, setting the
  // parameters to zero, while leaving any other configuration values the same.
  virtual void Resize(int32 input_dim, int32 output_dim);

  // The following functions are used for collapsing multiple layers
  // together.  They return a pointer to a new Component equivalent to
  // the sequence of two components.  We haven't implemented this for
  // FixedLinearComponent yet.

 protected:

  void Normalize();

  const LinearSoftmaxNormalizedComponent &operator =
     (const LinearSoftmaxNormalizedComponent &other); // Disallow.
  CuMatrix<BaseFloat> linear_params_;
//  CuVector<BaseFloat> normalizer_;
  CuMatrix<BaseFloat> actual_params_;
//  bool normalized_;
};

class AffineSampleLogSoftmaxComponent: public LmOutputComponent {
 public:
  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  AffineSampleLogSoftmaxComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "AffineSampleLogSoftmaxComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kLinearInParameters|
        kBackpropNeedsInput|kBackpropAdds;
  }


  virtual void Propagate(const CuMatrixBase<BaseFloat> &in,
                         const vector<int> &indexes,
                         CuMatrixBase<BaseFloat> *out) const;

  void Propagate(const CuMatrixBase<BaseFloat> &in,
                 bool normalize,
                 CuMatrixBase<BaseFloat> *out) const;

  virtual void Backprop(const vector<int> &indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        LmOutputComponent *to_update_in,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual LmComponent* Copy() const;

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const LmComponent &other);
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const LmComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.

  // This new function is used when mixing up:
  virtual void SetParams(const CuMatrixBase<BaseFloat> &bias,
                         const CuMatrixBase<BaseFloat> &linear);
  const CuMatrix<BaseFloat> &BiasParams() { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() { return linear_params_; }
  explicit AffineSampleLogSoftmaxComponent(const AffineSampleLogSoftmaxComponent &other);
  // The next constructor is used in converting from nnet1.
  AffineSampleLogSoftmaxComponent(const CuMatrixBase<BaseFloat> &linear_params,
                  const CuMatrixBase<BaseFloat> &bias_params,
                  BaseFloat learning_rate);
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev);
  void Init(std::string matrix_filename);

  // This function resizes the dimensions of the component, setting the
  // parameters to zero, while leaving any other configuration values the same.
  virtual void Resize(int32 input_dim, int32 output_dim);

  // The following functions are used for collapsing multiple layers
  // together.  They return a pointer to a new Component equivalent to
  // the sequence of two components.  We haven't implemented this for
  // FixedLinearComponent yet.

 protected:
  // This function Update() is for extensibility; child classes may override
  // this, e.g. for natural gradient update.
//  virtual void Update(
//      const CuMatrixBase<BaseFloat> &in_value,
//      const CuMatrixBase<BaseFloat> &out_deriv) {
//    UpdateSimple(in_value, out_deriv);
//  }
//  // UpdateSimple is used when *this is a gradient.  Child classes may override
//  // this if needed, but typically won't need to.
//  virtual void UpdateSimple(
//      const CuMatrixBase<BaseFloat> &in_value,
//      const CuMatrixBase<BaseFloat> &out_deriv);

  const AffineSampleLogSoftmaxComponent &operator = (const AffineSampleLogSoftmaxComponent &other); // Disallow.
  CuMatrix<BaseFloat> linear_params_;
  CuMatrix<BaseFloat> bias_params_;  // a 1 * dim() matrix
};


} // namespace rnnlm
} // namespace kaldi


#endif
