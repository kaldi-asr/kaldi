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

//using namespace nnet3;
//using nnet3::LmUpdatableComponent;
//using nnet3::LmNonlinearComponent;
//using nnet3::Component;
//using nnet3::ConfigLine;
//using nnet3::ComponentPrecomputedIndexes;
//using nnet3::kSimpleComponent;
//using nnet3::kLmUpdatableComponent;
//using nnet3::kLinearInParameters;
//using nnet3::kBackpropNeedsInput;
//using nnet3::kBackpropAdds;
//using nnet3::kBackpropNeedsOutput;
//using nnet3::kStoresStats;
//using nnet3::ConfigLine;

class LmFixedAffineSampleLogSoftmaxComponent;
class LmSoftmaxComponent;
class LmLogSoftmaxComponent;

// Affine means a linear function plus an offset.
// Note: although this class can be instantiated, it also
// functions as a base-class for more specialized versions of
// AffineSampleLogSoftmaxComponent.
class AffineSampleLogSoftmaxComponent: public LmUpdatableComponent {
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


  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const MatrixBase<BaseFloat> &in,
                         MatrixBase<BaseFloat> *out) const;

//  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
//                         const SparseMatrix<BaseFloat> &in,
//                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &, // out_value
                        const MatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        MatrixBase<BaseFloat> *in_deriv) const;

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const SparseMatrix<BaseFloat> &in,
                         MatrixBase<BaseFloat> *out) const {
    KALDI_ASSERT(false);
  }

  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const SparseMatrix<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &, // out_value
                        const MatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        MatrixBase<BaseFloat> *in_deriv) const {
    KALDI_ASSERT(false);
  }

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual LmComponent* Copy() const;


  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const LmComponent &other);
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const LmUpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.

  // This new function is used when mixing up:
  virtual void SetParams(const VectorBase<BaseFloat> &bias,
                         const MatrixBase<BaseFloat> &linear);
  const Vector<BaseFloat> &BiasParams() { return bias_params_; }
  const Matrix<BaseFloat> &LinearParams() { return linear_params_; }
  explicit AffineSampleLogSoftmaxComponent(const AffineSampleLogSoftmaxComponent &other);
  // The next constructor is used in converting from nnet1.
  AffineSampleLogSoftmaxComponent(const MatrixBase<BaseFloat> &linear_params,
                  const VectorBase<BaseFloat> &bias_params,
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
  virtual void Update(
      const std::string &debug_info,
      const MatrixBase<BaseFloat> &in_value,
      const MatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }
  // UpdateSimple is used when *this is a gradient.  Child classes may override
  // this if needed, but typically won't need to.
  virtual void UpdateSimple(
      const MatrixBase<BaseFloat> &in_value,
      const MatrixBase<BaseFloat> &out_deriv);

  const AffineSampleLogSoftmaxComponent &operator = (const AffineSampleLogSoftmaxComponent &other); // Disallow.
  Matrix<BaseFloat> linear_params_;
  Vector<BaseFloat> bias_params_;
};

// Affine means a linear function plus an offset.
// Note: although this class can be instantiated, it also
// functions as a base-class for more specialized versions of
// AffineSampleLogSoftmaxComponent.
class LmLinearComponent: public LmUpdatableComponent {
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


  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const MatrixBase<BaseFloat> &in,
                         MatrixBase<BaseFloat> *out) const;

  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &, // out_value
                        const MatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        MatrixBase<BaseFloat> *in_deriv) const;

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const SparseMatrix<BaseFloat> &in,
                         MatrixBase<BaseFloat> *out) const;

  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const SparseMatrix<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &, // out_value
                        const MatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        MatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual LmComponent* Copy() const;


  // Some functions from base-class LmUpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const LmComponent &other);
  virtual void SetZero(bool treat_as_gradient);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const LmUpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.

  // This new function is used when mixing up:
  virtual void SetParams(//const VectorBase<BaseFloat> &bias,
                         const MatrixBase<BaseFloat> &linear);
//  const Vector<BaseFloat> &BiasParams() { return bias_params_; }
  const Matrix<BaseFloat> &LinearParams() { return linear_params_; }
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

  // The following functions are used for collapsing multiple layers
  // together.  They return a pointer to a new Component equivalent to
  // the sequence of two components.  We haven't implemented this for
  // FixedLinearComponent yet.
  LmComponent *CollapseWithNext(const LmLinearComponent &next) const ;
//  Component *CollapseWithNext(const LmFixedAffineSampleLogSoftmaxComponent &next) const;
//  Component *CollapseWithNext(const FixedScaleComponent &next) const;
//  Component *CollapseWithPrevious(const LmFixedAffineSampleLogSoftmaxComponent &prev) const;

 protected:
//  friend class NaturalGradientAffineSampleLogSoftmaxComponent;
  // This function Update() is for extensibility; child classes may override
  // this, e.g. for natural gradient update.
  virtual void Update(
      const std::string &debug_info,
      const MatrixBase<BaseFloat> &in_value,
      const MatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }

  virtual void Update(
      const std::string &debug_info,
      const SparseMatrix<BaseFloat> &in_value,
      const MatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }
  // UpdateSimple is used when *this is a gradient.  Child classes may override
  // this if needed, but typically won't need to.
  virtual void UpdateSimple(
      const MatrixBase<BaseFloat> &in_value,
      const MatrixBase<BaseFloat> &out_deriv);

  virtual void UpdateSimple(
      const SparseMatrix<BaseFloat> &in_value,
      const MatrixBase<BaseFloat> &out_deriv);

  const LmLinearComponent &operator = (const LmLinearComponent &other); // Disallow.
  Matrix<BaseFloat> linear_params_;
//  Vector<BaseFloat> bias_params_;
};

class LmSoftmaxComponent: public LmNonlinearComponent {
 public:
  explicit LmSoftmaxComponent(const LmSoftmaxComponent &other):
      LmNonlinearComponent(other) { }
  LmSoftmaxComponent() { }
  virtual std::string Type() const { return "LmSoftmaxComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsOutput|kStoresStats;
  }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const MatrixBase<BaseFloat> &in,
                         MatrixBase<BaseFloat> *out) const;
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const SparseMatrix<BaseFloat> &in,
                         MatrixBase<BaseFloat> *out) const {
    KALDI_ASSERT(0);
  }
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        MatrixBase<BaseFloat> *in_deriv) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const SparseMatrix<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        MatrixBase<BaseFloat> *in_deriv) const {
    KALDI_ASSERT(0);
  }
  virtual void StoreStats(const MatrixBase<BaseFloat> &out_value);

  virtual LmComponent* Copy() const { return new LmSoftmaxComponent(*this); }
 private:
  LmSoftmaxComponent &operator = (const LmSoftmaxComponent &other); // Disallow.
};

class LmLogSoftmaxComponent: public LmNonlinearComponent {
 public:
  explicit LmLogSoftmaxComponent(const LmLogSoftmaxComponent &other):
      LmNonlinearComponent(other) { }
  LmLogSoftmaxComponent() { }
  virtual std::string Type() const { return "LmLogSoftmaxComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsOutput|kStoresStats;
  }
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const MatrixBase<BaseFloat> &in,
                         MatrixBase<BaseFloat> *out) const;
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const SparseMatrix<BaseFloat> &in,
                         MatrixBase<BaseFloat> *out) const {
    KALDI_ASSERT(0);
  }
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        MatrixBase<BaseFloat> *in_deriv) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const SparseMatrix<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &out_value,
                        const MatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        MatrixBase<BaseFloat> *in_deriv) const {
    KALDI_ASSERT(0);
  }

  virtual LmComponent* Copy() const { return new LmLogSoftmaxComponent(*this); }
 private:
  LmLogSoftmaxComponent &operator = (const LmLogSoftmaxComponent &other); // Disallow.
};

/*
/// Keywords: natural gradient descent, NG-SGD, naturalgradient.  For
/// the top-level of the natural gradient code look here, and also in
/// nnet-precondition-online.h.
/// NaturalGradientAffineSampleLogSoftmaxComponent is
/// a version of AffineSampleLogSoftmaxComponent that has a non-(multiple of unit) learning-rate
/// matrix.  See nnet-precondition-online.h for a description of the technique.
/// It is described, under the name Online NG-SGD, in the paper "Parallel
/// training of DNNs with Natural Gradient and Parameter Averaging" (ICLR
/// workshop, 2015) by Daniel Povey, Xiaohui Zhang and Sanjeev Khudanpur.
class NaturalGradientAffineSampleLogSoftmaxComponent: public AffineSampleLogSoftmaxComponent {
 public:
  virtual std::string Type() const { return "NaturalGradientAffineSampleLogSoftmaxComponent"; }
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  void Init(int32 input_dim, int32 output_dim,
            BaseFloat param_stddev, BaseFloat bias_stddev, BaseFloat bias_mean,
            int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history, BaseFloat alpha,
            BaseFloat max_change_per_sample);
  void Init(int32 rank_in, int32 rank_out, int32 update_period,
            BaseFloat num_samples_history,
            BaseFloat alpha, BaseFloat max_change_per_sample,
            std::string matrix_filename);
  // this constructor does not really initialize, use Init() or Read().
  NaturalGradientAffineSampleLogSoftmaxComponent();
  virtual void Resize(int32 input_dim, int32 output_dim);
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Info() const;
  virtual Component* Copy() const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  // copy constructor
  explicit NaturalGradientAffineSampleLogSoftmaxComponent(
      const NaturalGradientAffineSampleLogSoftmaxComponent &other);
  virtual void ZeroStats();

 private:
  // disallow assignment operator.
  NaturalGradientAffineSampleLogSoftmaxComponent &operator= (
      const NaturalGradientAffineSampleLogSoftmaxComponent&);

  // Configs for preconditioner.  The input side tends to be better conditioned ->
  // smaller rank needed, so make them separately configurable.
  int32 rank_in_;
  int32 rank_out_;
  int32 update_period_;
  BaseFloat num_samples_history_;
  BaseFloat alpha_;

  OnlineNaturalGradient preconditioner_in_;

  OnlineNaturalGradient preconditioner_out_;

  // If > 0, max_change_per_sample_ is the maximum amount of parameter
  // change (in L2 norm) that we allow per sample, averaged over the minibatch.
  // This was introduced in order to control instability.
  // Instead of the exact L2 parameter change, for
  // efficiency purposes we limit a bound on the exact
  // change.  The limit is applied via a constant <= 1.0
  // for each minibatch, A suitable value might be, for
  // example, 10 or so; larger if there are more
  // parameters.
  BaseFloat max_change_per_sample_;

  // update_count_ records how many updates we have done.
  double update_count_;

  // active_scaling_count_ records how many updates we have done,
  // where the scaling factor is active (not 1.0).
  double active_scaling_count_;

  // max_change_scale_stats_ records the sum of scaling factors
  // in each update, so we can compute the averaged scaling factor
  // in Info().
  double max_change_scale_stats_;

  // Sets the configs rank, alpha and eta in the preconditioner objects,
  // from the class variables.
  void SetNaturalGradientConfigs();

  virtual void Update(
      const std::string &debug_info,
      const MatrixBase<BaseFloat> &in_value,
      const MatrixBase<BaseFloat> &out_deriv);
};
*/

/// FixedAffineSampleLogSoftmaxComponent is an affine transform that is supplied
/// at network initialization time and is not trainable.
class LmFixedAffineSampleLogSoftmaxComponent: public LmComponent {
 public:
  LmFixedAffineSampleLogSoftmaxComponent() { }
  virtual std::string Type() const { return "LmFixedAffineSampleLogSoftmaxComponent"; }
  virtual std::string Info() const;

  /// matrix should be of size input-dim+1 to output-dim, last col is offset
  void Init(const MatrixBase<BaseFloat> &matrix);

  // The ConfigLine cfl contains just the option matrix=<string>,
  // where the string is the filename of a Kaldi-format matrix to read.
  virtual void InitFromConfig(ConfigLine *cfl);

  virtual int32 Properties() const { return kSimpleComponent|kBackpropAdds; }
  virtual int32 InputDim() const { return linear_params_.NumCols(); }
  virtual int32 OutputDim() const { return linear_params_.NumRows(); }

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const MatrixBase<BaseFloat> &in,
                         MatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const MatrixBase<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &, // out_value
                        const MatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        MatrixBase<BaseFloat> *in_deriv) const;

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const SparseMatrix<BaseFloat> &in,
                         MatrixBase<BaseFloat> *out) const {
    KALDI_ASSERT(0);
  } //TODO
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const SparseMatrix<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &, // out_value
                        const MatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        MatrixBase<BaseFloat> *in_deriv) const { // TODO
    KALDI_ASSERT(0);
  } //TODO


  virtual LmComponent* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  // Function to provide access to linear_params_.
  const Matrix<BaseFloat> &LinearParams() const { return linear_params_; }
 protected:
  friend class AffineSampleLogSoftmaxComponent;
  Matrix<BaseFloat> linear_params_;
  Vector<BaseFloat> bias_params_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(LmFixedAffineSampleLogSoftmaxComponent);
};


} // namespace rnnlm
} // namespace kaldi


#endif
