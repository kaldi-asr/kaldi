// nnet3/nnet-component-itf.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Guoguo Chen
//                2015  Xiaohui Zhang

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

#ifndef KALDI_RNNLM_RNNLM_COMPONENT_ITF_H_
#define KALDI_RNNLM_RNNLM_COMPONENT_ITF_H_

#include "nnet3/nnet-common.h"
#include "rnnlm/nnet-parse.h"
#include "nnet3/nnet-parse.h"
#include "base/kaldi-error.h"
#include "thread/kaldi-mutex.h"

//#include "nnet3/nnet-component-itf.h"
//#include "rnnlm/rnnlm-component-itf.h"
//#include "nnet3/nnet-computation-graph.h"
#include <iostream>

namespace kaldi {

namespace nnet3 {
  class IndexSet;
}

namespace rnnlm {


using std::vector;
using nnet3::MiscComputationInfo;
using nnet3::Index;
using nnet3::ConfigLine;
using nnet3::SummarizeVector;
using nnet3::ExpectOneOrTwoTokens;

enum ComponentProperties {
  kSimpleComponent = 0x001,  // true if number of rows of input equals number of rows
                             // of output and this component doesn't care about the indexes
                             // (i.e. it maps each row of input to each row of output without
                             // regard to the index values).  Will normally be true.
  kUpdatableComponent = 0x002,  // true if the component has parameters that can
                                // be updated.  Components that return this flag
                                // must be dynamic_castable to type
                                // UpdatableComponent (but components of type
                                // UpdatableComponent do not have to return this
                                // flag, e.g.  if this instance is not really
                                // updatable).
  kLinearInInput = 0x004,    // true if the component's output is always a
                             // linear function of its input, i.e. alpha times
                             // input gives you alpha times output.
  kLinearInParameters = 0x008, // true if an updatable component's output is always a
                               // linear function of its parameters, i.e. alpha times
                               // parameters gives you alpha times output.  This is true
                               // for all updatable components we envisage.
  kPropagateInPlace = 0x010,  // true if we can do the propagate operation in-place
                              // (input and output matrices are the same).
                              // Note: if doing backprop, you'd also need to check
                              // that the kBackpropNeedsInput property is not true.
  kPropagateAdds = 0x020,  // true if the Propagate function adds to, rather
                           // than setting, its output.  The Component chooses
                           // whether to add or set, and the calling code has to
                           // accommodate it.
  kReordersIndexes = 0x040,  // true if the ReorderIndexes function might reorder
                             // the indexes (otherwise we can skip calling it).
                             // Must not be set for simple components.
  kBackpropAdds = 0x080,   // true if the Backprop function adds to, rather than
                           // setting, the "in_deriv" output.  The Component
                           // chooses whether to add or set, and the calling
                           // code has to accommodate it.  Note: in the case of
                           // in-place backprop, this flag has no effect.
  kBackpropNeedsInput = 0x100,  // true if backprop operation needs access to
                                // forward-pass input.
  kBackpropNeedsOutput = 0x200,  // true if backprop operation needs access to
                                 // forward-pass output (e.g. true for Sigmoid).
  kBackpropInPlace = 0x400,   // true if we can do the backprop operation in-place
                             // (input and output matrices may be the same).
  kStoresStats = 0x800,      // true if the StoreStats operation stores
                             // statistics e.g. on average node activations and
                             // derivatives of the nonlinearity, (as it does for
                             // Tanh, Sigmoid, ReLU and Softmax).
  kInputContiguous = 0x1000,  // true if the component requires its input data (and
                              // input derivatives) to have Stride()== NumCols().
  kOutputContiguous = 0x2000  // true if the component requires its input data (and
                              // output derivatives) to have Stride()== NumCols().
};

class ComponentPrecomputedIndexes {
 public:
  virtual ComponentPrecomputedIndexes *Copy() const = 0;
  virtual void Write(std::ostream &os, bool binary) const = 0;
  virtual void Read(std::istream &os, bool binary) = 0;
  virtual std::string Type() const = 0;
  static ComponentPrecomputedIndexes* ReadNew(std::istream &is, bool binary);
  // cpi stands for component_precomputed_indexes
  static ComponentPrecomputedIndexes* NewComponentPrecomputedIndexesOfType(
                                           const std::string &cpi_type);
  virtual ~ComponentPrecomputedIndexes() { }
};


class LmComponent {
 public:
  LmComponent(const LmComponent &other):
      learning_rate_(other.learning_rate_),
      learning_rate_factor_(other.learning_rate_factor_),
      is_gradient_(other.is_gradient_) { }
  LmComponent(): learning_rate_(0.001), learning_rate_factor_(1.0),
                        is_gradient_(false) { }

  /// \brief This function may store stats on average activation values, and for
  ///        some component types, the average value of the derivative of the
  ///        nonlinearity.  It only does something for those components that
  ///        have nonzero Properties()&kStoresStats.  It only needs as input
  ///        the value at the output of the nonlinearity.

  virtual void StoreStats(const MatrixBase<BaseFloat> &out_value) { }

  /// \brief Components that provide an implementation of StoreStats should also
  ///        provide an implementation of ZeroStats(), to set those stats to
  ///        zero.  Other components that store other types of statistics
  ///        (e.g. regarding gradient clipping) are free to implement ZeroStats()
  ///        also.
  virtual void ZeroStats() { }



  /// \brief  This function only does something interesting for non-simple Components.
  ///   For a given index at the output of the component, tells us what indexes
  ///   are required at its input (note: "required" encompasses also optionally-required
  ///   things; it will enumerate all things that we'd like to have).  See also
  ///   IsComputable().
  /// \param [in] misc_info  This argument is supplied to handle things that the
  ///       framework can't very easily supply: information like which time
  ///       indexes are needed for AggregateComponent, which time-indexes are
  ///       available at the input of a recurrent network, and so on.  We will
  ///       add members to misc_info as needed.
  /// \param [in] output_index  The Index at the output of the component, for
  ///       which we are requesting the list of indexes at the component's input.
  /// \param [out] desired_indexes  A list of indexes that are desired at the input.
  ///       By "desired" we mean required or optionally-required.
  ///
  /// The default implementation of this function is suitable for any
  /// SimpleComponent; it just copies the output_index to a single identical
  /// element in input_indexes.
//  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
//                               const Index &output_index,
//                               std::vector<Index> *desired_indexes) const;

  /// \brief This function only does something interesting for non-simple
  ///    Components, and it exists to make it possible to manage
  ///    optionally-required inputs.  It tells the user whether a given output
  ///    index is computable from a given set of input indexes, and if so,
  ///    says which input indexes will be used in the computation.
  ///
  ///    Implementations of this function are required to have the property that
  ///    adding an element to "input_index_set" can only ever change IsComputable
  ///    from false to true, never vice versa.
  ///
  ///    @param [in] misc_info  Some information specific to the computation, such as
  ///              minimum and maximum times for certain components to do adaptation on;
  ///              it's a place to put things that don't easily fit in the framework.
  ///    @param [in] output_index  The index that is to be computed at the output
  ///              of this Component.
  ///    @param [in] input_index_set  The set of indexes that is available at the
  ///              input of this Component.
  ///    @param [out] used_inputs  If non-NULL, then if the output is computable
  ///       this will be set to the list of input indexes that will actually be
  ///       used in the computation.
  ///    @return Returns true iff this output is computable from the provided
  ///          inputs.
  ///
  ///   The default implementation of this function is suitable for any
  ///   SimpleComponent: it just returns true if output_index is in
  ///   input_index_set, and if so sets used_inputs to vector containing that
  ///   one Index.

  /// \brief Returns a string such as "SigmoidComponent", describing
  ///        the type of the object.
  virtual std::string Type() const = 0;

  /// \brief  Initialize, from a ConfigLine object.
  /// \param [in] cfl  A ConfigLine containing any parameters that
  ///            are needed for initialization. For example:
  ///            "dim=100 param-stddev=0.1"
  virtual void InitFromConfig(ConfigLine *cfl) = 0;

  /// \brief Returns input-dimension of this component.
  virtual int32 InputDim() const = 0;

  /// \brief Returns output-dimension of this component.
  virtual int32 OutputDim() const = 0;

  /// \brief Return bitmask of the component's properties.
  ///   These properties depend only on the component's type.
  ///   See enum ComponentProperties.
  virtual int32 Properties() const = 0;

  /// \brief Read component from stream (works out its type).  Dies on error.
  static LmComponent* ReadNew(std::istream &is, bool binary);

  /// \brief Copies component (deep copy).
  virtual LmComponent* Copy() const = 0;

  /// \brief Returns a new Component of the given type e.g. "SoftmaxComponent",
  ///   or NULL if no such component type exists.
  static LmComponent *NewComponentOfType(const std::string &type);

  /// \brief Read function (used after we know the type of the Component);
  ///   accepts input that is missing the token that describes the component
  ///   type, in case it has already been consumed.
  virtual void Read(std::istream &is, bool binary) = 0;

  /// \brief Write component to stream
  virtual void Write(std::ostream &os, bool binary) const = 0;

  /// \brief Returns some text-form information about this component, for diagnostics.
  ///     Starts with the type of the component.  E.g. "SigmoidComponent dim=900",
  ///     although most components will have much more info.
  virtual std::string Info() const;

  /// This virtual function when called by
  //    -- an LmUpdatableComponent scales the parameters
  ///      by "scale" when called by an LmUpdatableComponent.
  //    -- a Nonlinear component it relates to scaling activation stats, not parameters.
  virtual void Scale(BaseFloat scale) {};

  /// This virtual function when called by
  ///    -- an LmUpdatableComponent adds the parameters of
  ///      another updatable component, times some constant, to the current
  ///      parameters.
  ///    -- a LmNonlinearComponent it relates to adding stats
  /// Otherwise it should do nothing.
  virtual void Add(BaseFloat alpha, const LmComponent &other) {};

//  LmComponent() { }

  virtual ~LmComponent() { }

  virtual BaseFloat DotProduct(const LmComponent &other) const = 0;

  /// This function is to be used in testing.  It adds unit noise times "stddev"
  /// to the parameters of the component.
  virtual void PerturbParams(BaseFloat stddev) = 0;

  /// Sets the learning rate of gradient descent- gets multiplied by
  /// learning_rate_factor_.
  virtual void SetUnderlyingLearningRate(BaseFloat lrate) {
    learning_rate_ = lrate * learning_rate_factor_;
  }

  /// Sets the learning rate directly, bypassing learning_rate_factor_.
  virtual void SetActualLearningRate(BaseFloat lrate) { learning_rate_ = lrate; }

  /// Gets the learning rate of gradient descent.  Note: if you call
  /// SetLearningRate(x), and learning_rate_factor_ != 1.0,
  /// a different value than x will returned.
  BaseFloat LearningRate() const { return learning_rate_; }

  /// The following new virtual function returns the total dimension of
  /// the parameters in this class.
  virtual int32 NumParameters() const { KALDI_ASSERT(0); return 0; }

  /// Turns the parameters into vector form.  We put the vector form on the CPU,
  /// because in the kinds of situations where we do this, we'll tend to use
  /// too much memory for the GPU.
  virtual void Vectorize(VectorBase<BaseFloat> *params) const { KALDI_ASSERT(0); }
  /// Converts the parameters from vector form.
  virtual void UnVectorize(const VectorBase<BaseFloat> &params) {
    KALDI_ASSERT(0);
  }

// private:
//  KALDI_DISALLOW_COPY_AND_ASSIGN(LmComponent);

 protected:
  BaseFloat learning_rate_; ///< learning rate (typically 0.0..0.01)
  BaseFloat learning_rate_factor_; ///< learning rate factor (normally 1.0, but
                                   ///< can be set to another < value so that
                                   ///when < you call SetLearningRate(), that
                                   ///value will be scaled by this factor.
  bool is_gradient_;  ///< True if this component is to be treated as a gradient rather
                      ///< than as parameters.  Its main effect is that we disable
                      ///< any natural-gradient update and just compute the standard
                      ///< gradient.
};

class LmInputComponent: public LmComponent {
 public:
  LmInputComponent(const LmInputComponent &other):
    LmComponent(other) {}

  /// \brief Sets parameters to zero, and if treat_as_gradient is true,
  ///  sets is_gradient_ to true and sets learning_rate_ to 1, ignoring
  ///  learning_rate_factor_.
  virtual void SetZero(bool treat_as_gradient) = 0;

  LmInputComponent(): LmComponent() {}

  virtual ~LmInputComponent() { }

  virtual void Propagate(const SparseMatrix<BaseFloat> &in,
                         MatrixBase<BaseFloat> *out) const = 0;

  virtual void Backprop(
                        const SparseMatrix<BaseFloat> &in_value,
                        const MatrixBase<BaseFloat> &, // out_value
                        const MatrixBase<BaseFloat> &out_deriv,
                        LmComponent *to_update,
                        MatrixBase<BaseFloat> *in_deriv) const = 0;

  virtual string Info() const;

 protected:
  // to be called from child classes, extracts any learning rate information
  // from the config line and sets them appropriately.
  void InitLearningRatesFromConfig(ConfigLine *cfl);

  // To be used in child-class Read() functions, this function reads the opening
  // tag <ThisComponentType> and the learning-rate factor and the learning-rate.
  void ReadUpdatableCommon(std::istream &is, bool binary);

  // To be used in child-class Write() functions, writes the opening
  // <ThisComponentType> tag and the learning-rate factor (if not 1.0) and the
  // learning rate;
  void WriteUpdatableCommon(std::ostream &is, bool binary) const;


 private:
  const LmInputComponent &operator = (const LmInputComponent &other); // Disallow.
};


class LmOutputComponent: public LmComponent {
 public:
  LmOutputComponent(const LmOutputComponent &other):
    LmComponent(other) {}

  /// \brief Sets parameters to zero, and if treat_as_gradient is true,
  ///  sets is_gradient_ to true and sets learning_rate_ to 1, ignoring
  ///  learning_rate_factor_.
  virtual void SetZero(bool treat_as_gradient) = 0;

  LmOutputComponent(): LmComponent() {}

  virtual ~LmOutputComponent() { }

  virtual void Propagate(const MatrixBase<BaseFloat> &in,
                 const vector<vector<int> > &indexes, // objf is computed on the chosen indexes
                 vector<vector<BaseFloat> > *out) const = 0;

  virtual void Backprop(int k,
             const vector<vector<int> > &indexes,
             const MatrixBase<BaseFloat> &in_value,
             const MatrixBase<BaseFloat> &, // out_value
             const vector<vector<BaseFloat> > &out_deriv,
             LmOutputComponent *to_update_in,
             MatrixBase<BaseFloat> *in_deriv) const = 0;

  virtual string Info() const;

 protected:
  // to be called from child classes, extracts any learning rate information
  // from the config line and sets them appropriately.
  void InitLearningRatesFromConfig(ConfigLine *cfl);

  // To be used in child-class Read() functions, this function reads the opening
  // tag <ThisComponentType> and the learning-rate factor and the learning-rate.
  void ReadUpdatableCommon(std::istream &is, bool binary);

  // To be used in child-class Write() functions, writes the opening
  // <ThisComponentType> tag and the learning-rate factor (if not 1.0) and the
  // learning rate;
  void WriteUpdatableCommon(std::ostream &is, bool binary) const;

 private:
  const LmOutputComponent &operator = (const LmOutputComponent &other); // Disallow.
};


} // namespace nnet3
} // namespace kaldi


#endif
