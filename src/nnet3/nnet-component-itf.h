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

#ifndef KALDI_NNET3_NNET_COMPONENT_ITF_H_
#define KALDI_NNET3_NNET_COMPONENT_ITF_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-parse.h"
#include "base/kaldi-error.h"
#include "thread/kaldi-mutex.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

// enum used to store various binary component properties.
// We give it a name ComponentProperties, but don't use this
// type for the bitmasks: instead use int32 for this type, e.g.
// int32 properties = kSimpleComponent|kBackpropNeedsOutput.
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


// This is a base class for a helper-class of class Component, which is used to
// store any pre-computed indexes it needs for its forward and backward
// computations.  For components which are not "Simple" components (i.e. the
// kSimpleComponent property is false), and which may therefore "care" about
// which index the input and output matrix's rows represent (i.e. about
// which "struct Index" each row corresponds to), their CreateIndexes() function
// will be called prior to Propagate() and Backprop(), to create an object which
// must be a child class of class ComponentPrecomputedIndexes, where they
// can store any indexes that they need.
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


class IndexSet;  // Forward declaration; declared in nnet-computation-graph.h.

/// Abstract base-class for neural-net components.
class Component {
 public:
  /// \brief Propagate function.
  ///   \param [in] indexes  A pointer to some information output by this class's
  ///      PrecomputeIndexes function (will be NULL for simple components,
  ///      i.e. those that don't do things like splicing).
  ///   \param [in] in   The input to this component.  Num-columns == InputDim().
  ///   \param [out] out  The output of this component.  Num-columns == OutputDim().
  ///      Note: output of this component will be added to the initial value of
  ///      "out" if Properties()&kPropagateAdds != 0; otherwise the output will
  ///      be set and the initial value ignored.  Each Component chooses whether
  ///      it is more convenient implementation-wise to add or set, and the
  ///      calling code has to deal with it.
  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const = 0;

  /// \brief Backprop function; depending on which of the arguments 'to_update'
  ///     and 'in_deriv' are non-NULL, this can compute input-data derivatives
  ///     and/or perform model update.
  ///
  ///   \param [in] debug_info  The component name, to be printed out in any
  ///       warning messages.
  ///   \param [in] indexes     A pointer to some information output by this
  ///      class's PrecomputeIndexes function (will be NULL for simple
  ///      components, i.e. those that don't do things like splicing).
  ///   \param [in] in_value    The matrix that was given as input to the
  ///      Propagate function.  Will be ignored (and may be empty) if
  ///      Properties()&kBackpropNeedsInput == 0.
  ///   \param [in] out_value   The matrix that was output from the Propagate
  ///      function.  Will be ignored (and may be empty) if
  ///      Properties()&kBackpropNeedsOutput == 0
  ///   \param [in] out_deriv  The derivative at the output of this component.
  ///   \param [out] to_update  If model update is desired, the Component
  ///       to be updated, else NULL.  Does not have to be identical to this.
  ///       If supplied, you can assume that
  ///       to_update->Properties() & kUpdatableComponent is nonzero.
  ///   \param [out] in_deriv   The derivative at the input of this component,
  ///       if needed (else NULL).   If  Properties()&kBackpropInPlace, may be
  ///       the same matrix as out_deriv.  If Properties()&kBackpropAdds, this
  ///       is added to by the Backprop routine, else it is set.  The component
  ///       code chooses which mode to work in, based on convenience.
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update, // may be NULL; may be identical
                                              // to "this" or different.
                        CuMatrixBase<BaseFloat> *in_deriv) const = 0;


  /// \brief This function may store stats on average activation values, and for
  ///        some component types, the average value of the derivative of the
  ///        nonlinearity.  It only does something for those components that
  ///        have nonzero Properties()&kStoresStats.  It only needs as input
  ///        the value at the output of the nonlinearity.

  virtual void StoreStats(const CuMatrixBase<BaseFloat> &out_value) { }

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
  virtual void GetInputIndexes(const MiscComputationInfo &misc_info,
                               const Index &output_index,
                               std::vector<Index> *desired_indexes) const;

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
  virtual bool IsComputable(const MiscComputationInfo &misc_info,
                            const Index &output_index,
                            const IndexSet &input_index_set,
                            std::vector<Index> *used_inputs) const;

  /// \brief This function only does something interesting for non-simple
  ///  Components.  It provides an opportunity for a Component to reorder the
  ///  indexes at its input and output.  This might be useful, for instance, if
  ///  a component requires a particular ordering of the indexes that doesn't
  ///  correspond to their natural ordering.  Components that might modify the
  ///  indexes are brequired to return the kReordersIndexes flag in their
  ///  Properties().
  ///
  ///  \param [in,out]  Indexes at the input of the Component.
  ///  \param [in,out]  Indexes at the output of the Component
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const {}



  /// \brief This function must return NULL for simple Components.  Returns a
  ///     pointer to a class that may contain some precomputed
  ///     component-specific and computation-specific indexes to be in used in
  ///     the Propagate and Backprop functions.
  ///
  /// \param [in] misc_info  This argument is supplied to handle things that the
  ///       framework can't very easily supply: information like which time
  ///       indexes are needed for AggregateComponent, which time-indexes are
  ///       available at the input of a recurrent network, and so on.  misc_info
  ///       may not even ever be used here.  We will add members to misc_info as
  ///       needed.
  /// \param [in] input_indexes  A vector of indexes that explains
  ///       what time-indexes (and other indexes) each row of the
  ///       in/in_value/in_deriv matrices given to Propagate and Backprop will
  ///       mean.
  /// \param [in] output_indexes  A vector of indexes that explains
  ///       what time-indexes (and other indexes) each row of the
  ///       out/out_value/out_deriv matrices given to Propagate and Backprop will
  ///       mean.
  /// \param [in] need_backprop  True if we might need to do backprop
  ///       with this component, so that if any different indexes are needed
  ///       for backprop then those should be computed too.
  /// \return  Returns a child-class of class ComponentPrecomputedIndexes, or
  ///       NULL if this component for does not need to precompute any indexes
  ///       (e.g. if it is a simple component and does not care about indexes).
  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const { return NULL;  }


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
  static Component* ReadNew(std::istream &is, bool binary);

  /// \brief Copies component (deep copy).
  virtual Component* Copy() const = 0;

  /// \brief Returns a new Component of the given type e.g. "SoftmaxComponent",
  ///   or NULL if no such component type exists.
  static Component *NewComponentOfType(const std::string &type);

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
  //    -- an UpdatableComponent scales the parameters
  ///      by "scale" when called by an UpdatableComponent.
  //    -- a Nonlinear component it relates to scaling activation stats, not parameters.
  virtual void Scale(BaseFloat scale) {};

  /// This virtual function when called by
  ///    -- an UpdatableComponent adds the parameters of
  ///      another updatable component, times some constant, to the current
  ///      parameters.
  ///    -- a NonlinearComponent it relates to adding stats
  /// Otherwise it should do nothing.
  virtual void Add(BaseFloat alpha, const Component &other) {};

  Component() { }

  virtual ~Component() { }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(Component);
};


class RandomComponent: public Component {
 public:
  // This function is required in testing code and in other places we need
  // consistency in the random number generation (e.g. when optimizing
  // validation-set performance), but check where else we call srand().  You'll
  // need to call srand as well as making this call.
  void ResetGenerator() { random_generator_.SeedGpu(); }
 protected:
  CuRand<BaseFloat> random_generator_;
};

/**
 * Class UpdatableComponent is a Component which has trainable parameters; it
 * extends the interface of Component.  This is a base-class for Components with
 * parameters.  See comment by declaration of kUpdatableComponent.
 * The functions in this interface must only be called if the component returns
 * the kUpdatable flag.
 */
class UpdatableComponent: public Component {
 public:
  UpdatableComponent(const UpdatableComponent &other):
      learning_rate_(other.learning_rate_),
      learning_rate_factor_(other.learning_rate_factor_),
      is_gradient_(other.is_gradient_), max_change_(other.max_change_) { }

  /// \brief Sets parameters to zero, and if treat_as_gradient is true,
  ///  sets is_gradient_ to true and sets learning_rate_ to 1, ignoring
  ///  learning_rate_factor_.
  virtual void SetZero(bool treat_as_gradient) = 0;

  UpdatableComponent(): learning_rate_(0.001), learning_rate_factor_(1.0),
                        is_gradient_(false), max_change_(0.0) { }

  virtual ~UpdatableComponent() { }

  /// \brief Computes dot-product between parameters of two instances of a
  ///  Component.  Can be used for computing parameter-norm of an
  ///  UpdatableComponent.
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const = 0;

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

  /// Gets per-component max-change value. Note: the components themselves do
  /// not enforce the per-component max-change; it's enforced in class
  /// NnetTrainer by querying the max-changes for each component.
  /// See NnetTrainer::UpdateParamsWithMaxChange() in nnet3/nnet-training.cc.
  BaseFloat MaxChange() const { return max_change_; }

  virtual std::string Info() const;

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

  BaseFloat learning_rate_; ///< learning rate (typically 0.0..0.01)
  BaseFloat learning_rate_factor_; ///< learning rate factor (normally 1.0, but
                                   ///< can be set to another < value so that
                                   ///when < you call SetLearningRate(), that
                                   ///value will be scaled by this factor.
  bool is_gradient_;  ///< True if this component is to be treated as a gradient rather
                      ///< than as parameters.  Its main effect is that we disable
                      ///< any natural-gradient update and just compute the standard
                      ///< gradient.
  BaseFloat max_change_; ///< configuration value for imposing max-change

 private:
  const UpdatableComponent &operator = (const UpdatableComponent &other); // Disallow.
};

/// This kind of Component is a base-class for things like sigmoid, softmax and
/// ReLU: nonlinearities that don't change the dimension.  It takes care of
/// storing statistics on the average activations and derivatives encountered
/// during training.
class NonlinearComponent: public Component {
 public:

  NonlinearComponent();
  explicit NonlinearComponent(const NonlinearComponent &other);

  virtual int32 InputDim() const { return dim_; }
  virtual int32 OutputDim() const { return dim_; }

  // We implement InitFromConfig at this level.
  // supported config parameters and their defaults:
  //   dim=-1  self-repair-lower-threshold=-1000  self-repair-upper-threshold=-1000
  //     self-repair-constant=0.0
  // the 'self-repair' stuff is 'self-repairing' nonlinearities-- they add small
  // quantities to the derivative to attempt to keep the average value (for
  // bounded nonlinearities) or average derivative (for ReLU) for each
  // dimension within a given range.  The default ranges (if you don't
  // specify self-repair-lower-threshold or self-repair-upper-threshold) are
  // dependent on the nonlinearity and are set in their Backprop functions.
  // To activate this code you have to set self-repair-constant to a number >0 like
  // 0.0001 when initializing the ReLU (this is a scaling factor on the 'fake
  // derivative').  This code is only activated if derivative and value stats
  // are present in the model, which will typically only be the case
  // if the 'store-stats' code is activated
  // (e.g. --optimization.store-stats=true) because it needs the stats.  To be
  // activated this code also requires that is_gradient_ is false (i.e. you're
  // not computing exact gradients).

  virtual void InitFromConfig(ConfigLine *cfl);

  /// We implement Read at this level as it just needs the Type().
  virtual void Read(std::istream &is, bool binary);

  virtual void ZeroStats();

  virtual std::string Info() const;

  /// Write component to stream.
  virtual void Write(std::ostream &os, bool binary) const;

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);

  // The following functions are unique to NonlinearComponent.
  // They mostly relate to diagnostics.
  const CuVector<double> &ValueSum() const { return value_sum_; }
  const CuVector<double> &DerivSum() const { return deriv_sum_; }

  double Count() const { return count_; }

 protected:
  enum { kUnsetThreshold = -1000 };

  friend class SigmoidComponent;
  friend class TanhComponent;
  friend class SoftmaxComponent;
  friend class LogSoftmaxComponent;
  friend class RectifiedLinearComponent;

  // This function updates the stats "value_sum_", "deriv_sum_", and
  // count_. (If deriv == NULL, it won't update "deriv_sum_").
  // It will be called from the Backprop function of child classes.
  void StoreStatsInternal(const CuMatrixBase<BaseFloat> &out_value,
                          const CuMatrixBase<BaseFloat> *deriv = NULL);


  const NonlinearComponent &operator = (const NonlinearComponent &other); // Disallow.
  int32 dim_;
  CuVector<double> value_sum_; // stats at the output.
  CuVector<double> deriv_sum_; // stats of the derivative of the nonlinearity
                               // (only applicable to element-by-element
                               // nonlinearities, not Softmax.
  double count_;

  // some stats for self-repairing nonlinearities.
  double num_dims_self_repaired_;
  double num_dims_processed_;

  // some configuration values relating to self-repairing nonlinearities.
  BaseFloat self_repair_lower_threshold_;
  BaseFloat self_repair_upper_threshold_;
  BaseFloat self_repair_scale_;

  // The mutex is used in UpdateStats, only for resizing vectors.
  Mutex mutex_;
};

} // namespace nnet3
} // namespace kaldi


#endif
