// nnet3/nnet-combined-component.h

// Copyright      2018  Johns Hopkins University (author: Daniel Povey)
//                2018  Hang Lyu

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

#ifndef KALDI_NNET3_NNET_SPECIAL_COMPONENT_H_
#define KALDI_NNET3_NNET_SPECIAL_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  nnet-combined-component.h
///   You can view this as an overflow from nnet-simple-component.h.
///   It contains components which meet the definition of "simple"
///   components, i.e. they set the kSimpleComponent flag, but
///   which are more special-purpose, i.e. they are specific to
///   special layer types such as LSTMs, CNNs and GRUs.



/**
 * WARNING, this component is deprecated in favor of
 *  TimeHeightConvolutionComponent, and will be deleted.
 * ConvolutionalComponent implements 2d-convolution.
 * It uses 3D filters on 3D inputs, but the 3D filters hop only over
 * 2 dimensions as it has same size as the input along the 3rd dimension.
 * Input : A matrix where each row is a  vectorized 3D-tensor.
 *        The 3D tensor has dimensions
 *        x: (e.g. time)
 *        y: (e.g. frequency)
 *        z: (e.g. channels like features/delta/delta-delta)
 *
 *        The component supports input vectorizations of type zyx and yzx.
 *        The default vectorization type is zyx.
 *        e.g. for input vectorization of type zyx the input is vectorized by
 *        spanning axes z, y and x of the tensor in that order.
 *        Given 3d tensor A with sizes (2, 2, 2) along the three dimensions
 *        the zyx vectorized input looks like
 *  A(0,0,0) A(0,0,1) A(0,1,0) A(0,1,1) A(1,0,0) A(1,0,1) A(1,1,0) A(1,1,1)
 *
 *
 * Output : The output is also a 3D tensor vectorized in the zyx format.
 *          The channel axis (z) in the output corresponds to the output of
 *          different filters. The first channel corresponds to the first filter
 *          i.e., first row of the filter_params_ matrix.
 *
 * Note: The component has to support yzx input vectorization as the binaries
 * like add-deltas generate yz vectorized output. These input vectors are
 * concatenated using the Append descriptor across time steps to form a yzx
 * vectorized 3D tensor input.
 * e.g. Append(Offset(input, -1), input, Offset(input, 1))
 *
 *
 * For information on the hyperparameters and parameters of this component see
 * the variable declarations.
 *
 * Propagation:
 * ------------
 * Convolution operation consists of a dot-products between the filter tensor
 * and input tensor patch, for various shifts of filter tensor along the x and y
 * axes input tensor. (Note: there is no shift along z-axis as the filter and
 * input tensor have same size along this axis).
 *
 * For a particular shift (i,j) of the filter tensor
 * along input tensor dimensions x and y, the elements of the input tensor which
 * overlap with the filter form the input tensor patch. This patch is vectorized
 * in zyx format. All the patches corresponding to various samples in the
 * mini-batch are stacked into a matrix, where each row corresponds to one
 * patch. Let this matrix be represented by X_{i,j}. The dot products with
 * various filters are computed simultaneously by computing the matrix product
 * with the filter_params_ matrix (W)
 * Y_{i,j} = X_{i,j}*W^T.
 * Each row of W corresponds to one filter 3D tensor vectorized in zyx format.
 *
 * All the matrix products corresponding to various shifts (i,j) of the
 * filter tensor are computed simultaneously using the AddMatMatBatched
 * call of CuMatrixBase class.
 *
 * BackPropagation:
 * ----------------
 *  Backpropagation to compute the input derivative (\nabla X_{i,j})
 *  consists of the a series of matrix products.
 *  \nablaX_{i,j} = \nablaY_{i,j}*W where \nablaY_{i,j} corresponds to the
 *   output derivative for a particular shift of the filter.
 *
 *   Once again these matrix products are computed simultaneously.
 *
 * Update:
 * -------
 *  The weight gradient is computed as
 *  \nablaW = \Sum_{i,j} (X_{i,j}^T *\nablaY_{i,j})
 *
 */
class ConvolutionComponent: public UpdatableComponent {
 public:
  enum TensorVectorizationType  {
    kYzx = 0,
    kZyx = 1
  };

  ConvolutionComponent();
  // constructor using another component
  ConvolutionComponent(const ConvolutionComponent &component);
  // constructor using parameters
  ConvolutionComponent(
    const CuMatrixBase<BaseFloat> &filter_params,
    const CuVectorBase<BaseFloat> &bias_params,
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 filt_x_dim, int32 filt_y_dim,
    int32 filt_x_step, int32 filt_y_step,
    TensorVectorizationType input_vectorization,
    BaseFloat learning_rate);

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "ConvolutionComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kBackpropNeedsInput|
           kBackpropAdds|kPropagateAdds;
  }

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update_in,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
  void Update(const std::string &debug_info,
              const CuMatrixBase<BaseFloat> &in_value,
              const CuMatrixBase<BaseFloat> &out_deriv,
              const std::vector<CuSubMatrix<BaseFloat> *>& out_deriv_batch);


  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.
  void SetParams(const VectorBase<BaseFloat> &bias,
                 const MatrixBase<BaseFloat> &filter);
  const CuVector<BaseFloat> &BiasParams() const { return bias_params_; }
  const CuMatrix<BaseFloat> &LinearParams() const { return filter_params_; }
  void Init(int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
            int32 filt_x_dim, int32 filt_y_dim,
            int32 filt_x_step, int32 filt_y_step, int32 num_filters,
            TensorVectorizationType input_vectorization,
            BaseFloat param_stddev, BaseFloat bias_stddev);
  // there is no filt_z_dim parameter as the length of the filter along
  // z-dimension is same as the input
  void Init(int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
            int32 filt_x_dim, int32 filt_y_dim,
            int32 filt_x_step, int32 filt_y_step,
            TensorVectorizationType input_vectorization,
            std::string matrix_filename);

  // resize the component, setting the parameters to zero, while
  // leaving any other configuration values the same
  void Resize(int32 input_dim, int32 output_dim);

  void Update(const std::string &debug_info,
              const CuMatrixBase<BaseFloat> &in_value,
              const CuMatrixBase<BaseFloat> &out_deriv);


 private:
  int32 input_x_dim_;   // size of the input along x-axis
                        // (e.g. number of time steps)

  int32 input_y_dim_;   // size of input along y-axis
                        // (e.g. number of mel-frequency bins)

  int32 input_z_dim_;   // size of input along z-axis
                        // (e.g. number of channels is 3 if the input has
                        // features + delta + delta-delta features

  int32 filt_x_dim_;    // size of the filter along x-axis

  int32 filt_y_dim_;    // size of the filter along y-axis

  // there is no filt_z_dim_ as it is always assumed to be
  // the same as input_z_dim_

  int32 filt_x_step_;   // the number of steps taken along x-axis of input
                        //  before computing the next dot-product
                        //  of filter and input

  int32 filt_y_step_;   // the number of steps taken along y-axis of input
                        // before computing the next dot-product of the filter
                        // and input

  // there is no filt_z_step_ as only dot product is possible along this axis

  TensorVectorizationType input_vectorization_; // type of vectorization of the
  // input 3D tensor. Accepts zyx and yzx formats

  CuMatrix<BaseFloat> filter_params_;
  // the filter (or kernel) matrix is a matrix of vectorized 3D filters
  // where each row in the matrix corresponds to one filter.
  // The 3D filter tensor is vectorizedin zyx format.
  // The first row of the matrix corresponds to the first filter and so on.
  // Keep in mind the vectorization type and order of filters when using file
  // based initialization.

  CuVector<BaseFloat> bias_params_;
  // the filter-specific bias vector (i.e., there is a seperate bias added
  // to the output of each filter).

  void InputToInputPatches(const CuMatrixBase<BaseFloat>& in,
                           CuMatrix<BaseFloat> *patches) const;
  void InderivPatchesToInderiv(const CuMatrix<BaseFloat>& in_deriv_patches,
                               CuMatrixBase<BaseFloat> *in_deriv) const;
  const ConvolutionComponent &operator = (const ConvolutionComponent &other); // Disallow.
};


/*
  LstmNonlinearityComponent is a component that implements part of an LSTM, by
  combining together the sigmoids and tanh's, plus some diagonal terms, into
  a single block.
  We will refer to the LSTM formulation used in

  Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling"
  by H. Sak et al,
  http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf.

  Suppose the cell dimension is C.  Then outside this component, we compute
  the 4 * C-dimensional quantity consisting of 4 blocks as follows, by a single
  matrix multiplication:

  i_part = W_{ix} x_t + W_{im} m_{t-1} + b_i
  f_part = W_{fx} x_t + W_{fm} m_{t-1} + b_f
  c_part = W_{cx} x_t + W_{cm} m_{t-1} + b_c
  o_part = W_{cx} x_t + W_{om} m_{t-1} + b_o

  The part of the computation that takes place in this component is as follows.
  Its input is of dimension 5C [however, search for 'dropout' below],
  consisting of 5 blocks: (i_part, f_part, c_part, o_part, and c_{t-1}).  Its
  output is of dimension 2C, consisting of 2 blocks: c_t and m_t.

  To recap: the input is (i_part, f_part, c_part, o_part, c_{t-1}); the output is (c_t, m_t).

  This component has parameters, 3C of them in total: the diagonal matrices w_i, w_f
  and w_o.


  In the forward pass (Propagate), this component computes the following:

     i_t = Sigmoid(i_part + w_{ic}*c_{t-1})   (1)
     f_t = Sigmoid(f_part + w_{fc}*c_{t-1})   (2)
     c_t = f_t*c_{t-1} + i_t * Tanh(c_part)   (3)
     o_t = Sigmoid(o_part + w_{oc}*c_t)       (4)
     m_t = o_t * Tanh(c_t)                    (5)
    # note: the outputs are just c_t and m_t.

  [Note regarding dropout: optionally the input-dimension may be 5C + 3 instead
  of 5C in this case, the last three input dimensions will be interpreted as
  per-frame dropout masks on i_t, f_t and o_t respectively, so that on the RHS of
  (3), i_t is replaced by i_t * i_t_scale, and likewise for f_t and o_t.]

  The backprop is as you would think, but for the "self-repair" we need to pass
  in additional vectors (of the same dim as the parameters of the layer) that
  dictate whether or not we add an additional term to the backpropagated
  derivatives.  (This term helps force the input to the nonlinearities into the
  range where the derivatives are not too small).

  This component stores stats of the same form as are normally stored by the
  StoreStats() functions for the sigmoid and tanh units, i.e. averages of the
  activations and derivatives, but this is done inside the Backprop() functions.
  [the StoreStats() functions don't take the input data as an argument, so
  storing this data that way is impossible, and anyway it's more efficient to
  do it as part of backprop.]

  Configuration values accepted:
         cell-dim          e.g. cell-dim=1024  Cell dimension.  The input
                          dimension of this component is cell-dim * 5, and the
                          output dimension is cell-dim * 2.  Note: this
                          component implements only part of the LSTM layer,
                          see comments above.
         param-stddev     Standard deviation for random initialization of
                          the diagonal matrices (AKA peephole connections).
                          default=1.0, which is probably too high but
                          we couldn't see any reliable gain from decreasing it.
         tanh-self-repair-threshold   Equivalent to the self-repair-lower-threshold
                          in a TanhComponent; applies to both the tanh nonlinearities.
                          default=0.2, you probably won't want to changethis.
         sigmoid-self-repair-threshold   Equivalent to self-repair-lower-threshold
                          in a SigmoidComponent; applies to all three of the sigmoid
                          nonlinearities.  default=0.05, you probably won't want to
                          change this.
         self-repair-scale Equivalent to the self-repair-scale in a SigmoidComponent
                          or TanhComponent; applies to both the sigmoid and tanh
                          nonlinearities.  default=1.0e-05, which you probably won't
                          want to change unless dealing with an objective function
                          that has smaller or larger dynamic range than normal, in
                          which case you might want to make it smaller or larger.
*/
class LstmNonlinearityComponent: public UpdatableComponent {
 public:

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;
  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  LstmNonlinearityComponent(): use_dropout_(false) { }
  virtual std::string Type() const { return "LstmNonlinearityComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kBackpropNeedsInput;
  }

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update_in,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;

  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
  virtual void ZeroStats();
  virtual void FreezeNaturalGradient(bool freeze);

  // Some functions that are specific to this class:
  explicit LstmNonlinearityComponent(
      const LstmNonlinearityComponent &other);

  void Init(int32 cell_dim, bool use_dropout,
            BaseFloat param_stddev,
            BaseFloat tanh_self_repair_threshold,
            BaseFloat sigmoid_self_repair_threshold,
            BaseFloat self_repair_scale);

  virtual void ConsolidateMemory();

 private:

  // Initializes the natural-gradient object with the configuration we
  // use for this object, which for now is hardcoded at the C++ level.
  void InitNaturalGradient();

  // Notation: C is the cell dimension; it equals params_.NumCols().

  // The dimension of the parameter matrix is (3 x C);
  // it contains the 3 diagonal parameter matrices w_i, w_f and w_o.
  CuMatrix<BaseFloat> params_;

  // If true, we expect an extra 3 dimensions on the input, for dropout masks
  // for i_t and f_t.
  bool use_dropout_;

  // Of dimension 5 * C, with a row for each of the Sigmoid/Tanh functions in
  // equations (1) through (5), this is the sum of the values of the nonliearities
  // (used for diagnostics only).  It is comparable to value_sum_ vector
  // in base-class NonlinearComponent.
  CuMatrix<double> value_sum_;

  // Of dimension 5 * C, with a row for each of the Sigmoid/Tanh functions in
  // equations (1) through (5), this is the sum of the derivatives of the
  // nonliearities (used for diagnostics and to control self-repair).  It is
  // comparable to the deriv_sum_ vector in base-class
  // NonlinearComponent.
  CuMatrix<double> deriv_sum_;

  // This matrix has dimension 10.  The contents are a block of 5 self-repair
  // thresholds (typically "0.05 0.05 0.2 0.05 0.2"), then a block of 5
  // self-repair scales (typically all 0.00001).  These are for each of the 5
  // nonlinearities in the LSTM component in turn (see comments in cu-math.h for
  // more info).
  CuVector<BaseFloat> self_repair_config_;

  // This matrix has dimension 5.  For each of the 5 nonlinearities in the LSTM
  // component (see comments in cu-math.h for more info), it contains the total,
  // over all frames represented in count_, of the number of dimensions that
  // were subject to self_repair.  To get the self-repair proportion you should
  // divide by (count_ times cell_dim_).
  CuVector<double> self_repair_total_;

  // The total count (number of frames) corresponding to the stats in value_sum_
  // and deriv_sum_.
  double count_;

  // Preconditioner for the parameters of this component [operates in the space
  // of dimension C].
  // The preconditioner stores its own configuration values; we write and read
  // these, but not the preconditioner object itself.
  OnlineNaturalGradient preconditioner_;

  const LstmNonlinearityComponent &operator
      = (const LstmNonlinearityComponent &other); // Disallow.
};




/*
 * WARNING, this component is deprecated as it's not compatible with
 *   TimeHeightConvolutionComponent, and it will eventually be deleted.
 * MaxPoolingComponent :
 * Maxpooling component was firstly used in ConvNet for selecting an
 * representative activation in an area. It inspired Maxout nonlinearity.
 * Each output element of this component is the maximum of a block of
 * input elements where the block has a 3D dimension (pool_x_size_,
 * pool_y_size_, pool_z_size_).
 * Blocks could overlap if the shift value on any axis is smaller
 * than its corresponding pool size (e.g. pool_x_step_ < pool_x_size_).
 * If the shift values are euqal to their pool size, there is no
 * overlap; while if they all equal 1, the blocks overlap to
 * the greatest possible extent.
 *
 * This component is designed to be used after a ConvolutionComponent
 * so that the input matrix is propagated from a 2d-convolutional layer.
 * This component implements 3d-maxpooling which performs
 * max pooling along the three axes.
 * Input : A matrix where each row is a vectorized 3D-tensor.
 *        The 3D tensor has dimensions
 *        x: (e.g. time)
 *        y: (e.g. frequency)
 *        z: (e.g. channels like number of filters in the ConvolutionComponent)
 *
 *        The component assumes input vectorizations of type zyx
 *        which is the default output vectorization type of a ConvolutionComponent.
 *        e.g. for input vectorization of type zyx the input is vectorized by
 *        spanning axes z, y and x of the tensor in that order.
 *        Given 3d tensor A with sizes (2, 2, 2) along the three dimensions
 *        the zyx vectorized input looks like
 *  A(0,0,0) A(0,0,1) A(0,1,0) A(0,1,1) A(1,0,0) A(1,0,1) A(1,1,0) A(1,1,1)
 *
 * Output : The output is also a 3D tensor vectorized in the zyx format.
 *
 * For information on the hyperparameters and parameters of this component see
 * the variable declarations.
 *
 *
 */
class MaxpoolingComponent: public Component {
 public:

  MaxpoolingComponent(): input_x_dim_(0), input_y_dim_(0), input_z_dim_(0),
                           pool_x_size_(0), pool_y_size_(0), pool_z_size_(0),
                           pool_x_step_(0), pool_y_step_(0), pool_z_step_(0) { }
  // constructor using another component
  MaxpoolingComponent(const MaxpoolingComponent &component);

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual std::string Type() const { return "MaxpoolingComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kBackpropNeedsInput|kBackpropNeedsOutput|
           kBackpropAdds;
  }

  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *, // to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary); // This Read function
  // requires that the Component has the correct type.

  /// Write component to stream
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Component* Copy() const { return new MaxpoolingComponent(*this); }


 protected:
  void InputToInputPatches(const CuMatrixBase<BaseFloat>& in,
                           CuMatrix<BaseFloat> *patches) const;
  void InderivPatchesToInderiv(const CuMatrix<BaseFloat>& in_deriv_patches,
                               CuMatrixBase<BaseFloat> *in_deriv) const;
  virtual void Check() const;


  int32 input_x_dim_;   // size of the input along x-axis
  // (e.g. number of time steps)
  int32 input_y_dim_;   // size of input along y-axis
  // (e.g. number of mel-frequency bins)
  int32 input_z_dim_;   // size of input along z-axis
  // (e.g. number of filters in the ConvolutionComponent)

  int32 pool_x_size_;    // size of the pooling window along x-axis
  int32 pool_y_size_;    // size of the pooling window along y-axis
  int32 pool_z_size_;    // size of the pooling window along z-axis

  int32 pool_x_step_;   // the number of steps taken along x-axis of input
  //  before computing the next pool
  int32 pool_y_step_;   // the number of steps taken along y-axis of input
  // before computing the next pool
  int32 pool_z_step_;   // the number of steps taken along z-axis of input
  // before computing the next pool

};


/**
  GruNonlinearityComponent is a component that implements part of a
  Gated Recurrent Unit (GRU).  This is more efficient in time and
  memory than stitching it together using more basic components.
  For a brief summary of what this actually computes, search
  for 'recap' below; the first part of this comment establishes
  the context.

  This component supports two cases: the regular GRU
 (as described in "Empirical Evaluation of
 Gated Recurrent Neural Networks on Sequence Modeling",
 https://arxiv.org/pdf/1412.3555.pdf),
  and our "projected GRU" which takes ideas from the
 paper we'll abbreviate as "LSTM based RNN architectures for LVCSR",
 https://arxiv.org/pdf/1402.1128.pdf.

 Before describing what this component does, we'll establish
 some notation for the GRU.

 First, the regular (non-projected) GRU.  In order to unify the notation with
 our "projected GRU", we'll use slightly different variable names.  We'll also
 ignore the bias terms for purposes of this exposition (let them be implicit).


  Regular GRU:

   z_t = \sigmoid ( U^z x_t + W^z y_{t-1} )   # update gate, dim == cell_dim
   r_t = \sigmoid ( U^r x_t + W^r y_{t-1} )   # reset gate, dim == cell_dim
   h_t = \tanh ( U^h x_t + W^h ( y_{t-1} \dot r_t ) )   # dim == cell_dim
   y_t = ( 1 - z_t ) \dot h_t  +  z_t \dot y_{t-1}  # dim == cell_dim

 For the "projected GRU", the 'cell_dim x cell_dim' full-matrix expressions W^z
 W^r and W^h that participate in the expressions for z_t, r_t and h_t are
 replaced with skinny matrices of dimension 'cell_dim x recurrent_dim'
 (where recurrent_dim < cell_dim) and the output is replaced by
 a lower-dimension projection of the hidden state, of dimension
 'recurrent_dim + non_recurrent_dim < cell_dim', instead of the
 full 'cell_dim'.  We rename y_t to c_t (this name is inspired by LSTMs), and
 we now let the output (still called y_t) be a projection of c_t.
 s_t is a dimension range of the output y_t.    Parameters of the
 projected GRU:
           cell_dim > 0
           recurrent_dim > 0
           non_recurrent_dim > 0  (where non_recurrent_dim + recurrent_dim < cell_dim).


  Equations:

   z_t = \sigmoid ( U^z x_t + W^z s_{t-1} )   # update gate, dim(z_t) == cell_dim
   r_t = \sigmoid ( U^r x_t + W^r s_{t-1} )   # reset gate, dim(r_t) == recurrent_dim
   h_t = \tanh ( U^h x_t + W^h ( s_{t-1} \dot r_t ) )   # dim(h_t) == cell_dim
   c_t = ( 1 - z_t ) \dot h_t  +  z_t \dot c_{t-1}  # dim(c_t) == cell_dim
   y_t = W^y c_t      # dim(y_t) = recurrent_dim + non_recurrent_dim.  This is
                      # the output of the GRU.
   s_t = y_t[0:recurrent_dim-1]  # dimension range of y_t, dim(s_t) = recurrent_dim.


   Because we'll need it below, we define
    hpart_t = U^h x_t
   which is a subexpression of h_t.

   Our choice to make a "special" component for the projected GRU is to have
   it be a function from
     (z_t, r_t, hpart_t, c_{t-1}, s_{t-1}) -> (h_t, c_t)
   That is, the input to the component is all those things on the LHS
   appended together, and the output is the two things on the
   RHS appended together.  The dimensions are:
    (cell_dim, recurrent_dim, cell_dim, cell_dim, recurrent_dim) -> (cell_dim, cell_dim).
   The component computes the functions:
     h_t = \tanh( hpart_t + W^h (s_{t-1} \dot r_t))
     c_t = (1 - z_t) \dot h_t + z_t \dot c_{t-1}.

   Notice that 'W^h' is the only parameter that lives inside the component.

   You might also notice that the output 'h_t' is never actually used
   in any other part of the GRU, so the question arises: why is it
   necessary to have it be an output of the component?  This has to do with
   saving computation: because h_t is an output, and we'll be defining
   the kBackpropNeedsOutput flag, it is available in the backprop phase
   and this helps us avoid some computation (otherwise we'd have to do
   a redundant multiplication by W^h in the backprop phase that we already
   did in the forward phase).  We could have used the 'memo' mechanism to
   do this, but this is undesirable because the use of a memo disables
   'update consolidation' in the backprop so we'd lose a little
   speed there.

   In the case where it's a regular, not projected GRU, this component
   is a function from
      (z_t, r_t, hpart_t, y_{t-1}) -> (h_t, y_t)
   We can actually do this with the same code as the projected-GRU code,
   we just make sure that recurrent_dim == cell_dim, and the only structural
   difference is that c_{t-1} and s_{t-1} become the same variable (y_{t-1}),
   and we rename c_t to y_t.

   This component stores stats of the same form as are normally stored by the
   StoreStats() functions for the sigmoid and tanh units, i.e. averages of the
   activations and derivatives, but this is done inside the Backprop() functions.


  The main configuration values that are accepted:
         cell-dim         e.g. cell-dim=1024  Cell dimension.
         recurrent-dim    e.g. recurrent-dim=256.  If not specified, we assume
                          this is a non-projected GRU.
         param-stddev     Standard deviation for random initialization of
                          the matrix W^h.  Defaults to 1.0 / sqrt(d) where
                          d is recurrent-dim if specified, else cell-dim.
         self-repair-threshold   Equivalent to the self-repair-lower-threshold
                          in a TanhComponent; applies to the tanh nonlinearity.
                          default=0.2, you probably won't want to change this.
         self-repair-scale Equivalent to the self-repair-scale in a
                          TanhComponent; applies to the tanh nonlinearity.
                          default=1.0e-05, which you probably won't want to
                          change unless dealing with an objective function that
                          has smaller or larger dynamic range than normal, in
                          which case you might want to make it smaller or
                          larger.

  Values inherited from UpdatableComponent (see its declaration in
  nnet-component-itf.h for details):
      learning-rate
      learning-rate-factor
      max-change

   Natural-gradient related options are below; you won't normally have to
   set these.
      alpha                 Constant that determines how much we smooth the
                            Fisher-matrix estimates with the unit matrix.
                            Larger means more smoothing. default=4.0
      rank-in               Rank used in low-rank-plus-unit estimate of Fisher
                            matrix in the input space.  default=20.
      rank-out              Rank used in low-rank-plus-unit estimate of Fisher
                            matrix in the output-derivative space.  default=80.
      update-period         Determines the period (in minibatches) with which
                            we update the Fisher-matrix estimates;
                            making this > 1 saves a little time in training.
                            default=4.


   Recap of what this computes:
      If recurrent-dim is specified, this component implements
      the function
           (z_t, r_t, hpart_t, c_{t-1}, s_{t-1}) -> (h_t, c_t)
     of dims:
   (cell_dim, recurrent_dim, cell_dim, cell_dim, recurrent_dim) -> (cell_dim, cell_dim).
    where:
         h_t = \tanh( hpart_t + W^h (s_{t-1} \dot r_t))
         c_t = (1 - z_t) \dot h_t + z_t \dot c_{t-1}.
     If recurrent-dim is not specified, this component implements
     the function
        (z_t, r_t, hpart_t, y_{t-1}) -> (h_t, y_t)
   of dimensions
       (cell_dim, cell_dim, cell_dim, cell_dim) -> (cell_dim, cell_dim),
    where:
         h_t = \tanh( hpart_t + W^h (y_{t-1} \dot r_t))
         y_t = (1 - z_t) \dot h_t + z_t \dot y_{t-1}.
*/
class GruNonlinearityComponent: public UpdatableComponent {
 public:

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;
  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  GruNonlinearityComponent() { }
  virtual std::string Type() const { return "GruNonlinearityComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kBackpropNeedsInput|\
        kBackpropNeedsOutput|kBackpropAdds;
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update_in,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const { return new GruNonlinearityComponent(*this); }

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);

  // Some functions from base-class UpdatableComponent.
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
  virtual void ZeroStats();
  virtual void FreezeNaturalGradient(bool freeze);

  // Some functions that are specific to this class:
  explicit GruNonlinearityComponent(
      const GruNonlinearityComponent &other);

 private:

  void Check() const;  // checks dimensions, etc.

  /**
     This function stores value and derivative stats for the tanh
     nonlinearity that is a part of this component, and if needed
     adds the small 'self-repair' term to 'h_t_deriv'.
      @param [in] h_t The output of the tanh expression from the
                      forward pass.
      @param [in,out] h_t_deriv  To here will be added the small
                      self-repair term (this is a small value
                      that we use to push oversaturated neurons
                      back to the center).
     This function has side effects on the class instance, specifically the
     members value_sum_, deriv_sum, self_repair_total_, and count_.
   */
  void TanhStatsAndSelfRepair(const CuMatrixBase<BaseFloat> &h_t,
                              CuMatrixBase<BaseFloat> *h_t_deriv);

  /*  This function is responsible for updating the w_h_ matrix
      (taking into account the learning rate).
        @param [in] sdotr  The value of the expression (s_{t-1} \dot r_t).
        @param [in] h_t_deriv  The derivative of the objective
                        function w.r.t. the argument of the tanh
                        function, i.e. w.r.t. the expression
                        "hpart_t + W^h (s_{t-1} \dot r_t)".
                        This function is concerned with the second
                        term as it affects the derivative w.r.t. W^h.
   */
  void UpdateParameters(const CuMatrixBase<BaseFloat> &sdotr,
                        const CuMatrixBase<BaseFloat> &h_t_deriv);


  int32 cell_dim_;  // cell dimension, e.g. 1024.
  int32 recurrent_dim_;  // recurrent dimension, e.g. 256 for projected GRU;
                         // if it's the same as cell_dim it means we are
                         // implementing regular (non-projected) GRU


  // The matrix W^h, of dimension cell_dim_ by recurrent_dim_.
  // There is no bias term needed here because hpart_t comes from
  // an affine component that has a bias.
  CuMatrix<BaseFloat> w_h_;

  // Of dimension cell_dim_, this is comparable to the value_sum_ vector in
  // class NonlinearComponent.  It stores the sum of the tanh nonlinearity.
  // Normalize by dividing by count_.
  CuVector<double> value_sum_;

  // Of dimension cell_dim_, this is comparable to the deriv_sum_ vector in
  // class NonlinearComponent.  It stores the sum of the function-derivative of
  // the tanh nonlinearity.  Normalize by dividing by count_.
  CuVector<double> deriv_sum_;

  // This is part of the stats (along with value_sum_, deriv_sum_, and count_);
  // if you divide it by count_ it gives you the proportion of the time that an
  // average dimension was subject to self-repair.
  double self_repair_total_;

  // The total count (number of frames) corresponding to the stats in value_sum_,
  // deriv_sum_, and self_repair_total_.
  double count_;

  // A configuration parameter, this determines how saturated the derivative
  // has to be for a particular dimension, before we activate self-repair.
  // Default value is 0.2, the same as for TanhComponent.
  BaseFloat self_repair_threshold_;

  // A configuration parameter, this determines the maximum absolute value of
  // the extra term that we add to the input derivative of the tanh when doing
  // self repair.  The default value is 1.0e-05.
  BaseFloat self_repair_scale_;

  // Preconditioner for the input space when updating w_h_ (has dimension
  // recurrent_dim_ if use-natural-gradient was true, else not set up).
  // The preconditioner stores its own configuration values; we write and read
  // these, but not the preconditioner object itself.
  OnlineNaturalGradient preconditioner_in_;
  // Preconditioner for the output space when updating w_h_ (has dimension
  // recurrent_dim_ if use-natural-gradient was true, else not set up).

  OnlineNaturalGradient preconditioner_out_;

  const GruNonlinearityComponent &operator
      = (const GruNonlinearityComponent &other); // Disallow.
};


/**
  OutputGruNonlinearityComponent is a component that implements part of a
  Output Gated Recurrent Unit (OGRU).  Compare with the traditional GRU, it uses
  output gate instead reset gate, and the formula of h_t will be different. 
  You can regard it as a variant of GRU.
  This code is more efficient in time and memory than stitching it together
  using more basic components.
  For a brief summary of what this actually computes, search for 'recap' below;
  the first part of this comment establishes the context. For more information
  about GRU, please check the summary of GruNonlinearityComponent.

 Before describing what this component does, we'll establish
 some notation for the OGRU.

 We use the same notation with previous GRU. We'll also
 ignore the bias terms for purposes of this exposition (let them be implicit).


  Regular OGRU:

   z_t = \sigmoid ( U^z x_t + W^z y_{t-1} )   # update gate, dim == cell_dim
   o_t = \sigmoid ( U^o x_t + W^o y_{t-1} )   # output gate, dim == cell_dim
   h_t = \tanh ( U^h x_t + W^h \dot c_{t-1} )   # dim == cell_dim
   c_t = ( 1 - z_t ) \dot h_t  +  z_t \dot c_{t-1}  # dim == cell_dim
   y_t = ( c_t \dot o_t )

 For the "projected OGRU", the 'cell_dim x cell_dim' full-matrix expressions W^z
 W^o that participate in the expressions for z_t, o_t are
 replaced with skinny matrices of dimension 'cell_dim x recurrent_dim'
 (where recurrent_dim < cell_dim) and the output is replaced by
 a lower-dimension projection of the hidden state, of dimension
 'recurrent_dim + non_recurrent_dim < cell_dim', instead of the
 full 'cell_dim'.
 s_t is a dimension range of the output y_t.    Parameters of the
 projected OGRU:
           cell_dim > 0
           recurrent_dim > 0
           non_recurrent_dim > 0  (where non_recurrent_dim + recurrent_dim <= cell_dim).


  Equations:

   z_t = \sigmoid ( U^z x_t + W^z s_{t-1} )   # update gate, dim(z_t) == cell_dim
   o_t = \sigmoid ( U^o x_t + W^o s_{t-1} )   # output gate, dim(o_t) == cell_dim
   h_t = \tanh ( U^h x_t + W^h \dot c_{t-1} )   # dim(h_t) == cell_dim
   c_t = ( 1 - z_t ) \dot h_t  +  z_t \dot c_{t-1}  # dim(c_t) == cell_dim
   y_t = ( c_t \dot o_t) W^y  # dim(y_t) = recurrent_dim + non_recurrent_dim.
                              # This is the output of the OGRU.
   s_t = y_t[0:recurrent_dim-1]  # dimension range of y_t, dim(s_t) = recurrent_dim.


   Because we'll need it below, we define
    hpart_t = U^h x_t
   which is a subexpression of h_t.

   Our choice to make a "special" component for the projected OGRU is to have
   it be a function from
     (z_t, hpart_t, c_{t-1}) -> (h_t, c_t)
   That is, the input to the component is all those things on the LHS
   appended together, and the output is the two things on the
   RHS appended together.  The dimensions are:
    (cell_dim, cell_dim, cell_dim) -> (cell_dim, cell_dim).
   The component computes the functions:
     h_t = \tanh ( U^h x_t + W^h \dot c_{t-1} )
     c_t = ( 1 - z_t ) \dot h_t  +  z_t \dot c_{t-1}

   Notice that 'W^h' is the only parameter that lives inside the component.

   You might also notice that the output 'h_t' is never actually used
   in any other part of the GRU, so the question arises: why is it
   necessary to have it be an output of the component?  This has to do with
   saving computation: because h_t is an output, and we'll be defining
   the kBackpropNeedsOutput flag, it is available in the backprop phase
   and this helps us avoid some computation (otherwise we'd have to do
   a redundant multiplication by W^h in the backprop phase that we already
   did in the forward phase).  We could have used the 'memo' mechanism to
   do this, but this is undesirable because the use of a memo disables
   'update consolidation' in the backprop so we'd lose a little
   speed there.

   This component stores stats of the same form as are normally stored by the
   StoreStats() functions for the sigmoid and tanh units, i.e. averages of the
   activations and derivatives, but this is done inside the Backprop() functions.


  The main configuration values that are accepted:
         cell-dim         e.g. cell-dim=1024  Cell dimension.
         recurrent-dim    e.g. recurrent-dim=256.  If not specified, we assume
                          this is a non-projected GRU.
         param-stddev     Standard deviation for random initialization of
                          the matrix W^h.  Defaults to 1.0 / sqrt(d) where
                          d is recurrent-dim if specified, else cell-dim.
         self-repair-threshold   Equivalent to the self-repair-lower-threshold
                          in a TanhComponent; applies to the tanh nonlinearity.
                          default=0.2, you probably won't want to change this.
         self-repair-scale Equivalent to the self-repair-scale in a
                          TanhComponent; applies to the tanh nonlinearity.
                          default=1.0e-05, which you probably won't want to
                          change unless dealing with an objective function that
                          has smaller or larger dynamic range than normal, in
                          which case you might want to make it smaller or
                          larger.

  Values inherited from UpdatableComponent (see its declaration in
  nnet-component-itf.h for details):
      learning-rate
      learning-rate-factor
      max-change

   Natural-gradient related options are below; you won't normally have to
   set these.
      alpha                 Constant that determines how much we smooth the
                            Fisher-matrix estimates with the unit matrix.
                            Larger means more smoothing. default=4.0
      rank                  The rank of the correction to the unit matrix.
                            default=8.
      update-period         Determines the period (in minibatches) with which
                            we update the Fisher-matrix estimates;
                            making this > 1 saves a little time in training.
                            default=10.


   Recap of what this computes:
     This component implements the function
        (z_t, hpart_t, c_{t-1}) -> (h_t, c_t)
     of dimensions
        (cell_dim, cell_dim, cell_dim) -> (cell_dim, cell_dim),
    where:
         h_t = \tanh( hpart_t + W^h \dot c_{t-1} )
         c_t = (1 - z_t) \dot h_t + z_t \dot c_{t-1}.
*/
class OutputGruNonlinearityComponent: public UpdatableComponent {
 public:

  virtual int32 InputDim() const;
  virtual int32 OutputDim() const;
  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);
  OutputGruNonlinearityComponent() { }
  virtual std::string Type() const { return "OutputGruNonlinearityComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kBackpropNeedsInput|\
        kBackpropNeedsOutput|kBackpropAdds;
  }
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &in_value,
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update_in,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const { return new OutputGruNonlinearityComponent(*this); }

  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);

  // Some functions from base-class UpdatableComponent.
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
  virtual void ZeroStats();
  virtual void FreezeNaturalGradient(bool freeze);

  // Some functions that are specific to this class:
  explicit OutputGruNonlinearityComponent(
      const OutputGruNonlinearityComponent &other);

 private:

  void Check() const;  // checks dimensions, etc.

  /**
     This function stores value and derivative stats for the tanh
     nonlinearity that is a part of this component, and if needed
     adds the small 'self-repair' term to 'h_t_deriv'.
      @param [in] h_t The output of the tanh expression from the
                      forward pass.
      @param [in,out] h_t_deriv  To here will be added the small
                      self-repair term (this is a small value
                      that we use to push oversaturated neurons
                      back to the center).
     This function has side effects on the class instance, specifically the
     members value_sum_, deriv_sum, self_repair_total_, and count_.
   */
  void TanhStatsAndSelfRepair(const CuMatrixBase<BaseFloat> &h_t,
                              CuMatrixBase<BaseFloat> *h_t_deriv);

  /*  This function is responsible for updating the w_h_ matrix
      (taking into account the learning rate).
        @param [in] c_t1_value  The value of c_{t-1}.
        @param [in] h_t_deriv  The derivative of the objective
                        function w.r.t. the argument of the tanh
                        function, i.e. w.r.t. the expression
                        "hpart_t + W^h \dot c_t1".
                        This function is concerned with the second
                        term as it affects the derivative w.r.t. W^h.
   */
  void UpdateParameters(const CuMatrixBase<BaseFloat> &c_t1_value,
                        const CuMatrixBase<BaseFloat> &h_t_deriv);


  int32 cell_dim_;  // cell dimension, e.g. 1024.

  // The matrix W^h, of dimension cell_dim_ by recurrent_dim_.
  // There is no bias term needed here because hpart_t comes from
  // an affine component that has a bias.
  CuVector<BaseFloat> w_h_;

  // Of dimension cell_dim_, this is comparable to the value_sum_ vector in
  // class NonlinearComponent.  It stores the sum of the tanh nonlinearity.
  // Normalize by dividing by count_.
  CuVector<double> value_sum_;

  // Of dimension cell_dim_, this is comparable to the deriv_sum_ vector in
  // class NonlinearComponent.  It stores the sum of the function-derivative of
  // the tanh nonlinearity.  Normalize by dividing by count_.
  CuVector<double> deriv_sum_;

  // This is part of the stats (along with value_sum_, deriv_sum_, and count_);
  // if you divide it by count_ it gives you the proportion of the time that an
  // average dimension was subject to self-repair.
  double self_repair_total_;

  // The total count (number of frames) corresponding to the stats in value_sum_,
  // deriv_sum_, and self_repair_total_.
  double count_;

  // A configuration parameter, this determines how saturated the derivative
  // has to be for a particular dimension, before we activate self-repair.
  // Default value is 0.2, the same as for TanhComponent.
  BaseFloat self_repair_threshold_;

  // A configuration parameter, this determines the maximum absolute value of
  // the extra term that we add to the input derivative of the tanh when doing
  // self repair.  The default value is 1.0e-05.
  BaseFloat self_repair_scale_;

  // Unlike the GruNonlinearityComponent, there is only one dimension to
  // consider as the parameters are a vector not a matrix, so we only need one
  // preconditioner.
  OnlineNaturalGradient preconditioner_;

  const OutputGruNonlinearityComponent &operator
      = (const OutputGruNonlinearityComponent &other); // Disallow.
};


} // namespace nnet3
} // namespace kaldi


#endif
