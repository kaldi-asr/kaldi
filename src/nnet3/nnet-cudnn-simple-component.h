// nnet3/nnet-simple-component.h

// Copyright 2016  Daniel Galvez

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

#ifndef KALDI_NNET3_NNET_CUDNN_SIMPLE_COMPONENT_H_
#define KALDI_NNET3_NNET_CUDNN_SIMPLE_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "cudamatrix/cudnn-utils.h"
#include <cudnn.h>
#include <vector>

namespace kaldi {
namespace nnet3 {

class CuDNN3DConvolutionComponent: public UpdatableComponent {
 public:

// the following vectorization notation need to be modified
// as they are arranging the dimension from smallest to largest
// stride which is opposite to the convention used by a large group
// of people
  enum TensorVectorizationType  {
    kYzx = 0,
    kZyx = 1
  };

  virtual int32 InputDim() const { return input_num_filters_ * input_x_dim_ * input_y_dim_ * input_z_dim_; }
  virtual int32 OutputDim() const;

  virtual std::string Info() const;
  virtual void InitFromConfig(ConfigLine *cfl);

  CuDNN3DConvolutionComponent(); // use Init to really initialize.
  virtual std::string Type() const { return "CuDNN3DConvolutionComponent"; }
  virtual int32 Properties() const {
    return kSimpleComponent|kUpdatableComponent|kBackpropNeedsInput|
      kBackpropAdds|kPropagateAdds|kInputContiguous|kOutputContiguous;
  }

  virtual void Propagate(const ComponentPrecomputedIndexes *indexes,
                 const CuMatrixBase<BaseFloat> &in,
                 CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, // in_value
                        const CuMatrixBase<BaseFloat> &, // out_value
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;

  void InitDescriptor(
    int32 filt_x_dim, int32 filt_y_dim, int32 filt_z_dim,
    int32 filt_x_stride, int32 filt_y_stride, int32 filt_z_stride,
    int32 pad_x_dim, int32 pad_y_dim, int32 pad_z_dim,
    int32 upscale_x_dim, int32 upscale_y_dim, int32 upscale_z_dim);
  void FillInputStrides(int32 n_dims, const int32 *shape, int32 *strides) const;
  void FillOutputStrides(int32 n_dims, const int32 *shape, int32 *strides) const;
  void Init(int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
            int32 input_num_filters,
            int32 filt_x_dim, int32 filt_y_dim, int32 filt_z_dim,
            int32 filt_x_stride, int32 filt_y_stride, int32 filt_z_stride,
            int32 num_filters,
            int32 pad_x_dim, int32 pad_y_dim, int32 pad_z_dim,
            int32 upscale_x_dim, int32 upscale_y_dim, int32 upscale_z_dim,
            TensorVectorizationType input_vectorization,
            BaseFloat param_stddev, BaseFloat bias_stddev);
  void Init(int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
            int32 input_num_filters,
            int32 filt_x_dim, int32 filt_y_dim, int32 filt_z_dim,
            int32 filt_x_stride, int32 filt_y_stride, int32 filt_z_stride,
            int32 pad_x_dim, int32 pad_y_dim, int32 pad_z_dim,
            int32 upscale_x_dim, int32 upscale_y_dim, int32 upscale_z_dim,
            TensorVectorizationType input_vectorization,
            std::string matrix_filename);

  // constructor using another component
  explicit CuDNN3DConvolutionComponent(const CuDNN3DConvolutionComponent &component);
  // constructor using parameters
  CuDNN3DConvolutionComponent(
    const CuMatrixBase<BaseFloat> &filter_params,
    const CuVectorBase<BaseFloat> &bias_params,
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 input_num_filters,
    int32 filt_x_dim, int32 filt_y_dim, int32 filt_z_dim,
    int32 filt_x_step, int32 filt_y_step, int32 filt_z_step,
    TensorVectorizationType input_vectorization,
    BaseFloat learning_rate);
  ~CuDNN3DConvolutionComponent();

  virtual Component *Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual void SetZero(bool treat_as_gradient);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual int32 NumParameters() const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void Vectorize(VectorBase<BaseFloat> *params) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
  virtual void PerturbParams(BaseFloat stddev);
 private:
  std::vector<int32> GetOutputDims() const;
  std::vector<int32> GetFilterDims() const;
  void Update(const CuMatrixBase<BaseFloat> &in_value,
              const CuMatrixBase<BaseFloat> &out_deriv,
              const cudnnTensorDescriptor_t in_desc,
              const cudnnTensorDescriptor_t out_desc);

  int32 input_x_dim_;   // size of the input along x-axis
                        // (e.g. number of time steps)
  int32 input_y_dim_;   // size of input along y-axis
                        // (e.g. number of mel-frequency bins)
  int32 input_z_dim_;   // size of input along z-axis
                        // (e.g. number of channels is 3 if the input has
                        // features + delta + delta-delta features
  int32 input_num_filters_;

  CuMatrix<BaseFloat> filter_params_;
  // the filter (or kernel) matrix is a matrix of vectorized 3D filters
  // where each row in the matrix corresponds to one filter.
  // The 3D filter tensor is vectorized in zyx format.
  // The first row of the matrix corresponds to the first filter and so on.
  // Keep in mind the vectorization type and order of filters when using file
  // based initialization.

  CuVector<BaseFloat> bias_params_;
  // the filter-specific bias vector (i.e., there is a seperate bias added
  // to the output of each filter).

  int32 num_filters_;
  void *work_space_;
  uint32 work_space_size_; // I belive size_t (used in cudnn.h) == kaldi::uint32 == uint32_t
  bool is_gradient_;

  TensorVectorizationType input_vectorization_; // type of vectorization of the
  // input 3D tensor. Accepts zyx and yzx formats

  // cudnn specific variables.
  static const int32 kConvolutionDimension_ = 3; // the number of dimensions that the filter is in
  // This is volumteric convolution, so the filter is 3D.

  // When we serialize, we'll need to call the getter functions on
  // these, since they are opaque types whose implementations we do
  // not know.  We could also store the data stored within these as members of the class
  // as well, but I do not want to have redundant data lying around.
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t forward_algo_;
  cudnnConvolutionBwdFilterAlgo_t backward_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t backward_data_algo_;

  const CuDNN3DConvolutionComponent &operator = (const CuDNN3DConvolutionComponent &other); // Disallow.
};

} // namespace nnet3
} // namespace kaldi

#endif
