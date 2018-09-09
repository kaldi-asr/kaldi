// nnet3/convolution-cudnn.cc

// Copyright      2018  Daniel Galvez

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

#include "nnet3/convolution-cudnn.h"

namespace kaldi {
namespace nnet3 {
namespace cudnn {


namespace {
  const BaseFloat ONE(1.0);
  const BaseFloat ZERO(0.0);
}

  ConvolutionComputation::
  ConvolutionComputation(int32 num_channels_out, int32 num_channels_in,
                         int32 filter_height, int32 filter_width,
                         int32 filter_stride_height, int32 filter_stride_width,
                         // dilation?
                         int32 filter_dilation_height,
                         int32 filter_dilation_width,
                         int32 num_images,
                         int32 input_image_height, int32 input_image_width,
                         int32 zero_padding_height, int32 zero_padding_width) {
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc_));
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc_));
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&params_desc_));
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_SAFE_CALL(cudnnCreateActivationDescriptor(&activation_desc_));

    CUDNN_SAFE_CALL(
      cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NHWC,
                                 CUDNN_DATA_FLOAT, num_images,
                                 num_channels_in, input_image_width,
                                 input_image_height));

    int32 out_kaldi_height_cudnn_width = OutputImageHeight();
    int32 out_kaldi_width_cudnn_height = OutputImageWidth();
    CUDNN_SAFE_CALL(
    cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT, num_images,
                               num_channels_in, out_kaldi_width_cudnn_height,
                               out_kaldi_height_cudnn_width));
    CUDNN_SAFE_CALL(
    cudnnSetFilter4dDescriptor(params_desc_, CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NHWC, num_channels_out,
                               num_channels_in, filter_width, filter_height));
    int32 bias_stride = 1;
    CUDNN_SAFE_CALL(
    cudnnSetTensorNdDescriptor(bias_desc_, CUDNN_DATA_FLOAT, 1,
                               &num_channels_out, &bias_stride));
    CUDNN_SAFE_CALL(
    cudnnSetConvolution2dDescriptor(conv_desc_,
                                    zero_padding_width, zero_padding_height,
                                    filter_stride_width, filter_stride_height,
                                    filter_dilation_width, filter_dilation_height,
                                    CUDNN_CROSS_CORRELATION, // TODO: Double check this!
                                    CUDNN_DATA_FLOAT));

    const double DONT_CARE = 0;
    CUDNN_SAFE_CALL(
      cudnnSetActivationDescriptor(activation_desc_, CUDNN_ACTIVATION_IDENTITY,
                                   CUDNN_PROPAGATE_NAN, DONT_CARE));

    int32 requested_algo_count, returned_algo_count;
    CUDNN_SAFE_CALL(cudnnGetConvolutionForwardAlgorithmMaxCount(
      CuDevice::Instantiate().GetCudnnHandle(), &requested_algo_count));

    cudnnConvolutionFwdAlgoPerf_t *forward_results =
      new cudnnConvolutionFwdAlgoPerf_t[requested_algo_count];
    CUDNN_SAFE_CALL(cudnnFindConvolutionForwardAlgorithm(
      CuDevice::Instantiate().GetCudnnHandle(),
      input_desc_,
      params_desc_,
      conv_desc_,
      output_desc_,
      requested_algo_count,
      &returned_algo_count,
      forward_results));

    KALDI_ASSERT(returned_algo_count > 0);
    const cudnnConvolutionFwdAlgoPerf_t& best_forward = forward_results[0];
    fwd_algo_ = best_forward.algo;
    delete [] forward_results;

    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
      CuDevice::Instantiate().GetCudnnHandle(), &requested_algo_count));
    cudnnConvolutionBwdFilterAlgoPerf_t *backward_filter_results =
      new cudnnConvolutionBwdFilterAlgoPerf_t[requested_algo_count];
    CUDNN_SAFE_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(
      CuDevice::Instantiate().GetCudnnHandle(),
      input_desc_,
      output_desc_,
      conv_desc_,
      params_desc_,
      requested_algo_count,
      &returned_algo_count,
      backward_filter_results));
    KALDI_ASSERT(returned_algo_count > 0);
    const cudnnConvolutionBwdFilterAlgoPerf_t& best_backward_filter =
      backward_filter_results[0];
    bwd_filter_algo_ = best_backward_filter.algo;
    delete [] backward_filter_results;

    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
      CuDevice::Instantiate().GetCudnnHandle(), &requested_algo_count));
    cudnnConvolutionBwdDataAlgoPerf_t *backward_data_results =
      new cudnnConvolutionBwdDataAlgoPerf_t[requested_algo_count];
    CUDNN_SAFE_CALL(cudnnFindConvolutionBackwardDataAlgorithm(
      CuDevice::Instantiate().GetCudnnHandle(),
      params_desc_,
      output_desc_,
      conv_desc_,
      input_desc_,
      requested_algo_count,
      &returned_algo_count,
      backward_data_results));
    KALDI_ASSERT(returned_algo_count > 0);
    const cudnnConvolutionBwdDataAlgoPerf_t& best_backward_data =
      backward_data_results[0];
    bwd_data_algo_ = best_backward_data.algo;
    delete [] backward_data_results;
  }

  ConvolutionComputation::~ConvolutionComputation() {
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc_));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc_));
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(params_desc_));
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
    CUDNN_SAFE_CALL(cudnnDestroyActivationDescriptor(activation_desc_));
  }

  int32 ConvolutionComputation::OutputImageHeight() const {
    int32 unused;
    int32 kaldi_height_cudnn_width;
    CUDNN_SAFE_CALL(
      cudnnGetConvolution2dForwardOutputDim(conv_desc_, input_desc_,
                                            params_desc_,
                                            &unused, &unused,
                                            &kaldi_height_cudnn_width,
                                            &unused));
    return kaldi_height_cudnn_width;
  }

  int32 ConvolutionComputation::OutputImageWidth() const {
    int32 unused;
    int32 kaldi_width_cudnn_height;
    CUDNN_SAFE_CALL(
      cudnnGetConvolution2dForwardOutputDim(conv_desc_, input_desc_,
                                            params_desc_,
                                            &unused, &unused,
                                            &unused,
                                            &kaldi_width_cudnn_height));
    return kaldi_width_cudnn_height;
  }

  size_t ConvolutionComputation::TempSpaceRequiredForward() const {
    size_t workspace_size_bytes;
    CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      CuDevice::Instantiate().GetCudnnHandle(),
      input_desc_,
      params_desc_,
      conv_desc_,
      output_desc_,
      fwd_algo_,
      &workspace_size_bytes));
    return workspace_size_bytes;
  }

  size_t ConvolutionComputation::TempSpaceRequiredBackwardData() const {
    size_t workspace_size_bytes;
    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
      CuDevice::Instantiate().GetCudnnHandle(),
      params_desc_,
      output_desc_,
      conv_desc_,
      input_desc_,
      bwd_data_algo_,
      &workspace_size_bytes));
    return workspace_size_bytes;
  }


  size_t ConvolutionComputation::TempSpaceRequiredBackwardFilter() const {
    size_t workspace_size_bytes;
    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      CuDevice::Instantiate().GetCudnnHandle(),
      input_desc_,
      output_desc_,
      conv_desc_,
      params_desc_,
      bwd_filter_algo_,
      &workspace_size_bytes));
    return workspace_size_bytes;
  }



  void ConvolutionComputation::
  ConvolveForward(const CuMatrixBase<BaseFloat> &input,
                  const CuMatrixBase<BaseFloat> &params,
                  const CuVectorBase<BaseFloat> &bias,
                  CuVectorBase<BaseFloat> *temp_space,
                  CuMatrixBase<BaseFloat> *output) const {
    CUDNN_SAFE_CALL(cudnnConvolutionBiasActivationForward(
      CuDevice::Instantiate().GetCudnnHandle(),
      &ONE,
      input_desc_,
      input.Data(),
      params_desc_,
      params.Data(),
      conv_desc_,
      fwd_algo_,
      temp_space->Data(),
      temp_space->Dim() * sizeof(BaseFloat),
      &ZERO,
      output_desc_,
      output->Data(),
      bias_desc_,
      bias.Data(),
      activation_desc_,
      output_desc_,
      output->Data()));
  }

  void ConvolutionComputation::
  ConvolveBackwardData(const CuMatrixBase<BaseFloat> &params,
                       const CuMatrixBase<BaseFloat> &output_deriv,
                       CuVectorBase<BaseFloat> *temp,
                       CuMatrixBase<BaseFloat> *input_deriv) const {
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardData(
      CuDevice::Instantiate().GetCudnnHandle(),
      &ONE,
      params_desc_,
      params.Data(),
      output_desc_,
      output_deriv.Data(),
      conv_desc_,
      bwd_data_algo_,
      temp->Data(),
      temp->Dim() * sizeof(BaseFloat),
      &ZERO,
      input_desc_,
      input_deriv->Data()));
  }

  void ConvolutionComputation::
  ConvolveBackwardParams(const CuMatrixBase<BaseFloat> &output_deriv,
                         const CuMatrixBase<BaseFloat> &input,
                         BaseFloat alpha,
                         CuVectorBase<BaseFloat> *temp,
                         CuMatrixBase<BaseFloat> *params_deriv) const {
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardFilter(
      CuDevice::Instantiate().GetCudnnHandle(),
      &alpha,
      input_desc_,
      input.Data(),
      output_desc_,
      output_deriv.Data(),
      conv_desc_,
      bwd_filter_algo_,
      temp->Data(),
      temp->Dim() * sizeof(BaseFloat),
      &ONE,
      params_desc_,
      params_deriv->Data()));
  }

  void ConvolutionComputation::
  ConvolveBackwardBias(const CuMatrixBase<BaseFloat> &output_deriv,
                       BaseFloat alpha,
                       CuVectorBase<BaseFloat> *bias_deriv) const {
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardBias(
      CuDevice::Instantiate().GetCudnnHandle(),
      &alpha,
      output_desc_,
      output_deriv.Data(),
      &ONE,
      bias_desc_,
      bias_deriv->Data()));
  }

} // namespace cudnn
} // namespace nnet3
} // namespace kaldi
