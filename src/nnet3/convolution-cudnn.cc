// nnet3/convolution-cudnn.cc

// Copyright      2018  Daniel Galvez
//                2018  Johns Hopkins University (author: Daniel Povey)

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
namespace cudnn_convolution {


namespace {
// Note: anonymous namespaces are now preferred (by the C++ standard) over
// static variables.
const BaseFloat ONE(1.0);
const BaseFloat ZERO(0.0);
}

ConvolutionComputation::
ConvolutionComputation(int32 num_channels_out, int32 num_channels_in,
                       int32 filter_height, int32 filter_width,
                       int32 filter_stride_vertical, int32 filter_stride_horizontal,
                       int32 filter_dilation_height,
                       int32 filter_dilation_width,
                       int32 num_images,
                       int32 input_image_height, int32 input_image_width,
                       int32 zero_padding_height, int32 zero_padding_width):
    num_channels_out_(num_channels_out),
    num_channels_in_(num_channels_in),
    filter_height_(filter_height),
    filter_width_(filter_width),
    filter_stride_vertical_(filter_stride_vertical),
    filter_stride_horizontal_(filter_stride_horizontal),
    filter_dilation_height_(filter_dilation_height),
    filter_dilation_width_(filter_dilation_width),
    num_images_(num_images),
    input_image_height_(input_image_height),
    input_image_width_(input_image_width),
    zero_padding_height_(zero_padding_height),
    zero_padding_width_(zero_padding_width) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    InitCudnn();
  }
#endif
  // The following is called whether or not we are using CUDA.
  ComputeOutputImageHeight();
  ComputeOutputImageWidth();
}

#if HAVE_CUDA == 1
void ConvolutionComputation::InitCudnn() {
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc_));
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc_));
  CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&params_desc_));
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&bias_desc_));
  CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
  CUDNN_SAFE_CALL(cudnnCreateActivationDescriptor(&activation_desc_));

  CUDNN_SAFE_CALL(
      cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NHWC,
                                 CUDNN_DATA_FLOAT, num_images_,
                                 num_channels_in_, input_image_width_,
                                 input_image_height_));
  CUDNN_SAFE_CALL(
      cudnnSetConvolution2dDescriptor(conv_desc_,
                                      zero_padding_width_, zero_padding_height_,
                                      filter_stride_horizontal_, filter_stride_vertical_,
                                      filter_dilation_width_, filter_dilation_height_,
                                      CUDNN_CROSS_CORRELATION, // TODO: Double check this!
                                      CUDNN_DATA_FLOAT));
  CUDNN_SAFE_CALL(
      cudnnSetFilter4dDescriptor(params_desc_, CUDNN_DATA_FLOAT,
                                 CUDNN_TENSOR_NCHW, num_channels_out_,
                                 num_channels_in_, filter_width_, filter_height_));

  // These two member functions depend only on input_desc_,
  // conv_desc_, and params_desc_, so they are safe to call now.
  int32 out_kaldi_height_cudnn_width = OutputImageHeight();
  int32 out_kaldi_width_cudnn_height = OutputImageWidth();
  CUDNN_SAFE_CALL(
      cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NHWC,
                                 CUDNN_DATA_FLOAT, num_images_,
                                 num_channels_in_, out_kaldi_width_cudnn_height,
                                 out_kaldi_height_cudnn_width));
  const int32 bias_stride[] = {1};
  CUDNN_SAFE_CALL(
      cudnnSetTensorNdDescriptor(bias_desc_, CUDNN_DATA_FLOAT, 1,
                                 &num_channels_out_, bias_stride));

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

  KALDI_ASSERT(returned_algo_count > 0 &&
               "No algorithms were returned by CUDNN.");
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
  KALDI_ASSERT(returned_algo_count > 0 &&
               "No algorithms were returned by CUDNN.");
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
  KALDI_ASSERT(returned_algo_count > 0 &&
               "No algorithms were returned by CUDNN.");
  const cudnnConvolutionBwdDataAlgoPerf_t& best_backward_data =
      backward_data_results[0];
  bwd_data_algo_ = best_backward_data.algo;
  delete [] backward_data_results;
  ComputeTempSpaceSizes();
}
#endif

#if HAVE_CUDA == 1
void ConvolutionComputation::ComputeTempSpaceSizes() {
  CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      CuDevice::Instantiate().GetCudnnHandle(),
      input_desc_,
      params_desc_,
      conv_desc_,
      output_desc_,
      fwd_algo_,
      &temp_space_required_forward_));

  CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
      CuDevice::Instantiate().GetCudnnHandle(),
      params_desc_,
      output_desc_,
      conv_desc_,
      input_desc_,
      bwd_data_algo_,
      &temp_space_required_backward_data_));

  CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      CuDevice::Instantiate().GetCudnnHandle(),
      input_desc_,
      output_desc_,
      conv_desc_,
      params_desc_,
      bwd_filter_algo_,
      &temp_space_required_backward_filter_));
}
#endif

#if HAVE_CUDA == 1
void ConvolutionComputation::DestroyCudnn() {
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(params_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(bias_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyActivationDescriptor(activation_desc_));
}
#endif

ConvolutionComputation::~ConvolutionComputation() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled())
    DestroyCudnn();
#endif
}

void ConvolutionComputation::ComputeOutputImageHeight() {
  // 'filter_height_reduction' is the amount by which the height of the filter patch
  // reduces the effective height of the input image.  It's the distance between
  // the first and last pixels of the filter patch.  E.g. in a 3x3 kernel it
  // would be 2.
  int32 filter_height_reduction = (filter_height_ - 1) * filter_dilation_height_;
  // 'modified_input_height' is the number of times we can shift the filter patch
  // (not yet taking account of any filter stride).  It's a kind of augmented input-image
  // height, after applying zero-padding and subtracting filter_height_reduction.
  int32 modified_input_height =
        input_image_height_ - filter_height_reduction + (zero_padding_height_ * 2),
      s = filter_stride_vertical_;

  // output_image_height_ equals reduced_input_height divided by s (but rounding
  // up), which is the number of times we can shift the filter patch by
  // filter_stride_vertical_.
  output_image_height_ = (modified_input_height + s - 1) / s;

#if HAVE_CUDA == 1
  // Check that CUDA has the same idea of what the output image height is, as we
  // do.  This helps check that the CPU and GPU computations are compatible.
  int32 unused;
  int32 kaldi_height_cudnn_width;
  CUDNN_SAFE_CALL(
      cudnnGetConvolution2dForwardOutputDim(conv_desc_, input_desc_,
                                            params_desc_,
                                            &unused, &unused,
                                            &unused,
                                            &kaldi_height_cudnn_width));
  if (kaldi_height_cudnn_width != output_image_height_) {
    KALDI_ERR << "Code error: the height from CUDNN " << kaldi_height_cudnn_width
              << " does not match our value " << output_image_height_;
  }
#endif
}

void ConvolutionComputation::ComputeOutputImageWidth() {
  // 'filter_width_reduction' is the amount by which the width of the filter patch
  // reduces the effective width of the input image.  It's the distance between
  // the first and last pixels of the filter patch.  E.g. in a 3x3 kernel it
  // would be 2.
  int32 filter_width_reduction = (filter_width_ - 1) * filter_dilation_width_;
  // 'modified_input_width' is the number of times we can shift the filter patch
  // (not yet taking account of any filter stride).  It's a kind of augmented input-image
  // width, after applying zero-padding and subtracting filter_width_reduction.
  int32 modified_input_width =
        input_image_width_ - filter_width_reduction + (zero_padding_width_ * 2),
      s = filter_stride_horizontal_;

  // output_image_width equals reduced_input_width divided by s (but rounding
  // up), which is the number of times we can shift the filter patch by
  // filter_stride_horizontal_.
  output_image_width_ = (modified_input_width + s - 1) / s;
#if HAVE_CUDA == 1
  int32 unused;
  int32 kaldi_width_cudnn_height;
  CUDNN_SAFE_CALL(
      cudnnGetConvolution2dForwardOutputDim(conv_desc_, input_desc_,
                                            params_desc_,
                                            &unused, &unused,
                                            &kaldi_width_cudnn_height,
                                            &unused));
  if (kaldi_width_cudnn_height != output_image_width_) {
    KALDI_ERR << "Code error: the height from CUDNN " << kaldi_width_cudnn_height
              << " does not match our value " << output_image_width_;
  }
#endif
}


void ConvolutionComputation::Write(std::ostream &os, bool binary) const {
  // TODO: write just num_channels_out_ through zero_padding_width_;

}

void ConvolutionComputation::Read(std::istream &is, bool binary) {
  // TODO: read just num_channels_out_ through zero_padding_width_;

#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    InitCudnn();
  }
#endif
  // The following are called whether or not we have CUDA.
  ComputeOutputImageHeight();
  ComputeOutputImageWidth();
}


void ConvolutionComputation::
ConvolveForward(const CuMatrixBase<BaseFloat> &input,
                const CuMatrixBase<BaseFloat> &params,
                const CuVectorBase<BaseFloat> &bias,
                CuMatrixBase<BaseFloat> *output) const {
  // Check some dimensions.
  KALDI_ASSERT(
      input.NumRows() == num_images_ * input_image_width_ &&
      input.NumCols() == input_image_height_ * num_channels_in_ &&
      input.Stride() == input.NumCols() &&
      params.NumRows() == num_channels_out_ &&
      params.NumCols() == num_channels_in_ * filter_height_ * filter_width_ &&
      params.Stride() == params.NumCols() &&
      bias.Dim() == num_channels_out_ &&
      output->NumRows() == num_images_ * input_image_height_ &&
      output->NumCols() == input_image_width_ * num_channels_out_ &&
      output->Stride() == output->NumCols());

#ifdef HAVE_CUDNN
  if (CuDevice::Instantiate().Enabled()) {
    CuVector<BaseFloat> temp_space(temp_space_required_forward_ /
                                   sizeof(BaseFloat), kUndefined);
    CUDNN_SAFE_CALL(cudnnConvolutionBiasActivationForward(
        CuDevice::Instantiate().GetCudnnHandle(),
        &ONE,
        input_desc_,
        input.Data(),
        params_desc_,
        params.Data(),
        conv_desc_,
        fwd_algo_,
        temp_space.Data(),
        temp_space.Dim() * sizeof(BaseFloat),
        &ZERO,
        output_desc_,
        output->Data(),
        bias_desc_,
        bias.Data(),
        activation_desc_,
        output_desc_,
        output->Data()));
  } else
#endif
  {
    ConvolveForward(input.Mat(), params.Mat(), bias.Vec(),
                    &(output->Mat()));
  }
}


void ConvolutionComputation::
ConvolveForward(const MatrixBase<BaseFloat> &input,
                const MatrixBase<BaseFloat> &params,
                const VectorBase<BaseFloat> &bias,
                MatrixBase<BaseFloat> *output) const {
  // Check some dimensions.
  KALDI_ASSERT(
      input.NumRows() == num_images_ * input_image_width_ &&
      input.NumCols() == input_image_height_ * num_channels_in_ &&
      input.Stride() == input.NumCols() &&
      params.NumRows() == num_channels_out_ &&
      params.NumCols() == num_channels_in_ * filter_height_ * filter_width_ &&
      params.Stride() == params.NumCols() &&
      bias.Dim() == num_channels_out_ &&
      output->NumRows() == num_images_ * input_image_height_ &&
      output->NumCols() == input_image_width_ * num_channels_out_ &&
      output->Stride() == output->NumCols());


  {  // Deal with the bias.
    SubMatrix<BaseFloat> output_rearranged(
        output->Data(),
        num_images_ * input_image_width_ * input_image_height_,
        num_channels_out_, num_channels_out_);
    output_rearranged.CopyRowsFromVec(bias);
  }

  Matrix<BaseFloat> params_rearranged(filter_width_ * filter_height_,
                                      num_channels_out_ * num_channels_in_,
                                      kUndefined, kStrideEqualNumCols);
  ConvertParams(params, &params_rearranged);

  // We're using variable names w (as in width) for horizontal positions and h
  // (as in height) for vertical positions.  This is perhaps not ideal.
  for (int32 output_w = 0; output_w < output_image_width_; output_w++) {
    for (int32 output_h = 0; output_h < output_image_height_; output_h++) {
      for (int32 filter_h = 0; filter_h < filter_height_; filter_h++) {
        int32 filter_h_flipped = filter_height_ - 1 - filter_h;
        int32 input_h = output_h * filter_stride_vertical_
            - zero_padding_height_
            + filter_h * filter_dilation_height_;
        if (input_h < 0 || input_h >= input_image_height_)
          continue;
        for (int32 filter_w = 0; filter_w < filter_width_; filter_w++) {
          int32 filter_w_flipped = filter_width_ - 1 - filter_w;
          int32 input_w = output_w * filter_stride_horizontal_
              - zero_padding_width_
              + filter_w * filter_dilation_width_;

          if (input_w < 0 || input_w >= input_image_width_)
            continue;

          const BaseFloat *params_data = params_rearranged.RowData(
              filter_w_flipped * filter_height_ + filter_h_flipped);
          SubMatrix<BaseFloat> this_params(params_data,
                                           num_channels_out_,
                                           num_channels_in_, num_channels_in_);
          const BaseFloat *input_data = input.Data() +
              input_w * input_image_height_ * num_channels_in_ +
              input_h * num_channels_in_;
          SubMatrix<BaseFloat> this_input_pixel(input_data,
                                                num_images_,
                                                num_channels_in_,
                                                num_channels_in_);
          SubMatrix<BaseFloat> this_output_pixel(input_data,
                                                 num_images_,
                                                 num_channels_in_,
                                                 num_channels_in_);
          this_output_pixel.AddMatMat(1.0, this_input_pixel, kNoTrans,
                                      this_params, kTrans, 1.0);
        }
      }
    }
  }
}

void ConvolutionComputation::
ConvolveBackwardData(const CuMatrixBase<BaseFloat> &params,
                     const CuMatrixBase<BaseFloat> &output_deriv,
                     CuMatrixBase<BaseFloat> *input_deriv) const {
#ifdef HAVE_CUDNN
  if (CuDevice::Instantiate().Enabled()) {
    CuVector<BaseFloat> temp_space(temp_space_required_backward_data_ /
                                   sizeof(BaseFloat), kUndefined);
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardData(
        CuDevice::Instantiate().GetCudnnHandle(),
        &ONE,
        params_desc_,
        params.Data(),
        output_desc_,
        output_deriv.Data(),
        conv_desc_,
        bwd_data_algo_,
        temp_space.Data(),
        temp_space.Dim() * sizeof(BaseFloat),
        &ZERO,
        input_desc_,
        input_deriv->Data()));
  } else
#endif
  {
    // TODO
  }
}

void ConvolutionComputation::
ConvolveBackwardData(const MatrixBase<BaseFloat> &params,
                     const MatrixBase<BaseFloat> &output_deriv,
                     MatrixBase<BaseFloat> *input_deriv) const {
  // TODO
}



void ConvolutionComputation::
ConvolveBackwardParams(const CuMatrixBase<BaseFloat> &output_deriv,
                       const CuMatrixBase<BaseFloat> &input,
                       BaseFloat alpha,
                       CuMatrixBase<BaseFloat> *params_deriv) const {
#ifdef HAVE_CUDNN
  if (CuDevice::Instantiate().Enabled()) {
    CuVector<BaseFloat> temp_space(temp_space_required_backward_params_ /
                                   sizeof(BaseFloat), kUndefined);
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardFilter(
        CuDevice::Instantiate().GetCudnnHandle(),
        &alpha,
        input_desc_,
        input.Data(),
        output_desc_,
        output_deriv.Data(),
        conv_desc_,
        bwd_filter_algo_,
        temp_space.Data(),
        temp_space.Dim() * sizeof(BaseFloat),
        &ONE,
        params_desc_,
        params_deriv->Data()));
  } else
#endif
  {
    ConvolveBackwardParams(output_deriv.Mat(), input.Mat(),
                           alpha, &(params_deriv->Mat()));
  }
}


void ConvolutionComputation::
ConvolveBackwardParams(const MatrixBase<BaseFloat> &output_deriv,
                       const MatrixBase<BaseFloat> &input,
                       BaseFloat alpha,
                       MatrixBase<BaseFloat> *params_deriv) const {
  // TODO
}


void ConvolutionComputation::
ConvolveBackwardBias(const CuMatrixBase<BaseFloat> &output_deriv,
                     BaseFloat alpha,
                     CuVectorBase<BaseFloat> *bias_deriv) const {
#ifdef HAVE_CUDNN
  if (CuDevice::Instantiate().Enabled()) {
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardBias(
        CuDevice::Instantiate().GetCudnnHandle(),
        &alpha,
        output_desc_,
        output_deriv.Data(),
        &ONE,
        bias_desc_,
        bias_deriv->Data()));
  } else
#endif
  {
    ConvolveBackwardBias(output_deriv.Mat(), alpha, &(bias_deriv->Vec()));
  }
}

void ConvolutionComputation::
ConvolveBackwardBias(const MatrixBase<BaseFloat> &output_deriv,
                     BaseFloat alpha,
                     VectorBase<BaseFloat> *bias_deriv) const {
  // TODO.
}


// This function, called only if we are not using the GPU, converts
// the params from KCWH format to WHKC format (which is more convenient
// when using the CPU.  Note: K == channels-out, C == channels-in.
void ConvolutionComputation::ConvertParams(
    const MatrixBase<BaseFloat> &params,
    MatrixBase<BaseFloat> *params_rearranged) const {
  KALDI_ASSERT(params.NumRows() == num_channels_out_ &&
               params.Stride() == params.NumCols() &&
               params_rearranged->NumRows() == filter_width_ * filter_height_ &&
               params_rearranged->Stride() == params_rearranged->NumCols());

  // Reinterpret params as params_reinterpret which is of dimension KC * WH (instead of  K * CWH).
  SubMatrix<BaseFloat> params_reinterpret(params.Data(),
                                          num_channels_out_ * num_channels_in_,
                                          filter_width_ * filter_height_,
                                          filter_width_ * filter_height_);
  params_rearranged->CopyFromMat(params_reinterpret, kTrans);
}


}  // namespace cudnn_convolution
}  // namespace nnet3
}  // namespace kaldi
