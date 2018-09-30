// nnet3/convolution-cudnn.h

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

#ifndef KALDI_NNET3_NNET_CUDNN_CONVOLUTION_H_
#define KALDI_NNET3_NNET_CUDNN_CONVOLUTION_H_


#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "nnet3/convolution.h"

// TODO: Consider forward declaring types like
// cudnnTensorDescriptor_t, so that this header file doesn't depend on
// cudnn.h
#include <cudnn.h>

namespace kaldi {
namespace nnet3 {
namespace cudnn {

class ConvolutionComputation final {
public:
  // Represents structural information about a convolution computation,
  // with filters, padding, striding, inputs and outputs of a specified size.  The same interface
  // is usable on both GPU and CPU.  You create this object only after you know the
  // number of images and input and output sizes, and it will be stored as part of
  // a NnetComputation (i.e. a compiled computation) and re-used between different
  // minibatches.  This object is lightweight.
  //
  // In the following docstrings, consider:
  // N to be equivalent to num_images
  // C to be equivalent to num_channels_in
  // K to be equivalent to num_channels_out
  // H to be equivalent to input_image_height, or filter_height,
  //   depending on context
  // W to be equivalent to input_image_width, or filter_width,
  //   depending on context
  //
  //  @param [in] num_channels_out   Number of output channels, e.g. 64.
  //  @param [in] num_channels_in   Number of input channels, e.g. 32.
  //  @param [in] filter_height  Height of filter patch, e.g. 3 (for 3x3 kernel).  Corresponds
  //                              to the 'frequency' dimension in normal speech applications, or
  //                              height in OCR applications.
  //  @param [in] filter_width  Width of filter patch, e.g. 3 (for 3x3 kernel).  Corresponds
  //                              to the 'time' dimension in normal speech applications.
  //  @param [in] filter_stride_height   Filter stride in the height ('frequency') dimension.
  //                              Will normally be 1 in speech and OCR applications.
  //  @param [in] filter_stride_width  Filter stride in the width ('time') dimension.
  //                              Will usually be 1 in most layers, but may be 2 or 3 if
  //                              we are doing subsampling on this layer (e.g. in
  //                              reduced-frame-rate models like chain models).
  //  @param [in] filter_dilation_height  Filter dilation in the height ('frequency')
  //                              dimension.  Equals the stride, in the input image, of
  //                              individual elements of the filter patch. Will
  //                              normally be 1.
  //  @param [in] filter_dilation_width  Filter dilation in the width ('time')
  //                              dimension.  Will normally be 1, but could
  //                              be more than one if, for instance, you have components
  //                              with time-stride > 1 which for some reason are required
  //                              to be evaluated on every frame.
  //  @param [in] num_images      The number of images we are processing, generally
  //                              equal to the minibatch size.
  //  @param [in] input_image_height  The height of the input images.  Corresponds to
  //                              the number of frequency bins, in speech applications.
  //  @param [in] input_image_width  The width of the input images.  Corresponds to
  //                              the number of time frames on the input, in speech
  //                              applications.
  //  @param [in] zero_padding_height  The number of pixels that we zero-pad with on
  //                              the bottom, and on the top, of the image (the
  //                              frequency dimension, in speech applications).  Would
  //                              be 1, for instance, if you are using a 3x3 kernel
  //                              and don't want to lose frequency bins.
  //  @param [in] zero_padding_width  The number of frames that we zero-pad with on
  //                              the left, and on the right, of the image (time
  //                              dimension).  Likely to be 0 in many speech applications,
  //                              since we normally deal with edge effects by padding
  //                              with repeats of the first and last frame; but
  //                              padding is supported by the component.
  ConvolutionComputation(int32 num_channels_out, int32 num_channels_in,
                         int32 filter_height, int32 filter_width,
                         int32 filter_stride_height, int32 filter_stride_width,
                         int32 filter_dilation_height,
                         int32 filter_dilation_width,
                         int32 num_images,
                         int32 input_image_height, int32 input_image_width,
                         int32 zero_padding_height, int32 zero_padding_width);
  ~ConvolutionComputation();
  int32 OutputImageHeight() const;
  int32 OutputImageWidth() const;

  /**
   * Returns the size of the workspace required for each stage, in
   * bytes (_not_ 32-bit words).
   */
  size_t TempSpaceRequiredForward() const;
  size_t TempSpaceRequiredBackwardData() const;
  size_t TempSpaceRequiredBackwardFilter() const;

  /**
   *  @param [in] input NWHC fully-packed tensor, with N == NumRows()
   *  @param [in] params KCWH fully-packed tensor, with K == NumRows()
   *  @param [in] bias vector of length K
   *  @param [in/out] temp_space Pointer to pre-allocated memory of size at least 
   *                             this->TempSpaceRequiredForward() bytes
   *  @param [out] output Pre-allocated NWHK fully-packed tensor, with N == NumRows()
   */
  void ConvolveForward(const CuMatrixBase<BaseFloat> &input,
                       const CuMatrixBase<BaseFloat> &params,
                       const CuVectorBase<BaseFloat> &bias,
                       CuVectorBase<BaseFloat> *temp_space,
                       CuMatrixBase<BaseFloat> *output) const;

  /**
   *  @param [in] params KCWH fully-packed tensor, with K == NumRows()
   *  @param [in] output_deriv NWHK fully-packed tensor, with N == NumRows()
   *  @param [in/out] temp_space Pointer to pre-allocated memory of size at least 
   *                             this->TempSpaceRequiredBackwardData() bytes
   *  @param [out] input_deriv Pre-allocated NWHC fully-packed tensor, with N == NumRows()
   */
  void ConvolveBackwardData(const CuMatrixBase<BaseFloat> &params,
                            const CuMatrixBase<BaseFloat> &output_deriv,
                            CuVectorBase<BaseFloat> *temp_space,
                            CuMatrixBase<BaseFloat> *input_deriv) const;

  /**
   *  @param [in] output_deriv NWHK fully-packed tensor, with N == NumRows()
   *  @param [in] input NWHC fully-packed tensor, with N == NumRows()
   *  @param [in] alpha 
   *              params_deriv := alpha * gradient_computed + params_deriv
   *  @param [in] params KCWH fully-packed tensor, with K == NumRows()
   *  @param [in/out] temp_space Pointer to pre-allocated memory of size at least 
   *                             this->TempSpaceRequiredBackwardFilter() bytes
   *  @param [out] params_deriv Pre-allocated KCWH fully-packed tensor, with K == NumRows()
   */
  void ConvolveBackwardParams(const CuMatrixBase<BaseFloat> &output_deriv,
                              const CuMatrixBase<BaseFloat> &input,
                              BaseFloat alpha,
                              CuVectorBase<BaseFloat> *temp_space,
                              CuMatrixBase<BaseFloat> *params_deriv) const;

  /**
   *  @param [in] output_deriv NWHK fully-packed tensor, with N == NumRows()
   *  @param [in] alpha 
   *              bias_deriv := alpha * gradient_computed + bias_deriv
   *  @param [out] bias_deriv Pre-allocated vector of length K
   */
  void ConvolveBackwardBias(const CuMatrixBase<BaseFloat> &output_deriv,
                            BaseFloat alpha,
                            CuVectorBase<BaseFloat> *bias_deriv) const;

private:
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnFilterDescriptor_t params_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnActivationDescriptor_t activation_desc_;

  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
};

} // namespace cudnn
} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_CUDNN_CONVOLUTION_H_
