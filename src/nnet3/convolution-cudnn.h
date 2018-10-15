// nnet3/convolution-cudnn.h

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

#ifndef KALDI_NNET3_NNET_CUDNN_CONVOLUTION_H_
#define KALDI_NNET3_NNET_CUDNN_CONVOLUTION_H_


#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "nnet3/convolution.h"
#if HAVE_CUDA == 1
#include <cudnn.h>
#endif


namespace kaldi {
namespace nnet3 {
namespace cudnn_convolution {

/**
   Represents structural information about a convolution computation, with
   filters, padding, striding, inputs and outputs of a specified size.  The
   same interface is usable on both GPU and CPU.  You create this object only
   after you know the number of images and input and output sizes, and it will
   be stored as part of a NnetComputation (i.e. a compiled computation) and
   re-used between different minibatches.  This object is lightweight; it
   doesn't contain data, only a few integers and descriptors.

   In the following docstrings:
    N is equivalent to num_images
    C is equivalent to num_channels_in
    K is equivalent to num_channels_out
    H is equivalent to input_image_height or output_image_height (for images) or
       filter_height (for filter parameters).
    W is equivalent to input_image_width or output_image_width (for images) or
       filter_width (for filter parameters).

    @param [in] num_channels_out   Number of output channels, e.g. 64.
    @param [in] num_channels_in   Number of input channels, e.g. 32.
    @param [in] filter_height  Height of filter patch, e.g. 3 (for 3x3 kernel).  Corresponds
                                to the 'frequency' dimension in normal speech applications, or
                                height in OCR applications.
    @param [in] filter_width  Width of filter patch, e.g. 3 (for 3x3 kernel).  Corresponds
                                to the 'time' dimension in normal speech applications.
    @param [in] filter_stride_vertical   Filter stride in the vertical ('frequency') dimension.
                                Will normally be 1 in speech and OCR applications.
    @param [in] filter_stride_horizontal  Filter stride in the horizontal ('time') dimension.
                                Will usually be 1 in most layers, but may be 2 or 3 if
                                we are doing subsampling on this layer (e.g. in
                                reduced-frame-rate models like chain models).
    @param [in] filter_dilation_height  Filter dilation in the vertical ('frequency')
                                dimension.  Equals the stride, in the input image, of
                                individual elements of the filter patch. Will
                                normally be 1.
    @param [in] filter_dilation_width  Filter dilation in the horizontal ('time')
                                dimension.  Will normally be 1, but could
                                be more than one if, for instance, you have components
                                with time-stride > 1 which for some reason are required
                                to be evaluated on every frame.
    @param [in] num_images      The number of images we are processing, generally
                                equal to the minibatch size.
    @param [in] input_image_height  The height of the input images.  Corresponds to
                                the number of frequency bins, in speech applications.
    @param [in] input_image_width  The width of the input images.  Corresponds to
                                the number of time frames on the input, in speech
                                applications.
    @param [in] zero_padding_height  The number of pixels that we zero-pad with on
                                the bottom, and on the top, of the image (the
                                frequency dimension, in speech applications).  Would
                                be 1, for instance, if you are using a 3x3 kernel
                                and don't want to lose frequency bins.
    @param [in] zero_padding_width  The number of frames that we zero-pad with on
                                the left, and on the right, of the image (time
                                dimension).  Likely to be 0 in many speech applications,
                                since we normally deal with edge effects by padding
                                with repeats of the first and last frame; but
                                padding is supported by this object.
*/
class ConvolutionComputation final {
public:
  ConvolutionComputation(int32 num_channels_out, int32 num_channels_in,
                         int32 filter_height, int32 filter_width,
                         int32 filter_stride_vertical, int32 filter_stride_horizontal,
                         int32 filter_dilation_height,
                         int32 filter_dilation_width,
                         int32 num_images,
                         int32 input_image_height, int32 input_image_width,
                         int32 zero_padding_height, int32 zero_padding_width);
  ~ConvolutionComputation();

  int32 OutputImageHeight() const { return output_image_height_; }
  int32 OutputImageWidth() const { return output_image_width_; }

  /**
   * For an explanation of the notation below (e.g. NWHC), see the
   * explanation for those variable names in the documentation for this
   * class above.  Variables that come first have the higher stride.
   *
   * Caution: for convenience, given the way nnet3 works, we flip the notion of
   * height and width that CUDNN uses, so our height is CUDNN's width, and vice
   * versa.  This is not visible to the user; we mention it just in case
   * those familiar with CUDNN get surprised at the order
   *
   *  @param [in] input NWHC fully-packed tensor, with NumRows() == N * W
   *  @param [in] params KCWH fully-packed tensor, with NumRows() == K.
   *  @param [in] bias vector of length K
   *  @param [out] output Pre-allocated NWHK fully-packed tensor, with N == NumRows()
   */
  void ConvolveForward(const CuMatrixBase<BaseFloat> &input,
                       const CuMatrixBase<BaseFloat> &params,
                       const CuVectorBase<BaseFloat> &bias,
                       CuMatrixBase<BaseFloat> *output) const;

  /**
   *  @param [in] params KCWH fully-packed tensor, with NumRows() == K
   *  @param [in] output_deriv NWHK fully-packed tensor, with NumRows() == N * W
   *  @param [out] input_deriv Pre-allocated NWHC fully-packed tensor, with
   *                           NumRows() == N * W
   */
  void ConvolveBackwardData(const CuMatrixBase<BaseFloat> &params,
                            const CuMatrixBase<BaseFloat> &output_deriv,
                            CuMatrixBase<BaseFloat> *input_deriv) const;

  /**
   *  @param [in] output_deriv NWHK fully-packed tensor, with NumRows() == N * W.
   *  @param [in] input NWHC fully-packed tensor, with NumRows() == N * W.
   *  @param [in] alpha
   *              params_deriv := alpha * gradient_computed + params_deriv
   *  @param [in] params KCWH fully-packed tensor, with NumRows() == K
   *  @param [out] params_deriv Pre-allocated KCWH fully-packed tensor,
   *                             with NumRows() == K.
   */
  void ConvolveBackwardParams(const CuMatrixBase<BaseFloat> &output_deriv,
                              const CuMatrixBase<BaseFloat> &input,
                              BaseFloat alpha,
                              CuMatrixBase<BaseFloat> *params_deriv) const;

  /**
   *  @param [in] output_deriv NWHK fully-packed tensor, with NumRows() * N * W.
   *  @param [in] alpha
   *              bias_deriv := alpha * gradient_computed + bias_deriv
   *  @param [out] bias_deriv Pre-allocated vector of length K
   */
  void ConvolveBackwardBias(const CuMatrixBase<BaseFloat> &output_deriv,
                            BaseFloat alpha,
                            CuVectorBase<BaseFloat> *bias_deriv) const;

  // The CPU versions of the functions declared above allow the user to use the
  // CPU even if a GPU is active.  They are also called by the versions of the
  // same name that take CuMatrix types, in the case when either we did not
  // compile for CUDA or we did but we are not using a GPU.
  void ConvolveForward(const MatrixBase<BaseFloat> &input,
                       const MatrixBase<BaseFloat> &params,
                       const VectorBase<BaseFloat> &bias,
                       MatrixBase<BaseFloat> *output) const;
  void ConvolveBackwardData(const MatrixBase<BaseFloat> &params,
                            const MatrixBase<BaseFloat> &output_deriv,
                            MatrixBase<BaseFloat> *input_deriv) const;
  void ConvolveBackwardParams(const MatrixBase<BaseFloat> &output_deriv,
                              const MatrixBase<BaseFloat> &input,
                              BaseFloat alpha,
                              MatrixBase<BaseFloat> *params_deriv) const;
  void ConvolveBackwardBias(const MatrixBase<BaseFloat> &output_deriv,
                            BaseFloat alpha,
                            VectorBase<BaseFloat> *bias_deriv) const;




  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &os, bool binary);

private:
#if HAVE_CUDA == 1
  // initialize the various descriptors; only called if compiled for CUDA
  // AND we are using a GPU.
  void InitCudnn();
  // ComputeTempSpaceSizes() is called from InitCudnn(); it sets the
  // temp_space_*_ member variables.
  void ComputeTempSpaceSizes();
  // Destroy the descriptors.
  void DestroyCudnn();
#endif

  // Called from the constructor and Read(), this sets output_image_height_.
  void ComputeOutputImageHeight();
  // Called from the constructor and Read(), this sets output_image_width_.
  void ComputeOutputImageWidth();


  // This function, called only if we are not using the GPU, converts
  // the params from KCWH format to WHKC format (which is more convenient
  // when using the CPU.  params and params_rearranged must both be
  // packed (Stride() == NumCols()), params must have num-rows equal to K
  // (num_channels_out_), and params_rearranged must have num-rows equal
  // to to WH (filter_width_ * filter_height_).
  void ConvertParams(const MatrixBase<BaseFloat> &params,
                     MatrixBase<BaseFloat> *params_rearranged) const;
  // This function does the opposite transformation of what ConvertParams()
  // does.
  void ConvertParamsBack(const MatrixBase<BaseFloat> &params_rearranged,
                         MatrixBase<BaseFloat> *params) const;



  // The following block of members are just copies of the args to the
  // constructor.  Please see the documentation of the constructor, and look for
  // the similarly named parameter, to understand the meaning of these
  // individual members.
  int32 num_channels_out_;
  int32 num_channels_in_;
  int32 filter_height_;
  int32 filter_width_;
  int32 filter_stride_vertical_;
  int32 filter_stride_horizontal_;
  int32 filter_dilation_height_;
  int32 filter_dilation_width_;
  int32 num_images_;
  int32 input_image_height_;
  int32 input_image_width_;
  int32 zero_padding_height_;
  int32 zero_padding_width_;
  int32 output_image_height_;
  int32 output_image_width_;

#if HAVE_CUDA == 1
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnFilterDescriptor_t params_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnActivationDescriptor_t activation_desc_;

  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;

  // The units of the following are all in bytes.
  size_t temp_space_required_forward_;
  size_t temp_space_required_backward_data_;
  size_t temp_space_required_backward_filter_;
#endif


};

}  // namespace cudnn_convolution
}  // namespace nnet3
}  // namespace kaldi

#endif // KALDI_NNET3_NNET_CUDNN_CONVOLUTION_H_
