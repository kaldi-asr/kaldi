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

/**  This struct contains information about a specific convolution
     computation.  It combines information about the model and the data.
     The examples below are very arbitrary.
 */
struct ConvolutionComputationConfig {
  // The number of images we are working on, e.g. 128.
  int32 num_images;
  // The number of input channels, e.g. 32
  int32 num_channels_in;
  // The number of output channels, e.g. 64.
  int32 num_channels_out;
  // The number of pixels in the filter patch in the vertical direction, e.g. 3.
  int32 filter_height;
  // The number of pixels in the filter patch in the horizontal direction, e.g. 3.
  int32 filter_width;
  // The vertical stride of the filter, normally 1 but might be (e.g.) 2 if we
  // are subsampling in the vertical (usually: frequency) dimension.
  int32 filter_stride_vertical;
  // The horizontal stride of the filter, normally 1 but might be (e.g.) 3 at a
  // certain layer of the network if we are training a chain model with a
  // frame-subsampling-factor of 3.
  int32 filter_stride_horizontal;
  // Normally 1, if this is more than 1 the pixels of the image patch will be
  // spaced apart from each other.
  int32 filter_dilation_vertical;
  // Normally 1, if this is more than 1 the pixels of the image patch will be
  // spaced apart from each other.
  int32 filter_dilation_horizontal;
  // The height of the input image, e.g. this is often 40 in speech applications,
  // subsampled to 20 or 10 later in the network.
  int32 input_image_height;
  // The width of the input image, which will be the same as the number of
  // frames being computed.
  int32 input_image_width;
  // The amount of zero-padding in the height (normally: frequency) dimension;
  // this number of zero frames are added at both top and bottom of the input.
  // Will often be 1, if you are using 3x3 kernels and don't want to
  // reduce the height of the image.
  int32 zero_padding_vertical;
  // The amount of zero-padding in the time dimension, meaning the number of
  // zero frames that we implicitly add to the beginning and end of the utterance.
  // This will normally be zero because in Kaldi ASR recipes we generally don't
  // do zero padding, but duplicate the first and last frame of the input to
  // match the amount of left and right context that the neural network requires.
  int32 zero_padding_horizontal;


  // The height of the output image.  The user does not have to set this;
  // it will be computed when you call ComputeOutputImageSize().
  int32 output_image_height;
  // The width of the output image.  The user does not have to set this;
  // it will be computed when you call ComputeOutputImageSize().
  int32 output_image_width;


  // Checks that all the configuration variables except output_image_height
  // and output_image_width have allowed values.
  void Check();

  // Computes output_image_height and output_image_width from the other
  // configuration values.
  void ComputeOutputImageSize();

  void Write(std::ostream &os, bool binary) const;

  // Note: Read() automatically calls ComputeOutputImageSize().
  void Read(std::istream &is, bool binary);

};


/**
   This object allows you to execute a convolution computation, and its backprop,
   on either a GPU (using CUDNN), or on CPU using a compatible interface.

   This object is quite lightweight: it only contains some structural data and a
   few smallish CUDNN descriptors that are derived from it.

*/
class ConvolutionComputation final {
public:
  // Note: you don't have to have done ComputeOutputImageSize() on 'config',
  // this class will do it in the constructor.
  ConvolutionComputation(const ConvolutionComputationConfig &config);

  // This constructor may be used prior to calling Read().
  ConvolutionComputation();

  const ConvolutionComputationConfig &Config() const { return config_; }

  ~ConvolutionComputation();

  /*
    For an explanation of the notation below (e.g. NWHC):

      N is equivalent to num_images
      C is equivalent to num_channels_in
      K is equivalent to num_channels_out
      H is equivalent to input_image_height or output_image_height (for images) or
         filter_height (for filter parameters).
      W is equivalent to input_image_width or output_image_width (for images) or
         filter_width (for filter parameters).
    and the order of letters is from highest to lowest stride, e.g in

    In NWHC, N would have the highest stride, and C a stride of 1.

    Caution: for convenience, given the way nnet3 works, we flip the notion of
    height and width that CUDNN uses, so our height is CUDNN's width, and vice
    versa.  This is not visible to the user; we mention it just in case
    those familiar with CUDNN get surprised at the order

      @param [in] input NWHC fully-packed tensor, with NumRows() == N * W
      @param [in] params KCWH fully-packed tensor, with NumRows() == K.
      @param [in] bias vector of length K
      @param [out] output Pre-allocated NWHK fully-packed tensor, with NumRows() == N * W.
   */
  void ConvolveForward(const CuMatrixBase<BaseFloat> &input,
                       const CuMatrixBase<BaseFloat> &params,
                       const CuVectorBase<BaseFloat> *bias,
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
                       const VectorBase<BaseFloat> *bias,
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




  void Write(std::ostream &os, bool binary) const { config_.Write(os, binary); }

  void Read(std::istream &is, bool binary);

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
  // when using the CPU).  params and params_rearranged must both be
  // packed (Stride() == NumCols()), params must have num-rows equal to K
  // (num_channels_out_), and params_rearranged must have num-rows equal
  // to to WH (filter_width_ * filter_height_).
  void ConvertParams(const MatrixBase<BaseFloat> &params,
                     MatrixBase<BaseFloat> *params_rearranged) const;
  // This function does the opposite transformation of what ConvertParams()
  // does.
  void ConvertParamsBack(const MatrixBase<BaseFloat> &params_rearranged,
                         MatrixBase<BaseFloat> *params) const;



  ConvolutionComputationConfig config_;


#if HAVE_CUDA == 1
  bool descriptors_initialized_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnFilterDescriptor_t params_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;

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
