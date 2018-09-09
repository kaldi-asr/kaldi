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

#include <cudnn.h>

namespace kaldi {
namespace nnet3 {
namespace cudnn {

class ConvolutionComputation {
public:
  ConvolutionComputation(int32 num_channels_out, int32 num_channels_in,
                         int32 filter_height, int32 filter_width,
                         int32 filter_stride_height, int32 filter_stride_width,
                         // dilation?
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
   * bytes (not 32-bit words).
   */
  size_t TempSpaceRequiredForward() const;
  size_t TempSpaceRequiredBackwardData() const;
  size_t TempSpaceRequiredBackwardFilter() const;

  // Why aren't these const methods? That would make things a lot simpler
  void ConvolveForward(const CuMatrixBase<BaseFloat> &input,
                       const CuMatrixBase<BaseFloat> &params,
                       const CuVectorBase<BaseFloat> &bias,
                       CuVectorBase<BaseFloat> *temp_space,
                       CuMatrixBase<BaseFloat> *output) const;

  // Why aren't these const methods? That would make things a lot simpler
  void ConvolveBackwardData(const CuMatrixBase<BaseFloat> &params,
                            const CuMatrixBase<BaseFloat> &output_deriv,
                            CuVectorBase<BaseFloat> *temp,
                            CuMatrixBase<BaseFloat> *input_deriv) const;

  // Why aren't these const methods? That would make things a lot simpler
  void ConvolveBackwardParams(const CuMatrixBase<BaseFloat> &output_deriv,
                              const CuMatrixBase<BaseFloat> &input,
                              BaseFloat alpha,
                              CuVectorBase<BaseFloat> *temp,
                              CuMatrixBase<BaseFloat> *params_deriv) const;

  // Why aren't these const methods? That would make things a lot simpler
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

/* /\** */
/*    This function does the compilation for a convolution computation; it's */
/*    a wrapper for the functions below, which should not have to be called */
/*    by the end user. */

/*    @param [in] model  The convolution model that this computation is for. */
/*    @param [in] input_indexes   The list of Indexes available at the input of */
/*                       the computation. */
/*    @param [in] output_indexes  The list of Indexes requested to be computed */
/*                       at the output of the computation.  It is an error if */
/*                       all dependencies are not satisfied (specifically: for */
/*                       each Index (n,t,x) in 'output_indexes', the Index */
/*                       (n,t+time_offset,x) must be present in 'input_indexes' */
/*                       for each time_offset in model.required_time_offsets. */
/*    @param [out] computation  If non-NULL, the compiled computation will be */
/*                       written to this location. */

/*  *\/ */
/* void CompileConvolutionComputation( */
/*     const ConvolutionModel& model, */
/*     const std::vector<Index> &input_indexes, */
/*     const std::vector<Index> &output_indexes, */
/*     const ConvolutionComputationOptions &opts, */
/*     cudnn::ConvolutionComputation *computation, */
/*     std::vector<Index> *input_indexes_modified, */
/*     std::vector<Index> *output_indexes_modified); */

} // namespace cudnn
} // namespace nnet3
} // namespace kaldi

#endif // KALDI_NNET3_NNET_CUDNN_CONVOLUTION_H_
