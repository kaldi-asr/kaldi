// cudamatrix/cudnn-convlolution.h

// Copyright      2016  Johns Hopkins University (author: Daniel Povey)
//                2016  Yiming Wang

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONdITIONS OF ANY
// KINd, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONdITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_CUDAMATRIX_CUDNN_CONVOLUTION_H_
#define KALDI_CUDAMATRIX_CUDNN_CONVOLUTION_H_

#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
#include "cudamatrix/cu-matrix.h"
#include "cudnn.h"


namespace kaldi {
namespace cudnn {
  void ConvolutionForward(cudnnHandle_t handle,
                          const void *alpha,
                          const cudnnTensorDescriptor_t &x_desc,
                          const void *x_data,
                          const cudnnFilterDescriptor_t &w_desc,
                          const void *w_data,
                          const cudnnConvolutionDescriptor_t &conv_desc,
                          const cudnnConvolutionFwdAlgo_t &algo,
                          void *work_space,
                          size_t work_space_size_in_bytes,
                          const void *beta,
                          const cudnnTensorDescriptor_t &y_desc,
                          void *y_data);

  void ConvolutionBackwardData(cudnnHandle_t handle,
                               const void *alpha,
                               const cudnnFilterDescriptor_t &w_desc,
                               const void *w_data,
                               const cudnnTensorDescriptor_t &dy_desc,
                               const void *dy_data,
                               const cudnnConvolutionDescriptor_t &conv_desc,
                               const cudnnConvolutionBwdDataAlgo_t &algo,
                               void *work_space,
                               size_t work_space_size_in_bytes,
                               const void *beta,
                               const cudnnTensorDescriptor_t &dx_desc,
                               void *dx_data);

  void ConvolutionBackwardFilter(cudnnHandle_t handle,
                                 const void *alpha,
                                 const cudnnTensorDescriptor_t &x_desc,
                                 const void *x,
                                 const cudnnTensorDescriptor_t &dy_desc,
                                 const void *dy,
                                 const cudnnConvolutionDescriptor_t &conv_desc,
                                 const cudnnConvolutionBwdFilterAlgo_t &algo,
                                 void *work_space,
                                 size_t work_space_size_in_bytes,
                                 const void *beta,
                                 const cudnnFilterDescriptor_t &dw_desc,
                                 void *dw_data);

  void ConvolutionBackwardBias(cudnnHandle_t                       handle,
                               const void                         *alpha,
                               const cudnnTensorDescriptor_t       dy_desc,
                               const void                         *dy_data,
                               const void                         *beta,
                               const cudnnTensorDescriptor_t       db_desc,
                               void                               *db_data);

  void PoolingForward(cudnnHandle_t handle,
                      const cudnnPoolingDescriptor_t &pooling_desc,
                      const void *alpha,
                      const cudnnTensorDescriptor_t &x_desc,
                      const void *x_data,
                      const void *beta,
                      const cudnnTensorDescriptor_t &y_desc,
                      void *y_data);

  void PoolingBackward(const cudnnPoolingDescriptor_t &pooling_desc,
                       const void *alpha,
                       const cudnnTensorDescriptor_t &y_desc,
                       const void *y_data,
                       const cudnnTensorDescriptor_t &dy_desc,
                       const void *dy_data,
                       const cudnnTensorDescriptor_t &x_desc,
                       const void *x_data,
                       const void *beta,
                       const cudnnTensorDescriptor_t &dx_desc,
                       void *dx_data);

  void FindBestConvolutionFwdAlgo(const cudnnTensorDescriptor_t &x_desc,
                                  const cudnnFilterDescriptor_t &w_desc,
                                  const cudnnConvolutionDescriptor_t &conv_desc,
                                  const cudnnTensorDescriptor_t &y_desc,
                                  int requested_algo_count,
                                  cudnnConvolutionFwdAlgo_t *algo);

  void FindBestConvolutionBwdDataAlgo(const cudnnFilterDescriptor_t &w_desc,
                                      const cudnnTensorDescriptor_t &dy_desc,
                                      const cudnnConvolutionDescriptor_t &conv_desc,
                                      const cudnnTensorDescriptor_t &dx_desc,
                                      int requested_algo_count,
                                      cudnnConvolutionBwdDataAlgo_t *algo);

  void FindBestConvolutionBwdFilterAlgo(const cudnnTensorDescriptor_t &x_desc,
                                        const cudnnTensorDescriptor_t &dy_desc,
                                        const cudnnConvolutionDescriptor_t &conv_desc,
                                        const cudnnFilterDescriptor_t &dw_desc,
                                        int requested_algo_count,
                                        cudnnConvolutionBwdFilterAlgo_t *algo);

  void GetConvolutionFwdWorkspaceSize(const cudnnTensorDescriptor_t &x_desc,
                                      const cudnnFilterDescriptor_t &w_desc,
                                      const cudnnConvolutionDescriptor_t &conv_desc,
                                      const cudnnTensorDescriptor_t &y_desc,
                                      cudnnConvolutionFwdAlgo_t algo,
                                      size_t *size_in_bytes);

  void GetConvolutionBwdDataWorkspaceSize(const cudnnFilterDescriptor_t &w_desc,
                                          const cudnnTensorDescriptor_t &dy_desc,
                                          const cudnnConvolutionDescriptor_t &conv_desc,
                                          const cudnnTensorDescriptor_t &dx_desc,
                                          cudnnConvolutionBwdDataAlgo_t algo,
                                          size_t *size_in_bytes);

  void GetConvolutionBwdFilterWorkspaceSize(const cudnnTensorDescriptor_t &x_desc,
                                            const cudnnTensorDescriptor_t &dy_desc,
                                            const cudnnConvolutionDescriptor_t &conv_desc,
                                            const cudnnFilterDescriptor_t &dw_desc,
                                            cudnnConvolutionBwdFilterAlgo_t algo,
                                            size_t *size_in_bytes);
} // namespace cudnn
} // namespace kaldi
#endif // HAVE_CUDA && HAVE_CUDNN
#endif
