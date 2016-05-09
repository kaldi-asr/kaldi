// cudamatrix/cudnn-convolution.cc

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
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
#include "cudamatrix/cudnn-convolution.h"
#include "cudamatrix/cu-device.h"

namespace kaldi {
namespace cudnn {

void ConvolutionForward(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t &x_desc, const void *x_data,
    const cudnnFilterDescriptor_t &w_desc, const void *w_data,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnConvolutionFwdAlgo_t &algo,
    void *work_space, size_t work_space_size_in_bytes,
    const void *beta,
    const cudnnTensorDescriptor_t &y_desc, void *y_data) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnConvolutionForward(handle, alpha, x_desc,
                 x_data, w_desc, w_data, conv_desc, algo, work_space,
                 work_space_size_in_bytes, beta, y_desc, y_data));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void ConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t &w_desc, const void *w_data,
    const cudnnTensorDescriptor_t &dy_desc, const void *dy_data,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnConvolutionBwdDataAlgo_t &algo,
    void *work_space, size_t work_space_size_in_bytes,
    const void *beta, const cudnnTensorDescriptor_t &dx_desc, void *dx_data) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardData(handle, alpha, w_desc,
                                              w_data, dy_desc, dy_data,
                                              conv_desc, algo,
                                              work_space,
                                              work_space_size_in_bytes, beta,
                                              dx_desc, dx_data));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void ConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t &x_desc, const void *x_data,
    const cudnnTensorDescriptor_t &dy_desc, const void *dy_data,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnConvolutionBwdFilterAlgo_t &algo,
    void *work_space, size_t work_space_size_in_bytes,
    const void *beta, const cudnnFilterDescriptor_t &dw_desc, void *dw_data) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardFilter(handle, alpha, x_desc,
                                                x_data, dy_desc, dy_data,
                                                conv_desc, algo,
                                                work_space,
                                                work_space_size_in_bytes, beta,
                                                dw_desc, dw_data));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

  void ConvolutionBackwardBias(cudnnHandle_t                       handle,
                               const void                         *alpha,
                               const cudnnTensorDescriptor_t       dy_desc,
                               const void                         *dy_data,
                               const void                         *beta,
                               const cudnnTensorDescriptor_t       db_desc,
                               void                               *db_data) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
    if (CuDevice::Instantiate().Enabled()) {
      Timer tim;
      CUDNN_SAFE_CALL(cudnnConvolutionBackwardBias(handle, alpha,
                                                   dy_desc, dy_data,
                                                   beta, db_desc, db_data
                                                   ));
      CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
    }
#endif
  }


void PoolingForward(
    const cudnnPoolingDescriptor_t &pooling_desc,
    const void *alpha,
    const cudnnTensorDescriptor_t &x_desc, const void *x_data,
    const void *beta,
    const cudnnTensorDescriptor_t &y_desc, void *y_data) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnPoolingForward(GetCudnnHandle(), pooling_desc, alpha,
                                      x_desc, x_data, beta, y_desc,
                                      y_data));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void PoolingBackward(
    const cudnnPoolingDescriptor_t &pooling_desc,
    const void *alpha,
    const cudnnTensorDescriptor_t &y_desc, const void *y_data,
    const cudnnTensorDescriptor_t &dy_desc, const void *dy_data,
    const cudnnTensorDescriptor_t &x_desc, const void *x_data,
    const void *beta,
    const cudnnTensorDescriptor_t &dx_desc, void *dx_data) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnPoolingBackward(GetCudnnHandle(), pooling_desc, alpha,
                                      y_desc, y_data,  dy_desc, dy_data,
                                      x_desc, x_data, beta, dx_desc,
                                      dx_data));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void FindBestConvolutionFwdAlgo(
    const cudnnTensorDescriptor_t &x_desc, const cudnnFilterDescriptor_t &w_desc,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnTensorDescriptor_t &y_desc,
    int requested_algo_count, cudnnConvolutionFwdAlgo_t *algo) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    cudnnConvolutionFwdAlgoPerf_t *perfResults =
        new cudnnConvolutionFwdAlgoPerf_t[requested_algo_count];
    int returnedAlgoCount = 0;
    Timer tim;
    CUDNN_SAFE_CALL(cudnnFindConvolutionForwardAlgorithm(GetCudnnHandle(), x_desc,
                                                      w_desc, conv_desc, y_desc,
                                                      requested_algo_count,
                                                      &returnedAlgoCount,
                                                      perfResults));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
    KALDI_ASSERT(returnedAlgoCount > 0);
    for (int i = 0; i < returnedAlgoCount; i++) {
      if (perfResults[i].status == CUDNN_STATUS_SUCCESS) {
        *algo = perfResults[i].algo;
        return;
      }
    }
    delete perfResults;
  }
#endif
}

void FindBestConvolutionBwdDataAlgo(
    const cudnnFilterDescriptor_t &w_desc, const cudnnTensorDescriptor_t &dy_desc,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnTensorDescriptor_t &dx_desc,
    int requested_algo_count, cudnnConvolutionBwdDataAlgo_t *algo) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults =
        new cudnnConvolutionBwdDataAlgoPerf_t[requested_algo_count];
    int returnedAlgoCount = 0;
    Timer tim;
    CUDNN_SAFE_CALL(cudnnFindConvolutionBackwardDataAlgorithm(GetCudnnHandle(),
                                                           w_desc, dy_desc,
                                                           conv_desc, dx_desc,
                                                           requested_algo_count,
                                                           &returnedAlgoCount,
                                                           perfResults));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
    KALDI_ASSERT(returnedAlgoCount > 0);
    for (int i = 0; i < returnedAlgoCount; i++) {
      if (perfResults[i].status == CUDNN_STATUS_SUCCESS) {
        *algo = perfResults[i].algo;
        return;
      }
    }
    delete perfResults;
  }
#endif
}

void FindBestConvolutionBwdFilterAlgo(
    const cudnnTensorDescriptor_t &x_desc, const cudnnTensorDescriptor_t &dy_desc,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnFilterDescriptor_t &dw_desc,
    int requested_algo_count, cudnnConvolutionBwdFilterAlgo_t *algo) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults =
        new cudnnConvolutionBwdFilterAlgoPerf_t[requested_algo_count];
    int returnedAlgoCount = 0;
    Timer tim;
    CUDNN_SAFE_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(GetCudnnHandle(),
                                                             x_desc, dy_desc,
                                                             conv_desc, dw_desc,
                                                             requested_algo_count,
                                                             &returnedAlgoCount,
                                                             perfResults));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
    KALDI_ASSERT(returnedAlgoCount > 0);
    for (int i = 0; i < returnedAlgoCount; i++) {
      if (perfResults[i].status == CUDNN_STATUS_SUCCESS) {
        *algo = perfResults[i].algo;
        return;
      }
    }
    delete perfResults;
  }
#endif
}

void GetConvolutionBwdDataWorkspaceSize(
    const cudnnFilterDescriptor_t &w_desc, const cudnnTensorDescriptor_t &dy_desc,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnTensorDescriptor_t &dx_desc,
    cudnnConvolutionBwdDataAlgo_t algo, size_t *size_in_bytes) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(GetCudnnHandle(),
                                                              w_desc, dy_desc,
                                                              conv_desc, dx_desc,
                                                              algo,
                                                              size_in_bytes));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

void GetConvolutionBwdFilterWorkspaceSize(
    const cudnnTensorDescriptor_t &x_desc, const cudnnTensorDescriptor_t &dy_desc,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnFilterDescriptor_t &dw_desc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *size_in_bytes) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(GetCudnnHandle(),
                                                                x_desc, dy_desc,
                                                                conv_desc, dw_desc,
                                                                algo,
                                                                size_in_bytes));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

} //namespace cudnn
} // namespace kaldi

#endif //HAVE_CUDA && HAVE_CUDNN
