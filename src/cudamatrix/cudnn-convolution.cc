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

template<typename Real> const Real CuDnnConvolution<Real>::one_ = 1;
template<typename Real> const Real CuDnnConvolution<Real>::zero_ = 0;

template<typename Real>
void CuDnnConvolution<Real>::InitializeTensorDescriptor(size_t nbDims,
    MatrixIndexT dimA[], MatrixIndexT strideA[],
    cudnnTensorDescriptor_t *tensor) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(tensor));
    CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(*tensor, ConvertType(), nbDims,
                                            dimA, strideA));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
    KALDI_ERR << "Not implemented for CPU.";
}

template<typename Real>
void CuDnnConvolution<Real>::InitializeFilterDescriptor(size_t nbDims,
    MatrixIndexT filterDimA[], cudnnFilterDescriptor_t *filter) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(filter));
    CUDNN_SAFE_CALL(cudnnSetFilterNdDescriptor(*filter, ConvertType(), nbDims,
                                            filterDimA));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
    KALDI_ERR << "Not implemented for CPU.";
}

template<typename Real>
void CuDnnConvolution<Real>::InitializeConvolutionDescriptor(
    size_t arrayLength, MatrixIndexT padA[], MatrixIndexT filterStrideA[],
    cudnnConvolutionMode_t mode, cudnnConvolutionDescriptor_t *conv) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    int *upscaleA = new int[arrayLength];
    for (size_t i = 0; i < arrayLength; i++)
      upscaleA[i] = 1;
    Timer tim;
    CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(conv));
    CUDNN_SAFE_CALL(cudnnSetConvolutionNdDescriptor(*conv, arrayLength, padA,
                                                 filterStrideA, upscaleA, mode,
                                                 ConvertType()));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
    delete upscaleA;
  } else
#endif
    KALDI_ERR << "Not implemented for CPU.";
}

template<typename Real>
void CuDnnConvolution<Real>::InitializePoolingDescriptor(
    size_t nbDims, MatrixIndexT windowDimA[], MatrixIndexT paddingA[],
    MatrixIndexT strideA[], cudnnPoolingMode_t mode,
    cudnnPoolingDescriptor_t *pool) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnCreatePoolingDescriptor(pool));
    CUDNN_SAFE_CALL(cudnnSetPoolingNdDescriptor(*pool, mode, nbDims, windowDimA,
                                             paddingA, strideA));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
    KALDI_ERR << "Not implemented for CPU.";
}

template<typename Real>
void CuDnnConvolution<Real>::DestroyTensorDescriptor(
    cudnnTensorDescriptor_t tensor) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(tensor));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real>
void CuDnnConvolution<Real>::DestroyFilterDescriptor(
    cudnnFilterDescriptor_t filter) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(filter));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real>
void CuDnnConvolution<Real>::DestroyConvolutionDescriptor(
    cudnnConvolutionDescriptor_t conv) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real>
void CuDnnConvolution<Real>::DestroyPoolingDescriptor(
    cudnnPoolingDescriptor_t pool) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnDestroyPoolingDescriptor(pool));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real>
void CuDnnConvolution<Real>::ConvolutionForward(
    const cudnnTensorDescriptor_t &xDesc, const CuMatrixBase<Real> &x,
    const cudnnFilterDescriptor_t &wDesc, const CuMatrixBase<Real> &w,
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnConvolutionFwdAlgo_t &algo,
    CuMatrixBase<Real> *workSpace, size_t workSpaceSizeInBytes,
    const cudnnTensorDescriptor_t &yDesc, CuMatrixBase<Real> *y) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnConvolutionForward(GetCudnnHandle(), &one_, xDesc,
                 x.Data(), wDesc, w.Data(), convDesc, algo, workSpace->Data(),
                 workSpaceSizeInBytes, &zero_, yDesc, y->Data()));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
#endif
    KALDI_ERR << "Not implemented for CPU.";
}

template<typename Real>
void CuDnnConvolution<Real>::ConvolutionBackwardData(
    const cudnnFilterDescriptor_t &wDesc, const CuMatrixBase<Real> &w,
    const cudnnTensorDescriptor_t &dyDesc, const CuMatrixBase<Real> &dy,
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnConvolutionBwdDataAlgo_t &algo,
    CuMatrixBase<Real> *workSpace, size_t workSpaceSizeInBytes,
    const cudnnTensorDescriptor_t &dxDesc, CuMatrixBase<Real> *dx) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardData(GetCudnnHandle(), &one_, wDesc,
                                              w.Data(), dyDesc, dy.Data(),
                                              convDesc, algo,
                                              workSpace->Data(),
                                              workSpaceSizeInBytes, &zero_,
                                              dxDesc, dx->Data()));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real>
void CuDnnConvolution<Real>::ConvolutionBackwardFilter(
    const cudnnTensorDescriptor_t &xDesc, const CuMatrixBase<Real> &x,
    const cudnnTensorDescriptor_t &dyDesc, const CuMatrixBase<Real> &dy,
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnConvolutionBwdFilterAlgo_t &algo,
    CuMatrixBase<Real> *workSpace, size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t &dwDesc, CuMatrixBase<Real> *dw) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardFilter(GetCudnnHandle(), &one_, xDesc,
                                                x.Data(), dyDesc, dy.Data(),
                                                convDesc, algo,
                                                workSpace->Data(),
                                                workSpaceSizeInBytes, &zero_,
                                                dwDesc, dw->Data()));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real>
void CuDnnConvolution<Real>::GetConvolutionNdForwardOutputDim(
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnTensorDescriptor_t &inputTensorDesc,
    const cudnnFilterDescriptor_t &filterDesc,
    size_t nbDims, MatrixIndexT tensorOutputDimA[]) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                       inputTensorDesc,
                                                       filterDesc,
                                                       nbDims,
                                                       tensorOutputDimA));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real>
void CuDnnConvolution<Real>::PoolingForward(
    const cudnnPoolingDescriptor_t &poolingDesc,
    const cudnnTensorDescriptor_t &xDesc, const CuMatrixBase<Real> &x,
    const cudnnTensorDescriptor_t &yDesc, CuMatrixBase<Real> *y) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnPoolingForward(GetCudnnHandle(), poolingDesc, &one_,
                                      xDesc, x.Data(), &zero_, yDesc,
                                      y->Data()));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real>
void CuDnnConvolution<Real>::PoolingBackward(
    const cudnnPoolingDescriptor_t &poolingDesc,
    const cudnnTensorDescriptor_t &yDesc, const CuMatrixBase<Real> &y,
    const cudnnTensorDescriptor_t &dyDesc, const CuMatrixBase<Real> &dy,
    const cudnnTensorDescriptor_t &xDesc, const CuMatrixBase<Real> &x,
    const cudnnTensorDescriptor_t &dxDesc, CuMatrixBase<Real> *dx) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnPoolingBackward(GetCudnnHandle(), poolingDesc, &one_,
                                      yDesc, y.Data(),  dyDesc, dy.Data(),
                                      xDesc, x.Data(), &zero_, dxDesc,
                                      dx->Data()));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real>
void CuDnnConvolution<Real>::GetPoolingNdForwardOutputDim(
    const cudnnPoolingDescriptor_t &poolDesc,
    const cudnnTensorDescriptor_t &inputDesc,
    size_t nbDims, MatrixIndexT OutDimA[]) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetPoolingNdForwardOutputDim(poolDesc,
                                                   inputDesc, nbDims, OutDimA));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif
}

template<typename Real> 
void CuDnnConvolution<Real>::FindBestConvolutionFwdAlgo(
    const cudnnTensorDescriptor_t &xDesc, const cudnnFilterDescriptor_t &wDesc,
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnTensorDescriptor_t &yDesc,
    int requestedAlgoCount, cudnnConvolutionFwdAlgo_t *algo) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    cudnnConvolutionFwdAlgoPerf_t *perfResults =
        new cudnnConvolutionFwdAlgoPerf_t[requestedAlgoCount];
    int returnedAlgoCount = 0;
    Timer tim;
    CUDNN_SAFE_CALL(cudnnFindConvolutionForwardAlgorithm(GetCudnnHandle(), xDesc,
                                                      wDesc, convDesc, yDesc,
                                                      requestedAlgoCount,
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

template<typename Real>
void CuDnnConvolution<Real>::FindBestConvolutionBwdDataAlgo(
    const cudnnFilterDescriptor_t &wDesc, const cudnnTensorDescriptor_t &dyDesc,
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnTensorDescriptor_t &dxDesc,
    int requestedAlgoCount, cudnnConvolutionBwdDataAlgo_t *algo) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults =
        new cudnnConvolutionBwdDataAlgoPerf_t[requestedAlgoCount];
    int returnedAlgoCount = 0;
    Timer tim;
    CUDNN_SAFE_CALL(cudnnFindConvolutionBackwardDataAlgorithm(GetCudnnHandle(),
                                                           wDesc, dyDesc,
                                                           convDesc, dxDesc,
                                                           requestedAlgoCount,
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

template<typename Real> 
void CuDnnConvolution<Real>::FindBestConvolutionBwdFilterAlgo(
    const cudnnTensorDescriptor_t &xDesc, const cudnnTensorDescriptor_t &dyDesc,
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnFilterDescriptor_t &dwDesc,
    int requestedAlgoCount, cudnnConvolutionBwdFilterAlgo_t *algo) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults =
        new cudnnConvolutionBwdFilterAlgoPerf_t[requestedAlgoCount];
    int returnedAlgoCount = 0;
    Timer tim;
    CUDNN_SAFE_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(GetCudnnHandle(),
                                                             xDesc, dyDesc,
                                                             convDesc, dwDesc,
                                                             requestedAlgoCount,
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

template<typename Real>
void CuDnnConvolution<Real>::GetConvolutionFwdWorkspaceSize(
    const cudnnTensorDescriptor_t &xDesc,
    const cudnnFilterDescriptor_t &wDesc,
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnTensorDescriptor_t &yDesc,
    cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(GetCudnnHandle(),xDesc,
                                                         wDesc, convDesc, yDesc,
                                                         algo, sizeInBytes));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif 
}

template<typename Real>
void CuDnnConvolution<Real>::GetConvolutionBwdDataWorkspaceSize(
    const cudnnFilterDescriptor_t &wDesc, const cudnnTensorDescriptor_t &dyDesc,
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnTensorDescriptor_t &dxDesc,
    cudnnConvolutionBwdDataAlgo_t algo, size_t *sizeInBytes) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(GetCudnnHandle(),
                                                              wDesc, dyDesc,
                                                              convDesc, dxDesc,
                                                              algo,
                                                              sizeInBytes));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif 
}

template<typename Real>
void CuDnnConvolution<Real>::GetConvolutionBwdFilterWorkspaceSize(
    const cudnnTensorDescriptor_t &xDesc, const cudnnTensorDescriptor_t &dyDesc,
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnFilterDescriptor_t &dwDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
#if HAVE_CUDA == 1 && HAVE_CUDNN == 1
  if (CuDevice::Instantiate().Enabled()) {
    Timer tim;
    CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(GetCudnnHandle(),
                                                                xDesc, dyDesc,
                                                                convDesc, dwDesc,
                                                                algo,
                                                                sizeInBytes));
    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  }
#endif 
}





// Instantiate classes CuDnnConvolution for float and double.
template class CuDnnConvolution<float>;
template class CuDnnConvolution<double>;

} // namespace kaldi 

#endif //HAVE_CUDA && HAVE_CUDNN 
