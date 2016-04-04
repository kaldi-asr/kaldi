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

template<typename Real> 
class CuDnnConvolution {
 public:
  void InitializeTensorDescriptor(size_t nbDims, MatrixIndexT dimA[],
      MatrixIndexT strideA[], cudnnTensorDescriptor_t *tensor);

  void InitializeFilterDescriptor(size_t nbDims, MatrixIndexT filterDimA[],
      cudnnFilterDescriptor_t *filter);

  void InitializeConvolutionDescriptor(size_t arrayLength, MatrixIndexT padA[],
      MatrixIndexT filterStrideA[], cudnnConvolutionMode_t mode,
      cudnnConvolutionDescriptor_t *conv);

  void InitializePoolingDescriptor(size_t nbDims, MatrixIndexT windowDimA[],
      MatrixIndexT paddingA[], MatrixIndexT strideA[], cudnnPoolingMode_t mode, 
      cudnnPoolingDescriptor_t *pool);

  void DestroyTensorDescriptor(cudnnTensorDescriptor_t tensor);

  void DestroyFilterDescriptor(cudnnFilterDescriptor_t filter);

  void DestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t conv);

  void DestroyPoolingDescriptor(cudnnPoolingDescriptor_t pool);

  void ConvolutionForward(const cudnnTensorDescriptor_t &xDesc,
                          const CuMatrixBase<Real> &x,
                          const cudnnFilterDescriptor_t &wDesc,
                          const CuMatrixBase<Real> &w,
                          const cudnnConvolutionDescriptor_t &convDesc,
                          const cudnnConvolutionFwdAlgo_t &algo,
                          CuMatrixBase<Real> *workSpace,
                          size_t workSpaceSizeInBytes,
                          const cudnnTensorDescriptor_t &yDesc,
                          CuMatrixBase<Real> *y);

  void ConvolutionBackwardData(const cudnnFilterDescriptor_t &wDesc,
                               const CuMatrixBase<Real> &w,
                               const cudnnTensorDescriptor_t &dyDesc,
                               const CuMatrixBase<Real> &dy,
                               const cudnnConvolutionDescriptor_t &convDesc,
                               const cudnnConvolutionBwdDataAlgo_t &algo,
                               CuMatrixBase<Real> *workSpace,
                               size_t workSpaceSizeInBytes,
                               const cudnnTensorDescriptor_t &dxDesc,
                               CuMatrixBase<Real> *dx);

  void ConvolutionBackwardFilter(const cudnnTensorDescriptor_t &xDesc,
                                 const CuMatrixBase<Real> &x,
                                 const cudnnTensorDescriptor_t &dyDesc,
                                 const CuMatrixBase<Real> &dy,
                                 const cudnnConvolutionDescriptor_t &convDesc,
                                 const cudnnConvolutionBwdFilterAlgo_t &algo,
                                 CuMatrixBase<Real> *workSpace,
                                 size_t workSpaceSizeInBytes,
                                 const cudnnFilterDescriptor_t &dwDesc,
                                 CuMatrixBase<Real> *dw);

  void GetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t &convDesc,
                                        const cudnnTensorDescriptor_t &inputTensorDesc,
                                        const cudnnFilterDescriptor_t &filterDesc,
                                        size_t nbDims, MatrixIndexT tensorOutputDimA[]);

  void PoolingForward(const cudnnPoolingDescriptor_t &poolingDesc,
                      const cudnnTensorDescriptor_t &xDesc,
                      const CuMatrixBase<Real> &x,
                      const cudnnTensorDescriptor_t &yDesc,
                      CuMatrixBase<Real> *y);

  void PoolingBackward(const cudnnPoolingDescriptor_t &poolingDesc,
                       const cudnnTensorDescriptor_t &yDesc,
                       const CuMatrixBase<Real> &y,
                       const cudnnTensorDescriptor_t &dyDesc,
                       const CuMatrixBase<Real> &dy,
                       const cudnnTensorDescriptor_t &xDesc,
                       const CuMatrixBase<Real> &x,
                       const cudnnTensorDescriptor_t &dxDesc,
                       CuMatrixBase<Real> *dx);

  void GetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t &poolDesc,
                                    const cudnnTensorDescriptor_t &inputDesc,
                                    size_t nbDims, MatrixIndexT OutDimA[]);

  void FindBestConvolutionFwdAlgo(const cudnnTensorDescriptor_t &xDesc,
                                  const cudnnFilterDescriptor_t &wDesc,
                                  const cudnnConvolutionDescriptor_t &convDesc,
                                  const cudnnTensorDescriptor_t &yDesc,
                                  int requestedAlgoCount,
                                  cudnnConvolutionFwdAlgo_t *algo);

  void FindBestConvolutionBwdDataAlgo(const cudnnFilterDescriptor_t &wDesc,
                                      const cudnnTensorDescriptor_t &dyDesc,
                                      const cudnnConvolutionDescriptor_t &convDesc,
                                      const cudnnTensorDescriptor_t &dxDesc,
                                      int requestedAlgoCount,
                                      cudnnConvolutionBwdDataAlgo_t *algo);

  void FindBestConvolutionBwdFilterAlgo(const cudnnTensorDescriptor_t &xDesc,
                                        const cudnnTensorDescriptor_t &dyDesc,
                                        const cudnnConvolutionDescriptor_t &convDesc,
                                        const cudnnFilterDescriptor_t &dwDesc,
                                        int requestedAlgoCount,
                                        cudnnConvolutionBwdFilterAlgo_t *algo);

  void GetConvolutionFwdWorkspaceSize(const cudnnTensorDescriptor_t &xDesc,
                                      const cudnnFilterDescriptor_t &wDesc,
                                      const cudnnConvolutionDescriptor_t &convDesc,
                                      const cudnnTensorDescriptor_t &yDesc,
                                      cudnnConvolutionFwdAlgo_t algo,
                                      size_t *sizeInBytes);

  void GetConvolutionBwdDataWorkspaceSize(const cudnnFilterDescriptor_t &wDesc,
                                          const cudnnTensorDescriptor_t &dyDesc,
                                          const cudnnConvolutionDescriptor_t &convDesc,
                                          const cudnnTensorDescriptor_t &dxDesc,
                                          cudnnConvolutionBwdDataAlgo_t algo,
                                          size_t *sizeInBytes);

  void GetConvolutionBwdFilterWorkspaceSize(const cudnnTensorDescriptor_t &xDesc,
                                            const cudnnTensorDescriptor_t &dyDesc,
                                            const cudnnConvolutionDescriptor_t &convDesc,
                                            const cudnnFilterDescriptor_t &dwDesc,
                                            cudnnConvolutionBwdFilterAlgo_t algo,
                                            size_t *sizeInBytes);



 private:
  static const Real one_;
  static const Real zero_;

  inline cudnnDataType_t ConvertType() const {
    if (sizeof(Real) == sizeof(float))
      return CUDNN_DATA_FLOAT;
    else if (sizeof(Real) == sizeof(double))
      return CUDNN_DATA_DOUBLE;
    else {
      KALDI_ERR << "Unsupported type.";
      return CUDNN_DATA_FLOAT;
    }
  }

};

}  // namespace

#endif // HAVE_CUDA && HAVE_CUDNN
#endif
