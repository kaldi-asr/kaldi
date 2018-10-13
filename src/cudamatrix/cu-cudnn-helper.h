// cudamatrix/cu-cudnn-helper.h

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

#ifndef KALDI_CUDAMATRIX_CU_CUDNN_HELPER_H_
#define KALDI_CUDAMATRIX_CU_CUDNN_HELPER_H_

#if HAVE_CUDA == 1
#include <cudnn.h>
#else
typedef struct cudnnTensorStruct*          cudnnTensorDescriptor_t;
typedef struct cudnnConvolutionStruct*     cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct*         cudnnPoolingDescriptor_t;
typedef struct cudnnFilterStruct*          cudnnFilterDescriptor_t;
typedef struct cudnnLRNStruct*             cudnnLRNDescriptor_t;
typedef struct cudnnActivationStruct*      cudnnActivationDescriptor_t;
typedef struct cudnnSpatialTransformerStruct* cudnnSpatialTransformerDescriptor_t;
typedef struct cudnnOpTensorStruct*        cudnnOpTensorDescriptor_t;
typedef struct cudnnReduceTensorStruct*    cudnnReduceTensorDescriptor_t;
typedef struct cudnnCTCLossStruct*         cudnnCTCLossDescriptor_t;

typedef enum {} cudnnConvolutionBwdDataAlgo_t;
typedef enum {} cudnnConvolutionBwdFilterAlgo_t;
typedef enum {} cudnnConvolutionFwdAlgo_t;
#endif // HAVE_CUDA == 1

#endif // KALDI_CUDAMATRIX_CU_CUDNN_HELPER_H_
