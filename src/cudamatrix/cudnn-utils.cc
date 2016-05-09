// cudamatrix/cudnn-utils.h

// Copyright 2016  Daniel Galvez

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

#if HAVE_CUDNN == 1
#include "cudnn-utils.h"

namespace kaldi {
  namespace cudnn {
    cudnnTensorDescriptor_t CopyTensorDesc(const cudnnTensorDescriptor_t tensor_desc) {
      cudnnTensorDescriptor_t copy_desc;
      CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&copy_desc));

      cudnnDataType_t float_type;
      int32 numDimensions;
      int32 tensor_dims[CUDNN_DIM_MAX];
      int32 tensor_strides[CUDNN_DIM_MAX];
      CUDNN_SAFE_CALL(cudnnGetTensorNdDescriptor(tensor_desc,
                                                 CUDNN_DIM_MAX,
                                                 &float_type,
                                                 &numDimensions,
                                                 tensor_dims,
                                                 tensor_strides)
                      );
      KALDI_ASSERT(float_type == cudnn::GetDataType());

      CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(copy_desc,
                                                 float_type,
                                                 numDimensions,
                                                 tensor_dims,
                                                 tensor_strides
                                                 )
                      );

      return copy_desc;
    }

    cudnnFilterDescriptor_t CopyFilterDesc(const cudnnFilterDescriptor_t filter_desc) {
      cudnnFilterDescriptor_t copy_desc;
      CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&copy_desc));

      cudnnDataType_t float_type;
      int32 numDimensions;
      int32 filter_dims[CUDNN_DIM_MAX];
      CUDNN_SAFE_CALL(cudnnGetFilterNdDescriptor(filter_desc,
                                                 CUDNN_DIM_MAX,
                                                 &float_type,
                                                 &numDimensions,
                                                 filter_dims)
                      );
      KALDI_ASSERT(float_type == cudnn::GetDataType());

      CUDNN_SAFE_CALL(cudnnSetFilterNdDescriptor(copy_desc,
                                                 float_type,
                                                 numDimensions,
                                                 filter_dims
                                                 )
                      );

      return copy_desc;
    }

    cudnnConvolutionDescriptor_t CopyConvolutionDesc(const cudnnConvolutionDescriptor_t conv_desc) {
      cudnnConvolutionDescriptor_t copy_desc;
      CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&copy_desc));

      int32 numDimensions;
      int32 pad_dims[CUDNN_DIM_MAX];
      int32 stride_dims[CUDNN_DIM_MAX];
      int32 upscale_dims[CUDNN_DIM_MAX];
      cudnnConvolutionMode_t mode;
      cudnnDataType_t float_type;
      CUDNN_SAFE_CALL(cudnnGetConvolutionNdDescriptor(conv_desc,
                                                      // Documentation claims that 
                                                      // CUDNN_DIM_MAX is an acceptable
                                                      // number of dimensions to request,
                                                      // yet it returns as its status
                                                      // CUDNN_STATUS_NOT_SUPPORTED
                                                      CUDNN_DIM_MAX - 4,
                                                      &numDimensions,
                                                      pad_dims,
                                                      stride_dims,
                                                      upscale_dims,
                                                      &mode,
                                                      &float_type)
                      );
      KALDI_ASSERT(float_type == cudnn::GetDataType());

      CUDNN_SAFE_CALL(cudnnSetConvolutionNdDescriptor(copy_desc,
                                                      numDimensions,
                                                      pad_dims,
                                                      stride_dims,
                                                      upscale_dims,
                                                      mode,
                                                      float_type
                                                      )
                      );

      return copy_desc;
    }
  }
}
#endif // HAVE_CUDNN
