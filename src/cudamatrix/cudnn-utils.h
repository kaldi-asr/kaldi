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

#ifndef KALDI_CUDAMATRIX_CUDNN_UTILS_H_
#define KALDI_CUDAMATRIX_CUDNN_UTILS_H_
#if HAVE_CUDNN == 1
#include "base/kaldi-types.h"
#include "cudamatrix/cu-common.h"
#include <cudnn.h>

namespace kaldi {
  namespace cudnn {

    const BaseFloat one  = 1;
    const BaseFloat zero = 0;

    // is_same() is defined in the the ++ standard library, but only in C+11
    // and onward. We need these because we cannot assume that users have a
    // C++11 compiler.
    template<class T, class U>
      struct is_same {
        enum { value = 0 };
      };

    template<class T>
      struct is_same<T, T> {
      enum { value = 1 };
    };

    inline cudnnDataType_t GetDataType() {
      if (is_same<BaseFloat, float>::value)
        return CUDNN_DATA_FLOAT;
      else if (is_same<BaseFloat, double>::value)
        return CUDNN_DATA_DOUBLE;
      else {
        KALDI_ERR << "Unsupported type.";
        return CUDNN_DATA_FLOAT;
      }
    }

    /**
     * Creates a copy of the input descriptor. It is the caller's responsibility
     * to call cudnnDestroyTensorDescriptor() on the output copy when finished.
     */
    cudnnTensorDescriptor_t CopyTensorDesc(const cudnnTensorDescriptor_t tensor_desc);
    /**
     * Creates a copy of the input descriptor. It is the caller's responsibility
     * to call cudnnDestroyFilterDescriptor() on the output copy when finished.
     */
    cudnnFilterDescriptor_t CopyFilterDesc(const cudnnFilterDescriptor_t filter_desc);
    /**
     * Creates a copy of the input descriptor. It is the caller's responsibility
     * to call cudnnDestroyConvolutionDescriptor() on the output copy when
     * finished.
     */
    cudnnConvolutionDescriptor_t CopyConvolutionDesc(const cudnnConvolutionDescriptor_t conv_desc);

  } // end namespace cudnn
} // end namespace kaldi
#endif // HAVE_CUDNN
#endif // KALDI_CUDAMATRIX_CUDNN_UTILS_H_
