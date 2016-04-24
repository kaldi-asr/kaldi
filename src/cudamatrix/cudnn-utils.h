// TODO: License header

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

  } // end namespace cudnn
} // end namespace kaldi
#endif
#endif
