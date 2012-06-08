#ifndef KALDI_CUDAMATRIX_CU_MATRIXDIM_H_
#define KALDI_CUDAMATRIX_CU_MATRIXDIM_H_


#ifdef _MSC_VER
  typedef unsigned __int32 uint32_cuda;
  typedef __int32          int32_cuda;
#else
  #include <stdint.h>
  typedef uint32_t         uint32_cuda;
  typedef int32_t          int32_cuda;
#endif


extern "C" {
  typedef struct MatrixDim_ {
    int32_cuda rows;
    int32_cuda cols;
    int32_cuda stride;
  } MatrixDim;
}

#endif
