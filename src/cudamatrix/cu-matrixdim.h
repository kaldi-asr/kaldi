#ifndef KALDI_CUDAMATRIX_CU_MATRIXDIM_H_
#define KALDI_CUDAMATRIX_CU_MATRIXDIM_H_


#ifdef _MSC_VER
  typedef __int32          int32_cuda;
#else
  #include <stdint.h>
  typedef int32_t          int32_cuda;
#endif


extern "C" {
  typedef struct MatrixDim_ {
    int rows;
    int cols;
    int stride;
  } MatrixDim;
}

#endif
