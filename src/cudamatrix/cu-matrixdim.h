#ifndef KALDI_CUDAMATRIX_CU_MATRIXDIM_H_
#define KALDI_CUDAMATRIX_CU_MATRIXDIM_H_

extern "C" {
  typedef struct MatrixDim_ {
    int rows;
    int cols;
    int stride;
  } MatrixDim;
}

#endif
