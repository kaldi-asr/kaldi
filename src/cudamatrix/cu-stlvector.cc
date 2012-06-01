
#include "cudamatrix/cu-stlvector.h"

#if HAVE_CUDA==1
  #include "cudamatrix/cu-kernels.h"
#endif

namespace kaldi {


template<> void CuStlVector<int32>::Set(int32 value) {
  #if HAVE_CUDA==1
  if (CuDevice::Instantiate().Enabled()) { 
    Timer tim;

    dim3 dimBlock(CUBLOCK);
    dim3 dimGrid(n_blocks(Dim(), CUBLOCK));
    ::MatrixDim d = { 1, Dim(), Dim() };

    cudaI32_set_const(dimGrid, dimBlock, data_, value, d);
    cuSafeCall(cudaGetLastError());

    CuDevice::Instantiate().AccuProfile(__func__, tim.Elapsed());
  } else
  #endif
  {
    vec_.assign(vec_.size(), value);
  }
}

}// namespace kaldi
