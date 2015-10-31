// ctc/cctc-kernels.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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



#ifndef KALDI_CTC_CCTC_KERNELS_H_
#define KALDI_CTC_CCTC_KERNELS_H_

#if HAVE_CUDA == 1

#include "base/kaldi-error.h"
#include "ctc/cctc-kernels-ansi.h"

/*
 * In this file are C++ wrappers of the ANSI-C CUDA kernels
 */

namespace kaldi {


inline void cuda_rearrange_3d_tensor(dim3 Gr, dim3 Bl, int32_cuda xdim,
                                      int32_cuda xstride_in, int32_cuda ystride_in,
                                      int32_cuda zstride_in, int32_cuda xstride_out,
                                      int32_cuda ystride_out, int32_cuda zstride_out,
                                      const float *src, float *dst) {
  cudaF_rearrange_3d_tensor(Gr, Bl, xdim, xstride_in, ystride_in, zstride_in,
                            xstride_out, ystride_out, zstride_out, src, dst);
}

inline void cuda_rearrange_3d_tensor(dim3 Gr, dim3 Bl, int32_cuda xdim,
                                     int32_cuda xstride_in, int32_cuda ystride_in,
                                     int32_cuda zstride_in, int32_cuda xstride_out,
                                     int32_cuda ystride_out, int32_cuda zstride_out,
                                     const double *src, double *dst) {
  cudaD_rearrange_3d_tensor(Gr, Bl, xdim, xstride_in, ystride_in, zstride_in,
                            xstride_out, ystride_out, zstride_out, src, dst);
}



// The functions cuda_ctc_hmm_forward and cuda_ctc_hmm_backward are not wrapped
// here because we don't have separate float and double versions of them, they
// use BaseFloat.


} // namespace kaldi



#endif // HAVE_CUDA

#endif
