// cudamatrix/cu-common.h

// Copyright 2009-2011  Karel Vesely

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



#ifndef KALDI_CUDAMATRIX_CUCOMMON_H_
#define KALDI_CUDAMATRIX_CUCOMMON_H_


#if HAVE_CUDA==1


#include <iostream>
#include <sstream>

#include <cuda_runtime_api.h>

#include "base/kaldi-error.h"


#define cuSafeCall(fun) \
{ \
  int ret; \
  if ((ret = (fun)) != 0) { \
    KALDI_ERR << "CUDA ERROR #" << ret << " " << cudaGetErrorString((cudaError_t)ret) << " '" << #fun << "'"; \
  } \
  cudaThreadSynchronize(); \
} 





namespace kaldi {

  /** The size of edge of CUDA square block **/
  static const int CUBLOCK = 16;

  /** Number of blocks in which the task of size 'size' is splitted **/
  inline int n_blocks(int size, int block_size) { 
    return size / block_size + ((size % block_size == 0)? 0 : 1); 
  }

}
#endif //HAVE_CUDA

#endif
