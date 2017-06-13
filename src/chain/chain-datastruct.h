// chain/chain-datastruct.h

// Copyright 2015    Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_CHAIN_CHAIN_DATASTRUCT_H_
#define KALDI_CHAIN_CHAIN_DATASTRUCT_H_
#include "cudamatrix/cu-matrixdim.h" // for CU1DBLOCK and CU2DBLOCK, and int32_cuda

/**
   This header is for declaring "C" structures that are to be used in the
   CUDA interface for things in this directory.  We put it in a separate header from
   the CUDA stuff as it may be needed regardless of whether we're actually compiling with
   CUDA.
 */

extern "C" {
  // "C" version of the BaseFloat typedef-- this saves us having to write
  // multiple versions of these kernels.
#if (KALDI_DOUBLEPRECISION != 0)
  typedef double  BaseFloat;
#else
  typedef float   BaseFloat;
#endif

  struct DenominatorGraphTransition {
    BaseFloat transition_prob;  // language-model part of the probability (not
                                // in log)
    int32_cuda pdf_id;   // pdf-id on the transition.
    int32_cuda hmm_state;  // source, or destination, HMM state.
  };


  // Search for this in chain-kernels.cu for an explanation.
  enum { kThresholdingPowerOfTwo = 14 };

}



#endif
