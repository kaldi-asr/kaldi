// gmm/model-test-common.h

// Copyright 2009-2011  Saarland University
// Author:  Arnab Ghoshal

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


#ifndef KALDI_GMM_MODEL_TEST_COMMON_H_
#define KALDI_GMM_MODEL_TEST_COMMON_H_

#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"

namespace kaldi {
namespace unittest {

void RandPosdefSpMatrix(int32 dim, SpMatrix<BaseFloat> *matrix,
                        TpMatrix<BaseFloat> *matrix_sqrt = NULL,
                        BaseFloat *logdet = NULL);
void RandDiagGaussFeatures(int32 num_samples,
                           const VectorBase<BaseFloat> &mean,
                           const VectorBase<BaseFloat> &sqrt_var,
                           MatrixBase<BaseFloat> *feats);
void RandFullGaussFeatures(int32 num_samples,
                           const VectorBase<BaseFloat> &mean,
                           const TpMatrix<BaseFloat> &sqrt_var,
                           MatrixBase<BaseFloat> *feats);
void InitRandDiagGmm(int32 dim, int32 num_comp, DiagGmm *gmm);
void InitRandFullGmm(int32 dim, int32 num_comp, FullGmm *gmm);

}  // End namespace unittest
}  // End namespace kaldi


#endif  // KALDI_GMM_MODEL_TEST_COMMON_H_
