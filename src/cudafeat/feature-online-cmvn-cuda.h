// cudafeat/feature-online-cmvn-cuda.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_CUDAFEAT_FEATURE_ONLINE_CMVN_CUDA_H_
#define KALDI_CUDAFEAT_FEATURE_ONLINE_CMVN_CUDA_H_

#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "feat/online-feature.h"

namespace kaldi {

struct CudaOnlineCmvnState {
  // The following is the global CMVN stats, in the usual
  // format, of dimension 2 x (dim+1), as [  sum-stats          count
  //                                       sum-sqared-stats   0    ]
  CuMatrix<float> global_cmvn_stats;
  CuMatrix<float> speaker_cmvn_stats;

  CudaOnlineCmvnState(){};
  CudaOnlineCmvnState(const OnlineCmvnState &cmvn_state)
      : global_cmvn_stats(cmvn_state.global_cmvn_stats),
        speaker_cmvn_stats(cmvn_state.speaker_cmvn_stats) {}

  CudaOnlineCmvnState(const CudaOnlineCmvnState &cmvn_state)
      : global_cmvn_stats(cmvn_state.global_cmvn_stats),
        speaker_cmvn_stats(cmvn_state.speaker_cmvn_stats) {}
};

class CudaOnlineCmvn {
 public:
  CudaOnlineCmvn(const OnlineCmvnOptions &opts, const CudaOnlineCmvnState &cmvn_state)
      : opts_(opts), cmvn_state_(cmvn_state){};
  ~CudaOnlineCmvn(){};

  void ComputeFeatures(const CuMatrixBase<BaseFloat> &feats_in,
                       CuMatrix<BaseFloat> *feats_out);

 private:
  const OnlineCmvnOptions &opts_;
  const CudaOnlineCmvnState cmvn_state_;
};
}

#endif
