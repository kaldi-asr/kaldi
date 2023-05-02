// cudafeat/feature-online-batched-cmvn-cuda.cc
//
// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

#include "cudafeat/feature-online-batched-cmvn-cuda.h"
#include "cudafeat/feature-online-batched-cmvn-cuda-kernels.h"

namespace kaldi {

CudaOnlineBatchedCmvn::CudaOnlineBatchedCmvn(
    const OnlineCmvnOptions &opts, const CudaOnlineCmvnState &cmvn_state,
    int32_t feat_dim, int32_t chunk_size, int32_t num_channels,
    int32_t stats_coarsening_factor)
    : opts_(opts),
      cmvn_state_(cmvn_state),
      chunk_size_(chunk_size),
      num_channels_(num_channels),
      stats_coarsening_factor_(stats_coarsening_factor) {
  // This constraint could probably be removed by estimating partial frames
  KALDI_ASSERT(opts_.cmn_window % stats_coarsening_factor_ == 0);
  KALDI_ASSERT(chunk_size % stats_coarsening_factor_ == 0);

  // Number of fragements we need to keep around for each chunk
  num_fragments_ = (chunk_size + opts_.cmn_window) / stats_coarsening_factor_;

  stats_fragments_.Resize(num_channels_, feat_dim * num_fragments_ * 2);
}

// Computes a chunk of features for each channel included in channels
void CudaOnlineBatchedCmvn::ComputeFeaturesBatched(
    int32_t num_lanes, const LaneDesc *lanes,
    const CuMatrixBase<BaseFloat> &feats_in, CuMatrix<BaseFloat> *feats_out) {
  if (num_lanes == 0) return;

  // Step 1:
  // Compute windows sum/sum2 prefix along columns of feets
  // For audio chunk compute sum, sum2 and fill in stats
  // need to coarsen data by coarsening factor,
  // 1 out of coarsening factor threads actually write prefixs out
  // need to handle rolling window (modular indexing)
  // if partial frame of audio use mixture of global/current stats to fill in
  compute_cmvn_stats(feats_in.NumCols(), chunk_size_, stats_coarsening_factor_,
                     opts_.cmn_window, feats_in.Data(), feats_in.Stride(),
                     feats_in.Stride() * chunk_size_, stats_fragments_.Data(),
                     stats_fragments_.Stride(), lanes, num_lanes);

  // Step 2:
  // Apply CMVN
  const CuMatrix<float> &gstats = cmvn_state_.global_cmvn_stats;
  const CuMatrix<float> &sstats = cmvn_state_.speaker_cmvn_stats;

  int global_frames = opts_.global_frames;
  int speaker_frames = opts_.speaker_frames;

  if (gstats.NumRows() == 0) global_frames = 0;
  if (sstats.NumRows() == 0) speaker_frames = 0;

  apply_cmvn(opts_.cmn_window, opts_.normalize_variance, opts_.normalize_mean,
             feats_in.NumCols(), chunk_size_, stats_coarsening_factor_,
             feats_in.Data(), feats_in.Stride(),
             feats_in.Stride() * chunk_size_, stats_fragments_.Data(),
             stats_fragments_.Stride(), gstats.Data(), gstats.Stride(),
             global_frames, sstats.Data(), sstats.Stride(), speaker_frames,
             feats_out->Data(), feats_out->Stride(),
             feats_out->Stride() * chunk_size_, lanes, num_lanes);
}

}  // namespace kaldi
