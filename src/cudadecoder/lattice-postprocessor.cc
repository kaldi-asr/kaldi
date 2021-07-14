// cudadecoder/lattice-postprocessor.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun
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

#include "cudadecoder/lattice-postprocessor.h"

#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "util/common-utils.h"

#if HAVE_CUDA == 1

namespace kaldi {
namespace cuda_decoder {

LatticePostprocessor::LatticePostprocessor(
    const LatticePostprocessorConfig &config)
    : config_(config), decoder_frame_shift_(0.0) {
  ApplyConfig();
}

void LatticePostprocessor::ApplyConfig() {
  // Lattice scale
  lattice_scales_.resize(2);
  lattice_scales_[0].resize(2);
  lattice_scales_[1].resize(2);
  lattice_scales_[0][0] = config_.lm_scale;
  lattice_scales_[0][1] = config_.acoustic2lm_scale;
  lattice_scales_[1][0] = config_.lm2acoustic_scale;
  lattice_scales_[1][1] = config_.acoustic_scale;

  use_lattice_scale_ =
      (config_.lm_scale != 1.0 || config_.acoustic2lm_scale != 0.0 ||
       config_.lm2acoustic_scale != 0.0 || config_.acoustic_scale != 1.0);

  // Word boundary
  if (!config_.word_boundary_rxfilename.empty())
    LoadWordBoundaryInfo(config_.word_boundary_rxfilename);
}

bool LatticePostprocessor::GetPostprocessedLattice(
    CompactLattice &clat, CompactLattice *out_clat) const {
  // Nothing to do for empty lattice
  if (clat.NumStates() == 0) return true;

  bool ok = true;
  // Scale lattice
  if (use_lattice_scale_) fst::ScaleLattice(lattice_scales_, &clat);

  // Word insertion penalty
  if (config_.word_ins_penalty > 0.0)
    AddWordInsPenToCompactLattice(config_.word_ins_penalty, &clat);

  // Word align
  int32 max_states;
  if (config_.max_expand > 0)
    max_states = 1000 + config_.max_expand * clat.NumStates();
  else
    max_states = 0;

  KALDI_ASSERT(tmodel_ &&
               "SetTransitionModel() must be called (typically by pipeline)");

  KALDI_ASSERT(decoder_frame_shift_ != 0.0 &&
               "SetDecoderFrameShift() must be called (typically by pipeline)");

  if (!word_info_)
    KALDI_ERR << "You must set --word-boundary-rxfilename in the lattice "
                 "postprocessor config";
  // ok &=
  // Ignoring the return false for now (but will print a warning),
  // because the doc says we can, and it can happen when using endpointing
  WordAlignLattice(clat, *tmodel_, *word_info_, max_states, out_clat);
  return ok;
}

bool LatticePostprocessor::GetCTM(CompactLattice &clat,
                                  CTMResult *ctm_result) const {
  // Empty CTM output for empty lattice
  if (clat.NumStates() == 0) return true;

  CompactLattice postprocessed_lattice;
  GetPostprocessedLattice(clat, &postprocessed_lattice);

  // MBR
  MinimumBayesRisk mbr(postprocessed_lattice, config_.mbr_opts);
  ctm_result->conf = std::move(mbr.GetOneBestConfidences());
  ctm_result->words = std::move(mbr.GetOneBest());
  ctm_result->times_seconds = std::move(mbr.GetOneBestTimes());

  // Convert timings to seconds
  for (auto &p : ctm_result->times_seconds) {
    p.first *= decoder_frame_shift_;
    p.second *= decoder_frame_shift_;
  }

  return true;
}

void SetResultUsingLattice(
    CompactLattice &clat, const int result_type,
    const std::shared_ptr<LatticePostprocessor> &lattice_postprocessor,
    CudaPipelineResult *result) {
  if (result_type & CudaPipelineResult::RESULT_TYPE_LATTICE) {
    if (lattice_postprocessor) {
      CompactLattice postprocessed_clat;
      lattice_postprocessor->GetPostprocessedLattice(clat, &postprocessed_clat);
      result->SetLatticeResult(std::move(postprocessed_clat));
    } else {
      result->SetLatticeResult(std::move(clat));
    }
  }

  if (result_type & CudaPipelineResult::RESULT_TYPE_CTM) {
    CTMResult ctm_result;
    KALDI_ASSERT(lattice_postprocessor &&
                 "A lattice postprocessor must be set with "
                 "SetLatticePostprocessor() to use RESULT_TYPE_CTM");
    lattice_postprocessor->GetCTM(clat, &ctm_result);
    result->SetCTMResult(std::move(ctm_result));
  }
}

}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // HAVE_CUDA
