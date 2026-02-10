// cudadecoder/cuda-fst.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary
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

#if !HAVE_CUDA
#error CUDA support must be configured to compile this library.
#endif

#include "cudadecoder/cuda-fst.h"
#include "cudamatrix/cu-common.h"

#include <cuda_runtime_api.h>
#include <nvToolsExt.h>

namespace kaldi {
namespace cuda_decoder {

CudaFst::CudaFst(const fst::StdFst &fst,
                 const TransitionInformation *trans_model /* = nullptr */) {
  nvtxRangePushA("CudaFst constructor");

  start_ = fst.Start();
  KALDI_ASSERT(start_ != fst::kNoStateId);

  ComputeOffsets(fst);
  AllocateData(fst);
  // Temporarily allocating data for this vector
  // We just need it during CSR generation. We will clear it
  // at the end of Initialize
  h_arc_pdf_ilabels_.resize(arc_count_);
  PopulateArcs(fst);
  if (trans_model) ApplyTransitionModelOnIlabels(*trans_model);

  KALDI_ASSERT(d_e_offsets_);
  KALDI_ASSERT(d_ne_offsets_);
  KALDI_ASSERT(d_final_);
  KALDI_ASSERT(d_arc_weights_);
  KALDI_ASSERT(d_arc_nextstates_);
  KALDI_ASSERT(d_arc_pdf_ilabels_);

  CopyDataToDevice();

  // Making sure the graph is ready
  cudaDeviceSynchronize();
  KALDI_DECODER_CUDA_CHECK_ERROR();
  h_arc_pdf_ilabels_.clear();  // we don't need those on host

  nvtxRangePop();
}

void CudaFst::ComputeOffsets(const fst::StdFst &fst) {
  // count states since Fst doesn't provide this functionality
  num_states_ = 0;
  for (fst::StateIterator<fst::StdFst> iter(fst); !iter.Done(); iter.Next()) {
    ++num_states_;
  }

  // allocate and initialize offset arrays
  h_final_.resize(num_states_);
  h_e_offsets_.resize(num_states_ + 1);
  h_ne_offsets_.resize(num_states_ + 1);

  // iterate through states and arcs and count number of arcs per state
  e_count_ = 0;
  ne_count_ = 0;

  // Init first offsets
  h_ne_offsets_[0] = 0;
  h_e_offsets_[0] = 0;
  for (uint32 i = 0; i < num_states_; ++i) {
    h_final_[i] = fst.Final(i).Value();
    // count emiting and non_emitting arcs
    for (fst::ArcIterator<fst::StdFst> aiter(fst, i); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      Label ilabel = arc.ilabel;
      if (ilabel != 0) {  // emitting
        e_count_++;
      } else {  // non-emitting
        ne_count_++;
      }
    }
    h_ne_offsets_[i + 1] = ne_count_;
    h_e_offsets_[i + 1] = e_count_;
  }

  // We put the emitting arcs before the nonemitting arcs in the arc list
  // adding offset to the non emitting arcs
  // we go to num_states_+1 to take into account the last offset
  for (uint32 i = 0; i < num_states_ + 1; ++i)
    h_ne_offsets_[i] += e_count_;  // e_arcs before

  arc_count_ = e_count_ + ne_count_;
}

void CudaFst::AllocateData(const fst::StdFst &fst) {
  void *temp_pointer;
  CU_SAFE_CALL(cudaMalloc(&temp_pointer, (num_states_ + 1) * sizeof(uint32)));
  d_e_offsets_.reset(static_cast<uint32*>(temp_pointer));
  CU_SAFE_CALL(cudaMalloc(&temp_pointer, (num_states_ + 1) * sizeof(uint32)));
  d_ne_offsets_.reset(static_cast<uint32*>(temp_pointer));
  CU_SAFE_CALL(cudaMalloc(&temp_pointer, num_states_ * sizeof(float)));
  d_final_.reset(static_cast<float*>(temp_pointer));

  h_arc_weights_.resize(arc_count_);
  h_arc_nextstate_.resize(arc_count_);
  // ilabels (id indexing)
  h_arc_id_ilabels_.resize(arc_count_);
  h_arc_olabels_.resize(arc_count_);

  CU_SAFE_CALL(cudaMalloc(&temp_pointer, arc_count_ * sizeof(CostType)));
  d_arc_weights_.reset(static_cast<CostType*>(temp_pointer));
  CU_SAFE_CALL(cudaMalloc(&temp_pointer, arc_count_ * sizeof(StateId)));
  d_arc_nextstates_.reset(static_cast<StateId*>(temp_pointer));
  // Only the ilabels for the e_arc are needed on the device
  CU_SAFE_CALL(cudaMalloc(&temp_pointer, e_count_ * sizeof(int32)));
  d_arc_pdf_ilabels_.reset(static_cast<int32*>(temp_pointer));
}

void CudaFst::PopulateArcs(const fst::StdFst &fst) {
  // now populate arc data
  uint32 e_idx = 0;
  uint32 ne_idx = e_count_;  // starts where e_offsets_ ends
  for (uint32 i = 0; i < num_states_; ++i) {
    for (fst::ArcIterator<fst::StdFst> aiter(fst, i); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      uint32 idx;
      if (arc.ilabel != 0) {  // emitting
        idx = e_idx++;
      } else {
        idx = ne_idx++;
      }
      h_arc_weights_[idx] = arc.weight.Value();
      h_arc_nextstate_[idx] = arc.nextstate;
      h_arc_id_ilabels_[idx] = arc.ilabel;
      // For now we consider id indexing == pdf indexing
      // If the two are differents, we'll call ApplyTransModelOnIlabels with a
      // TransitionModel
      h_arc_pdf_ilabels_[idx] = arc.ilabel;
      h_arc_olabels_[idx] = arc.olabel;
    }
  }
}

void CudaFst::ApplyTransitionModelOnIlabels(
    const TransitionInformation &trans_model) {
  // Converting ilabel here, to avoid reindexing when reading nnet3 output
  // We only need to convert the emitting arcs
  // The emitting arcs are the first e_count_ arcs
  for (uint32 iarc = 0; iarc < e_count_; ++iarc) {
    h_arc_pdf_ilabels_[iarc] =
        trans_model.TransitionIdToPdf(h_arc_id_ilabels_[iarc]);
  }
}

void CudaFst::CopyDataToDevice() {
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_e_offsets_.get(), &h_e_offsets_[0],
      (num_states_ + 1) * sizeof(*d_e_offsets_),
      cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_ne_offsets_.get(), &h_ne_offsets_[0],
      (num_states_ + 1) * sizeof(*d_ne_offsets_), cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(d_final_.get(), &h_final_[0],
                                                num_states_ * sizeof(*d_final_),
                                                cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaMemcpy(d_arc_weights_.get(), &h_arc_weights_[0],
                 arc_count_ * sizeof(*d_arc_weights_), cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_arc_nextstates_.get(), &h_arc_nextstate_[0],
      arc_count_ * sizeof(*d_arc_nextstates_), cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_arc_pdf_ilabels_.get(), &h_arc_pdf_ilabels_[0],
      e_count_ * sizeof(*d_arc_pdf_ilabels_), cudaMemcpyHostToDevice));
}

}  // namespace cuda_decoder
}  // namespace kaldi
