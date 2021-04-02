// cudadecoder/cuda-fst.h
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

#ifndef KALDI_CUDA_DECODER_CUDA_FST_H_
#define KALDI_CUDA_DECODER_CUDA_FST_H_
#include "cudadecoder/cuda-decoder-common.h"
#include "cudamatrix/cu-device.h"
#include "lat/kaldi-lattice.h"
#include "nnet3/decodable-online-looped.h"  // TransitionModel

namespace kaldi {
namespace cuda_decoder {

typedef fst::StdArc StdArc;
typedef StdArc::Weight StdWeight;
typedef StdArc::Label Label;

// FST in both device and host memory
// Converting the OpenFst format to the CSR Compressed Sparse Row (CSR) Matrix
// format.
// https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
// Where states = rows and arcs = columns.
// This format allows us to store the FST in a compact form, and leads to clean
// memory accesses
// For instance, when loading the arcs from a given source, we can load all arc
// informations (destination, weight, etc.) with coalesced reads
// Emitting arcs and non-emitting arcs are stored as separate matrices for
// efficiency
// We then copy the FST to the device (while keeping its original copy on host)
class CudaFst {
 public:
  CudaFst()
      : d_e_offsets_(nullptr),
        d_ne_offsets_(nullptr),
        d_arc_weights_(nullptr),
        d_arc_nextstates_(nullptr),
        d_arc_pdf_ilabels_(nullptr),
        d_final_(nullptr){};
  // Creates a CSR representation of the FST,
  // then copies it to the GPU
  // If a TransitionModel is passed, we'll use it to convert the ilabels id
  // indexes into pdf indexes
  // If no TransitionModel is passed, we'll assume TransitionModel == identity
  // Important: The CudaDecodable won't apply the TransitionModel. If you use a
  // TransitionModel, you need to apply it now
  void Initialize(const fst::Fst<StdArc> &fst,
                  const TransitionModel *trans_model = NULL);
  void Finalize();

  inline uint32_t NumStates() const { return num_states_; }
  inline StateId Start() const { return start_; }

 private:
  friend class CudaDecoder;
  // Counts arcs and computes offsets of the fst passed in
  void ComputeOffsets(const fst::Fst<StdArc> &fst);
  // Allocates memory to store FST
  void AllocateData(const fst::Fst<StdArc> &fst);
  // Populate the arcs data (arc.destination, arc.weights, etc.)
  void PopulateArcs(const fst::Fst<StdArc> &fst);
  // Converting the id ilabels into pdf ilabels using the transition model
  // It allows the CudaDecoder to read the acoustic model loglikelihoods at the
  // right indexes
  void ApplyTransitionModelOnIlabels(const TransitionModel &trans_model);
  // Copies fst to device into the pre-allocated datastructures
  void CopyDataToDevice();
  // Total number of states
  unsigned int num_states_;
  // Starting state of the FST
  // Computation should start from state start_
  StateId start_;
  // Number of emitting, non-emitting, and total number of arcs
  unsigned int e_count_, ne_count_, arc_count_;
  // This data structure is similar to a CSR matrix format
  // with 2 offsets matrices (one emitting one non-emitting).
  // Offset arrays are num_states_+1 in size (last state needs
  // its +1 arc_offset)
  // Arc values for state i are stored in the range of [offset[i],offset[i+1][
  unsigned int *d_e_offsets_;  // Emitting offset arrays
  std::vector<unsigned int> h_e_offsets_;
  unsigned int *d_ne_offsets_;  // Non-emitting offset arrays
  std::vector<unsigned int> h_ne_offsets_;
  // These are the values for each arc.
  // Arcs belonging to state i are found in the range of [offsets[i],
  // offsets[i+1][
  // Use e_offsets or ne_offsets depending on what you need
  // (emitting/nonemitting)
  // The ilabels arrays are of size e_count_, not arc_count_
  std::vector<CostType> h_arc_weights_;
  CostType *d_arc_weights_;
  std::vector<StateId> h_arc_nextstate_;
  StateId *d_arc_nextstates_;
  std::vector<int32> h_arc_id_ilabels_;
  int32 *d_arc_pdf_ilabels_;
  std::vector<int32> h_arc_olabels_;
  // Final costs
  // final cost of state i is h_final_[i]
  std::vector<CostType> h_final_;
  CostType *d_final_;

  // ilabels (pdf indexing)
  // only populate during CSR generation, cleared after (not needed on host)
  std::vector<int32> h_arc_pdf_ilabels_;
};

}  // end namespace cuda_decoder
}  // end namespace kaldi
#endif  // KALDI_CUDA_DECODER_CUDA_FST_H_
