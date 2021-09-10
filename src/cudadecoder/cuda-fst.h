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

#ifndef KALDI_CUDADECODER_CUDA_FST_H_
#define KALDI_CUDADECODER_CUDA_FST_H_

#if HAVE_CUDA

#include <vector>

#include "base/kaldi-utils.h"
#include "cudadecoder/cuda-decoder-common.h"
#include "cudamatrix/cu-common.h"
#include "lat/kaldi-lattice.h"
#include "itf/transition-information.h"

namespace kaldi {
namespace cuda_decoder {

//
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

///\brief Represent the HCLG FST in both device and host memory.
///
/// The standard Kaldi's representation of the HCLG machine in fst::StdFst is
/// converted to the Compressed Sparse Row (CSR), a.k.a. [Yale format matrix](
/// https://en.wikipedia.org/w/index.php?title=Sparse_matrix&oldid=1023781875#Compressed_sparse_row_%28CSR%2C_CRS_or_Yale_format%29),
/// where states correcpond to rows and arcs to columns. This format allows
/// storing the FST in a compact form, and leads to clean memory accesses.
///
/// For instance, when loading the arcs from a given source, we can load all
/// arcs information (destination, weight, etc.) with coalesced reads. Emitting
/// and non-emitting arcs are stored as separate matrices for efficiency.
///
/// The object stores this representation in both host and device memory for
/// its lifetime.

class CudaFst {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;

  ///\brief Creates a CSR representation of the FST, then copies it to the GPU.
  ///
  /// If a non-null \p trans_model is passed, we'll use it to convert the
  /// ilabel ID indexes into PDF indexes. Otherwise we assume
  /// TransitionInformation == identity.
  ///
  ///\warning The CudaDecodable won't apply the TransitionInformation. If you use a
  ///  TransitionInformation, you need to apply it now.
  CudaFst(const fst::StdFst &fst,
          const TransitionInformation *trans_model = nullptr);

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaFst);

  uint32_t NumStates() const { return num_states_; }
  StateId Start() const { return start_; }

 private:
  // TODO(kkm): The typedef and the deleter for unique_ptr to device memory
  // should better be integerated into cu-device or cu-allocator. This
  // is currently a proof-of-concept only.
  template<typename T>
  struct CuDeleter {
    constexpr CuDeleter() noexcept = default;
    void operator()(T* ptr) const { CU_SAFE_CALL(cudaFree(ptr)); }
  };
  /// A uniquely owned, movable pointer to device-allocated memory.
  template<typename T>
  using unique_device_ptr = std::unique_ptr<T, CuDeleter<T>>;


  friend class CudaDecoder;
  // Counts arcs and computes offsets of the fst passed in
  void ComputeOffsets(const fst::StdFst &fst);
  // Allocates memory to store FST
  void AllocateData(const fst::StdFst &fst);
  // Populate the arcs data (arc.destination, arc.weights, etc.)
  void PopulateArcs(const fst::StdFst &fst);
  // Converting the id ilabels into pdf ilabels using the transition model
  // It allows the CudaDecoder to read the acoustic model loglikelihoods at the
  // right indexes
  void ApplyTransitionModelOnIlabels(const TransitionInformation &trans_model);
  // Copies fst to device into the pre-allocated datastructures
  void CopyDataToDevice();
  // Total number of states
  uint32 num_states_;
  // Starting state of the FST
  // Computation should start from state start_
  StateId start_;
  // Number of emitting, non-emitting, and total number of arcs
  uint32 e_count_, ne_count_, arc_count_;
  // This data structure is similar to a CSR matrix format
  // with 2 offsets matrices (one emitting one non-emitting).
  // Offset arrays are num_states_+1 in size (last state needs
  // its +1 arc_offset)
  // Arc values for state i are stored in the range of [offset[i],offset[i+1]]
  unique_device_ptr<uint32> d_e_offsets_;  // Emitting offset arrays
  std::vector<uint32> h_e_offsets_;
  unique_device_ptr<uint32> d_ne_offsets_;  // Non-emitting offset arrays
  std::vector<uint32> h_ne_offsets_;
  // These are the values for each arc.
  // Arcs belonging to state i are found in the range of [offsets[i],
  // offsets[i+1][
  // Use e_offsets or ne_offsets depending on what you need
  // (emitting/nonemitting)
  // The ilabels arrays are of size e_count_, not arc_count_
  std::vector<CostType> h_arc_weights_;
  unique_device_ptr<CostType> d_arc_weights_;
  std::vector<StateId> h_arc_nextstate_;
  unique_device_ptr<StateId> d_arc_nextstates_;
  std::vector<int32> h_arc_id_ilabels_;
  unique_device_ptr<int32> d_arc_pdf_ilabels_;
  std::vector<int32> h_arc_olabels_;
  // Final costs
  // final cost of state i is h_final_[i]
  std::vector<CostType> h_final_;
  unique_device_ptr<CostType> d_final_;

  // ilabels (pdf indexing)
  // only populate during CSR generation, cleared after (not needed on host)
  std::vector<int32> h_arc_pdf_ilabels_;
};

}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // HAVE_CUDA
#endif  // KALDI_CUDADECODER_CUDA_FST_H_
