// decoder/cuda-decoder-utils.cc

// Copyright      2018  Zhehuai Chen

// See ../../COPYING for clarification regarding multiple authors
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http:// www.apache.org/licenses/LICENSE-2.0
// 
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "fst/fstlib.h"
#include "cuda-decoder-utils.h"

namespace kaldi {

void get_free_memory_stat(char *prefix) {
  int32 act_gpu_id;
  cudaError_t e = cudaGetDevice(&act_gpu_id);
  char name[128];
  CuDevice::Instantiate().DeviceGetName(name,128,act_gpu_id);
  // get GPU memory stats
  int64 free, total; 
  std::string mem_stats;
  mem_stats = CuDevice::Instantiate().GetFreeMemory(&free, &total);
  float mem_ratio = total > 0 ? free/(float)total : 0;
  // log
  KALDI_LOG << prefix << "\tcudaSetDevice(" << act_gpu_id << "): "
            << name << "\t" << mem_ratio << " % "
            << "\t" << mem_stats;
}




// CudaFst Implementation
HOST DEVICE float CudaFst::Final(StateId state) const {
#ifdef __CUDA_ARCH__
  return final_d[state];
#else
  return final_h[state];
#endif
}
void CudaFst::Initialize(const fst::Fst<StdArc> &fst) {
  PUSH_RANGE("CudaFst constructor", 1)
  bytes_cudaMalloc = 0;
  numStates = 0;
  for ( fst::StateIterator<fst::Fst<StdArc> > iter(fst); !iter.Done();
        iter.Next()) {
    numStates++;
  }
  start = fst.Start();
  cudaMallocHost(&final_h, sizeof(float) * numStates);
  cudaMalloc(&final_d, sizeof(float) * numStates);

  // allocate and initialize offset arrays
  e_offsets_h = (unsigned int *)malloc(sizeof(unsigned int) * (numStates + 1));
  ne_offsets_h = (unsigned int *)malloc(sizeof(unsigned int) * (numStates + 1));

  cudaMalloc((void**)&e_offsets_d, sizeof(unsigned int) * (numStates + 1));
  bytes_cudaMalloc += sizeof(unsigned int) * (numStates + 1);
  cudaMalloc((void**)&ne_offsets_d, sizeof(unsigned int) * (numStates + 1));
  bytes_cudaMalloc += sizeof(unsigned int) * (numStates + 1);

  memset(e_offsets_h, 0, sizeof(unsigned int) * (numStates + 1));
  memset(ne_offsets_h, 0, sizeof(unsigned int) * (numStates + 1));

  // iterate through states and arcs and count number of arcs per state
  e_count = 0;
  ne_count = 0;
  max_ilabel = 0;
  for (int i = 0; i < numStates; i++) {
    final_h[i] = fst.Final(i).Value();
    // count emmiting and non_emitting arcs
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done();
         aiter.Next()) {
      StdArc arc = aiter.Value();
      int32 ilabel = arc.ilabel;
      int32 olabel = arc.olabel;
      if (ilabel > max_ilabel) {
        max_ilabel = ilabel;
      }
      if (ilabel != 0) { // emitting
        e_count++;
      } else { // non-emitting
        ne_count++;
      }
    }
    ne_offsets_h[i + 1] = ne_count;
    e_offsets_h[i + 1] = e_count;
  }

  // offset ne_offsets by the number of emitting arcs
  for (int i = 0; i < numStates + 1; i++) {
    e_offsets_h[i] += 1;        // add dummy arc at the beginingg.
    ne_offsets_h[i] += e_count + 1; // add dummy arc and put e_arcs before
  }

  arc_count = e_count + ne_count + 1;
  numArcs = arc_count;

  cudaMemcpyAsync(final_d, final_h, sizeof(float)*numStates, cudaMemcpyHostToDevice,
                  cudaStreamPerThread);
  cudaMemcpyAsync(e_offsets_d, e_offsets_h, sizeof(unsigned int) * (numStates + 1),
                  cudaMemcpyHostToDevice, cudaStreamPerThread);
  cudaMemcpyAsync(ne_offsets_d, ne_offsets_h,
                  sizeof(unsigned int) * (numStates + 1), cudaMemcpyHostToDevice,
                  cudaStreamPerThread);

  // Allocate non-zero arrays
  cudaMallocHost(&arc_weights_h, arc_count * sizeof(BaseFloat));
  cudaMallocHost(&arc_nextstates_h, arc_count * sizeof(StateId));
  cudaMallocHost(&arc_ilabels_h, arc_count * sizeof(int32));
  cudaMallocHost(&arc_olabels_h, arc_count * sizeof(int32));

  cudaMalloc((void**)&arc_weights_d, arc_count * sizeof(BaseFloat));
  bytes_cudaMalloc += arc_count * sizeof(BaseFloat);
  cudaMalloc((void**)&arc_nextstates_d, arc_count * sizeof(StateId));
  bytes_cudaMalloc += arc_count * sizeof(StateId);
  cudaMalloc((void**)&arc_ilabels_d, arc_count * sizeof(int32));
  bytes_cudaMalloc += arc_count * sizeof(int32);
  cudaMalloc((void**)&arc_olabels_d, arc_count * sizeof(int32));
  bytes_cudaMalloc += arc_count * sizeof(int32);

  // now populate arc data
  int e_idx = 1;        // save room for dummy arc (so start at 1)
  int ne_idx = e_count + 1; // starts where e_offsets ends

  // create dummy arc
  arc_weights_h[0] = StdWeight::One().Value();
  arc_nextstates_h[0] = fst.Start();
  arc_ilabels_h[0] = 0;
  arc_olabels_h[0] = 0;

  for (int i = 0; i < numStates; i++) {
    // count emmiting and non_emitting arcs
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done();
         aiter.Next()) {
      StdArc arc = aiter.Value();
      int idx;
      if (arc.ilabel != 0) { // emitting
        idx = e_idx++;
      } else {
        idx = ne_idx++;
      }
      arc_weights_h[idx] = arc.weight.Value();
      arc_nextstates_h[idx] = arc.nextstate;
      arc_ilabels_h[idx] = arc.ilabel;
      arc_olabels_h[idx] = arc.olabel;
    }
  }

  cudaMemcpyAsync(arc_weights_d, arc_weights_h, arc_count * sizeof(BaseFloat),
                  cudaMemcpyHostToDevice, cudaStreamPerThread);
  cudaMemcpyAsync(arc_nextstates_d, arc_nextstates_h, arc_count * sizeof(StateId),
                  cudaMemcpyHostToDevice, cudaStreamPerThread);
  cudaMemcpyAsync(arc_ilabels_d, arc_ilabels_h, arc_count * sizeof(int32),
                  cudaMemcpyHostToDevice, cudaStreamPerThread);
  cudaMemcpyAsync(arc_olabels_d, arc_olabels_h, arc_count * sizeof(int32),
                  cudaMemcpyHostToDevice, cudaStreamPerThread);

  cudaStreamSynchronize(cudaStreamPerThread);
  POP_RANGE
}

void CudaFst::Finalize() {
  PUSH_RANGE("CudaFst destructor", 0);
  cudaFreeHost(final_h);
  cudaFree(final_d);
  free(e_offsets_h);
  free(ne_offsets_h);

  cudaFree(e_offsets_d);
  cudaFree(ne_offsets_d);

  cudaFreeHost(arc_weights_h);
  cudaFreeHost(arc_nextstates_h);
  cudaFreeHost(arc_ilabels_h);
  cudaFreeHost(arc_olabels_h);

  cudaFree(arc_weights_d);
  cudaFree(arc_nextstates_d);
  cudaFree(arc_ilabels_d);
  cudaFree(arc_olabels_d);
  POP_RANGE
}

} // end namespace kaldi.

