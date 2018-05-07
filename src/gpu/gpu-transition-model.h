#ifndef KALDI_HMM_GPU_TRANSITION_MODEL_H_
#define KALDI_HMM_GPU_TRANSITION_MODEL_H_

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "gpucommons/gpu-vector.h"

#include "hmm/transition-model.h"

namespace kaldi{

struct GPUTransitionModel{

  thrust::device_vector<int32> id2pdf_id_;
  int32* id2pdf_id;
  int32 num_pdfs_;

  GPUTransitionModel();
  GPUTransitionModel(TransitionModel& t);
  
  __host__ __device__ int32 NumPdfs() const;

  __device__ int32 TransitionIdToPdf(int32 trans_id);

};

}

#endif