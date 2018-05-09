#ifndef KALDI_GMM_GPU_AM_DIAG_GMM_H_
#define KALDI_GMM_GPU_AM_DIAG_GMM_H_ 1

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "gpu/gpu-diag-gmm.h"

namespace kaldi{

struct GPUAmDiagGmm{

  thrust::device_vector<GPUDiagGmm*> densities_;
  GPUDiagGmm** densities;

  GPUAmDiagGmm();

  void AddPdf(GPUDiagGmm *gpugmm);
  __device__ BaseFloat LogLikelihood(const int32 pdf_index, BaseFloat* data, int32 num_data) const;
};

}

#endif
