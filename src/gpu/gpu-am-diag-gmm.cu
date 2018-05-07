#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "gpu/gpu-am-diag-gmm.h"

namespace kaldi{
  
GPUAmDiagGmm::GPUAmDiagGmm() {
  densities = densities_.data().get();
}
GPUAmDiagGmm::~GPUAmDiagGmm() {
  for(int i = 0;i < densities_.size(); ++i){
    GPUDiagGmm* gpugmm = densities_[i];
    delete gpugmm;
  }
  densities_.clear();
}

void GPUAmDiagGmm::AddPdf(GPUDiagGmm &gpugmm){
  // if (densities_.size() != 0)  // not the first gmm
  //   KALDI_ASSERT(gpugmm.Dim() == this->Dim());
  densities_.push_back(&gpugmm);
}

__device__ BaseFloat GPUAmDiagGmm::LogLikelihood(const int32 pdf_index, BaseFloat* data, int32 num_data) const {
  return densities[pdf_index]->LogLikelihood(data, num_data);
}

}
