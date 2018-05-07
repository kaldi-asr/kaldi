#include "gpu-transition-model.h"

namespace kaldi{

GPUTransitionModel::GPUTransitionModel() {}
GPUTransitionModel::GPUTransitionModel(TransitionModel& t) : 
  id2pdf_id_(t.id2pdf_id()),
  num_pdfs_(t.NumPdfs())
  {
    id2pdf_id = id2pdf_id_.data().get();
  }

__host__ __device__ int32 GPUTransitionModel::NumPdfs() const { return num_pdfs_; }

__device__ int32 GPUTransitionModel::TransitionIdToPdf(int32 trans_id) const {
  return id2pdf_id[trans_id];
}


}