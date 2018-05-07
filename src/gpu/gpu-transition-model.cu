#include "gpu-transition-model.h"

namespace kaldi{

GPUTransitionModel::GPUTransitionModel() {}
GPUTransitionModel::GPUTransitionModel(TransitionModel& t) : 
  topo_(t.GetTopo()),
  tuples_(t.tuples()),
  state2id_(t.state2id()),
  id2state_(t.id2state()),
  id2pdf_id_(t.id2pdf_id()),
  num_pdfs_(t.NumPdfs()),
  log_probs_(t.log_probs()),
  non_self_loop_log_probs_(t.non_self_loop_log_probs()) {
    id2pdf_id = id2pdf_id_.data().get();
  }


int32 GPUTransitionModel::NumPdfs() const { return num_pdfs_; }

__host__ __device__ int32 GPUTransitionModel::TransitionIdToPdf(int32 trans_id) const {
  return id2pdf_id[trans_id];
}


}