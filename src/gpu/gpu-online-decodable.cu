#include "gpu/gpu-online-decodable.h"

namespace kaldi{

GPUOnlineDecodableDiagGmmScaled::GPUOnlineDecodableDiagGmmScaled() {}

GPUOnlineDecodableDiagGmmScaled::GPUOnlineDecodableDiagGmmScaled(
  GPUAmDiagGmm* gpu_ac_model_,
  GPUTransitionModel* gpu_transition_model_,
  BaseFloat ac_scale_
) : 
  ac_model_(gpu_ac_model_),
  transition_model_(gpu_transition_model_),
  ac_scale_(ac_scale_) {}

/* TODO Optimasi :
 * 1. Pake Cachenya sama Locknya berarti 
 */
__device__ BaseFloat GPUOnlineDecodableDiagGmmScaled::LogLikelihood(int32 frame, int32 index){
  int32 pdf_id = transition_model_->TransitionIdToPdf(index);
  BaseFloat ans = ac_model_->LogLikelihood(pdf_id, cur_feats_.data, cur_feats_.Dim()) * ac_scale_;
  return ans;
}

}
