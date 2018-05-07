#include "gpu/gpu-online-decodable.h"

namespace kaldi{

GPUOnlineDecodableDiagGmmScaled::GPUOnlineDecodableDiagGmmScaled() {}
GPUOnlineDecodableDiagGmmScaled::GPUOnlineDecodableDiagGmmScaled(
  const GPUAmDiagGmm& gpu_ac_model_,
  const GPUTransitionModel& gpu_transition_model_,
  BaseFloat ac_scale_
) : 
  ac_model_(gpu_ac_model_),
  transition_model_(gpu_transition_model_),
  ac_scale_(ac_scale_) 
{
  int32 num_pdfs = transition_model_.NumPdfs();
  cache_.resize(num_pdfs, std::pair<int32,BaseFloat>(-1, 0.0));
}

/* TODO Optimasi :
 * 1. Pake Cachenya sama Locknya berarti 
 */
__device__ BaseFloat GPUOnlineDecodableDiagGmmScaled::LogLikelihood(int32 frame, int32 index){
  // KALDI_ASSERT(frame == cur_frame_);
  int32 pdf_id = transition_model_.TransitionIdToPdf(index);
  // if (cache_[pdf_id].first == frame)
  //   return cache_[pdf_id].second;
  BaseFloat ans = ac_model_.LogLikelihood(pdf_id, cur_feats_) * ac_scale_;
  // cache_[pdf_id].first = frame;
  // cache_[pdf_id].second = ans;
  return ans;
}

}