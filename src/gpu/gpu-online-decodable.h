#ifndef KALDI_ONLINE_GPU_ONLINE_DECODABLE_H_
#define KALDI_ONLINE_GPU_ONLINE_DECODABLE_H_

#include "online/online-decodable.h"

#include "gpu_commons/gpu_vector.hpp"

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace kaldi{

struct GPUOnlineDecodableDiagGmmScaled {

  // nggak perlu masukin OnlineFeatureMatrixnya jadinya (parsial aja)

  const GPUAmDiagGmm& ac_model_;
  const GPUTransitionModel& transition_model_;
  BaseFloat ac_scale_;
  const int32 feat_dim_;

  GPUVector<BaseFloat> cur_feats_;//Vector<BaseFloat> cur_feats_;
  int32 cur_frame_;
  thrust::device_vector<std::pair<int32, BaseFloat> > cache_;

  GPUOnlineDecodableDiagGmmScaled() {}
  GPUOnlineDecodableDiagGmmScaled(
    const GPUAmDiagGmm& gpu_ac_model_,
    const GPUTransitionModel& gpu_transition_model_,
    BaseFloat ac_scale_
  ) : 
  ac_model_(gpu_ac_model_),
  transition_model_(gpu_transition_model_),
  ac_scale_(ac_scale_) {

    int32 num_pdfs = transition_model_.NumPdfs();
    cache_.resize(num_pdfs, std::pair<int32,BaseFloat>(-1, 0.0));

  }

  __host__ __device__ BaseFloat LogLikelihood(int32 frame, int32 index);
}

__host__ __device__ BaseFloat GPUOnlineDecodableDiagGmmScaled::LogLikelihood(int32 frame, int32 index){
  KALDI_ASSERT(frame == cur_frame_);
  int32 pdf_id = transition_model_.TransitionIdToPdf(index);
  // if (cache_[pdf_id].first == frame)
  //   return cache_[pdf_id].second;
  BaseFloat ans = ac_model_.LogLikelihood(pdf_id, cur_feats_) * ac_scale_;
  // cache_[pdf_id].first = frame;
  // cache_[pdf_id].second = ans;
  return ans;












}

};  





}

#endif
