#ifndef KALDI_ONLINE_GPU_ONLINE_DECODABLE_H_
#define KALDI_ONLINE_GPU_ONLINE_DECODABLE_H_

#include "online/online-decodable.h"

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace kaldi{

struct GPUOnlineDecodableDiagGmmScaled {

  // nggak perlu masukin OnlineFeatureMatrixnya jadinya

  const GPUAmDiagGmm& ac_model_;
  BaseFloat ac_scale_;
  const GPUTransitionModel& transition_model_;
  const int32 feat_dim_;

  thrust::device_vector<BaseFloat> cur_feats_;//Vector<BaseFloat> cur_feats_;
  int32 cur_frame_;
  thrust::device_vector<std::pair<int32, BaseFloat> > cache_;

  GPUOnlineDecodableDiagGmmScaled() {}
  GPUOnlineDecodableDiagGmmScaled(
    const GPUAmDiagGmm& gpu_ac_model_,
    BaseFloat ac_scale_,
    const GPUTransitionModel& gpu_transition_model_,
    const int32 feat_dim_
  ) : 
  ac_model_(gpu_ac_model_),
  ac_scale_(ac_scale_),
  transition_model_(gpu_transition_model_),
  feat_dim_(feat_dim_) {}

};  

}

#endif