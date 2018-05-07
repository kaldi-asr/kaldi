#ifndef KALDI_ONLINE_GPU_ONLINE_DECODABLE_H_
#define KALDI_ONLINE_GPU_ONLINE_DECODABLE_H_

#include "gpucommons/gpu-vector.h"
#include "gpu/gpu-am-diag-gmm.h"
#include "gpu/gpu-transition-model.h"

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace kaldi{

struct GPUOnlineDecodableDiagGmmScaled {

  // nggak perlu masukin OnlineFeatureMatrixnya jadinya (parsial aja)

  GPUAmDiagGmm ac_model_;
  GPUTransitionModel transition_model_;
  BaseFloat ac_scale_;
  int32 feat_dim_;

  GPUVector<BaseFloat> cur_feats_;//Vector<BaseFloat> cur_feats_;

  int32 cur_frame_;

  GPUOnlineDecodableDiagGmmScaled();
  GPUOnlineDecodableDiagGmmScaled(
    const GPUAmDiagGmm& gpu_ac_model_,
    const GPUTransitionModel& gpu_transition_model_,
    BaseFloat ac_scale_
  );

  __device__ BaseFloat LogLikelihood(int32 frame, int32 index);
};

}

#endif
