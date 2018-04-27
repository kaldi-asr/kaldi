#ifndef KALDI_ONLINE_GPU_ONLINE_FEAT_INPUT_H_
#define KALDI_ONLINE_GPU_ONLINE_FEAT_INPUT_H_

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "online/online-feat-input.h"

namespace kaldi{

struct GPUOnlineFeatureMatrix{

  const OnlineFeatureMatrixOptions opts_;
  OnlineFeatInputItf *input_; // TODO : ini harus diganti
  int32 feat_dim_;
  thrust::device_vector<BaseFloat> feat_matrix_;//Matrix<BaseFloat> feat_matrix_;
  int32 feat_offset_; // the offset of the first frame in the current batch
  bool finished_; // True if there are no more frames to be got from the input.

  GPUOnlineFeatureMatrix(OnlineFeatureMatrix& ofm):
    opts_(ofm.opts()),
    feat_dim_(ofm.feat_dim()),
    feat_offset_(ofm.feat_offset()),
    finished_(ofm.finished())
  {
    const size_t feat_matrix_dim = t.feat_matrix().SizeInBytes() / sizeof(BaseFloat);
    BaseFloat* feat_matrix_data = t.feat_matrix().Data();
    thrust::copy(
      feat_matrix_data,
      feat_matrix_data + feat_matrix_dim,
      feat_matrix_.begin()
    );
  }
};

}


#endif