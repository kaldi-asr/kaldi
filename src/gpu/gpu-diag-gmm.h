#ifndef KALDI_GMM_GPU_DIAG_GMM_H_
#define KALDI_GMM_GPU_DIAG_GMM_H_

#include "gmm/diag-gmm.h"
#include "gpucommons/gpu-matrix.h"
#include "gpucommons/gpu-vector.h"
#include "base/kaldi-math.h"

#include <algorithm>

namespace kaldi{

struct GPUDiagGmm{
  GPUVector<BaseFloat> gconsts_;
  GPUVector<BaseFloat> weights_;
  GPUMatrix<BaseFloat> inv_vars_;
  GPUMatrix<BaseFloat> means_invvars_;

  bool valid_gconsts_;  // bool valid_gconsts_;   ///< Recompute gconsts_ if false

  GPUDiagGmm() {}
  GPUDiagGmm(DiagGmm &d);

  __device__
  int32 Dim() const;

  __device__
  BaseFloat LogLikelihood(BaseFloat *data, int32 num_data);

};

}

#endif
