#ifndef KALDI_GMM_GPU_DIAG_GMM_H_
#define KALDI_GMM_GPU_DIAG_GMM_H_

#include "gmm/diag-gmm.h"
#include "gpucommons/gpu-matrix.h"
#include "gpucommons/gpu-vector.h"
#include "base/kaldi-math.h"

#include <algorithm>

namespace kaldi{

struct _GPUDiagGmm{
  GPUVector<BaseFloat> gconsts_;
  GPUVector<BaseFloat> weights_;
  GPUMatrix<BaseFloat> inv_vars_;
  GPUMatrix<BaseFloat> means_invvars_;

  bool valid_gconsts_;  // bool valid_gconsts_;   ///< Recompute gconsts_ if false

  _GPUDiagGmm(DiagGmm &d);

  __host__ __device__
  int32 Dim() const;

  __host__ __device__
  BaseFloat LogLikelihood(BaseFloat *data, int32 num_data);

};

typedef struct _GPUDiagGmm GPUDiagGmm;

}

#endif
