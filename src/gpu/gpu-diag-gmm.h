#ifndef KALDI_GMM_GPU_DIAG_GMM_H_
#define KALDI_GMM_GPU_DIAG_GMM_H_ 1

#include "gmm/diag-gmm.h"
#include "gpu_commons/gpu_matrix.hpp"
#include "gpu_commons/gpu_vector.hpp"

namespace kaldi{

struct GPUDiagGmm{

  GPUVector<BaseFloat> gconsts_;
  GPUVector<BaseFloat> weights_;
  GPUMatrix<BaseFloat> inv_vars_;
  GPUMatrix<BaseFloat> means_invvars_;

  bool valid_gconsts_;  // bool valid_gconsts_;   ///< Recompute gconsts_ if false

  GPUDiagGmm(DiagGmm &d):
    valid_gconsts_(d.valid_gconsts_()),
    gconsts_(d.gconsts()),
    weights_(d.weights()),
    inv_vars_(d.inv_vars()),
    means_invvars_(d.means_invvars()) {}

  // TODO : Implement this!
  __device__ BaseFloat LogLikelihood(BaseFloat *data, int32 num_data){
    if (!valid_gconsts_)
      KALDI_ERR << "Must call ComputeGconsts() before computing likelihood";
    BaseFloat* loglikes;
    int32 num_loglikes;
    LogLikelihoods(data, num_data, loglikes, &num_loglikes);
    
    BaseFloat log_sum = loglikes.LogSumExp(); // TODO : implement ini disini
    if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
      KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";

    free(loglikes);
    return log_sum;
  }

  __device__ void LogLikelihoods(BaseFloat* data, int32 num_data, BaseFloat* loglikes, int32* num_loglikes){
    loglikes = malloc(gconsts_.Dim() * sizeof(BaseFloat));
    *num_loglikes = gconsts_.Dim();
    for(int i = 0;i < *num_loglikes; ++i) loglikes[i] = gconsts_[i];

    if (num_data != Dim()) {
      KALDI_ERR << "DiagGmm::ComponentLogLikelihood, dimension "
                << "mismatch " << num_data << " vs. "<< Dim();
    }
    
    BaseFloat* data_sq = malloc(num_data * sizeof(BaseFloat));
    
    for(int i = 0;i < num_data; ++i) data_sq[i] = data[i] * data[i];
    
    for(int i = 0;i < gconsts_.Dim(); ++i){
      for(int j = 0;j < num_data; ++j){
        loglikes[i] += means_invvars_[Index(i, j)] * data[j];
        loglikes[i] -= 0.5 * inv_vars_[Index(i, j)] * data_sq[j];
      }
    }

    free(data_sq);
  }
};

}

#endif