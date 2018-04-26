#ifndef KALDI_GMM_GPU_DIAG_GMM_H_
#define KALDI_GMM_GPU_DIAG_GMM_H_ 1

#include <vector>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "gmm/diag-gmm.h"

namespace kaldi{

struct GPUDiagGmm{

  thrust::device_vector<BaseFloat> gconsts_; // Vector<BaseFloat> gconsts_;
  bool valid_gconsts_;  // bool valid_gconsts_;   ///< Recompute gconsts_ if false
  thrust::device_vector<BaseFloat> weights_; // Vector<BaseFloat> weights_;        ///< weights (not log).
  thrust::device_vector<BaseFloat> inv_vars_;  // Matrix<BaseFloat> inv_vars_;       ///< Inverted (diagonal) variances
  thrust::device_vector<BaseFloat> means_invvars_; // Matrix<BaseFloat> means_invvars_;  ///< Means times inverted variance

  GPUDiagGmm(DiagGmm &d):
    valid_gconsts_(d.valid_gconsts_()),
  {
    int verbose = 0;
    
    // copy gconsts_
    const size_t gconsts_dim = d.gconsts().SizeInBytes() / sizeof(BaseFloat);
    BaseFloat* gconsts_data = d.gconsts().Data();
    thrust::copy(gconsts_data, gconsts_data + gconsts_dim, gconsts_.begin());

    // copy weights_
    const size_t weights_dim = d.weights().SizeInBytes() / sizeof(BaseFloat);
    BaseFloat* weights_data = d.weights().Data();
    thrust::copy(weights_data, weights_data + weights_dim, weights_.begin());

    // copy inv_vars_
    const size_t inv_vars_dim = d.inv_vars().SizeInBytes() / sizeof(BaseFloat);
    BaseFloat* inv_vars_data = d.inv_vars().Data();
    thrust::copy(inv_vars_data, inv_vars_data + inv_vars_dim, inv_vars_.begin());

    // copy means_invvars_ 
    const size_t means_invvars_dim = d.means_invvars().SizeInBytes() / sizeof(BaseFloat);
    BaseFloat* means_invvars_data = d.means_invvars().Data();
    thrust::copy(means_invvars_data, means_invvars_data + means_invvars_dim, means_invvars_.begin());    
  }

  // TODO : FUNGSI
};

}

#endif