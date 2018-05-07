#include "gpu/gpu-diag-gmm.h"

#include <algorithm>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace kaldi{

struct _GPUDiagGmm{

  GPUVector<BaseFloat> gconsts_;
  GPUVector<BaseFloat> weights_;
  GPUMatrix<BaseFloat> inv_vars_;
  GPUMatrix<BaseFloat> means_invvars_;

  bool valid_gconsts_;  // bool valid_gconsts_;   ///< Recompute gconsts_ if false

  int32 Dim() const { return means_invvars_.NumCols(); }

  _GPUDiagGmm(DiagGmm &d):
    valid_gconsts_(d.valid_gconsts_()),
    gconsts_(d.gconsts()),
    weights_(d.weights()),
    inv_vars_(d.inv_vars()),
    means_invvars_(d.means_invvars()) {}

  // TODO : Implement this!
  __host__ __device__ BaseFloat LogLikelihood(BaseFloat *data, int32 num_data){
    if (!valid_gconsts_)
      KALDI_ERR << "Must call ComputeGconsts() before computing likelihood";

    /* BEGIN LogLikelihoods */
    int32 num_loglikes = gconsts_.Dim();
    BaseFloat* loglikes = new BaseFloat[num_loglikes];
    for(int32 i = 0;i < num_loglikes; ++i) loglikes[i] = gconsts_.data[i];

    if (num_data != Dim()) {
      KALDI_ERR << "DiagGmm::ComponentLogLikelihood, dimension "
                << "mismatch " << num_data << " vs. "<< Dim();
    }

    BaseFloat* data_sq = new BaseFloat[num_data];
    for(int32 i = 0;i < num_data; ++i) data_sq[i] = data[i] * data[i];

    for(int i = 0;i < gconsts_.Dim(); ++i){
      for(int j = 0;j < num_data; ++j){
        loglikes[i] += means_invvars_.data[means_invvars_.Index(i, j)] * data[j];
        loglikes[i] -= 0.5 * inv_vars_.data[inv_vars_.Index(i, j)] * data_sq[j];
      }
    }

    /* END LogLikelihoods */

    /* Begin Log Sum Exp */
    BaseFloat max_elem = *(std::max_element(loglikes, loglikes + num_loglikes));

    BaseFloat cutoff;
    if (sizeof(Real) == 4) cutoff = max_elem + kMinLogDiffFloat;
    else cutoff = max_elem + kMinLogDiffDouble;
    double sum_relto_max_elem = 0.0;

    for (int32 i = 0; i < num_loglikes; i++) {
      BaseFloat f = loglikes[i];
      if (f >= cutoff)
        sum_relto_max_elem += Exp(f - max_elem);
    }
    BaseFloat log_sum = max_elem + Log(sum_relto_max_elem);
    /* End Log Sum Exp */

    if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
      KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";

    delete [] loglikes;
    delete [] data_sq;
    return log_sum;
  }
};

}