// sgmm2/fmllr-sgmm2.h

// Copyright 2009-2012     Saarland University (author: Arnab Ghoshal)
//                         Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifndef KALDI_SGMM2_FMLLR_SGMM2_H_
#define KALDI_SGMM2_FMLLR_SGMM2_H_

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "sgmm2/am-sgmm2.h"
#include "transform/transform-common.h"
#include "util/kaldi-table.h"
#include "util/kaldi-holder.h"
#include "itf/options-itf.h"

namespace kaldi {

/** \struct Sgmm2FmllrConfig
 *  Configuration variables needed in the estimation of FMLLR for SGMMs.
 */
struct Sgmm2FmllrConfig {
  int32 fmllr_iters;  ///< Number of iterations in FMLLR estimation.
  int32 step_iters;  ///< Iterations to find optimal FMLLR step size.
  /// Minimum occupancy count to estimate FMLLR using basis matrices.
  BaseFloat fmllr_min_count_basis;
  /// Minimum occupancy count to estimate FMLLR without basis matrices.
  BaseFloat fmllr_min_count;
  /// Minimum occupancy count to stop using FMLLR bases and switch to
  /// regular FMLLR estimation.
  BaseFloat fmllr_min_count_full;
  /// Number of basis matrices to use for FMLLR estimation. Can only *reduce*
  /// the number of bases present. Overridden by the 'bases_occ_scale' option.
  int32 num_fmllr_bases;
  /// Scale per-speaker count to determine number of CMLLR bases.
  BaseFloat bases_occ_scale;

  Sgmm2FmllrConfig() {
    fmllr_iters = 5;
    step_iters = 10;
    fmllr_min_count_basis = 100.0;
    fmllr_min_count = 1000.0;
    fmllr_min_count_full = 5000.0;
    num_fmllr_bases = 50;
    bases_occ_scale = 0.2;
  }

  void Register(OptionsItf *opts);
};

inline void Sgmm2FmllrConfig::Register(OptionsItf *opts) {
  std::string module = "Sgmm2FmllrConfig: ";
  opts->Register("fmllr-iters", &fmllr_iters, module+
                 "Number of iterations in FMLLR estimation.");
  opts->Register("fmllr-step-iters", &step_iters, module+
                 "Number of iterations to find optimal FMLLR step size.");
  opts->Register("fmllr-min-count-bases", &fmllr_min_count_basis, module+
                 "Minimum occupancy count to estimate FMLLR using basis matrices.");
  opts->Register("fmllr-min-count", &fmllr_min_count, module+
                 "Minimum occupancy count to estimate FMLLR (without bases).");
  opts->Register("fmllr-min-count-full", &fmllr_min_count_full, module+
                 "Minimum occupancy count to stop using basis matrices for FMLLR.");
  opts->Register("fmllr-num-bases", &num_fmllr_bases, module+
                 "Number of FMLLR basis matrices.");
  opts->Register("fmllr-bases-occ-scale", &bases_occ_scale, module+
                 "Scale per-speaker count to determine number of CMLLR bases.");
}


/** \class Sgmm2FmllrGlobalParams
 *  Global adaptation parameters.
 */
class Sgmm2FmllrGlobalParams {
 public:
  void Init(const AmSgmm2 &sgmm, const Vector<BaseFloat> &state_occs);
  void Write(std::ostream &out_stream, bool binary) const;
  void Read(std::istream &in_stream, bool binary);
  bool IsEmpty() const {
    return (pre_xform_.NumRows() == 0 || inv_xform_.NumRows() == 0 ||
            mean_scatter_.Dim() == 0);
  }
  bool HasBasis() const { return fmllr_bases_.size() != 0; }

  /// Pre-transform matrix. Dim is [D][D+1].
  Matrix<BaseFloat> pre_xform_;
  /// Inverse of pre-transform. Dim is [D][D+1].
  Matrix<BaseFloat> inv_xform_;
  /// Diagonal of mean-scatter matrix. Dim is [D]
  Vector<BaseFloat> mean_scatter_;
  /// \tilde{W}_b.  [b][d][d], dim is [B][D][D+1].
  std::vector< Matrix<BaseFloat> > fmllr_bases_;
};

inline void Sgmm2FmllrGlobalParams::Init(const AmSgmm2 &sgmm,
                                        const Vector<BaseFloat> &state_occs) {
  sgmm.ComputeFmllrPreXform(state_occs, &pre_xform_, &inv_xform_,
                            &mean_scatter_);
}

/** \class FmllrSgmm2Accs
 *  Class for computing the accumulators needed for the maximum-likelihood
 *  estimate of FMLLR transforms for a subspace GMM acoustic model.
 */
class FmllrSgmm2Accs {
 public:
  FmllrSgmm2Accs() : dim_(-1) {}
  ~FmllrSgmm2Accs() {}

  void Init(int32 dim, int32 num_gaussians);
  void SetZero() { stats_.SetZero(); }

  void Write(std::ostream &out_stream, bool binary) const;
  void Read(std::istream &in_stream, bool binary, bool add);

  /// Accumulation routine that computes the Gaussian posteriors and calls
  /// the AccumulateFromPosteriors function with the computed posteriors.
  /// The 'data' argument is not FMLLR-transformed and is needed in addition
  /// to the the 'frame_vars' since the latter only contains a copy of the
  /// transformed feature vector.
  BaseFloat Accumulate(const AmSgmm2 &sgmm,                       
                       const VectorBase<BaseFloat> &data,
                       const Sgmm2PerFrameDerivedVars &frame_vars,
                       int32 state_index,
                       BaseFloat weight,
                       Sgmm2PerSpkDerivedVars *spk);

  void AccumulateFromPosteriors(const AmSgmm2 &sgmm,
                                const Sgmm2PerSpkDerivedVars &spk,
                                const VectorBase<BaseFloat> &data,
                                const std::vector<int32> &gauss_select,
                                const Matrix<BaseFloat> &posteriors,
                                int32 state_index);

  void AccumulateForFmllrSubspace(const AmSgmm2 &sgmm,
                                  const Sgmm2FmllrGlobalParams &fmllr_globals,
                                  SpMatrix<double> *grad_scatter);

  BaseFloat FmllrObjGradient(const AmSgmm2 &sgmm,
                             const Matrix<BaseFloat> &xform,
                             Matrix<BaseFloat> *grad_out,
                             Matrix<BaseFloat> *G_out) const;

  /// Computes the FMLLR transform from the accumulated stats, using the
  /// pre-transforms in fmllr_globals. Expects the transform matrix out_xform
  /// to be initialized to the correct size. Returns true if the transform was
  /// updated (i.e. had enough counts).
  bool Update(const AmSgmm2 &model,
              const Sgmm2FmllrGlobalParams &fmllr_globals,
              const Sgmm2FmllrConfig &opts, Matrix<BaseFloat> *out_xform,
              BaseFloat *frame_count, BaseFloat *auxf_improv) const;

  /// Accessors
  int32 Dim() const { return dim_; }
  const AffineXformStats &stats() const { return stats_; }

 private:
  AffineXformStats stats_;  ///< Accumulated stats
  int32 dim_;  ///< Dimension of feature vectors

  // Cannot have copy constructor and assigment operator
  KALDI_DISALLOW_COPY_AND_ASSIGN(FmllrSgmm2Accs);
};

/// Computes the fMLLR basis matrices given the scatter of the vectorized
/// gradients (eq: B.10). The result is stored in 'fmllr_globals'.
/// The actual number of bases may be less than 'num_fmllr_bases' depending
/// on the feature dimension and number of eigenvalues greater than 'min_eig'.
void EstimateSgmm2FmllrSubspace(const SpMatrix<double> &fmllr_grad_scatter,
                               int32 num_fmllr_bases, int32 feat_dim,
                               Sgmm2FmllrGlobalParams *fmllr_globals,
                               double min_eig = 0.0);

}  // namespace kaldi

#endif  // KALDI_SGMM2_FMLLR_SGMM2_H_
