// sgmm/estimate-am-sgmm-multi.h

// Copyright 2012       Arnab Ghoshal

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

#ifndef KALDI_SGMM_ESTIMATE_AM_SGMM_MULTI_H_
#define KALDI_SGMM_ESTIMATE_AM_SGMM_MULTI_H_ 1

#include <string>
#include <vector>

#include "sgmm/am-sgmm.h"
#include "sgmm/estimate-am-sgmm.h"
#include "gmm/model-common.h"

namespace kaldi {

/** \class MleAmSgmmGlobalAccs
 *  Class for the accumulators associated with SGMM global parameters (e.g.
 *  phonetic-, weight- and speaker-projections; and covariances). This is
 *  used when the global parameters are updated using stats from multiple
 *  models.
 */
class MleAmSgmmGlobalAccs {
 public:
  explicit MleAmSgmmGlobalAccs()
      : feature_dim_(0), phn_space_dim_(0), spk_space_dim_(0),
        num_gaussians_(0), total_frames_(0.0), total_like_(0.0) {}

  /// Resizes the accumulators to the correct sizes given the model. The flags
  /// argument control which accumulators to resize.
  void ResizeAccumulators(const AmSgmm &model, SgmmUpdateFlagsType flags);

  /// Set the accumulators specified by the flags argument to zero.
  void ZeroAccumulators(SgmmUpdateFlagsType flags);

  /// Add another accumulator object
  void AddAccumulators(const AmSgmm &model, const MleAmSgmmAccs &acc,
                       SgmmUpdateFlagsType flags);

  int32 FeatureDim() const { return feature_dim_; }
  int32 PhoneSpaceDim() const { return phn_space_dim_; }
  int32 NumGauss() const { return num_gaussians_; }

 private:
  /// The stats which are not tied to any state.
  /// Stats Y_{i} for phonetic-subspace projections M; Dim is [I][D][S].
  std::vector< Matrix<double> > Y_;
  /// Stats Z_{i} for speaker-subspace projections N. Dim is [I][D][T].
  std::vector< Matrix<double> > Z_;
  /// R_{i}, quadratic term for speaker subspace estimation. Dim is [I][T][T]
  std::vector< SpMatrix<double> > R_;
  /// S_{i}^{-}, scatter of adapted feature vectors x_{i}(t). Dim is [I][D][D].
  std::vector< SpMatrix<double> > S_;
  /// Total occupancies gamma_i for each Gaussian. Dim is [I]
  Vector<double> gamma_i_;

  /// Q_{i}, quadratic term for phonetic subspace estimation. Dim is [I][S][S]
  std::vector< SpMatrix<double> > Q_;
  /// Eq (74): S_{i}^{(means)}, scatter of substate mean vectors for estimating
  /// the shared covariance matrices. Dimension is [I][D][D].
  std::vector< SpMatrix<double> > S_means_;

  /// Dimensionality of various subspaces
  int32 feature_dim_, phn_space_dim_, spk_space_dim_;
  int32 num_gaussians_;  ///< Other model specifications

  double total_frames_, total_like_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmSgmmGlobalAccs);
  friend class MleAmSgmmUpdaterMulti;
};


/** \class MleAmSgmmUpdaterMulti
 *  Contains the functions needed to update the parameters for multiple SGMMs
 *  whose global parameters are tied.
 */
class MleAmSgmmUpdaterMulti {
 public:
  explicit MleAmSgmmUpdaterMulti(const AmSgmm &model,
                                 const MleAmSgmmOptions &options)
      : update_options_(options), global_SigmaInv_(model.SigmaInv_),
        global_M_(model.M_), global_N_(model.N_), global_w_(model.w_) {}

  void Update(const std::vector<MleAmSgmmAccs*> &accs,
              const std::vector<AmSgmm*> &models,
              SgmmUpdateFlagsType flags);

  /// Various model dimensions.
  int32 NumGauss() const { return global_M_.size(); }
  int32 PhoneSpaceDim() const { return global_w_.NumCols(); }
  int32 SpkSpaceDim() const {
    return (global_N_.size() > 0) ? global_N_[0].NumCols() : 0;
  }
  int32 FeatureDim() const { return global_M_[0].NumRows(); }

 private:
  MleAmSgmmOptions update_options_;

  /// SGMM global parameters that will be updated together and copied to the
  /// different models:
  std::vector< SpMatrix<BaseFloat> > global_SigmaInv_;
  std::vector< Matrix<BaseFloat> > global_M_;
  std::vector< Matrix<BaseFloat> > global_N_;
  Matrix<BaseFloat> global_w_;

  BaseFloat UpdateGlobals(const MleAmSgmmGlobalAccs &glob_accs,
                          SgmmUpdateFlagsType flags);

  double UpdateM(const MleAmSgmmGlobalAccs &accs);
  double UpdateN(const MleAmSgmmGlobalAccs &accs);
  double UpdateVars(const MleAmSgmmGlobalAccs &accs);
  double UpdateWParallel(const std::vector<MleAmSgmmAccs*> &accs,
                         const std::vector<AmSgmm*> &models);
//  double UpdateWSequential(const std::vector<MleAmSgmmAccs*> &accs,
//                           const std::vector<AmSgmm*> &models);

  void ComputeSmoothingTerms(const MleAmSgmmGlobalAccs &accs,
                             const std::vector<SpMatrix<double> > &H,
                             SpMatrix<double> *H_sm) const;
  void RenormalizeV(const SpMatrix<double> &H_sm,
                    const std::vector<AmSgmm*> &models);

  KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmSgmmUpdaterMulti);
  MleAmSgmmUpdaterMulti() {}  // Prevent unconfigured updater.
};

}  // namespace kaldi


#endif  // KALDI_SGMM_ESTIMATE_AM_SGMM_MULTI_H_
