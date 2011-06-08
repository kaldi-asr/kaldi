// sgmm/estimate-am-sgmm.h

// Copyright 2009-2011  Microsoft Corporation, Lukas Burget, Arnab Ghoshal (Saarland University), Ondrej Glembek,
//                 Yanmin Qian

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

#ifndef KALDI_SGMM_ESTIMATE_AM_SGMM_H_
#define KALDI_SGMM_ESTIMATE_AM_SGMM_H_ 1

#include <string>
#include <vector>

#include "sgmm/am-sgmm.h"
#include "gmm/model-common.h"
#include "util/parse-options.h"

namespace kaldi {

/** \struct MleAmSgmmOptions
 *  Configuration variables needed in the SGMM estimation process.
 */
struct MleAmSgmmOptions {
  /// Configuration Parameters.  See initialization code for more comments.
  BaseFloat tau_vec;  ///< Amount of smoothing for v_{jm} update
  BaseFloat tau_c;    ///< Tau value for smoothing substate weights (c)
  /// Floor covariance matrices Sigma_i to this times average cov.
  BaseFloat cov_floor;
  /// ratio to dim below which we use diagonal. default 2, set to inf for diag.
  BaseFloat cov_diag_ratio;
  /// Max on condition of matrices in update beyond which we do not update.
  /// Should probably be related to numerical properties of machine
  /// or BaseFloat type.
  BaseFloat max_cond;
  /// Limits condition of smoothing matrices H_sm (e.g. 100).
  /// Only really important on 1st iter if using priors.
  BaseFloat max_cond_H_sm;
  /// Fix for the smoothing approach, necessary if max_cond_H_sm != inf
  /// note: only has an effect if tau_vec != 0.
  bool fixup_Hk_sm;

  bool renormalize_V;
  bool renormalize_N;

  /// Number of iters when re-estimating weight projections "w".
  int weight_projections_iters;
  /// The "sequential" weight update that checks each i in turn.
  /// (if false, uses the "parallel" one).
  bool use_sequential_weight_update;

  int compress_m_dim;  // dimension of subspace to limit the M's to (if nonzero)
  int compress_n_dim;  // dimension of subspace to limit the N's to (if nonzero)
  int compress_vars_dim;  // dimension of subspace to limit the log SigmaInv's
  // to (if nonzero) [not used yet]

  BaseFloat epsilon;  ///< very small value used to prevent SVD crashing.

  MleAmSgmmOptions() {
    // tau value used in smoothing vector re-estimation (if no prior used).
    tau_vec = 0.0;
    tau_c  = 5.0;
    cov_floor = 0.025;
    cov_diag_ratio = 2.0;  // set to very large to get diagonal-cov models.
    max_cond = 1.0e+05;
    epsilon = 1.0e-40;
    max_cond_H_sm = 1.0e+05; // only real significance in normal situation is for diagnostics.
    fixup_Hk_sm = true;
    renormalize_V = true;
    renormalize_N = false;  // default to false since will invalidate spk vectors
    // on disk.
    weight_projections_iters = 3;
    use_sequential_weight_update = false;
    compress_m_dim = 0;
    compress_n_dim = 0;
    compress_vars_dim = 0;
  }

  void Register(ParseOptions *po) {
    std::string module = "MleAmSgmmOptions: ";
    po->Register("tau-vec", &tau_vec, module+
                 "Smoothing for phone vector estimation.");
    po->Register("tau-c", &tau_c, module+
                 "Smoothing for substate weights estimation.");
    po->Register("cov-floor", &cov_floor, module+
                 "Covariance floor (fraction of average covariance).");
    po->Register("cov-diag-ratio", &cov_diag_ratio, module+
                 "Minumum occ/dim ratio below which use diagonal covariances.");
    po->Register("max-cond", &max_cond, module+
                 "Maximum condition number beyond which matrices are not updated.");
    po->Register("weight-projections-iters", &weight_projections_iters, module+
                 "Number for iterations for weight projection estimation.");
    po->Register("renormalize-v", &renormalize_V, module+
                 "If true, renormalize the phonetic-subspace vectors to have meaningful sizes.");
    po->Register("renormalize-n", &renormalize_N, module+
                 "If true, renormalize the speaker subspace to have meaningful sizes.");
    po->Register("compress-m-dim", &compress_m_dim,
                 "If nonzero, limit the M matrices to a subspace of this dimension.");
    po->Register("compress-n-dim", &compress_n_dim,
                 "If nonzero, limit the N matrices to a subspace of this dimension.");
    po->Register("compress-vars-dim", &compress_vars_dim,
                 "If nonzero, limit the SigmaInv matrices to a subspace of this dimension.");
  }
};

/** \class MleAmSgmmAccs
 *  Class for the accumulators associated with the phonetic-subspace model
 *  parameters
 */
class MleAmSgmmAccs {
 public:
  explicit MleAmSgmmAccs(BaseFloat rand_prune = 1.0e-05)
      : total_frames_(0.0), total_like_(0.0), feature_dim_(0),
        phn_space_dim_(0), spk_space_dim_(0), num_gaussians_(0),
        num_states_(0), rand_prune_(rand_prune) {}

  MleAmSgmmAccs(const AmSgmm &model, SgmmUpdateFlagsType flags,
                BaseFloat rand_prune = 1.0e-05)
      : total_frames_(0.0), total_like_(0.0), rand_prune_(rand_prune) {
    ResizeAccumulators(model, flags);
  }

  ~MleAmSgmmAccs();

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Checks the various accumulators for correct sizes given a model. With
  /// wrong sizes, assertion failure occurs. When the show_properties argument
  /// is set to true, dimensions and presence/absence of the various
  /// accumulators are printed. For use when accumulators are read from file.
  void Check(const AmSgmm &model, bool show_properties = true) const;

  /// Resizes the accumulators to the correct sizes given the model. The flags
  /// argument control which accumulators to resize.
  void ResizeAccumulators(const AmSgmm &model, SgmmUpdateFlagsType flags);

  /// Set the accumulators specified by the flags argument to zero.
  void ZeroAccumulators(SgmmUpdateFlagsType flags);

  /// Returns likelihood.
  BaseFloat Accumulate(const AmSgmm &model,
                       const SgmmPerFrameDerivedVars& frame_vars,
                       const VectorBase<BaseFloat> &v_s,  // may be empty
                       int32 state_index, BaseFloat weight,
                       SgmmUpdateFlagsType flags);

  /// Returns count accumulated (may differ from posteriors.Sum()
  /// due to weight pruning).
  BaseFloat AccumulateFromPosteriors(const AmSgmm &model,
                                     const SgmmPerFrameDerivedVars& frame_vars,
                                     const Matrix<BaseFloat> &posteriors,
                                     const VectorBase<BaseFloat> &v_s,  // may be empty
                                     int32 state_index,
                                     SgmmUpdateFlagsType flags);

  /// Accumulates global stats for the current speaker (if applicable).
  /// If flags contains kSgmmSpeakerProjections (N), must call
  /// this after finishing the speaker's data.
  void CommitStatsForSpk(const AmSgmm& model,
                         const VectorBase<BaseFloat> &v_s);

  /// Accessors
  void GetStateOccupancies(Vector<BaseFloat> *occs) const;
  int32 FeatureDim() const { return feature_dim_; }
  int32 PhoneSpaceDim() const { return phn_space_dim_; }
  int32 NumStates() const { return num_states_; }
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

  /// The SGMM state specific stats.
  /// Statistics y_{jm} for state vectors v_{jm}. dimension is [J][M_{j}[S].
  std::vector< Matrix<double> > y_;
  /// Gaussian occupancies gamma_{jmi} for each substate. Dim is [J][M_{j}][I].
  std::vector< Matrix<double> > gamma_;

  /// gamma_{i}^{(s)}.  Per-speaker counts for each Gaussian. Dimension is [I]
  /// Needed for stats R_.
  Vector<double> gamma_s_;

  double total_frames_, total_like_;

  /// Dimensionality of various subspaces
  int32 feature_dim_, phn_space_dim_, spk_space_dim_;
  int32 num_gaussians_, num_states_;  ///< Other model specifications

  BaseFloat rand_prune_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmSgmmAccs);
  friend class MleAmSgmmUpdater;
};

/** \class MleAmSgmmUpdater
 *  Contains the functions needed to update the SGMM parameters.
 */
class MleAmSgmmUpdater {
 public:
  explicit MleAmSgmmUpdater(const MleAmSgmmOptions &options)
      : update_options_(options) {}
  void Reconfigure(const MleAmSgmmOptions &options) {
    update_options_ = options;
  }

  void Update(const MleAmSgmmAccs &accs,
              AmSgmm *model,
              SgmmUpdateFlagsType flags);

 private:
  MleAmSgmmOptions update_options_;
  /// Q_{i}, quadratic term for phonetic subspace estimation. Dim is [I][S][S]
  std::vector< SpMatrix<double> > Q_;
  /// Eq (74): S_{i}^{(means)}, scatter of substate mean vectors for estimating
  /// the shared covariance matrices. Dimension is [I][D][D].
  std::vector< SpMatrix<double> > S_means_;
  Vector<double> gamma_j_;  ///< State occupancies

  /** Compute the Q_{i} (Eq. 64), S_{i}^{(means)} (Eq. 74), and H_{i}^{spk}
   * (Eq. 82) stats needed to update the phonetic subspace, covariances, and
   * speaker vectors respectively.
   */
  void PreComputeStats(const MleAmSgmmAccs &accs,
                       const AmSgmm &model, SgmmUpdateFlagsType flags);

  void ComputeSmoothingTerms(const MleAmSgmmAccs &accs,
                             const AmSgmm &model,
                             const std::vector< SpMatrix<double> > &H,
                             SpMatrix<double> *H_sm,
                             Vector<double> *y_sm) const;

  double UpdatePhoneVectors(const MleAmSgmmAccs &accs,
                               AmSgmm *model,
                               const std::vector<SpMatrix<double> > &H,
                               const SpMatrix<double> &H_sm,
                               const Vector<double> &y_sm);

  double UpdateM(const MleAmSgmmAccs &accs, AmSgmm *model);
  double UpdateMCompress(const MleAmSgmmAccs &accs, AmSgmm *model);

  void RenormalizeV(const MleAmSgmmAccs &accs, AmSgmm *model,
                    const SpMatrix<double> &H_sm);
  double UpdateN(const MleAmSgmmAccs &accs, AmSgmm *model);
  double UpdateNCompress(const MleAmSgmmAccs &accs, AmSgmm *model);
  void RenormalizeN(const MleAmSgmmAccs &accs, AmSgmm *model);
  double UpdateVars(const MleAmSgmmAccs &accs, AmSgmm *model);
  double UpdateVarsCompress(const MleAmSgmmAccs &accs, AmSgmm *model);
  double UpdateWParallel(const MleAmSgmmAccs &accs, AmSgmm *model);
  double UpdateWSequential(const MleAmSgmmAccs &accs,
                              AmSgmm *model);
  double UpdateSubstateWeights(const MleAmSgmmAccs &accs,
                                  AmSgmm *model);

  KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmSgmmUpdater);
  MleAmSgmmUpdater() {}  // Prevent unconfigured updater.
};


/** \class MleSgmmSpeakerAccs
 *  Class for the accumulators required to update the speaker
 *  vectors v_s.
 *  Note: if you have multiple speakers you will want to initialize
 *  this just once and call Clear() after you're done with each speaker,
 *  rather than creating a new object for each speaker, since the
 *  initialization function does nontrivial work.
 */

class MleSgmmSpeakerAccs {
 public:
  /// Initialize the object.  Error if speaker subspace not set up.
  MleSgmmSpeakerAccs(const AmSgmm &model, BaseFloat rand_prune_ = 1.0e-05);

  /// Clear the statistics.
  void Clear();

  /// Accumulate statistics.  Returns per-frame log-likelihood.
  BaseFloat Accumulate(const AmSgmm &model,
                       const SgmmPerFrameDerivedVars& frame_vars,
                       int32 state_index, BaseFloat weight);

  /// Accumulate statistics, given posteriors.  Returns total
  /// count accumulated, which may differ from posteriors.Sum()
  /// due to randomized pruning.
  BaseFloat AccumulateFromPosteriors(const AmSgmm &model,
                                     const SgmmPerFrameDerivedVars& frame_vars,
                                     const Matrix<BaseFloat> &posteriors,
                                     int32 state_index);

  /// Update speaker vector.  If v_s was empty, will assume it started as zero
  /// and will resize it to the speaker-subspace size.
  void Update(BaseFloat min_count,  // e.g. 100
              Vector<BaseFloat> *v_s,
              BaseFloat *objf_impr_out,
              BaseFloat *count_out);

 private:
  /// Statistics for speaker adaptation (vectors), stored per-speaker.
  /// Per-speaker stats for vectors, y^{(s)}. Dimension [T].
  Vector<double> y_s_;
  /// gamma_{i}^{(s)}.  Per-speaker counts for each Gaussian. Dimension is [I]
  Vector<double> gamma_s_;

  /// The following variable does not change per speaker.
  /// Eq. (82): H_{i}^{spk} = N_{i}^T \Sigma_{i}^{-1} N_{i}
  std::vector< SpMatrix<double> > H_spk_;

  /// N_i^T \Sigma_{i}^{-1}. Needed for y^{(s)}
  std::vector< Matrix<double> > NtransSigmaInv_;

  /// small constant to randomly prune tiny posteriors
  BaseFloat rand_prune_;
};

}  // namespace kaldi


#endif  // KALDI_SGMM_ESTIMATE_AM_SGMM_H_
