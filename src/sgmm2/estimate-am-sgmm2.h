// sgmm2/estimate-am-sgmm2.h

// Copyright 2009-2011  Microsoft Corporation;  Lukas Burget;
//                      Saarland University (Author: Arnab Ghoshal);
//                      Ondrej Glembek;  Yanmin Qian;
// Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey)
//                      Liang Lu;  Arnab Ghoshal

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

#ifndef KALDI_SGMM2_ESTIMATE_AM_SGMM2_H_
#define KALDI_SGMM2_ESTIMATE_AM_SGMM2_H_ 1

#include <string>
#include <vector>

#include "sgmm2/am-sgmm2.h"
#include "gmm/model-common.h"
#include "itf/options-itf.h"
#include "thread/kaldi-thread.h"

namespace kaldi {

/** \struct MleAmSgmm2Options
 *  Configuration variables needed in the SGMM estimation process.
 */
struct MleAmSgmm2Options {
  /// Smoothing constant for sub-state weights [count to add to each one].
  BaseFloat tau_c;
  /// Floor covariance matrices Sigma_i to this times average cov.
  BaseFloat cov_floor;
  /// ratio to dim below which we use diagonal. default 2, set to inf for diag.
  BaseFloat cov_diag_ratio;
  /// Max on condition of matrices in update beyond which we do not update.
  /// Should probably be related to numerical properties of machine
  /// or BaseFloat type.
  BaseFloat max_cond;

  bool renormalize_V;  // Renormalize the phonetic space.
  bool renormalize_N;  // Renormalize the speaker space.

  /// Number of iters when re-estimating weight projections "w".
  int weight_projections_iters;

  BaseFloat epsilon;  ///< very small value used to prevent SVD crashing.
  BaseFloat max_impr_u; ///< max improvement per frame allowed in update of u.

  BaseFloat tau_map_M;  ///< For MAP update of the phonetic subspace M
  int map_M_prior_iters;  ///< num of iterations to update the prior of M
  bool full_row_cov;  ///< Estimate row covariance instead of using I
  bool full_col_cov;  ///< Estimate col covariance instead of using I

  MleAmSgmm2Options() {
    cov_floor = 0.025;
    tau_c  = 2.0;
    cov_diag_ratio = 2.0;  // set this to very large to get diagonal-cov models.
    max_cond = 1.0e+05;
    epsilon = 1.0e-40;
    renormalize_V = true;
    renormalize_N = false;  // default to false since will invalidate spk vectors
    // on disk.
    weight_projections_iters = 3;
    max_impr_u = 0.25;

    map_M_prior_iters = 5;
    tau_map_M = 0.0;  // No MAP update by default (~500-1000 depending on prior)
    full_row_cov = false;
    full_col_cov = false;
  }

  void Register(OptionsItf *opts) {
    std::string module = "MleAmSgmm2Options: ";
    opts->Register("tau-c", &tau_c, module+
                   "Count for smoothing weight update.");
    opts->Register("cov-floor", &cov_floor, module+
                   "Covariance floor (fraction of average covariance).");
    opts->Register("cov-diag-ratio", &cov_diag_ratio, module+
                   "Minimum occ/dim ratio below which use diagonal covariances.");
    opts->Register("max-cond", &max_cond, module+"Maximum condition number used to "
                   "regularize the solution of certain quadratic auxiliary functions.");
    opts->Register("weight-projections-iters", &weight_projections_iters, module+
                   "Number for iterations for weight projection estimation.");
    opts->Register("renormalize-v", &renormalize_V, module+"If true, renormalize "
                   "the phonetic-subspace vectors to have meaningful sizes.");
    opts->Register("renormalize-n", &renormalize_N, module+"If true, renormalize "
                   "the speaker subspace to have meaningful sizes.");
    opts->Register("max-impr-u", &max_impr_u, module+"Maximum objective function "
                   "improvement per frame allowed in update of u (to "
                   "maintain stability.");

    opts->Register("tau-map-M", &tau_map_M, module+"Smoothing for MAP estimate "
                   "of M (0 means ML update).");
    opts->Register("map-M-prior-iters", &map_M_prior_iters, module+
                   "Number of iterations to estimate prior covariances for M.");
    opts->Register("full-row-cov", &full_row_cov, module+
                   "Estimate row covariance instead of using I.");
    opts->Register("full-col-cov", &full_col_cov, module+
                   "Estimate column covariance instead of using I.");
  }
};

/** \class MleAmSgmm2Accs
 *  Class for the accumulators associated with the phonetic-subspace model
 *  parameters
 */
class MleAmSgmm2Accs {
 public:
  explicit MleAmSgmm2Accs(BaseFloat rand_prune = 1.0e-05)
      : total_frames_(0.0), total_like_(0.0), feature_dim_(0),
        phn_space_dim_(0), spk_space_dim_(0), num_gaussians_(0),
        num_pdfs_(0), num_groups_(0), rand_prune_(rand_prune) {}

  MleAmSgmm2Accs(const AmSgmm2 &model, SgmmUpdateFlagsType flags,
                 bool have_spk_vecs,
                 BaseFloat rand_prune = 1.0e-05)
      : total_frames_(0.0), total_like_(0.0), rand_prune_(rand_prune) {
    ResizeAccumulators(model, flags, have_spk_vecs);
  }

  ~MleAmSgmm2Accs();

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Checks the various accumulators for correct sizes given a model. With
  /// wrong sizes, assertion failure occurs. When the show_properties argument
  /// is set to true, dimensions and presence/absence of the various
  /// accumulators are printed. For use when accumulators are read from file.
  void Check(const AmSgmm2 &model, bool show_properties = true) const;

  /// Resizes the accumulators to the correct sizes given the model. The flags
  /// argument controls which accumulators to resize.
  void ResizeAccumulators(const AmSgmm2 &model, SgmmUpdateFlagsType flags,
                          bool have_spk_vecs);

  /// Returns likelihood.
  BaseFloat Accumulate(const AmSgmm2 &model,
                       const Sgmm2PerFrameDerivedVars &frame_vars,
                       int32 pdf_index, // == j2.
                       BaseFloat weight,
                       Sgmm2PerSpkDerivedVars *spk_vars);

  /// Returns count accumulated (may differ from posteriors.Sum()
  /// due to weight pruning).
  BaseFloat AccumulateFromPosteriors(const AmSgmm2 &model,
                                     const Sgmm2PerFrameDerivedVars &frame_vars,
                                     const Matrix<BaseFloat> &posteriors,
                                     int32 pdf_index, // == j2.
                                     Sgmm2PerSpkDerivedVars *spk_vars);

  /// Accumulates global stats for the current speaker (if applicable).  If
  /// flags contains kSgmmSpeakerProjections (N), or
  /// kSgmmSpeakerWeightProjections (u), must call this after finishing the
  /// speaker's data.
  void CommitStatsForSpk(const AmSgmm2 &model,
                         const Sgmm2PerSpkDerivedVars &spk_vars);

  /// Accessors
  void GetStateOccupancies(Vector<BaseFloat> *occs) const;
  int32 FeatureDim() const { return feature_dim_; }
  int32 PhoneSpaceDim() const { return phn_space_dim_; }
  int32 NumPdfs() const { return num_pdfs_; } // returns J2
  int32 NumGroups() const { return num_groups_; } // returns J1
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
  /// Statistics y_{jm} for state vectors v_{jm}. dimension is [J1][#mix][S].
  std::vector< Matrix<double> > y_;
  /// Gaussian occupancies gamma_{jmi} for each substate and Gaussian index,
  /// pooled over groups. Dim is [J1][#mix][I].
  std::vector< Matrix<double> > gamma_;

  /// [SSGMM] These a_{jmi} quantities are dimensionally the same
  /// as the gamma quantities.  They're needed to estimate the v_{jm}
  /// and w_i quantities in the symmetric SGMM.  Dimension is [J1][#mix][S]
  std::vector< Matrix<double> > a_;

  /// [SSGMM] each row is one of the t_i quantities in the less-exact
  /// version of the SSGMM update for the speaker weight projections.
  /// Dimension is [I][T]
  Matrix<double> t_;

  /// [SSGMM], this is a per-speaker variable storing the a_i^{(s)}
  /// quantities that we will use in order to compute the non-speaker-
  /// specific quantities [see eqs. 53 and 54 in techreport].  Note:
  /// there is a separate variable a_s_ in class MleSgmm2SpeakerAccs,
  /// which is the same thing but for purposes of computing
  /// the speaker-vector v^{(s)}.
  Vector<double> a_s_;

  /// the U_i quantities from the less-exact version of the SSGMM update for the
  /// speaker weight projections.  Dimension is [I][T][T]
  std::vector<SpMatrix<double> > U_;

  /// Sub-state occupancies gamma_{jm}^{(c)} for each sub-state.  In the
  /// SCTM version of the SGMM, for compactness we store two separate
  /// sets of gamma statistics, one to estimate the v_{jm} quantities
  /// and one to estimate the sub-state weights c_{jm}.
  std::vector< Vector<double> > gamma_c_;

  /// gamma_{i}^{(s)}.  Per-speaker counts for each Gaussian. Dimension is [I]
  /// Needed for stats R_.  This can be viewed as a temporary variable; it
  /// does not form part of the stats that we eventually dump to disk.
  Vector<double> gamma_s_;

  double total_frames_, total_like_;

  /// Dimensionality of various subspaces
  int32 feature_dim_, phn_space_dim_, spk_space_dim_;
  int32 num_gaussians_, num_pdfs_, num_groups_;  ///< Other model specifications

  BaseFloat rand_prune_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmSgmm2Accs);
  friend class MleAmSgmm2Updater;
  friend class EbwAmSgmm2Updater;
};

/** \class MleAmSgmmUpdater
 *  Contains the functions needed to update the SGMM parameters.
 */
class MleAmSgmm2Updater {
 public:
  explicit MleAmSgmm2Updater(const MleAmSgmm2Options &options)
      : options_(options) {}
  void Reconfigure(const MleAmSgmm2Options &options) {
    options_ = options;
  }

  void Update(const MleAmSgmm2Accs &accs,
              AmSgmm2 *model,
              SgmmUpdateFlagsType flags);

 private:
  friend class UpdateWClass;
  friend class UpdatePhoneVectorsClass;
  friend class EbwEstimateAmSgmm2;

  ///  Compute the Q_i quantities (Eq. 64).
  static void ComputeQ(const MleAmSgmm2Accs &accs,
                       const AmSgmm2 &model,
                       std::vector< SpMatrix<double> > *Q);

  /// Compute the S_means quantities, minus sum: (Y_i M_i^T + M_i Y_I^T).
  static void ComputeSMeans(const MleAmSgmm2Accs &accs,
                            const AmSgmm2 &model,
                            std::vector< SpMatrix<double> > *S_means);
  friend class EbwAmSgmm2Updater;

  MleAmSgmm2Options options_;

  // Called from UpdatePhoneVectors; updates a subset of states
  // (relates to multi-threading).
  void UpdatePhoneVectorsInternal(const MleAmSgmm2Accs &accs,
                                  const std::vector<SpMatrix<double> > &H,
                                  const std::vector<Matrix<double> > &log_a,
                                  AmSgmm2 *model,
                                  double *auxf_impr,
                                  int32 num_threads,
                                  int32 thread_id) const;

  double UpdatePhoneVectors(const MleAmSgmm2Accs &accs,
                            const std::vector<SpMatrix<double> > &H,
                            const std::vector<Matrix<double> > &log_a,
                            AmSgmm2 *model) const;

  double UpdateM(const MleAmSgmm2Accs &accs,
                 const std::vector< SpMatrix<double> > &Q,
                 const Vector<double> &gamma_i,
                 AmSgmm2 *model);

  void RenormalizeV(const MleAmSgmm2Accs &accs, AmSgmm2 *model,
                    const Vector<double> &gamma_i,
                    const std::vector<SpMatrix<double> > &H);

  double UpdateN(const MleAmSgmm2Accs &accs, const Vector<double> &gamma_i,
                 AmSgmm2 *model);
  void RenormalizeN(const MleAmSgmm2Accs &accs, const Vector<double> &gamma_i,
                    AmSgmm2 *model);
  double UpdateVars(const MleAmSgmm2Accs &accs,
                    const std::vector< SpMatrix<double> > &S_means,
                    const Vector<double> &gamma_i,
                    AmSgmm2 *model);
  // Update for the phonetic-subspace weight projections w_i
  double UpdateW(const MleAmSgmm2Accs &accs,
                 const std::vector<Matrix<double> > &log_a,
                 const Vector<double> &gamma_i,
                 AmSgmm2 *model);
  // Update for the speaker-subspace weight projections u_i [SSGMM]
  double UpdateU(const MleAmSgmm2Accs &accs, const Vector<double> &gamma_i,
                 AmSgmm2 *model);

  /// Called, multithreaded, inside UpdateW
  static
  void UpdateWGetStats(const MleAmSgmm2Accs &accs,
                       const AmSgmm2 &model,
                       const Matrix<double> &w,
                       const std::vector<Matrix<double> > &log_a,
                       Matrix<double> *F_i,
                       Matrix<double> *g_i,
                       double *tot_like,
                       int32 num_threads,
                       int32 thread_id);

  double UpdateSubstateWeights(const MleAmSgmm2Accs &accs,
                               AmSgmm2 *model);

  static void ComputeLogA(const MleAmSgmm2Accs &accs,
                          std::vector<Matrix<double> > *log_a); // [SSGMM]

  void ComputeMPrior(AmSgmm2 *model);  // TODO(arnab): Maybe make this static?
  double MapUpdateM(const MleAmSgmm2Accs &accs,
                    const std::vector< SpMatrix<double> > &Q,
                    const Vector<double> &gamma_i, AmSgmm2 *model);

  KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmSgmm2Updater);
  MleAmSgmm2Updater() {}  // Prevent unconfigured updater.
};


/** \class MleSgmm2SpeakerAccs
 *  Class for the accumulators required to update the speaker
 *  vectors v_s.
 *  Note: if you have multiple speakers you will want to initialize
 *  this just once and call Clear() after you're done with each speaker,
 *  rather than creating a new object for each speaker, since the
 *  initialization function does nontrivial work.
 */

class MleSgmm2SpeakerAccs {
 public:
  /// Initialize the object.  Error if speaker subspace not set up.
  MleSgmm2SpeakerAccs(const AmSgmm2 &model,
                      BaseFloat rand_prune_ = 1.0e-05);

  /// Clear the statistics.
  void Clear();

  /// Accumulate statistics.  Returns per-frame log-likelihood.
  BaseFloat Accumulate(const AmSgmm2 &model,
                       const Sgmm2PerFrameDerivedVars &frame_vars,
                       int32 pdf_index,
                       BaseFloat weight,
                       Sgmm2PerSpkDerivedVars *spk_vars);

  /// Accumulate statistics, given posteriors.  Returns total
  /// count accumulated, which may differ from posteriors.Sum()
  /// due to randomized pruning.
  BaseFloat AccumulateFromPosteriors(const AmSgmm2 &model,
                                     const Sgmm2PerFrameDerivedVars &frame_vars,
                                     const Matrix<BaseFloat> &posteriors,
                                     int32 pdf_index,
                                     Sgmm2PerSpkDerivedVars *spk_vars);

  /// Update speaker vector.  If v_s was empty, will assume it started as zero
  /// and will resize it to the speaker-subspace size.
  void Update(const AmSgmm2 &model,
              BaseFloat min_count,  // e.g. 100
              Vector<BaseFloat> *v_s,
              BaseFloat *objf_impr_out,
              BaseFloat *count_out);

 private:
  // Update without speaker-dependent weights (vectors u_i),
  // i.e. not symmetric SGMM (SSGMM)
  void UpdateNoU(Vector<BaseFloat> *v_s,
                 BaseFloat *objf_impr_out,
                 BaseFloat *count_out);
  // Update for SSGMM
  void UpdateWithU(const AmSgmm2 &model,
                   Vector<BaseFloat> *v_s,
                   BaseFloat *objf_impr_out,
                   BaseFloat *count_out);


  /// Statistics for speaker adaptation (vectors), stored per-speaker.
  /// Per-speaker stats for vectors, y^{(s)}. Dimension [T].
  Vector<double> y_s_;
  /// gamma_{i}^{(s)}.  Per-speaker counts for each Gaussian. Dimension is [I]
  Vector<double> gamma_s_;
  /// a_i^{(s)}.  For SSGMM.
  Vector<double> a_s_;

  /// The following variable does not change per speaker, it just
  /// relates to the speaker subspace.
  /// Eq. (82): H_{i}^{spk} = N_{i}^T \Sigma_{i}^{-1} N_{i}
  std::vector< SpMatrix<double> > H_spk_;

  /// N_i^T \Sigma_{i}^{-1}. Needed for y^{(s)}
  std::vector< Matrix<double> > NtransSigmaInv_;

  /// small constant to randomly prune tiny posteriors
  BaseFloat rand_prune_;
};

// This class, used in multi-core implementation of the updates of the "w_i"
// quantities, was previously in estimate-am-sgmm.cc, but is being moved to the
// header so it can be used in estimate-am-sgmm-ebw.cc.  It is responsible for
// computing, in parallel, the F_i and g_i quantities used in the updates of
// w_i.
class UpdateWClass: public MultiThreadable {
 public:
  UpdateWClass(const MleAmSgmm2Accs &accs,
               const AmSgmm2 &model,
               const Matrix<double> &w,
               const std::vector<Matrix<double> > &log_a,
               Matrix<double> *F_i,
               Matrix<double> *g_i,
               double *tot_like):
      accs_(accs), model_(model), w_(w), log_a_(log_a),
      F_i_ptr_(F_i), g_i_ptr_(g_i), tot_like_ptr_(tot_like) {
    tot_like_ = 0.0;
    F_i_.Resize(F_i->NumRows(), F_i->NumCols());
    g_i_.Resize(g_i->NumRows(), g_i->NumCols());
  }

  ~UpdateWClass() {
    F_i_ptr_->AddMat(1.0, F_i_, kNoTrans);
    g_i_ptr_->AddMat(1.0, g_i_, kNoTrans);
    *tot_like_ptr_ += tot_like_;
  }

  inline void operator() () {
    // Note: give them local copy of the sums we're computing,
    // which will be propagated to the total sums in the destructor.
    MleAmSgmm2Updater::UpdateWGetStats(accs_, model_, w_, log_a_,
                                      &F_i_, &g_i_, &tot_like_,
                                      num_threads_, thread_id_);
  }
 private:
  const MleAmSgmm2Accs &accs_;
  const AmSgmm2 &model_;
  const Matrix<double> &w_;
  const std::vector<Matrix<double> > &log_a_;
  Matrix<double> *F_i_ptr_;
  Matrix<double> *g_i_ptr_;
  Matrix<double> F_i_;
  Matrix<double> g_i_;
  double *tot_like_ptr_;
  double tot_like_;
};


}  // namespace kaldi


#endif  // KALDI_SGMM2_ESTIMATE_AM_SGMM2_H_
