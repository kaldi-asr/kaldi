// sgmm/estimate-am-sgmm.h

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

#ifndef KALDI_SGMM_ESTIMATE_AM_SGMM_H_
#define KALDI_SGMM_ESTIMATE_AM_SGMM_H_ 1

#include <string>
#include <vector>

#include "sgmm/am-sgmm.h"
#include "gmm/model-common.h"
#include "itf/options-itf.h"
#include "sgmm/sgmm-clusterable.h"
#include "thread/kaldi-thread.h"  // for MultiThreadable

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
  bool fixup_H_sm;
  /// Set check_v to true if you want to use the "checking" version of the update
  /// for the v's, in which it checks the "real" objective function value and
  /// backtracks if necessary;
  bool check_v;

  bool renormalize_V;  // Renormalize the phonetic space.
  bool renormalize_N;  // Renormalize the speaker space.

  /// Number of iters when re-estimating weight projections "w".
  int weight_projections_iters;
  /// The "sequential" weight update that checks each i in turn.
  /// (if false, uses the "parallel" one).
  bool use_sequential_weight_update;

  BaseFloat epsilon;  ///< very small value used to prevent SVD crashing.

  BaseFloat tau_map_M;  ///< For MAP update of the phonetic subspace M
  int map_M_prior_iters;  ///< num of iterations to update the prior of M
  bool full_row_cov;  ///< Estimate row covariance instead of using I
  bool full_col_cov;  ///< Estimate col covariance instead of using I

  MleAmSgmmOptions() {
    // tau value used in smoothing vector re-estimation (if no prior used).
    tau_vec = 0.0;
    tau_c  = 5.0;
    cov_floor = 0.025;
    cov_diag_ratio = 2.0;  // set to very large to get diagonal-cov models.
    max_cond = 1.0e+05;
    epsilon = 1.0e-40;
    max_cond_H_sm = 1.0e+05;  // only for diagnostics in normal situations.
    fixup_H_sm = true;
    check_v = false;  // for back-compat.
    renormalize_V = true;
    renormalize_N = false;  // default to false since will invalidate spk vectors
                            // on disk.
    weight_projections_iters = 3;
    use_sequential_weight_update = false;

    map_M_prior_iters = 5;
    tau_map_M = 0.0;  // No MAP update by default (~500-1000 depending on prior)
    full_row_cov = false;
    full_col_cov = false;
  }

  void Register(OptionsItf *po) {
    std::string module = "MleAmSgmmOptions: ";
    po->Register("tau-vec", &tau_vec, module+
                 "Smoothing for phone vector estimation.");
    po->Register("tau-c", &tau_c, module+
                 "Smoothing for substate weights estimation.");
    po->Register("cov-floor", &cov_floor, module+
                 "Covariance floor (fraction of average covariance).");
    po->Register("cov-diag-ratio", &cov_diag_ratio, module+
                 "Minimum occ/dim ratio below which use diagonal covariances.");
    po->Register("max-cond", &max_cond, module+"Maximum condition number beyond"
                 " which matrices are not updated.");
    po->Register("weight-projections-iters", &weight_projections_iters, module+
                 "Number for iterations for weight projection estimation.");
    po->Register("renormalize-v", &renormalize_V, module+"If true, renormalize "
                 "the phonetic-subspace vectors to have meaningful sizes.");
    po->Register("check-v", &check_v, module+"If true, check real auxf "
                 "improvement in update of v and backtrack if needed "
                 "(not compatible with smoothing v)");
    po->Register("renormalize-n", &renormalize_N, module+"If true, renormalize "
                 "the speaker subspace to have meaningful sizes.");

    po->Register("tau-map-M", &tau_map_M, module+"Smoothing for MAP estimate "
                 "of M (0 means ML update).");
    po->Register("map-M-prior-iters", &map_M_prior_iters, module+
                 "Number of iterations to estimate prior covariances for M.");
    po->Register("full-row-cov", &full_row_cov, module+
                 "Estimate row covariance instead of using I.");
    po->Register("full-col-cov", &full_col_cov, module+
                 "Estimate column covariance instead of using I.");
  }
};

/** \class MleAmSgmmAccs
 *  Class for the accumulators associated with the SGMM parameters except
 *  speaker vectors.
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

  /// Returns likelihood.
  BaseFloat Accumulate(const AmSgmm &model,
                       const SgmmPerFrameDerivedVars &frame_vars,
                       const VectorBase<BaseFloat> &v_s,  // spk-vec, may be empty
                       int32 state_index, BaseFloat weight,
                       SgmmUpdateFlagsType flags);

  /// Returns count accumulated (may differ from posteriors.Sum()
  /// due to weight pruning).
  BaseFloat AccumulateFromPosteriors(const AmSgmm &model,
                                     const SgmmPerFrameDerivedVars &frame_vars,
                                     const Matrix<BaseFloat> &posteriors,
                                     const VectorBase<BaseFloat> &v_s,  // may be empty
                                     int32 state_index,
                                     SgmmUpdateFlagsType flags);

  /// Accumulates global stats for the current speaker (if applicable).
  /// If flags contains kSgmmSpeakerProjections (N), must call
  /// this after finishing the speaker's data.
  void CommitStatsForSpk(const AmSgmm &model,
                         const VectorBase<BaseFloat> &v_s);

  /// Accessors
  void GetStateOccupancies(Vector<BaseFloat> *occs) const;
  const std::vector< Matrix<double> >& GetOccs() const {
    return gamma_;
  }
  int32 FeatureDim() const { return feature_dim_; }
  int32 PhoneSpaceDim() const { return phn_space_dim_; }
  int32 NumStates() const { return num_states_; }
  int32 NumGauss() const { return num_gaussians_; }
  double TotalFrames() const { return total_frames_; }
  double TotalLike() const { return total_like_; }

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
  friend class EbwAmSgmmUpdater;
  friend class MleAmSgmmGlobalAccs;
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

  /// Main update function: Computes some overall stats, does parameter updates
  /// and returns the total improvement of the different auxiliary functions.
  BaseFloat Update(const MleAmSgmmAccs &accs,
                   AmSgmm *model,
                   SgmmUpdateFlagsType flags);

  /// This function is like UpdatePhoneVectorsChecked, which supports
  /// objective-function checking and backtracking but no smoothing term, but it
  /// takes as input the stats used in SGMM-based tree clustering-- this is used
  /// in initializing an SGMM from the tree stats.  It's not part of the
  /// normal recipe.
  double UpdatePhoneVectorsCheckedFromClusterable(
      const std::vector<SgmmClusterable*> &stats,
      const std::vector<SpMatrix<double> > &H,
      AmSgmm *model);
  
 protected:
  friend class UpdateWParallelClass;
  friend class UpdatePhoneVectorsClass;
  friend class UpdatePhoneVectorsCheckedFromClusterableClass;
  friend class EbwEstimateAmSgmm;

  ///  Compute the Q_i quantities (Eq. 64).
  static void ComputeQ(const MleAmSgmmAccs &accs,
                       const AmSgmm &model,
                       std::vector< SpMatrix<double> > *Q);

  /// Compute the S_means quantities, minus sum: (Y_i M_i^T + M_i Y_I^T).
  static void ComputeSMeans(const MleAmSgmmAccs &accs,
                            const AmSgmm &model,
                            std::vector< SpMatrix<double> > *S_means);
  friend class EbwAmSgmmUpdater;
 private:
  MleAmSgmmOptions update_options_;
  /// Q_{i}, quadratic term for phonetic subspace estimation. Dim is [I][S][S]
  std::vector< SpMatrix<double> > Q_;

  /// Eq (74): S_{i}^{(means)}, scatter of substate mean vectors for estimating
  /// the shared covariance matrices. [Actually this variable contains also the
  /// term -(Y_i M_i^T + M_i Y_I^T).]  Dimension is [I][D][D].
  std::vector< SpMatrix<double> > S_means_;
  
  Vector<double> gamma_j_;  ///< State occupancies

  
  void ComputeSmoothingTerms(const MleAmSgmmAccs &accs,
                             const AmSgmm &model,
                             const std::vector< SpMatrix<double> > &H,
                             SpMatrix<double> *H_sm,
                             Vector<double> *y_sm) const;

  // UpdatePhoneVectors function that allows smoothing terms (but
  // no checking of proper auxiliary function RE weights)
  double UpdatePhoneVectors(const MleAmSgmmAccs &accs,
                            AmSgmm *model,
                            const std::vector<SpMatrix<double> > &H,
                            const SpMatrix<double> &H_sm,
                            const Vector<double> &y_sm);

  
  // Called from UpdatePhoneVectors; updates a subset of states
  // (relates to multi-threading).
  void UpdatePhoneVectorsInternal(const MleAmSgmmAccs &accs,
                                  AmSgmm *model,
                                  const std::vector<SpMatrix<double> > &H,
                                  const SpMatrix<double> &H_sm,
                                  const Vector<double> &y_sm,
                                  double *auxf_impr,
                                  double *like_impr,
                                  int32 num_threads,
                                  int32 thread_id) const;
  
  // UpdatePhoneVectors function that does not support smoothing
  // terms, but allows checking of objective-function improvement,
  // and backtracking.
  double UpdatePhoneVectorsChecked(const MleAmSgmmAccs &accs,
                                   AmSgmm *model,
                                   const std::vector<SpMatrix<double> > &H);

  // Called (indirectly) from UpdatePhoneVectorsCheckedFromClusterable()
  void UpdatePhoneVectorsCheckedFromClusterableInternal(
      const std::vector<SgmmClusterable*> &stats,
      const std::vector< SpMatrix<double> > &H,
      AmSgmm *model,
      double *count_ptr,
      double *like_impr_ptr,
      int32 num_threads,
      int32 thread_id);
  
  double UpdateM(const MleAmSgmmAccs &accs, AmSgmm *model);

  void RenormalizeV(const MleAmSgmmAccs &accs, AmSgmm *model,
                    const SpMatrix<double> &H_sm);
  double UpdateN(const MleAmSgmmAccs &accs, AmSgmm *model);
  void RenormalizeN(const MleAmSgmmAccs &accs, AmSgmm *model);
  double UpdateVars(const MleAmSgmmAccs &accs, AmSgmm *model);
  double UpdateWParallel(const MleAmSgmmAccs &accs, AmSgmm *model);

  /// Called, multithreaded, inside UpdateWParallel
  static
  void UpdateWParallelGetStats(const MleAmSgmmAccs &accs,
                               const AmSgmm &model,
                               const Matrix<double> &w,
                               Matrix<double> *F_i,
                               Matrix<double> *g_i,
                               double *tot_like,
                               int32 num_threads, 
                               int32 thread_id);
  
  double UpdateWSequential(const MleAmSgmmAccs &accs,
                           AmSgmm *model);
  double UpdateSubstateWeights(const MleAmSgmmAccs &accs,
                               AmSgmm *model);

  void ComputeMPrior(AmSgmm *model);  // TODO(arnab): Maybe make this static?
  double MapUpdateM(const MleAmSgmmAccs &accs, AmSgmm *model);

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
                       const SgmmPerFrameDerivedVars &frame_vars,
                       int32 state_index, BaseFloat weight);

  /// Accumulate statistics, given posteriors.  Returns total
  /// count accumulated, which may differ from posteriors.Sum()
  /// due to randomized pruning.
  BaseFloat AccumulateFromPosteriors(const AmSgmm &model,
                                     const SgmmPerFrameDerivedVars &frame_vars,
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

// This class, used in multi-core implementation of the updates of the "w_i"
// quantities, was previously in estimate-am-sgmm.cc, but is being moved to the
// header so it can be used in estimate-am-sgmm-ebw.cc.  It is responsible for
// computing, in parallel, the F_i and g_i quantities used in the updates of
// w_i.
class UpdateWParallelClass: public MultiThreadable {
 public:
  UpdateWParallelClass(const MleAmSgmmAccs &accs,
                       const AmSgmm &model,
                       const Matrix<double> &w,
                       Matrix<double> *F_i,
                       Matrix<double> *g_i,
                       double *tot_like):
      accs_(accs), model_(model), w_(w),
      F_i_ptr_(F_i), g_i_ptr_(g_i), tot_like_ptr_(tot_like) {
    tot_like_ = 0.0;
    F_i_.Resize(F_i->NumRows(), F_i->NumCols());
    g_i_.Resize(g_i->NumRows(), g_i->NumCols());
  }
    
  ~UpdateWParallelClass() {
    F_i_ptr_->AddMat(1.0, F_i_, kNoTrans);
    g_i_ptr_->AddMat(1.0, g_i_, kNoTrans);
    *tot_like_ptr_ += tot_like_;
  }
  
  inline void operator() () {
    // Note: give them local copy of the sums we're computing,
    // which will be propagated to the total sums in the destructor.
    MleAmSgmmUpdater::UpdateWParallelGetStats(accs_, model_, w_,
                                              &F_i_, &g_i_, &tot_like_,
                                              num_threads_, thread_id_);
  }
 private:
  // MleAmSgmmUpdater *updater_;
  const MleAmSgmmAccs &accs_;
  const AmSgmm &model_;
  const Matrix<double> &w_;
  Matrix<double> *F_i_ptr_;
  Matrix<double> *g_i_ptr_;
  Matrix<double> F_i_;
  Matrix<double> g_i_;
  double *tot_like_ptr_;
  double tot_like_;
};


}  // namespace kaldi


#endif  // KALDI_SGMM_ESTIMATE_AM_SGMM_H_
