// ivector/ivector-extractor.h

// Copyright 2013-2014    Daniel Povey
//           2015         David Snyder


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

#ifndef KALDI_IVECTOR_IVECTOR_EXTRACTOR_H_
#define KALDI_IVECTOR_IVECTOR_EXTRACTOR_H_

#include <vector>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "itf/options-itf.h"
#include "util/common-utils.h"
#include "thread/kaldi-mutex.h"
#include "hmm/posterior.h"

namespace kaldi {

// Note, throughout this file we use SGMM-type notation because
// that's what I'm comfortable with.
// Dimensions:
//  D is the feature dim (e.g. D = 60)
//  I is the number of Gaussians (e.g. I = 2048)
//  S is the ivector dim (e.g. S = 400)



// Options for estimating iVectors, during both training and test.  Note: the
// "acoustic_weight" is not read by any class declared in this header; it has to
// be applied by calling IvectorExtractorUtteranceStats::Scale() before
// obtaining the iVector.
// The same is true of max_count: it has to be applied by programs themselves
// e.g. see ../ivectorbin/ivector-extract.cc.
struct IvectorEstimationOptions {
  double acoustic_weight;
  double max_count;
  IvectorEstimationOptions(): acoustic_weight(1.0), max_count(0.0) {}
  void Register(OptionsItf *opts) {
    opts->Register("acoustic-weight", &acoustic_weight,
                   "Weight on part of auxf that involves the data (e.g. 0.2); "
                   "if this weight is small, the prior will have more effect.");
    opts->Register("max-count", &max_count,
                   "Maximum frame count (affects prior scaling): if >0, the prior "
                   "term will be scaled up after the frame count exceeds this "
                   "value.  Note that this count is considered after posterior "
                   "scaling (e.g. --acoustic-weight option, or scale argument to "
                   "scale-post), so you would normally use a cutoff 10 times "
                   "smaller than the corresponding number of frames.");
  }
};


class IvectorExtractor;
class IvectorExtractorComputeDerivedVarsClass;

/// These are the stats for a particular utterance, i.e. the sufficient stats
/// for estimating an iVector (if need_2nd_order_stats == true, we can also
/// estimate the variance of the model; these 2nd order stats are not needed if
/// we just need the iVector.
class IvectorExtractorUtteranceStats {
 public:
  IvectorExtractorUtteranceStats(int32 num_gauss, int32 feat_dim,
                                 bool need_2nd_order_stats):
      gamma_(num_gauss), X_(num_gauss, feat_dim) {
    if (need_2nd_order_stats) {
      S_.resize(num_gauss);
      for (int32 i = 0; i < num_gauss; i++)
        S_[i].Resize(feat_dim);
    }
  }

  void AccStats(const MatrixBase<BaseFloat> &feats,
                const Posterior &post);

  void Scale(double scale); // Used to apply acoustic scale.

  double NumFrames() { return gamma_.Sum(); }

 protected:
  friend class IvectorExtractor;
  friend class IvectorExtractorStats;
  Vector<double> gamma_; // zeroth-order stats (summed posteriors), dimension [I]
  Matrix<double> X_; // first-order stats, dimension [I][D]
  std::vector<SpMatrix<double> > S_; // 2nd-order stats, dimension [I][D][D], if
                                     // required.
};


struct IvectorExtractorOptions {
  int ivector_dim;
  int num_iters;
  bool use_weights;
  IvectorExtractorOptions(): ivector_dim(400), num_iters(2),
                             use_weights(true) { }
  void Register(OptionsItf *opts) {
    opts->Register("num-iters", &num_iters, "Number of iterations in "
                   "iVector estimation (>1 needed due to weights)");
    opts->Register("ivector-dim", &ivector_dim, "Dimension of iVector");
    opts->Register("use-weights", &use_weights, "If true, regress the "
                   "log-weights on the iVector");
  }
};


// Forward declaration.  This class is used together with IvectorExtractor to
// compute iVectors in an online way, so we can update the estimate efficiently
// as we add frames.
class OnlineIvectorEstimationStats;

// Caution: the IvectorExtractor is not the only thing required to get an
// ivector.  We also need to get posteriors from a GMM, typically a FullGmm.
// Typically these will be obtained in a process that involves using a DiagGmm
// for Gaussian selection, followed by getting posteriors from the FullGmm.  To
// keep track of these, we keep them all in the same directory,
// e.g. final.{ubm,dubm,ie}.

class IvectorExtractor {
 public:
  friend class IvectorExtractorStats;
  friend class OnlineIvectorEstimationStats;

  IvectorExtractor(): prior_offset_(0.0) { }

  IvectorExtractor(
      const IvectorExtractorOptions &opts,
      const FullGmm &fgmm);

  /// Gets the distribution over ivectors (or at least, a Gaussian approximation
  /// to it).  The output "var" may be NULL if you don't need it.  "mean", and
  /// "var", if present, must be the correct dimension (this->IvectorDim()).
  /// If you only need a point estimate of the iVector, get the mean only.
  void GetIvectorDistribution(
      const IvectorExtractorUtteranceStats &utt_stats,
      VectorBase<double> *mean,
      SpMatrix<double> *var) const;

  /// The distribution over iVectors, in our formulation, is not centered at
  /// zero; its first dimension has a nonzero offset.  This function returns
  /// that offset.
  double PriorOffset() const { return prior_offset_; }

  /// Returns the log-likelihood objective function, summed over frames,
  /// for this distribution of iVectors (a point distribution, if var == NULL).
  double GetAuxf(const IvectorExtractorUtteranceStats &utt_stats,
                 const VectorBase<double> &mean,
                 const SpMatrix<double> *var = NULL) const;

  /// Returns the data-dependent part of the log-likelihood objective function,
  /// summed over frames.  If variance pointer is NULL, uses point value.
  double GetAcousticAuxf(const IvectorExtractorUtteranceStats &utt_stats,
                         const VectorBase<double> &mean,
                         const SpMatrix<double> *var = NULL) const;

  /// Returns the prior-related part of the log-likelihood objective function.
  /// Note: if var != NULL, this quantity is a *probability*, otherwise it is
  /// a likelihood (and the corresponding probability is zero).
  double GetPriorAuxf(const VectorBase<double> &mean,
                         const SpMatrix<double> *var = NULL) const;

  /// This returns just the part of the acoustic auxf that relates to the
  /// variance of the utt_stats (i.e. which would be zero if the utt_stats had
  /// zero variance.  This does not depend on the iVector, it's included as an
  /// aid to debugging.  We can only get this if we stored the S statistics.  If
  /// not we assume the variance is generated from the model.
  double GetAcousticAuxfVariance(
      const IvectorExtractorUtteranceStats &utt_stats) const;

  /// This returns just the part of the acoustic auxf that relates to the
  /// speaker-dependent means (and how they differ from the data means).
  double GetAcousticAuxfMean(
      const IvectorExtractorUtteranceStats &utt_stats,
      const VectorBase<double> &mean,
      const SpMatrix<double> *var = NULL) const;

  /// This returns the part of the acoustic auxf that relates to the
  /// gconsts of the Gaussians.
  double GetAcousticAuxfGconst(
      const IvectorExtractorUtteranceStats &utt_stats) const;

  /// This returns the part of the acoustic auxf that relates to the
  /// Gaussian-specific weights.  (impacted by the iVector only if
  /// we are using w_).
  double GetAcousticAuxfWeight(
      const IvectorExtractorUtteranceStats &utt_stats,
      const VectorBase<double> &mean,
      const SpMatrix<double> *var = NULL) const;


  /// Gets the linear and quadratic terms in the distribution over iVectors, but
  /// only the terms arising from the Gaussian means (i.e. not the weights
  /// or the priors).
  /// Setup is log p(x) \propto x^T linear -0.5 x^T quadratic x.
  /// This function *adds to* the output rather than setting it.
  void GetIvectorDistMean(
      const IvectorExtractorUtteranceStats &utt_stats,
      VectorBase<double> *linear,
      SpMatrix<double> *quadratic) const;

  /// Gets the linear and quadratic terms in the distribution over
  /// iVectors, that arise from the prior.  Adds to the outputs,
  /// rather than setting them.
  void GetIvectorDistPrior(
      const IvectorExtractorUtteranceStats &utt_stats,
      VectorBase<double> *linear,
      SpMatrix<double> *quadratic) const;

  /// Gets the linear and quadratic terms in the distribution over
  /// iVectors, that arise from the weights (if applicable).  The
  /// "mean" parameter is the iVector point that we compute
  /// the expansion around (it's a quadratic approximation of a
  /// nonlinear function, but with a "safety factor" (the "max" stuff).
  /// Adds to the outputs, rather than setting them.
  void GetIvectorDistWeight(
      const IvectorExtractorUtteranceStats &utt_stats,
      const VectorBase<double> &mean,
      VectorBase<double> *linear,
      SpMatrix<double> *quadratic) const;

  // Note: the function GetStats no longer exists due to code refactoring.
  // Instead of this->GetStats(feats, posterior, &utt_stats), call
  // utt_stats.AccStats(feats, posterior).

  int32 FeatDim() const;
  int32 IvectorDim() const;
  int32 NumGauss() const;
  bool IvectorDependentWeights() const { return w_.NumRows() != 0; }
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // Note: we allow the default assignment and copy operators
  // because they do what we want.
 protected:
  void ComputeDerivedVars();
  void ComputeDerivedVars(int32 i);
  friend class IvectorExtractorComputeDerivedVarsClass;

  // Imagine we'll project the iVectors with transformation T, so apply T^{-1}
  // where necessary to keep the model equivalent.  Used to keep unit variance
  // (like prior re-estimation).
  void TransformIvectors(const MatrixBase<double> &T,
                         double new_prior_offset);


  /// Weight projection vectors, if used.  Dimension is [I][S]
  Matrix<double> w_;

  /// If we are not using weight-projection vectors, stores the Gaussian mixture
  /// weights from the UBM.  This does not affect the iVector; it is only useful
  /// as a way of making sure the log-probs are comparable between systems with
  /// and without weight projection matrices.
  Vector<double> w_vec_;

  /// Ivector-subspace projection matrices, dimension is [I][D][S].
  /// The I'th matrix projects from ivector-space to Gaussian mean.
  /// There is no mean offset to add-- we deal with it by having
  /// a prior with a nonzero mean.
  std::vector<Matrix<double> > M_;

  /// Inverse variances of speaker-adapted model, dimension [I][D][D].
  std::vector<SpMatrix<double> > Sigma_inv_;

  /// 1st dim of the prior over the ivector has an offset, so it is not zero.
  /// This is used to handle the global offset of the speaker-adapted means in a
  /// simple way.
  double prior_offset_;

  // Below are *derived variables* that can be computed from the
  // variables above.

  /// The constant term in the log-likelihood of each Gaussian (not
  /// counting any weight).
  Vector<double> gconsts_;

  /// U_i = M_i^T \Sigma_i^{-1} M_i is a quantity that comes up
  /// in ivector estimation.  This is conceptually a
  /// std::vector<SpMatrix<double> >, but we store the packed-data
  /// in the rows of a matrix, which gives us an efficiency
  /// improvement (we can use matrix-multiplies).
  Matrix<double> U_;

  /// The product of Sigma_inv_[i] with M_[i].
  std::vector<Matrix<double> > Sigma_inv_M_;
 private:
  // var <-- quadratic_term^{-1}, but done carefully, first flooring eigenvalues
  // of quadratic_term to 1.0, which mathematically is the least they can be,
  // due to the prior term.
  static void InvertWithFlooring(const SpMatrix<double> &quadratic_term,
                                 SpMatrix<double> *var);
};

/**
   This class helps us to efficiently estimate iVectors in situations where the
   data is coming in frame by frame.
 */
class OnlineIvectorEstimationStats {
 public:
  // Search above for max_count to see an explanation; if nonzero, it will
  // put a higher weight on the prior (vs. the stats) once the count passes
  // that value.
  OnlineIvectorEstimationStats(int32 ivector_dim,
                               BaseFloat prior_offset,
                               BaseFloat max_count);

  OnlineIvectorEstimationStats(const OnlineIvectorEstimationStats &other);


  void AccStats(const IvectorExtractor &extractor,
                const VectorBase<BaseFloat> &feature,
                const std::vector<std::pair<int32, BaseFloat> > &gauss_post);

  int32 IvectorDim() const { return linear_term_.Dim(); }

  /// This function gets the current estimate of the iVector.  Internally it
  /// does some work to compute it (currently matrix inversion, but we are doing
  /// to use Conjugate Gradient which will increase the speed).  At entry,
  /// "ivector" must be a pointer to a vector dimension IvectorDim(), and free
  /// of NaN's.  For faster estimation, you can set "num_cg_iters" to some value
  /// > 0, which will limit how many iterations of conjugate gradient we use to
  /// re-estimate the iVector; in this case, you should make sure *ivector is
  /// set at entry to a recently estimated iVector from the same utterance,
  /// which will give the CG a better starting point.
  /// If num_cg_iters is set to -1, it will compute the iVector exactly; if it's
  /// set to a positive number, the number of conjugate gradient iterations will
  /// be limited to that number.  Note: the iVectors output still have a nonzero
  /// mean (first dim offset by PriorOffset()).
  void GetIvector(int32 num_cg_iters,
                  VectorBase<double> *ivector) const;

  double NumFrames() const { return num_frames_; }

  double PriorOffset() const { return prior_offset_; }

  /// ObjfChange returns the change in objective function *per frame* from
  /// using the default value [ prior_offset_, 0, 0, ... ] to
  /// using the provided value; should be >= 0, if "ivector" is
  /// a value we estimated.  This is for diagnostics.
  double ObjfChange(const VectorBase<double> &ivector) const;

  double Count() const { return num_frames_; }

  /// Scales the number of frames of stats by 0 <= scale <= 1, to make it
  /// as if we had fewer frames of adaptation data.  Note: it does not
  /// apply the scaling to the prior term.
  void Scale(double scale);

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  // Override the default assignment operator
  inline OnlineIvectorEstimationStats &operator=(const OnlineIvectorEstimationStats &other) {
    this->prior_offset_ = other.prior_offset_;
    this->max_count_ = other.max_count_;
    this->num_frames_ = other.num_frames_;
    this->quadratic_term_=other.quadratic_term_;
    this->linear_term_=other.linear_term_;
    return *this;
  }

 protected:
  /// Returns objective function per frame, at this iVector value.
  double Objf(const VectorBase<double> &ivector) const;

  /// Returns objective function evaluated at the point
  /// [ prior_offset_, 0, 0, 0, ... ]... this is used in diagnostics.
  double DefaultObjf() const;

  friend class IvectorExtractor;
  double prior_offset_;
  double max_count_;
  double num_frames_;  // num frames (weighted, if applicable).
  SpMatrix<double> quadratic_term_;
  Vector<double> linear_term_;
};


// This code obtains periodically (for each "ivector_period" frames, e.g. 10
// frames), an estimate of the iVector including all frames up to that point.
// This emulates what you could do in an online/streaming algorithm; its use is
// for neural network training in a way that's matched to online decoding.
// [note: I don't believe we are currently using the program,
// ivector-extract-online.cc, that calls this function, in any of the scripts.].
// Caution: this program outputs the raw iVectors, where the first component
// will generally be very positive.  You probably want to subtract PriorOffset()
// from the first element of each row of the output before writing it out.
// For num_cg_iters, we suggest 15.  It can be a positive number (more -> more
// exact, less -> faster), or if it's negative it will do the optimization
// exactly each time which is slower.
// It returns the objective function improvement per frame from the "default" iVector to
// the last iVector estimated.
double EstimateIvectorsOnline(
    const Matrix<BaseFloat> &feats,
    const Posterior &post,
    const IvectorExtractor &extractor,
    int32 ivector_period,
    int32 num_cg_iters,
    BaseFloat max_count,
    Matrix<BaseFloat> *ivectors);


/// Options for IvectorExtractorStats, which is used to update the parameters of
/// IvectorExtractor.
struct IvectorExtractorStatsOptions {
  bool update_variances;
  bool compute_auxf;
  int32 num_samples_for_weights;
  int cache_size;

  IvectorExtractorStatsOptions(): update_variances(true),
                         compute_auxf(true),
                         num_samples_for_weights(10),
                         cache_size(100) { }
  void Register(OptionsItf *opts) {
    opts->Register("update-variances", &update_variances, "If true, update the "
                   "Gaussian variances");
    opts->Register("compute-auxf", &compute_auxf, "If true, compute the "
                   "auxiliary functions on training data; can be used to "
                   "debug and check convergence.");
    opts->Register("num-samples-for-weights", &num_samples_for_weights,
                   "Number of samples from iVector distribution to use "
                   "for accumulating stats for weight update.  Must be >1");
    opts->Register("cache-size", &cache_size, "Size of an internal "
                   "cache (not critical, only affects speed/memory)");
  }
};


/// Options for training the IvectorExtractor, e.g. variance flooring.
struct IvectorExtractorEstimationOptions {
  double variance_floor_factor;
  double gaussian_min_count;
  int32 num_threads;
  bool diagonalize;
  IvectorExtractorEstimationOptions(): variance_floor_factor(0.1),
                                       gaussian_min_count(100.0),
                                       diagonalize(true) { }
  void Register(OptionsItf *opts) {
    opts->Register("variance-floor-factor", &variance_floor_factor,
                   "Factor that determines variance flooring (we floor each covar "
                   "to this times global average covariance");
    opts->Register("gaussian-min-count", &gaussian_min_count,
                   "Minimum total count per Gaussian, below which we refuse to "
                   "update any associated parameters.");
    opts->Register("diagonalize", &diagonalize,
                   "If true, diagonalize the quadratic term in the "
                   "objective function. This reorders the ivector dimensions"
                   "from most to least important.");
  }
};

class IvectorExtractorUpdateProjectionClass;
class IvectorExtractorUpdateWeightClass;

/// IvectorExtractorStats is a class used to update the parameters of the
/// ivector extractor
class IvectorExtractorStats {
 public:
  friend class IvectorExtractor;

  IvectorExtractorStats(): tot_auxf_(0.0), R_num_cached_(0), num_ivectors_(0) { }

  IvectorExtractorStats(const IvectorExtractor &extractor,
                        const IvectorExtractorStatsOptions &stats_opts);

  void Add(const IvectorExtractorStats &other);

  void AccStatsForUtterance(const IvectorExtractor &extractor,
                            const MatrixBase<BaseFloat> &feats,
                            const Posterior &post);

  // This version (intended mainly for testing) works out the Gaussian
  // posteriors from the model.  Returns total log-like for feats, given
  // unadapted fgmm.  You'd want to add Gaussian pruning and preselection using
  // the diagonal, GMM, for speed, if you used this outside testing code.
  double AccStatsForUtterance(const IvectorExtractor &extractor,
                              const MatrixBase<BaseFloat> &feats,
                              const FullGmm &fgmm);

  void Read(std::istream &is, bool binary, bool add = false);

  void Write(std::ostream &os, bool binary); // non-const version; relates to cache.

  // const version of Write; may use extra memory if we have stuff cached
  void Write(std::ostream &os, bool binary) const;

  /// Returns the objf improvement per frame.
  double Update(const IvectorExtractorEstimationOptions &opts,
                IvectorExtractor *extractor) const;

  double AuxfPerFrame() { return tot_auxf_ / gamma_.Sum(); }

  /// Prints the proportion of the variance explained by
  /// the Ivector model versus the Gaussians.
  void IvectorVarianceDiagnostic(const IvectorExtractor &extractor);

  // Copy constructor.
  explicit IvectorExtractorStats (const IvectorExtractorStats &other);

 protected:
  friend class IvectorExtractorUpdateProjectionClass;
  friend class IvectorExtractorUpdateWeightClass;


  // This is called by AccStatsForUtterance
  void CommitStatsForUtterance(const IvectorExtractor &extractor,
                               const IvectorExtractorUtteranceStats &utt_stats);

  /// This is called by CommitStatsForUtterance.  We commit the stats
  /// used to update the M matrix.
  void CommitStatsForM(const IvectorExtractor &extractor,
                       const IvectorExtractorUtteranceStats &utt_stats,
                       const VectorBase<double> &ivec_mean,
                       const SpMatrix<double> &ivec_var);

  /// Flushes the cache for the R_ stats.
  void FlushCache();

  /// Commit the stats used to update the variance.
  void CommitStatsForSigma(const IvectorExtractor &extractor,
                           const IvectorExtractorUtteranceStats &utt_stats);

  /// Commit the stats used to update the weight-projection w_-- this one
  /// takes a point sample, it's called from CommitStatsForW().
  void CommitStatsForWPoint(const IvectorExtractor &extractor,
                            const IvectorExtractorUtteranceStats &utt_stats,
                            const VectorBase<double> &ivector,
                            double weight);


  /// Commit the stats used to update the weight-projection w_.
  void CommitStatsForW(const IvectorExtractor &extractor,
                       const IvectorExtractorUtteranceStats &utt_stats,
                       const VectorBase<double> &ivec_mean,
                       const SpMatrix<double> &ivec_var);

  /// Commit the stats used to update the prior distribution.
  void CommitStatsForPrior(const VectorBase<double> &ivec_mean,
                           const SpMatrix<double> &ivec_var);

  // Updates M.  Returns the objf improvement per frame.
  double UpdateProjections(const IvectorExtractorEstimationOptions &opts,
                           IvectorExtractor *extractor) const;

  // This internally called function returns the objf improvement
  // for this Gaussian index.  Updates one M.
  double UpdateProjection(const IvectorExtractorEstimationOptions &opts,
                          int32 gaussian,
                          IvectorExtractor *extractor) const;

  // Updates the weight projections.  Returns the objf improvement per
  // frame.
  double UpdateWeights(const IvectorExtractorEstimationOptions &opts,
                       IvectorExtractor *extractor) const;

  // Updates the weight projection for one Gaussian index.  Returns the objf
  // improvement for this index.
  double UpdateWeight(const IvectorExtractorEstimationOptions &opts,
                      int32 gaussian,
                      IvectorExtractor *extractor) const;

  // Returns the objf improvement per frame.
  double UpdateVariances(const IvectorExtractorEstimationOptions &opts,
                         IvectorExtractor *extractor) const;



  // Updates the prior; returns obj improvement per frame.
  double UpdatePrior(const IvectorExtractorEstimationOptions &opts,
                     IvectorExtractor *extractor) const;

  // Called from UpdatePrior, separating out some code that
  // computes likelihood changes.
  double PriorDiagnostics(double old_prior_offset) const;


  void CheckDims(const IvectorExtractor &extractor) const;

  IvectorExtractorStatsOptions config_; /// Caution: if we read from disk, this
                               /// is not recovered.  Options will not be
                               /// used during the update phase anyway,
                               /// so this should not matter.

  /// Total auxiliary function over the training data-- can be
  /// used to check convergence, etc.
  double tot_auxf_;

  /// This mutex guards gamma_ and Y_ (for multi-threaded
  /// update)
  Mutex gamma_Y_lock_;

  /// Total occupation count for each Gaussian index (zeroth-order stats)
  Vector<double> gamma_;

  /// Stats Y_i for estimating projections M.  Dimension is [I][D][S].  The
  /// linear term in M.
  std::vector<Matrix<double> > Y_;

  /// This mutex guards R_ (for multi-threaded update)
  Mutex R_lock_;

  /// R_i, quadratic term for ivector subspace (M matrix)estimation.  This is a
  /// kind of scatter of ivectors of training speakers, weighted by count for
  /// each Gaussian.  Conceptually vector<SpMatrix<double> >, but we store each
  /// SpMatrix as a row of R_.  Conceptually, the dim is [I][S][S]; the actual
  /// dim is [I][S*(S+1)/2].
  Matrix<double> R_;

  /// This mutex guards R_num_cached_, R_gamma_cache_, R_ivec_cache_ (for
  /// multi-threaded update)
  Mutex R_cache_lock_;

  /// To avoid too-frequent rank-1 update of R, which is slow, we cache some
  /// quantities here.
  int32 R_num_cached_;
  /// dimension: [num-to-cache][I]
  Matrix<double> R_gamma_cache_;
  /// dimension: [num-to-cache][S*(S+1)/2]
  Matrix<double> R_ivec_scatter_cache_;

  /// This mutex guards Q_ and G_ (for multi-threaded update)
  Mutex weight_stats_lock_;

  /// Q_ is like R_ (with same dimensions), except used for weight estimation;
  /// the scatter of ivectors is weighted by the coefficient of the quadratic
  /// term in the expansion for w (the "safe" one, with the max expression).
  Matrix<double> Q_;

  /// G_ is the linear term in the weight projection matrix w_.  It has the same
  /// dim as w_, i.e. [I][S]
  Matrix<double> G_;

  /// This mutex guards S_ (for multi-threaded update)
  Mutex variance_stats_lock_;

  /// S_{i}, raw second-order stats per Gaussian which we will use to update the
  /// variances Sigma_inv_.
  std::vector< SpMatrix<double> > S_;


  /// This mutex guards num_ivectors_, ivector_sum_ and ivector_scatter_ (for multi-threaded
  /// update)
  Mutex prior_stats_lock_;

  /// Count of the number of iVectors we trained on.   Need for prior re-estimation.
  /// (make it double not int64 to more easily support weighting later.)
  double num_ivectors_;

  /// Sum of all the iVector means.  Needed for prior re-estimation.
  Vector<double> ivector_sum_;

  /// Second-order stats for the iVectors.  Needed for prior re-estimation.
  SpMatrix<double> ivector_scatter_;

 private:
  /// Computes an orthogonal matrix A from the iVector transform
  /// T such that T' = A*T is an alternative transform which diagonalizes the
  /// quadratic_term_ in the iVector estimation objective function. This
  /// reorders the dimensions of the iVector from most to least important,
  /// which may be more convenient to view. The transform should not
  /// affect the performance of systems which use iVectors.
  void GetOrthogonalIvectorTransform(const SubMatrix<double> &T,
                                     IvectorExtractor *extractor,
                                     Matrix<double> *A) const;

  IvectorExtractorStats &operator = (const IvectorExtractorStats &other);  // Disallow.
};



}  // namespace kaldi


#endif
