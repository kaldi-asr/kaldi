// sgmm2/am-sgmm2.h

// Copyright 2009-2011  Microsoft Corporation;  Lukas Burget;
//                      Saarland University (Author: Arnab Ghoshal);
//                      Ondrej Glembek;  Yanmin Qian;
// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey)
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

#ifndef KALDI_SGMM2_AM_SGMM2_H_
#define KALDI_SGMM2_AM_SGMM2_H_

#include <vector>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "itf/options-itf.h"
#include "util/table-types.h"
#include "thread/kaldi-thread.h"

namespace kaldi {
/*
  When reading this file, keep in mind two references: the paper
 "The Subspace Gaussian Mixture Model-- a Structured Model for Speech Recognition", by D. Povey,
  L. Burget et. al (Computer Speech and Language, 2011), and
  "The Symmetric Subspace Gaussian Mixture Model": Microsoft Research technical report MSR-TR-2010-138.
  We will refer to these as "the paper" [or "the CSL paper"] and "the techreport".

  (1) SSGMM
  
  We'll use the acronym SSGMM to refer to the Symmetric SGMM, and we'll mark in
  the code with "[SSGMM]" things that relate to it.  The technical report
  describes an extention to the originally described model where we have
  speaker-dependent mixture weights.  These are implemented here.  Note: we only
  implement the "more efficient" version of the update for the speaker
  projection vectors \u_i.  There is also an ICASSP paper that describes the
  stuff in the techreport (more briefly), with results, but we don't refer to
  any equation numbers in that.

  (2) SCTM

  What we implement here has another extension that was not in the CSL paper: an
  extension to the "state-clustered tied mixture" [SCTM] system-- a bit like BBN's
  style of system, except for SGMMs not Gaussians, at the sub-state not Gaussian level.
  We build a first
  tree, at which level the phonetic sub-state vectors are defined, and then a
  "more detailed" tree, at which level we share the sub-state mixture weights.
  In this class, NumPdfs() returns the real number of pdf's (i.e. the #leaves
  of the more detailed tree), and NumPdfGroups() returns the number of groups of
  pdf's that share the sub-state vectors.
  We use the index j2 for indexing 0...NumPdfs()-1 [as it's the "2nd level" of the tree],
  and j1 for indexing 0...NumPdfGroups()-1 [as it's the "1st level" of the tree].
  The weights are stored as c[j2][m].  There is a mapping Pdf2Group(j2) which returns
  the corresponding j1 for a given j2, and Group2PdfList(j1) which returns a vector<int32>
  consisting of the list of j2 indices for that j1. 
  
  The count quantities we store during the accumulation phase could most simply
  be stored as gamma[j2][m][i] (where m is the sub-state index), but this is
  inefficient.  Instead we store them separately as gamma1[j1][m][i] and gamma2[j2][m],
  so each count gets stored in two separate places; this makes the stats more compact.

  In this implementation, the normalizers n_{jmi} are now stored as n[j1][m][i],
  without including the log-weight term log c[j2][m].  In the computation of
  state likelihoods, we first compute the log-prob of the data given each of the
  sub-state vectors; and we compute the log-sum of this and the posteriors over
  each of the vectors [treating the weights as 1.0].  Call these
  "pseudo-posteriors".  Then to take into account the contribution of the
  weights in a state j2, we take the dot product of the weight-vector c[j2][...]
  with this vector of pseudo-posteriors.  The log of this dot-product gets added to the
  original log-sum.  
*/


struct Sgmm2SplitSubstatesConfig {
  int32 split_substates;
  BaseFloat perturb_factor;
  BaseFloat power;
  BaseFloat max_cond;
  BaseFloat min_count;
  Sgmm2SplitSubstatesConfig(): split_substates(0),
                               perturb_factor(0.01),
                               power(0.2),
                               max_cond(100.0),
                               min_count(40.0) { }
  void Register(OptionsItf *po) {
    po->Register("split-substates", &split_substates, "Increase number of "
                 "substates to this overall target.");
    po->Register("max-cond-split", &max_cond, "Max condition number of smoothing "
                "matrix used in substate splitting.");
    po->Register("perturb-factor", &perturb_factor, "Perturbation factor for "
                "state vectors while splitting substates.");
    po->Register("power", &power, "Exponent for substate occupancies used while "
                "splitting substates.");
    po->Register("min-count", &min_count, "Minimum allowed count, used in allocating "
                 "sub-states to state in mixture splitting.");
  }
};

// Caution: this config is probably not used in most of the setups, we generally do the Gaussian
// selection using separate programs
struct Sgmm2GselectConfig {
  /// Number of highest-scoring full-covariance Gaussians per frame.
  int32 full_gmm_nbest;
  /// Number of highest-scoring diagonal-covariance Gaussians per frame.
  int32 diag_gmm_nbest;

  Sgmm2GselectConfig() {
    full_gmm_nbest = 15;
    diag_gmm_nbest = 50;
  }

  void Register(OptionsItf *po) {
    po->Register("full-gmm-nbest", &full_gmm_nbest, "Number of highest-scoring"
        " full-covariance Gaussians selected per frame.");
    po->Register("diag-gmm-nbest", &diag_gmm_nbest, "Number of highest-scoring"
        " diagonal-covariance Gaussians selected per frame.");
  }
};

/** \struct Sgmm2PerFrameDerivedVars
 *  Holds the per-frame precomputed quantities x(t), x_{i}(t), z_{i}(t), and
 *  n_{i}(t) (cf. Eq. (33)-(36)) for the SGMM, as well as the cached Gaussian
 *  selection records.
 */
struct Sgmm2PerFrameDerivedVars {
  std::vector<int32> gselect;
  Vector<BaseFloat> xt;   ///< x'(t), FMLLR-adapted, dim = [D], eq.(33)
  Matrix<BaseFloat> xti;  ///< x_{i}(t) = x'(t) - o_i(s): dim = [I][D], eq.(34)
  Matrix<BaseFloat> zti;  ///< z_{i}(t), dim = [I][S], eq.(35)
  Vector<BaseFloat> nti;  ///< n_{i}(t), dim = [I], eq.(36) in CSL paper, but
                          ///< [SSGMM] with extra term log b_i^{(s)}, see eq. (24) of
                          ///< techreport.
  
  void Resize(int32 ngauss, int32 feat_dim, int32 phn_dim) { // resizes but does
    // not necessarily zero things.
    if (xt.Dim() != feat_dim) xt.Resize(feat_dim);
    if (xti.NumRows() != ngauss || xti.NumCols() != feat_dim)
      xti.Resize(ngauss, feat_dim);
    if (zti.NumRows() != ngauss || zti.NumCols() != phn_dim)
      zti.Resize(ngauss, phn_dim);
    if (nti.Dim() != ngauss)
      nti.Resize(ngauss);
  }
};

class AmSgmm2;

class Sgmm2PerSpkDerivedVars {
  // To set this up, call ComputePerSpkDerivedVars from the sgmm object.
 public:  
  void Clear() {
    v_s.Resize(0);
    o_s.Resize(0, 0);
    b_is.Resize(0);
    log_b_is.Resize(0);
    log_d_jms.resize(0);
  }
  bool Empty() { return v_s.Dim() == 0; }
  // caution: after SetSpeakerVector you typically want to
  // use the function AmSgmm::ComputePerSpkDerivedVars
  const Vector<BaseFloat> &GetSpeakerVector() { return v_s; }
  
  void SetSpeakerVector(const Vector<BaseFloat> &v_s_in) {
    v_s.Resize(v_s_in.Dim());
    v_s.CopyFromVec(v_s_in);
  }    
 protected:
  friend class AmSgmm2;
  friend class MleAmSgmm2Accs;
  Vector<BaseFloat> v_s;  ///< Speaker adaptation vector v_^{(s)}. Dim is [T]
  Matrix<BaseFloat> o_s;  ///< Per-speaker offsets o_{i}. Dimension is [I][D]
  Vector<BaseFloat> b_is; /// < [SSGMM]: Eq. (22) in techreport, b_i^{(s)} = \exp(\u_i^T \v^{(s)})
  Vector<BaseFloat> log_b_is; /// < [SSGMM] log of the above (more efficient to store both).
  std::vector<Vector<BaseFloat> > log_d_jms; ///< [SSGMM] normalizers per-speaker and per-substate;
                                             ///< indexed [j1][m].
};

/// Sgmm2LikelihoodCache caches SGMM likelihoods at two levels: the final
/// pdf likelihoods, and the sub-state level likelihoods, which means
/// that with the SCTM system we can avoid redundant computation.
/// You need to call NextFrame() on the cache, between frames.
struct Sgmm2LikelihoodCache {
 public:
  // you'll typically initialize with (sgmm.NumGroups(), sgmm.NumPdfs()).
  Sgmm2LikelihoodCache(int32 num_groups, int32 num_pdfs):
      substate_cache(num_groups), pdf_cache(num_pdfs), t(1) { }
  
  struct SubstateCacheElement { // indexed by j1.
    SubstateCacheElement(): t(0) { }
    // The "likes" and "remaining_log_like" quantities store the
    // log-like of the data given each substate vector, in a redundant
    // way, so the likelihood is likes(i) * exp(remaining_log_like).
    // This is to get around problems with numerical range.
    Vector<BaseFloat> likes; 
    BaseFloat remaining_log_like;
    int32 t; // used in detecting "freshness."
  };  
  struct PdfCacheElement { // indexed by j2.
    PdfCacheElement(): t(0) { }
    BaseFloat log_like;
    int32 t; // used in detecting "freshness."
  };

  void NextFrame(); // increments t.
  std::vector<SubstateCacheElement> substate_cache; // indexed by j1.
  std::vector<PdfCacheElement> pdf_cache; // indexed by j2.
  int32 t;
};


/** \class AmSgmm2
 *  Class for definition of the subspace Gmm acoustic model
 */
class AmSgmm2 {
 public:
  AmSgmm2() {}
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary,
             SgmmWriteFlagsType write_params) const;
  
  /// Checks the various components for correct sizes. With wrong sizes,
  /// assertion failure occurs. When the argument is set to true, dimensions of
  /// the various components are printed.
  void Check(bool show_properties = true);

  /// Initializes the SGMM parameters from a full-covariance UBM.
  /// The state2group vector maps from a state to the corresponding
  /// cluster of states [i.e. j2 to j1].  For conventionally structured
  /// systems (no 2-level tree), this can just be [ 0 1 ... n-1 ].
  void InitializeFromFullGmm(const FullGmm &gmm,
                             const std::vector<int32> &pdf2group,
                             int32 phn_subspace_dim,
                             int32 spk_subspace_dim,
                             bool speaker_dependent_weights,
                             BaseFloat self_weight); // self_weight relates to
  // initialization of the weights.  if self_weight == 1.0 it means we
  // just have 1 sub-state per group, otherwise we have one per pdf,
  // and each pdf has "self_weight" as its "own" weight.
  
  /// Copies the global parameters from the supplied model, but sets
  /// the state vectors to zero. 
  void CopyGlobalsInitVecs(const AmSgmm2 &other,
                           const std::vector<int32> &pdf2group,
                           BaseFloat self_weight);
  
  /// Used to copy models (useful in update)
  void CopyFromSgmm2(const AmSgmm2 &other,
                    bool copy_normalizers,
                    bool copy_weights);  // copy_weights is to copy w_{jmi} [which are
   // stored, in the symmetric SSGMM.]
  
  /// Computes the top-scoring Gaussian indices (used for pruning of later
  /// stages of computation). Returns frame log-likelihood given selected
  /// Gaussians from full UBM.
  BaseFloat GaussianSelection(const Sgmm2GselectConfig &config,
                              const VectorBase<BaseFloat> &data,
                              std::vector<int32> *gselect) const;
  
  /// This needs to be called with each new frame of data, prior to accumulation
  /// or likelihood evaluation: it computes various pre-computed quantities.
  void ComputePerFrameVars(const VectorBase<BaseFloat> &data,
                           const std::vector<int32> &gselect,
                           const Sgmm2PerSpkDerivedVars &spk_vars,
                           Sgmm2PerFrameDerivedVars *per_frame_vars) const;


  /// Computes the per-speaker derived vars; assumes vars->v_s is already
  /// set up.
  void ComputePerSpkDerivedVars(Sgmm2PerSpkDerivedVars *vars) const;
  
  /// This does a likelihood computation for a given state using the
  /// pre-selected Gaussian components (in per_frame_vars).  If the
  /// log_prune parameter is nonzero (e.g. 5.0), the LogSumExp() stage is
  /// pruned, which is a significant speedup... smaller values are faster.
  /// Note: you have to call cache->NextFrame() before calling this for
  /// a new frame of data.
  BaseFloat LogLikelihood(const Sgmm2PerFrameDerivedVars &per_frame_vars,
                          int32 j2, // pdf_id
                          Sgmm2LikelihoodCache *cache, // be careful to call NextFrame() when needed!
                          Sgmm2PerSpkDerivedVars *spk_vars,
                          BaseFloat log_prune = 0.0) const;
  
  /// Similar to LogLikelihood() function above, but also computes the posterior
  /// probabilities for the pre-selected Gaussian components and all substates.
  /// This one doesn't use caching to share computation for the groups of
  /// pdfs. [it's less necessary, as most of the time we're doing this from alignments,
  /// or lattices that are quite sparse, so we save little by sharing this.]
  BaseFloat ComponentPosteriors(const Sgmm2PerFrameDerivedVars &per_frame_vars,
                                int32 j2,
                                Sgmm2PerSpkDerivedVars *spk_vars,
                                Matrix<BaseFloat> *post) const;

  /// Increases the total number of substates based on the state occupancies.
  void SplitSubstates(const Vector<BaseFloat> &state_occupancies, // [indexed by pdf-id j2]
                      const Sgmm2SplitSubstatesConfig &config);

  /// Functions for increasing the phonetic and speaker space dimensions.
  /// The argument norm_xform is a LDA-like feature normalizing transform,
  /// computed by the ComputeFeatureNormalizingTransform function.
  void IncreasePhoneSpaceDim(int32 target_dim,
                             const Matrix<BaseFloat> &norm_xform);

  /// Increase the subspace dimension for speakers.  The
  /// boolean "speaker_dependent_weights" argument (for SSGMM)
  /// only makes a difference if increasing the subspace dimension
  /// from zero.
  void IncreaseSpkSpaceDim(int32 target_dim,
                           const Matrix<BaseFloat> &norm_xform,
                           bool speaker_dependent_weights);

  /// Computes (and initializes if necessary) derived vars...
  /// for now this is just the normalizers "n" and the diagonal UBM,
  /// and if we have the "u" matrix set up, also the w_jmi_
  /// quantities.
  void ComputeDerivedVars();

  /// Computes the data-independent terms in the log-likelihood computation
  /// for each Gaussian component and all substates. Eq. (31)
  void ComputeNormalizers();
  
  /// Computes the weights w_jmi_, which is needed for likelihood evaluation
  /// with SSGMMs.
  void ComputeWeights();

  /// Computes the LDA-like pre-transform and its inverse as well as the
  /// eigenvalues of the scatter of the means used in FMLLR estimation.
  void ComputeFmllrPreXform(const Vector<BaseFloat> &pdf_occs,
                            Matrix<BaseFloat> *xform,
                            Matrix<BaseFloat> *inv_xform,
                            Vector<BaseFloat> *diag_mean_scatter) const;
  
  /// Various model dimensions.
  int32 NumPdfs() const { return pdf2group_.size(); }
  int32 NumGroups() const { return group2pdf_.size(); } // relates to SCTM.  # pdf groups,
  // <= NumPdfs().
  int32 Pdf2Group(int32 j2) const; // relates to SCTM.
  int32 NumSubstatesForPdf(int32 j2) const {
    KALDI_ASSERT(j2 < NumPdfs()); return c_[j2].Dim();
  }
  int32 NumSubstatesForGroup(int32 j1) const {
    KALDI_ASSERT(j1 < NumGroups()); return v_[j1].NumRows();
  }
  int32 NumGauss() const { return M_.size(); }
  int32 PhoneSpaceDim() const { return w_.NumCols(); }
  int32 SpkSpaceDim() const { return (N_.size() > 0) ? N_[0].NumCols() : 0; }
  int32 FeatureDim() const { return M_[0].NumRows(); }

  /// True if doing SSGMM.
  bool HasSpeakerDependentWeights() const { return (u_.NumRows() != 0); }

  bool HasSpeakerSpace() const { return (!N_.empty()); }
  
  void RemoveSpeakerSpace() { N_.clear(); u_.Resize(0, 0); w_jmi_.clear(); }
  
  // [SSGMM] get the quantity d_{jm}^{(s)} and cache it with
  // spk vars if necessary.  Called in accumulation code.
  BaseFloat GetDjms(int32 j1, int32 m,
                    Sgmm2PerSpkDerivedVars *spk_vars) const;
  
  /// Accessors
  const FullGmm & full_ubm() const { return full_ubm_; }
  const DiagGmm & diag_ubm() const { return diag_ubm_; }
  
  
  /// Templated accessors (used to accumulate in different precision)
  template<typename Real>
  void GetInvCovars(int32 gauss_index, SpMatrix<Real> *out) const;

  template<typename Real>
  void GetSubstateMean(int32 j1, int32 m, int32 i,
                       VectorBase<Real> *mean_out) const;
    
  template<typename Real>
  void GetNtransSigmaInv(std::vector< Matrix<Real> > *out) const;

  template<typename Real>
  void GetSubstateSpeakerMean(int32 j1, int32 substate, int32 gauss,
                              const Sgmm2PerSpkDerivedVars &spk,
                              VectorBase<Real> *mean_out) const;
  
  template<typename Real>
  void GetVarScaledSubstateSpeakerMean(int32 j1, int32 substate,
                                       int32 gauss,
                                       const Sgmm2PerSpkDerivedVars &spk,
                                       VectorBase<Real> *mean_out) const;

  /// Computes quantities H = M_i Sigma_i^{-1} M_i^T.
  template<class Real>
  void ComputeH(std::vector< SpMatrix<Real> > *H_i) const;
  
 protected:
  std::vector<int32> pdf2group_;
  std::vector<std::vector<int32> > group2pdf_; // the reverse map.
  
  /// These contain the "background" model associated with the subspace GMM.
  DiagGmm diag_ubm_;
  FullGmm full_ubm_;

  /// Globally shared parameters of the subspace GMM.  The various quantities
  /// are: I = number of Gaussians, D = data dimension, S = phonetic subspace
  /// dimension, T = speaker subspace dimension, J2 = number of pdfs, J1 =
  /// number of groups of pdfs (for SCTM), #mix = number of substates [of state
  /// j2 or state-group j1, depending on context].

  /// Inverse within-class (full) covariances; dim is [I][D][D].
  std::vector< SpMatrix<BaseFloat> > SigmaInv_;
  /// Phonetic-subspace projections. Dimension is [I][D][S]
  std::vector< Matrix<BaseFloat> > M_;
  /// Speaker-subspace projections. Dimension is [I][D][T]
  std::vector< Matrix<BaseFloat> > N_;
  /// Phonetic-subspace weight projection vectors.  Dimension is [I][S]
  Matrix<BaseFloat> w_;
  /// [SSGMM] Speaker-subspace weight projection vectors. Dimension is [I][T]
  Matrix<BaseFloat> u_;
  
  /// The parameters in a particular SGMM state.

  /// v_{jm}, per-state phonetic-subspace vectors. Dimension is [J1][#mix][S].
  std::vector< Matrix<BaseFloat> > v_;
  /// c_{jm}, mixture weights. Dimension is [J2][#mix]
  std::vector< Vector<BaseFloat> > c_;
  /// n_{jim}, per-Gaussian normalizer. Dimension is [J1][I][#mix]
  std::vector< Matrix<BaseFloat> > n_;
  /// [SSGMM] w_{jmi}, dimension is [J1][#mix][I].  Computed from w_ and v_.
  std::vector< Matrix<BaseFloat> > w_jmi_;

  // Priors for MAP adaptation of M -- keeping them here for now but they may
  // be moved somewhere else eventually
  // These are parameters of a matrix-variate normal distribution. The means are
  // the unadapted M_i, and we have 2 separate covaraince matrices for the rows
  // and columns of M.
  std::vector< Matrix<BaseFloat> > M_prior_;  // Matrix-variate Gaussian mean
  SpMatrix<BaseFloat> row_cov_inv_;
  SpMatrix<BaseFloat> col_cov_inv_;

 private:
  /// Computes quasi-occupancies gamma_i from the state-level occupancies,
  /// assuming model correctness.
  void ComputeGammaI(const Vector<BaseFloat> &state_occupancies,
                     Vector<BaseFloat> *gamma_i) const;
  
  /// Called inside SplitSubstates(); splits substates of one group.
  void SplitSubstatesInGroup(const Vector<BaseFloat> &pdf_occupancies,
                             const Sgmm2SplitSubstatesConfig &opts,
                             const SpMatrix<BaseFloat> &sqrt_H_sm,
                             int32 j1, int32 M);
      
  /// Compute a subset of normalizers; used in multi-threaded implementation.
  void ComputeNormalizersInternal(int32 num_threads, int32 thread,
                                  int32 *entropy_count, double *entropy_sum);
  
  /// The code below is called internally from LogLikelihood() and
  /// ComponentPosteriors().  It computes the per-Gaussian log-likelihods
  /// given each sub-state of the state.  Note: the mixture weights
  /// are not included at this point.
  inline void ComponentLogLikes(const Sgmm2PerFrameDerivedVars &per_frame_vars,
                                int32 j1,
                                Sgmm2PerSpkDerivedVars *spk_vars,
                                Matrix<BaseFloat> *loglikes) const;

  
  /// Initializes the matrices M_ and w_.
  void InitializeMw(int32 phn_subspace_dim,
                     const Matrix<BaseFloat> &norm_xform);
  /// Initializes the matrices N_ and [if speaker_dependent_weights==true] u_ 
  void InitializeNu(int32 spk_subspace_dim,                    
                    const Matrix<BaseFloat> &norm_xform,
                    bool speaker_dependent_weights);
  void InitializeVecsAndSubstateWeights(BaseFloat self_weight);
  void InitializeCovars();  ///< initializes the within-class covariances.

  void ComputeHsmFromModel(
      const std::vector< SpMatrix<BaseFloat> > &H,
      const Vector<BaseFloat> &state_occupancies,
      SpMatrix<BaseFloat> *H_sm,
      BaseFloat max_cond) const;

  void ComputePdfMappings(); // sets up group2pdf_ from pdf2group_.
  /// maps from each pdf (index j2) to the corresponding group of
  /// pdfs (index j1) for SCTM.
  
  KALDI_DISALLOW_COPY_AND_ASSIGN(AmSgmm2);
  friend class ComputeNormalizersClass;
  friend class Sgmm2Project;
  friend class EbwAmSgmm2Updater;
  friend class MleAmSgmm2Accs;
  friend class MleAmSgmm2Updater;
  friend class MleSgmm2SpeakerAccs;
  friend class AmSgmm2Functions;  // misc functions that need access.
  friend class Sgmm2Feature;
};

template<typename Real>
inline void AmSgmm2::GetInvCovars(int32 gauss_index,
                                  SpMatrix<Real> *out) const {
  out->Resize(SigmaInv_[gauss_index].NumRows(), kUndefined);
  out->CopyFromSp(SigmaInv_[gauss_index]);
}


template<typename Real>
inline void AmSgmm2::GetSubstateMean(int32 j1, int32 m, int32 i,
                                    VectorBase<Real> *mean_out) const {
  KALDI_ASSERT(mean_out != NULL);
  KALDI_ASSERT(j1 < NumGroups() && m < NumSubstatesForGroup(j1)
               && i < NumGauss());
  KALDI_ASSERT(mean_out->Dim() == FeatureDim());
  Vector<BaseFloat> mean_tmp(FeatureDim());
  mean_tmp.AddMatVec(1.0, M_[i], kNoTrans, v_[j1].Row(m), 0.0);
  mean_out->CopyFromVec(mean_tmp);
}


template<typename Real>
inline void AmSgmm2::GetSubstateSpeakerMean(int32 j1, int32 m, int32 i,
                                            const Sgmm2PerSpkDerivedVars &spk,
                                           VectorBase<Real> *mean_out) const {
  GetSubstateMean(j1, m, i, mean_out);
  if (spk.v_s.Dim() != 0)  // have speaker adaptation...
    mean_out->AddVec(1.0, spk.o_s.Row(i));
}

template<typename Real>
void AmSgmm2::GetVarScaledSubstateSpeakerMean(int32 j1, int32 m, int32 i,
                                             const Sgmm2PerSpkDerivedVars &spk,
                                             VectorBase<Real> *mean_out) const {
  Vector<BaseFloat> tmp_mean(mean_out->Dim()), tmp_mean2(mean_out->Dim());
  GetSubstateSpeakerMean(j1, m, i, spk, &tmp_mean);
  tmp_mean2.AddSpVec(1.0, SigmaInv_[i], tmp_mean, 0.0);
  mean_out->CopyFromVec(tmp_mean2);
}


/// Computes the inverse of an LDA transform (without dimensionality reduction)
/// The computed transform is used in initializing the phonetic and speaker
/// subspaces, as well as while increasing the dimensions of those spaces.
void ComputeFeatureNormalizingTransform(const FullGmm &gmm, Matrix<BaseFloat> *xform);


/// This is the entry for a single time.
struct Sgmm2GauPostElement {
  // Need gselect info here, since "posteriors" is  relative to this set of
  // selected Gaussians.
  std::vector<int32> gselect;
  std::vector<int32> tids;  // transition-ids for each entry in "posteriors"
  std::vector<Matrix<BaseFloat> > posteriors;
};


/// indexed by time.
class Sgmm2GauPost: public std::vector<Sgmm2GauPostElement> {
 public:
  // Add the standard Kaldi Read and Write routines so
  // we can use KaldiObjectHolder with this type.
  explicit Sgmm2GauPost(size_t i) : std::vector<Sgmm2GauPostElement>(i) {}
  Sgmm2GauPost() {}
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};

typedef KaldiObjectHolder<Sgmm2GauPost> Sgmm2GauPostHolder;
typedef RandomAccessTableReader<Sgmm2GauPostHolder> RandomAccessSgmm2GauPostReader;
typedef SequentialTableReader<Sgmm2GauPostHolder> SequentialSgmm2GauPostReader;
typedef TableWriter<Sgmm2GauPostHolder> Sgmm2GauPostWriter;

}  // namespace kaldi


#endif  // KALDI_SGMM_AM_SGMM_H_
