// sgmm/am-sgmm.h

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

#ifndef KALDI_SGMM_AM_SGMM_H_
#define KALDI_SGMM_AM_SGMM_H_

#include <vector>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "itf/options-itf.h"
#include "util/table-types.h"

namespace kaldi {

struct SgmmGselectConfig {
  /// Number of highest-scoring full-covariance Gaussians per frame.
  int32 full_gmm_nbest;
  /// Number of highest-scoring diagonal-covariance Gaussians per frame.
  int32 diag_gmm_nbest;

  SgmmGselectConfig() {
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

/** \struct SgmmPerFrameDerivedVars
 *  Holds the per-frame precomputed quantities x(t), x_{i}(t), z_{i}(t), and
 *  n_{i}(t) (cf. Eq. (33)-(36)) for the SGMM, as well as the cached Gaussian
 *  selection records.
 */
struct SgmmPerFrameDerivedVars {
  std::vector<int32> gselect;
  Vector<BaseFloat> xt;   ///< x'(t), FMLLR-adapted, dim = [D], eq.(33)
  Matrix<BaseFloat> xti;  ///< x_{i}(t) = x'(t) - o_i(s): dim = [I][D], eq.(34)
  Matrix<BaseFloat> zti;  ///< z_{i}(t), dim = [I][S], eq.(35)
  Vector<BaseFloat> nti;  ///< n_{i}(t), dim = [I], eq.(36)

  SgmmPerFrameDerivedVars() : xt(0), xti(0, 0), zti(0, 0), nti(0) {}
  void Resize(int32 ngauss, int32 feat_dim, int32 phn_dim) {
    xt.Resize(feat_dim);
    xti.Resize(ngauss, feat_dim);
    zti.Resize(ngauss, phn_dim);
    nti.Resize(ngauss);
  }
  bool IsEmpty() const {
    return (xt.Dim() == 0 || xti.NumRows() == 0 || zti.NumRows() == 0
        || nti.Dim() == 0);
  }
  bool NeedsResizing(int32 ngauss, int32 feat_dim, int32 phn_dim) const {
    /*    if (xt.Dim() != feat_dim)
      KALDI_LOG << "xt dim = " << xt.Dim() << ", feat dim = " << feat_dim;
    if (xti.NumRows() != ngauss || xti.NumCols() != feat_dim)
      KALDI_LOG << "xti size = " << xti.NumRows() << ", " << xti.NumCols()
                << "; ngauss = " << ngauss << ", feat dim = " << feat_dim;
    if (zti.NumRows() != ngauss || zti.NumCols() != phn_dim)
      KALDI_LOG << "zti size = " << zti.NumRows() << ", " << zti.NumCols()
                << "; ngauss = " << ngauss << "; phn dim = " << phn_dim;
    if (nti.Dim() != ngauss)
      KALDI_LOG << "nti dim = " << nti.Dim() << ", ngauss = " << ngauss;
    */
    return (xt.Dim() != feat_dim || xti.NumRows() != ngauss
        || xti.NumCols() != feat_dim || zti.NumRows() != ngauss
        || zti.NumCols() != phn_dim || nti.Dim() != ngauss);
  }
};


struct SgmmPerSpkDerivedVars {
  // To set this up, call ComputePerSpkDerivedVars from the sgmm object.
  void Clear() {
    v_s.Resize(0);
    o_s.Resize(0, 0);
  }
  Vector<BaseFloat> v_s;  ///< Speaker adaptation vector v_^{(s)}. Dim is [T]
  Matrix<BaseFloat> o_s;  ///< Per-speaker offsets o_{i}. Dimension is [I][D]
};


/** \class AmSgmm
 *  Class for definition of the subspace Gmm acoustic model
 */
class AmSgmm {
 public:
  AmSgmm() {}
  void Read(std::istream &rIn, bool binary);
  void Write(std::ostream &out, bool binary,
             SgmmWriteFlagsType write_params) const;

  /// Checks the various components for correct sizes. With wrong sizes,
  /// assertion failure occurs. When the argument is set to true, dimensions of
  /// the various components are printed.
  void Check(bool show_properties = true);

  /// Initializes the SGMM parameters from a full-covariance UBM.
  void InitializeFromFullGmm(const FullGmm &gmm, int32 num_states,
                             int32 phn_subspace_dim, int32 spk_subspace_dim);

  /// Used to copy models (useful in update)
  void CopyFromSgmm(const AmSgmm &other, bool copy_normalizers);

  /// Copies the global parameters from the supplied model, but sets
  /// the state vectors to zero.  Supports reducing the phonetic
  /// and speaker subspace dimensions.
  void CopyGlobalsInitVecs(const AmSgmm &other, int32 phn_subspace_dim,
                           int32 spk_subspace_dim, int32 num_pdfs);

  /// Computes the top-scoring Gaussian indices (used for pruning of later
  /// stages of computation). Returns frame log-likelihood given selected
  /// Gaussians from full UBM.
  BaseFloat GaussianSelection(const SgmmGselectConfig &config,
                              const VectorBase<BaseFloat> &data,
                              std::vector<int32> *gselect) const;

  /// As GaussianSelection, but limiting it to a provided list of
  /// preselected Gaussians (e.g. for gender dependency).
  /// The list "preselect" must be sorted and uniq.
  BaseFloat GaussianSelectionPreselect(const SgmmGselectConfig &config,
                                       const VectorBase<BaseFloat> &data,
                                       const std::vector<int32> &preselect,
                                       std::vector<int32> *gselect) const;

  /// This needs to be called with each new frame of data, prior to accumulation
  /// or likelihood evaluation: it computes various pre-computed quantities. The
  /// 'logdet_s' term is the log determinant of the FMLLR transform, or 0.0 if
  /// no FMLLR is used or it's single-class fMLLR applied in the feature
  /// extraction, and we're not keeping track of it here.
  void ComputePerFrameVars(const VectorBase<BaseFloat> &data,
                           const std::vector<int32> &gselect,
                           const SgmmPerSpkDerivedVars &spk_vars,
                           BaseFloat logdet_s,
                           SgmmPerFrameDerivedVars *per_frame_vars) const;
  
  /// Computes the per-speaker derived vars; assumes vars->v_s is already
  /// set up.
  void ComputePerSpkDerivedVars(SgmmPerSpkDerivedVars *vars) const;

  /// This does a likelihood computation for a given state using the
  /// top-scoring Gaussian components (in per_frame_vars).  If the
  /// log_prune parameter is nonzero (e.g. 5.0), the LogSumExp() stage is
  /// pruned, which is a significant speedup... smaller values are faster.
  BaseFloat LogLikelihood(const SgmmPerFrameDerivedVars &per_frame_vars,
                          int32 state_index, BaseFloat log_prune = 0.0) const;

  /// Similar to LogLikelihood() function above, but also computes the posterior
  /// probabilities for the top-scoring Gaussian components and all substates.
  BaseFloat ComponentPosteriors(const SgmmPerFrameDerivedVars &per_frame_vars,
                                int32 state, Matrix<BaseFloat> *post) const;

  /// Increases the total number of substates based on the state occupancies.
  void SplitSubstates(const Vector<BaseFloat> &state_occupancies,
                      int32 target_nsubstates,
                      BaseFloat perturb,
                      BaseFloat power, 
                      BaseFloat cond);

  /// Functions for increasing the phonetic and speaker space dimensions.
  /// The argument norm_xform is a LDA-like feature normalizing transform,
  /// computed by the ComputeFeatureNormalizer function.
  void IncreasePhoneSpaceDim(int32 target_dim,
                             const Matrix<BaseFloat> &norm_xform);
  void IncreaseSpkSpaceDim(int32 target_dim,
                           const Matrix<BaseFloat> &norm_xform);

  /// Computes (and initializes if necessary) derived vars...
  /// for now this is just the normalizers "n" and the diagonal UBM.
  void ComputeDerivedVars();

  /// Computes the data-independent terms in the log-likelihood computation
  /// for each Gaussian component and all substates. Eq. (31)
  void ComputeNormalizers();

  /// Computes the normalizers, while normalizing the weights to one
  /// among each of the sets in "normalize_sets": these sets should
  /// be disjoint and their union should be all the indices 0 ... I-1.
  void ComputeNormalizersNormalized(
      const std::vector< std::vector<int32> > &normalize_sets);

  /// Computes the LDA-like pre-transform and its inverse as well as the
  /// eigenvalues of the scatter of the means used in FMLLR estimation.
  void ComputeFmllrPreXform(const Vector<BaseFloat> &state_occs,
                            Matrix<BaseFloat> *xform,
                            Matrix<BaseFloat> *inv_xform,
                            Vector<BaseFloat> *diag_mean_scatter) const;

  /// Various model dimensions.
  int32 NumPdfs() const { return c_.size(); }
  int32 NumSubstates(int32 j) const { return c_[j].Dim(); }
  int32 NumGauss() const { return M_.size(); }
  int32 PhoneSpaceDim() const { return w_.NumCols(); }
  int32 SpkSpaceDim() const { return (N_.size() > 0) ? N_[0].NumCols() : 0; }
  int32 FeatureDim() const { return M_[0].NumRows(); }

  void RemoveSpeakerSpace() { N_.clear(); }

  /// Accessors
  const FullGmm & full_ubm() const { return full_ubm_; }
  const DiagGmm & diag_ubm() const { return diag_ubm_; }

  const Matrix<BaseFloat>& StateVectors(int32 state_index) const {
    return v_[state_index];
  }
  const SpMatrix<BaseFloat>& GetInvCovars(int32 gauss_index) const {
    return SigmaInv_[gauss_index];
  }
  const Matrix<BaseFloat>& GetPhoneProjection(int32 gauss_index) const {
    return M_[gauss_index];
  }

  /// Templated accessors (used to accumulate in different precision)
  template<typename Real>
  void GetInvCovars(int32 gauss_index, SpMatrix<Real> *out) const;

  template<typename Real>
  void GetSubstateMean(int32 j, int32 m, int32 i,
                       VectorBase<Real> *mean_out) const;

  template<typename Real>
  void GetSubstateSpeakerMean(int32 state, int32 substate, int32 gauss,
                              const SgmmPerSpkDerivedVars &spk,
                              VectorBase<Real> *mean_out) const;

  template<typename Real>
  void GetVarScaledSubstateSpeakerMean(int32 state, int32 substate,
                                       int32 gauss,
                                       const SgmmPerSpkDerivedVars &spk,
                                       VectorBase<Real> *mean_out) const;
  
  template<typename Real>
  void GetNtransSigmaInv(std::vector< Matrix<Real> > *out) const;

  /// Computes quantities H = M_i Sigma_i^{-1} M_i^T.
  template<class Real>
  void ComputeH(std::vector< SpMatrix<Real> > *H_i) const;
  
 protected:
  friend class ComputeNormalizersClass;
 private:
  /// Compute a subset of normalizers; used in multi-threaded implementation.
  void ComputeNormalizersInternal(int32 num_threads, int32 thread,
                                  int32 *entropy_count, double *entropy_sum);
  

  /// Initializes the matrices M_ and w_
  void InitializeMw(int32 phn_subspace_dim,
                    const Matrix<BaseFloat> &norm_xform);
  /// Initializes the matrices N_
  void InitializeN(int32 spk_subspace_dim, const Matrix<BaseFloat> &norm_xform);
  void InitializeVecs(int32 num_states);  ///< Initializes the state-vectors.
  void InitializeCovars();  ///< initializes the within-class covariances.

  void ComputeSmoothingTermsFromModel(
      const std::vector< SpMatrix<BaseFloat> > &H,
      const Vector<BaseFloat> &state_occupancies, SpMatrix<BaseFloat> *H_sm,
      BaseFloat max_cond) const;

 private:
  /// These contain the "background" model associated with the subspace GMM.
  DiagGmm diag_ubm_;
  FullGmm full_ubm_;

  /// Globally shared parameters of the subspace GMM.
  /// The various quantities are: I = number of Gaussians, D = data dimension,
  /// S = phonetic subspace dimension, T = speaker subspace dimension,
  /// J = number of states, M_{j} = number of substates of state j.

  /// Inverse within-class (full) covariances; dim is [I][D][D].
  std::vector< SpMatrix<BaseFloat> > SigmaInv_;
  /// Phonetic-subspace projections. Dimension is [I][D][S]
  std::vector< Matrix<BaseFloat> > M_;
  /// Speaker-subspace projections. Dimension is [I][D][T]
  std::vector< Matrix<BaseFloat> > N_;
  /// Weight projection vectors. Dimension is [I][S]
  Matrix<BaseFloat> w_;

  /// The parameters in a particular SGMM state.

  /// v_{jm}, per-state phonetic-subspace vectors. Dimension is [J][M_{j}][S].
  std::vector< Matrix<BaseFloat> > v_;
  /// c_{jm}, mixture weights. Dimension is [J][M_{j}]
  std::vector< Vector<BaseFloat> > c_;
  /// n_{jim}, per-Gaussian normalizer. Dimension is [J][I][M_{j}]
  std::vector< Matrix<BaseFloat> > n_;

  // Priors for MAP adaptation of M -- keeping them here for now but they may
  // be moved somewhere else eventually
  // These are parameters of a matrix-variate normal distribution. The means are
  // the unadapted M_i, and we have 2 separate covaraince matrices for the rows
  // and columns of M.
  std::vector< Matrix<BaseFloat> > M_prior_;  // Matrix-variate Gaussian mean
  SpMatrix<BaseFloat> row_cov_inv_;
  SpMatrix<BaseFloat> col_cov_inv_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(AmSgmm);
  friend class EbwAmSgmmUpdater;
  friend class MleAmSgmmUpdater;
  friend class MleSgmmSpeakerAccs;
  friend class AmSgmmFunctions;  // misc functions that need access.
  friend class MleAmSgmmUpdaterMulti;
};

template<typename Real>
inline void AmSgmm::GetInvCovars(int32 gauss_index,
                                        SpMatrix<Real> *out) const {
  out->Resize(SigmaInv_[gauss_index].NumRows(), kUndefined);
  out->CopyFromSp(SigmaInv_[gauss_index]);
}

template<typename Real>
inline void AmSgmm::GetSubstateMean(int32 j, int32 m, int32 i,
                                    VectorBase<Real> *mean_out) const {
  KALDI_ASSERT(mean_out != NULL);
  KALDI_ASSERT(j < NumPdfs() && m < NumSubstates(j) && i < NumGauss());
  KALDI_ASSERT(mean_out->Dim() == FeatureDim());
  Vector<BaseFloat> mean_tmp(FeatureDim());
  mean_tmp.AddMatVec(1.0, M_[i], kNoTrans, v_[j].Row(m), 0.0);
  mean_out->CopyFromVec(mean_tmp);
}


template<typename Real>
inline void AmSgmm::GetSubstateSpeakerMean(int32 j, int32 m, int32 i,
                                           const SgmmPerSpkDerivedVars &spk,
                                           VectorBase<Real> *mean_out) const {
  GetSubstateMean(j, m, i, mean_out);
  if (spk.v_s.Dim() != 0)  // have speaker adaptation...
    mean_out->AddVec(1.0, spk.o_s.Row(i));
}

template<typename Real>
void AmSgmm::GetVarScaledSubstateSpeakerMean(int32 j, int32 m, int32 i,
                                             const SgmmPerSpkDerivedVars &spk,
                                             VectorBase<Real> *mean_out) const {
  Vector<BaseFloat> tmp_mean(mean_out->Dim()), tmp_mean2(mean_out->Dim());
  GetSubstateSpeakerMean(j, m, i, spk, &tmp_mean);
  tmp_mean2.AddSpVec(1.0, SigmaInv_[i], tmp_mean, 0.0);
  mean_out->CopyFromVec(tmp_mean2);
}


/// Computes the inverse of an LDA transform (without dimensionality reduction)
/// The computed transform is used in initializing the phonetic and speaker
/// subspaces, as well as while increasing the dimensions of those spaces.
void ComputeFeatureNormalizer(const FullGmm &gmm, Matrix<BaseFloat> *xform);


/// This is the entry for a single time.
struct SgmmGauPostElement {
  // Need gselect info here, since "posteriors" is  relative to this set of
  // selected Gaussians.
  std::vector<int32> gselect;
  std::vector<int32> tids;  // transition-ids for each entry in "posteriors"
  std::vector<Matrix<BaseFloat> > posteriors;
};


/// indexed by time.
class SgmmGauPost: public std::vector<SgmmGauPostElement> {
 public:
  // Add the standard Kaldi Read and Write routines so
  // we can use KaldiObjectHolder with this type.
  explicit SgmmGauPost(size_t i) : std::vector<SgmmGauPostElement>(i) {}
  SgmmGauPost() {}
  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};

typedef KaldiObjectHolder<SgmmGauPost> SgmmGauPostHolder;
typedef RandomAccessTableReader<SgmmGauPostHolder> RandomAccessSgmmGauPostReader;
typedef SequentialTableReader<SgmmGauPostHolder> SequentialSgmmGauPostReader;
typedef TableWriter<SgmmGauPostHolder> SgmmGauPostWriter;

/// Class for misc functions that need access to SGMM private variables.
class AmSgmmFunctions {
 public:
  /// Computes matrix of approximated K-L divergences,
  /// of size [#states x #states], as described in
  /// "State-Level Data Borrowing for Low-Resource Speech Recognition based on
  ///  Subspace GMMs", by Yanmin Qian et. al, Interspeech 2011.
  /// Model must have one substate per state.
  static void ComputeDistances(const AmSgmm &model,
                               const Vector<BaseFloat> &state_occs,
                               MatrixBase<BaseFloat> *dists);
};

}  // namespace kaldi


#endif  // KALDI_SGMM_AM_SGMM_H_
