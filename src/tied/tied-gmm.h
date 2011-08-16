// tied/tied-gmm.h

// Copyright 2011  Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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

#ifndef KALDI_TIED_TIED_GMM_H_
#define KALDI_TIED_TIED_GMM_H_ 1

#include <vector>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "util/stl-utils.h"

namespace kaldi {

/** \struct TiedGmmPerFrameVars
 *  Holds the per-frame derived variables, e.g. the codebook loglikelihoods.
 */
struct TiedGmmPerFrameVars {
  TiedGmmPerFrameVars() { }
  
  ~TiedGmmPerFrameVars() {
    DeletePointers(&ll);
  }
    
  /// Resize the loglikelihood vector to have num_pdf entries (but initialized with NULL)
  void Resize(int32 num_pdfs) {
    if (ll.size() > 0)
      DeletePointers(&ll);
    
    ll.resize(num_pdfs, NULL);
  }
  
  /// Resize the loglikelihood vector of the given pdf
  void Resize(int32 pdf_index, int32 num_gauss) {
    if (ll[pdf_index] != NULL)
      delete ll[pdf_index];
    
    ll[pdf_index] = new Vector<BaseFloat>(num_gauss);
  }
  
  void Clear() {
    DeletePointers(&ll);
  }
  
  Vector<BaseFloat> x;
  
  std::vector<Vector<BaseFloat> *> ll;
};

/** \class TiedGmm 
 *  Definition for tied Gaussian mixture models
 */
class TiedGmm {
 public:
  /// Empty constructor.
  TiedGmm() : valid_gconsts_(false) { }

  /// Resizes arrays to this dim. Does not initialize data.
  void Setup(int32 pdf_index, int32 nMix);

  /// Returns the number of mixture components in the GMM
  int32 NumGauss() const { return weights_.Dim(); }
  
  /// Copies from given DiagGmm
  void CopyFromTiedGmm(const TiedGmm &copy);

  /// Returns the log-likelihood of a data point (vector) given the tied GMM component scores
  /// caveat: The argument contains the codebook component scores, /NOT/ the data!
  BaseFloat LogLikelihood(const VectorBase<BaseFloat> &scores) const;

  /// Outputs the per-component log-likelihoods given the tied GMM component scores
/// caveat: The argument contains the codebook component scores, /NOT/ the data!
  void LogLikelihoods(const VectorBase<BaseFloat> &scores,
                      Vector<BaseFloat> *loglikes) const;

  /// Outputs the per-component log-likelihoods of a subset
  /// of mixture components.  Note: indices.size() will
  /// equal loglikes->Dim() at output.  loglikes[i] will 
  /// correspond to the log-likelihood of the Gaussian
  /// indexed indices[i].
  /// caveat: The argument contains the codebook component scores, /NOT/ the data!
  void LogLikelihoodsPreselect(const VectorBase<BaseFloat> &scores,
                               const std::vector<int32> &indices,
                               Vector<BaseFloat> *loglikes) const;

  
  /// Computes the posterior probabilities of all Gaussian components
  /// caveat: The argument contains the codebook component scores, /NOT/ the data!
  BaseFloat ComponentPosteriors(const VectorBase<BaseFloat> &scores,
                                Vector<BaseFloat> *posteriors) const;

  /// Computes the log-likelihood of a data point given a single Gaussian
  /// component. NOTE: Currently we make no guarantees about what happens if
  /// one of the variances is zero.
  BaseFloat ComponentLogLikelihood(const VectorBase<BaseFloat> &scores,
                                   int32 comp_id) const;

  /// Sets the gconsts.  Returns the number that are "invalid" e.g. because of
  /// zero weights or variances.
  int32 ComputeGconsts();

  void Write(std::ostream &rOut, bool binary) const;
  void Read(std::istream &rIn, bool binary);

  /// Const accessors
  const Vector<BaseFloat>& gconsts() const {
    KALDI_ASSERT(valid_gconsts_);
    return gconsts_;
  }

  /// Const accessors
  const Vector<BaseFloat>& weights() const { return weights_; }
  const int32& pdf_index() const { return pdf_index_; }

  /// Mutators for both float or double
  template<class Real>
  void SetWeights(const VectorBase<Real>& w);    ///< Set mixure weights

  /// Mutators for single component, supports float or double
  /// Set weight for single component.
  inline void SetComponentWeight(int32 gauss, BaseFloat weight);
  
  void SetPdfIndex(int32 pdf_index) { pdf_index_ = pdf_index; }

 private:
  int32 pdf_index_;            ///< index of the respective codebook (within the AM)
  Vector<BaseFloat> gconsts_; ///< This is actually log(weight) - log(1/nMix), to fix the codebook's gconsts_
  bool valid_gconsts_;        ///< Recompute gconsts_ if false
  Vector<BaseFloat> weights_; ///< weights (not log).

  KALDI_DISALLOW_COPY_AND_ASSIGN(TiedGmm);
};

/// ostream operator that calls TiedGMM::Write()
std::ostream &
operator << (std::ostream & rOut, const kaldi::TiedGmm &gmm);
/// istream operator that calls TiedGMM::Read()
std::istream &
operator >> (std::istream & rIn, kaldi::TiedGmm &gmm);

}  // End namespace kaldi

#include "tied/tied-gmm-inl.h"  // templated functions.

#endif  // KALDI_TIED_TIED_GMM_H_
