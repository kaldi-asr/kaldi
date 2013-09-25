// tied/tied-gmm.h

// Copyright 2011  Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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

#ifndef KALDI_TIED_TIED_GMM_H_
#define KALDI_TIED_TIED_GMM_H_ 1

#include <vector>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "util/stl-utils.h"

namespace kaldi {

/** \struct TiedGmmPerFrameVars
 *  Holds the per-frame derived variables, namely the current feature vector,
 *  posteriors of the soft vector quantizer (svq) and an indicator if the
 *  latter are current w.r.t. the feature vector.
 */
struct TiedGmmPerFrameVars {
  /// Initialize the TiedGmmPerFrameVars to the given feature dim and number of
  /// codebooks
  void Setup(int32 dim, int32 num_codebooks) {
    x.Resize(dim);
    c.Resize(num_codebooks);
    current.clear();
    current.resize(num_codebooks, false);
    svq.resize(num_codebooks);
  }

  /// Resize the loglikelihood vector of the given codebook
  void ResizeSvq(int32 codebook_index, int32 num_gauss) {
    svq[codebook_index].Resize(num_gauss);
  }

  /// Feature vector
  Vector<BaseFloat> x;

  /// offsets of the svq
  Vector<BaseFloat> c;

  /// cache indicator if the requested SVQ is already computed
  std::vector<bool> current;

  /// soft vector quantizer -- store the posteriors of the codebook(s)
  std::vector<Vector<BaseFloat> > svq;
};

/** \class TiedGmm
 *  Definition for tied Gaussian mixture models
 */
class TiedGmm {
 public:
  /// Empty constructor.
  TiedGmm() { }

  /// Resizes arrays to this dim. Does not initialize data.
  void Setup(int32 codebook_index, int32 nMix);

  /// Returns the number of mixture components in the GMM
  int32 NumGauss() const { return weights_.Dim(); }

  /// Copies from given DiagGmm
  void CopyFromTiedGmm(const TiedGmm &copy);

  /// Returns the log-likelihood of a data point (vector) given the tied GMM
  /// component scores
  /// Note: the argument contains the svq scores, *not* the data!
  BaseFloat LogLikelihood(BaseFloat c, const VectorBase<BaseFloat> &svq) const;

  /// Computes the posterior probabilities of all Gaussian components and
  /// returns loglike.
  /// Note: the argument contains the svq scores, *not* the data!
  BaseFloat ComponentPosteriors(BaseFloat c, const VectorBase<BaseFloat> &svq,
                                Vector<BaseFloat> *posteriors) const;

  /// this = rho x source + (1-rho) x this
  void Interpolate(BaseFloat rho, const TiedGmm &source);

  /// Split the tied GMM weights based on the split sequence of the codebook
  void Split(std::vector<int32> *sequence);

  /// Merge the tied GMM weights based on the merge sequence of the codebook
  void Merge(std::vector<int32> *sequence);

  void Write(std::ostream &rOut, bool binary) const;
  void Read(std::istream &rIn, bool binary);

  /// Const accessors
  const Vector<BaseFloat> &weights() const { return weights_; }
  const int32 &codebook_index() const { return codebook_index_; }

  /// Mutators for both float or double
  template<class Real>
  void SetWeights(const VectorBase<Real> &w);    ///< Set mixure weights

  /// Mutators for single component, supports float or double
  /// Set weight for single component.
  inline void SetComponentWeight(int32 gauss, BaseFloat weight);

  void SetCodebookIndex(int32 codebook_index) {
    codebook_index_ = codebook_index;
  }

 private:
  int32 codebook_index_;       ///< index of the codebook in the AM
  Vector<BaseFloat> weights_;  ///< weights (not log).

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
