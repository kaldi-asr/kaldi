// sgmm/sgmm-clusterable.h

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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

#ifndef KALDI_SGMM_SGMM_CLUSTERABLE_H_
#define KALDI_SGMM_SGMM_CLUSTERABLE_H_

#include <vector>
#include <queue>

#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"
#include "itf/clusterable-itf.h"

namespace kaldi {

/// This header defines an object that can be used to create decision
/// trees using a form of SGMM statistics.  It is analogous to the
/// GaussClusterable object, but uses the SGMM.  The auxiliary function
/// it uses is related to the normal SGMM auxiliary function, but for
/// efficiency it uses a simpler model on the weights, which is equivalent
/// to assuming the weights w_{ji} [there no index m since we assume one
/// mixture per state!] are directly estimated using ML, instead of being
/// computed from v_j and w_i as in the actual SGMM.

class SgmmClusterable: public Clusterable {
 public:
  SgmmClusterable(const AmSgmm &sgmm,
                  const std::vector< SpMatrix<double> > &H): // H can be empty vector
      // at initialization.  Used to cache something from the model.
      sgmm_(sgmm),
      H_(H),
      gamma_(sgmm.NumGauss()),
      y_(sgmm.PhoneSpaceDim()) { }
  virtual std::string Type() const { return "sgmm"; }

  /// compare with the Accumulate function of MleAmSgmmAccs
  /// Note: the pdf-index j, relating to the original SGMM
  /// in sgmm_, is only needed to select the right vector to
  /// compute Gaussian-level alignments with.
  void Accumulate(const SgmmPerFrameDerivedVars &frame_vars,
                  int32 j, 
                  BaseFloat weight);
  
  virtual BaseFloat Objf() const;
  virtual void SetZero();
  virtual void Add(const Clusterable &other_in);
  virtual void Sub(const Clusterable &other_in);
  virtual BaseFloat Normalizer() const;
  virtual Clusterable *Copy() const;
  virtual void Scale(BaseFloat f);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual Clusterable *ReadNew(std::istream &is, bool binary) const;
  virtual ~SgmmClusterable() {}

  const Vector<double> &gamma () const { return gamma_; }
  const Vector<double> &y() const { return y_; }
 private:
  void ComputeH(); // Compute the quantity my_H_, from gamma_ and H_.
  
  const AmSgmm &sgmm_;  // Reference to the SGMM object, needed to compute
  // objective functions.
  const std::vector< SpMatrix<double> > &H_; // Reference to a vector of SpMatrix which
  // should have been computed from the model using ComputeH().  Needed for Objf() function.
  Vector<double> gamma_; // Occupation counts for each Gaussian index.  Comparable
  // to the gamma_{jmi} statistics in the SGMM paper.
  Vector<double> y_; // Statistics comparable to the y_{jm} statistics in the SGMM
  // paper.

  SpMatrix<double> my_H_; // This quantity is a weighted sum over the H quantities,
  // weighted by gamma_(i).  It's only nonempty if the H_ matrix is nonempty.
  // This quantity is never written to disk; it is to be viewed as a kind of
  // cache, present only for purposes of fast objective-function computation.
};


/// Comparable to AccumulateTreeStats, but this version
/// accumulates stats of type SgmmClusterable.  Returns
/// true on success.
bool AccumulateSgmmTreeStats(const TransitionModel &trans_model,
                             const AmSgmm &am_sgmm,
                             const std::vector<SpMatrix<double> > &H, // this is a ref. to temp.
                             // storage needed in the clusterable class... can be empty
                             // during accumulation as it doesn't call Objf().
                             int N, // context window size.
                             int P, // central position.
                             const std::vector<int32> &ci_phones, // must be sorted
                             const std::vector<int32> &alignment,
                             const std::vector<std::vector<int32> > &gselect,
                             const SgmmPerSpkDerivedVars &per_spk_vars,
                             const Matrix<BaseFloat> &features,
                             std::map<EventType, SgmmClusterable*> *stats);


} // end namespace kaldi

#endif  // KALDI_SGMM_SGMM_CLUSTERABLE_H_
