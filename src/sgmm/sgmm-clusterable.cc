// sgmm/sgmm-clusterable.cc

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


#include "sgmm/sgmm-clusterable.h"
#include "hmm/hmm-utils.h"

namespace kaldi {

void SgmmClusterable::Accumulate(
    const SgmmPerFrameDerivedVars &per_frame_vars,
    int32 j, // state index in original SGMM.
    BaseFloat weight) {
  Matrix<BaseFloat> post;
  KALDI_ASSERT(weight >= 0.0); // Doesn't make sense to use negative weights here.
  // Compute Gaussian-level posteriors.
  // Note: "post" is indexed by Gaussian-selection index.
  sgmm_.ComponentPosteriors(per_frame_vars, j, &post);
  if (weight != 1.0) post.Scale(weight);
  const std::vector<int32> &gselect = per_frame_vars.gselect;
  for (int32 ki = 0; ki < gselect.size(); ki++) {
    int32 i = gselect[ki];
    BaseFloat gamma = 0.0;  // Sum the weight over all the vectors (index m) in
    // the state.  In sensible cases there should be just one vector per state
    // at the point where we do this, though.
    for (int32 m = 0; m < post.NumCols(); m++) gamma += post(ki, m);
    gamma_(i) += gamma;
    y_.AddVec(gamma, per_frame_vars.zti.Row(ki));
  }
  // Invalidate my_H_, if present, since it's not efficient to
  // keep it updated during accumulation.
  if (my_H_.NumRows() != 0) 
    my_H_.Resize(0);
}

BaseFloat SgmmClusterable::Objf() const {
  // Objective function consists of the expected log-likelihood of
  // a weight (assuming we estimate the weights directly as parameters
  // instead of the whole subspace thing on the weights), plus
  // the auxiliary function improvement we would get from estimating
  // the state vector v_j starting from zero.  Note: zero is an
  // arbitrary starting point-- we could use any value as long as
  // we were consistent.
  KALDI_ASSERT(static_cast<int32>(H_.size()) == sgmm_.NumGauss());
  if (my_H_.NumRows() == 0.0) {
    SgmmClusterable *s = static_cast<SgmmClusterable*>(this->Copy()); // will
    // set up my_H_, which we need.
    BaseFloat ans = s->Objf();
    delete s;
    return ans;
  }
  double ans = 0.0;
  double tot_gamma = gamma_.Sum(), tot_gamma2 = 0.0;
  if (tot_gamma == 0.0) return 0.0;
  int32 I = gamma_.Dim();
  
  for (int32 i = 0; i < I; i++) {
    double gamma = gamma_(i);
    if (gamma > 0.0) { // Note: should not be negative-- if it is, due to
      double prob = gamma / tot_gamma;
      if (prob > 0.0) { // Note: prob could be zero due to underflow-- this
        // happened! [we can get tiny values due to floating-point roundoff
        // while subtracting clusterable objects].
        ans += gamma * log(gamma / tot_gamma);
      }        
    }
    tot_gamma2 += gamma;
  }
  if (tot_gamma2 == 0.0)
    return 0.0; // No positive elements... maybe small negative ones were from
  // round off.

  // objf improvement is y^T H^{-1} y.
  // We'll try to compute this using Cholesky, first, which is more
  // efficient; if this fails or appears to lead to big values,
  // we'll back off to a more efficient SVD-based implementation.
  try {
    TpMatrix<double> C(my_H_.NumRows());
    C.Cholesky(my_H_);
    C.Invert();
    for (int32 i = 0; i < C.NumRows(); i++)
      if (fabs(C(i, i)) > 100.0) {
        KALDI_VLOG(3) << "Condion-number probably bad: element is "
                      << C(i, i);
        throw std::runtime_error("Bad condition number"); // back off to SVD.
      }
    // Note: assuming things are well preconditioned, the elements
    // C(i,i) should be of the rough magnitude 1/sqrt(count).
    Vector<double> yC(C.NumRows());
    // Note: if we decompose H = C C^T, then the line below
    // does yC = C^{-1} y.  Note: we are computing the inner
    // product y^T H^{-1} y.  H^{-1} = C^{-T} C^{-1}, so
    // y^T H^{-1} y = y^T C^{-T} C^{-1} y = yC^T yC.
    yC.AddTpVec(1.0, C, kNoTrans, y_, 0.0);
    ans += 0.5 * VecVec(yC, yC);
  } catch (...) { // Choleksy threw, or we detected bad condition.
    // we'll do this using an SVD-based implementation that will
    // deal with non-invertible matrices.
    KALDI_VLOG(3) << "Backing off to SVD-based objective computation.";
    Vector<double> v(y_.Dim()); // Initialized automatically to zero.
    ans += SolveQuadraticProblem(my_H_, y_, SolverOptions(), &v); // The objective function
    // change from estimating this vector.
  }
  return ans;
}

void SgmmClusterable::SetZero() {
  gamma_.SetZero();
  y_.SetZero();
  my_H_.SetZero(); // Should work even if empty.
}

void SgmmClusterable::Add(const Clusterable &other_in) {
  const SgmmClusterable *other =
      static_cast<const SgmmClusterable*>(&other_in);
  gamma_.AddVec(1.0, other->gamma_);
  y_.AddVec(1.0, other->y_);
  if (!H_.empty()) { // we need to compute my_H_.
    if (my_H_.NumRows() != 0 && other->my_H_.NumRows() != 0)
      my_H_.AddSp(1.0, other->my_H_);
    else {
      my_H_.Resize(0);
      ComputeH();
    }
  }
}

void SgmmClusterable::Sub(const Clusterable &other_in) {
  const SgmmClusterable *other =
      static_cast<const SgmmClusterable*>(&other_in);
  gamma_.AddVec(-1.0, other->gamma_);
  y_.AddVec(-1.0, other->y_);
  if (!H_.empty()) {
    if (my_H_.NumRows() != 0 && other->my_H_.NumRows() != 0)
      my_H_.AddSp(-1.0, other->my_H_);
    else {
      my_H_.Resize(0);
      ComputeH();
    }
  }
}

BaseFloat SgmmClusterable::Normalizer() const {
  return gamma_.Sum();
}

Clusterable *SgmmClusterable::Copy() const {
  SgmmClusterable *ans = new SgmmClusterable(sgmm_, H_);
  ans->gamma_.CopyFromVec(gamma_);
  ans->y_.CopyFromVec(y_);
  if (!H_.empty()) {
    if (my_H_.NumRows() == 0.0) ans->ComputeH();
    else {
      ans->my_H_.Resize(my_H_.NumRows());
      ans->my_H_.CopyFromSp(my_H_);
    }
  }
  return ans;
}

void SgmmClusterable::Scale(BaseFloat f) {
  KALDI_ASSERT(f >= 0.0);
  gamma_.Scale(f);
  y_.Scale(f);
  if (my_H_.NumRows() != 0) my_H_.Scale(f);
}

void SgmmClusterable::Write(std::ostream &os, bool binary) const {
  gamma_.Write(os, binary);
  y_.Write(os, binary);
}

Clusterable *SgmmClusterable::ReadNew(std::istream &is, bool binary) const {
  SgmmClusterable *ans = new SgmmClusterable(sgmm_, H_);
  ans->gamma_.Read(is, binary);
  ans->y_.Read(is, binary);
  if (!H_.empty()) ans->ComputeH();
  return ans;
}


bool AccumulateSgmmTreeStats(const TransitionModel &trans_model,
                             const AmSgmm &am_sgmm,
                             const std::vector<SpMatrix<double> > &H,
                             int N, // context window size.
                             int P, // central position.
                             const std::vector<int32> &ci_phones, // must be sorted
                             const std::vector<int32> &alignment,
                             const std::vector<std::vector<int32> > &gselect,
                             const SgmmPerSpkDerivedVars &per_spk_vars,
                             const Matrix<BaseFloat> &features,
                             std::map<EventType, SgmmClusterable*> *stats) {
  KALDI_ASSERT(IsSortedAndUniq(ci_phones));
  std::vector<std::vector<int32> > split_alignment;
  bool ans = SplitToPhones(trans_model, alignment, &split_alignment);
  if (!ans) {
    KALDI_WARN << "AccumulateTreeStats: bad alignment.";
    return false;
  }
  int t = 0;
  SgmmPerFrameDerivedVars per_frame_vars;
  
  KALDI_ASSERT(features.NumRows() == static_cast<int32>(alignment.size())
               && alignment.size() == gselect.size());
  for (int i = -N; i < static_cast<int>(split_alignment.size()); i++) {
    // consider window starting at i, only if i+P is within
    // list of phones.
    if (i + P >= 0 && i + P < static_cast<int>(split_alignment.size())) {
      int32 central_phone = trans_model.TransitionIdToPhone(split_alignment[i+P][0]);
      bool is_ctx_dep = ! std::binary_search(ci_phones.begin(),
                                             ci_phones.end(),
                                             central_phone);
      EventType evec;
      for (int j = 0; j < N; j++) {
        int phone;
        if (i + j >= 0 && i + j < static_cast<int>(split_alignment.size()))
          phone = trans_model.TransitionIdToPhone(split_alignment[i+j][0]);
        else
          phone = 0;  // ContextDependency class uses 0 to mean "out of window".
        
        if (is_ctx_dep || j == P)
          evec.push_back(std::make_pair(static_cast<EventKeyType>(j), static_cast<EventValueType>(phone)));
      }
      for (int j = 0; j < static_cast<int>(split_alignment[i+P].size());j++) {
        // for central phone of this window...
        EventType evec_more(evec);
        int32 pdf_id = trans_model.TransitionIdToPdf(split_alignment[i+P][j]),
            pdf_class = trans_model.TransitionIdToPdfClass(split_alignment[i+P][j]);
        // pdf_id represents the acoustic state in the current model.
        // pdf_class will normally by 0, 1 or 2 for a 3-state HMM.
        
        std::pair<EventKeyType, EventValueType> pr(kPdfClass, pdf_class);
        evec_more.push_back(pr);
        std::sort(evec_more.begin(), evec_more.end());  // these must be sorted!
        if (stats->count(evec_more) == 0)
          (*stats)[evec_more] = new SgmmClusterable(am_sgmm, H);

        am_sgmm.ComputePerFrameVars(features.Row(t), gselect[t], per_spk_vars, 0.0,
                                    &per_frame_vars);
        BaseFloat weight = 1.0; // weight is one, since we have alignment.
        (*stats)[evec_more]->Accumulate(per_frame_vars, pdf_id, weight);
        t++;
      }
    }
  }
  KALDI_ASSERT(t == static_cast<int>(alignment.size()));
  return true;
}

void SgmmClusterable::ComputeH() {
  // We're computing my_H_, as a weighted sum of H_, with gamma_ as the
  // weights.  
  KALDI_ASSERT(!H_.empty() && my_H_.NumRows() == 0); // Invalid to call this if H_ empty,
  // or my_H_ already set up.
  my_H_.Resize(H_[0].NumRows()); // will initialize to zero.
  KALDI_ASSERT(static_cast<int32>(H_.size()) == gamma_.Dim());
  for (int32 i = 0; i < gamma_.Dim(); i++) {
    double gamma = gamma_(i);
    if (gamma > 0.0) my_H_.AddSp(gamma, H_[i]);
  }
}


} // end namespace kaldi
