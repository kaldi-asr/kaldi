// dec-wrap/dec-wrap-utils-inl.h

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

#ifndef KALDI_DEC_WRAP_UTILS_INL_H_
#define KALDI_DEC_WRAP_UTILS_INL_H_

#include "lat/kaldi-lattice.h"
#include <vector>
// Do not include this file directly.  It is included by factor.h.

namespace kaldi {
// FIXME Copied from lat/lattice-functions.cc
// There is no no declaration in lat/lattice-functions.h!
// Computes (normal or Viterbi) alphas and betas; returns (total-prob, or
// best-path negated cost)
// Note: in either case, the alphas and betas are negated costs.
// Requires that lat be topologically sorted.  This code
// will work for either CompactLattice or Latice.
template<typename LatticeType>
static double ComputeLatticeAlphasAndBetas(const LatticeType &lat,
                                           vector<double> *alpha,
                                           vector<double> *beta) {
  typedef typename LatticeType::Arc Arc;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;

  StateId num_states = lat.NumStates();
  KALDI_ASSERT(lat.Properties(fst::kTopSorted, true) == fst::kTopSorted);
  KALDI_ASSERT(lat.Start() == 0);
  alpha->resize(num_states, kLogZeroDouble);
  beta->resize(num_states, kLogZeroDouble);

  double tot_forward_prob = kLogZeroDouble;
  (*alpha)[0] = 0.0;
  // Propagate alphas forward.
  for (StateId s = 0; s < num_states; s++) {
    double this_alpha = (*alpha)[s];
    for (fst::ArcIterator<LatticeType> aiter(lat, s); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight);
      (*alpha)[arc.nextstate] = LogAdd((*alpha)[arc.nextstate],
                                        this_alpha + arc_like);
    }
    Weight f = lat.Final(s);
    if (f != Weight::Zero()) {
      double final_like = this_alpha - ConvertToCost(f);
      tot_forward_prob = LogAdd(tot_forward_prob, final_like);
    }
  }
  for (StateId s = num_states-1; s >= 0; s--) { // it's guaranteed signed.
    double this_beta = -ConvertToCost(lat.Final(s));
    for (fst::ArcIterator<LatticeType> aiter(lat, s); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight),
          arc_beta = (*beta)[arc.nextstate] + arc_like;
      this_beta = LogAdd(this_beta, arc_beta);
    }
    (*beta)[s] = this_beta;
  }
  double tot_backward_prob = (*beta)[lat.Start()];
  if (!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-8)) {
    KALDI_WARN << "Total forward probability over lattice = " << tot_forward_prob
               << ", while total backward probability = " << tot_backward_prob;
  }
  // Split the difference when returning... they should be the same.
  return 0.5 * (tot_backward_prob + tot_forward_prob);
}

} // namespace kaldi

#endif
