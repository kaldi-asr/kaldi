// fstext/rescale-inl.h

// Copyright 2009-2011  Microsoft Corporation;  Jan Silovsky

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

#ifndef KALDI_FSTEXT_RESCALE_INL_H_
#define KALDI_FSTEXT_RESCALE_INL_H_
#include <cstring>
#include "base/kaldi-common.h"
#include "util/stl-utils.h"
#include "fstext/fstext-utils.h"

namespace fst {


template<class Arc>
inline typename Arc::Weight
ComputeTotalWeight(ExpandedFst<Arc> &fst, typename Arc::Weight max_weight, float delta) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  if (fst.Start() == kNoStateId) return Weight::Zero();
  StateId num_states = fst.NumStates();

  float zero = Weight::Zero().Value();

  // Should probably use Weight instead of float here, but would
  // involve some painful comparators.
  vector<float> cur_tot(num_states, zero);
  vector<float> cur_delta(num_states, zero);
  vector<bool> queued(num_states, false);

  std::queue<StateId> q;  // FIFO queue.

  Weight total_final = Weight::Zero();
  {
    float f = static_cast<float>(Weight::One().Value());
    q.push(fst.Start());
    cur_delta[fst.Start()] = cur_tot[fst.Start()] = f;
    queued[fst.Start()] = true;
  }

  while (!q.empty()) {
    StateId s = q.front();
    q.pop();
    Weight w = cur_delta[s];
    cur_delta[s] = zero;
    queued[s] = false;
    cur_tot[s] = Plus(w, Weight(cur_tot[s])).Value();

    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      Weight next_weight = Times(w, arc.weight);
      cur_delta[arc.nextstate] = Plus(Weight(cur_delta[arc.nextstate]),
                                      next_weight).Value();
      if (!queued[arc.nextstate] &&
         !ApproxEqual(Weight(cur_tot[arc.nextstate]),
                      Plus(Weight(cur_delta[arc.nextstate]),
                           Weight(cur_tot[arc.nextstate])), delta)) {
        queued[arc.nextstate] = true;
        q.push(arc.nextstate);
      }
    }
    Weight final = fst.Final(s);
    if (final != Weight::Zero()) {
      total_final = Plus(total_final, Times(w, final));
      if (total_final.Value() < max_weight.Value()) {  // Note: this means that total_final is MORE THAN max_weight.
        // assuming the weight is of the normal type.
        return max_weight;
      }
    }
  }
  return total_final;
}


  
template<class Arc>
inline void Rescale(MutableFst<Arc> *fst, typename Arc::Weight rescale) {
  typedef typename Arc::StateId StateId;
  // Multiplies all arcs and final-probs in the FST by this rescaling amount.
  // Typically useful with non-stochastic FST, in conjunction with pushing: i.e.
  // we rescale, push, and rescale again.
  for (StateIterator<MutableFst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    fst->SetFinal(s, Times(rescale, fst->Final(s)));
    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, s);
        !aiter.Done();
        aiter.Next()) {
      Arc arc = aiter.Value();
      arc.weight = Times(arc.weight, rescale);
      aiter.SetValue(arc);
    }
  }
}

inline LogWeight RescaleToStochastic(MutableFst<LogArc> *fst,
                                     float approx_delta,
                                     float delta) {
  // Rescales the FST until, to within "approx_delta", it sums to one (within
  // its own semiring).  Returns the amount it rescaled by.  Must be of the
  // LogArc or StdArc type.
  typedef LogArc::Weight Weight;

  if (fst->Start() == kNoStateId)
    return Weight::One();  // can't rescale empty FST.

  // total weight).
  Weight max = Weight(-log(2.0));

  // upper_bound and lower_bound are in terms of weight.Value(),
  // in terms of weight they would have the reversed names.
  float upper_bound = 1.0e+20, lower_bound = -1.0e+20;
  bool have_lower = false, have_upper = false;

  Weight last_rescale = Weight::One();
  Weight last_tot = Weight::Zero();

  Weight cur_rescale = Weight::One();
  Weight cur_tot;
  while (1) {
    cur_tot = ComputeTotalWeight(*fst, max, delta);
    std::cerr << "Current rescaling factor is " << cur_rescale <<", total is: " << cur_tot << '\n';
    if (cur_tot.Value() < Weight::One().Value()) {  // read as: cur_tot > 1.
      if (cur_rescale.Value() > lower_bound) {
        lower_bound = cur_rescale.Value();
        have_lower = true;
      }
    } else {
      if (cur_rescale.Value() < upper_bound) {
        upper_bound = cur_rescale.Value();
        have_upper = true;
      }
    }

    Weight factor;  // next factor by which we multiply.
    if (ApproxEqual(cur_tot, Weight::One())) return cur_rescale;  // Succeeded.
    else if (cur_tot == max) {  // Maxed out.
      if (last_tot == Weight::Zero()) {  // First iter.
        factor = Weight(0.2);
      } else if (last_tot != max) {  // Didn't max out last time...
        assert(cur_tot.Value() < last_tot.Value());  // or doesn't make sense.
        factor = Weight( (last_rescale.Value() - cur_rescale.Value()) / 2.0);
      } else {  // Maxed out last time.  Use larger interval.
        factor = Weight(0.5);
      }
    } else {
      // cur_tot != max.
      if (last_tot == Weight::Zero() || last_tot == max) {  // No derivative information.
        if (cur_tot.Value() > 0) factor = Weight(-0.01);  // enough to approximate derivative.
        else factor = Weight(0.01);
      } else {
        float dtot_drescale = (cur_tot.Value()-last_tot.Value()) /
            (cur_rescale.Value() - last_rescale.Value());
        std::cerr << "dtot_drescale = " <<dtot_drescale<<'\n';
        if (dtot_drescale < 1.0) {
          LOG(WARN) << "Rescale: derivative does not have expected value: " << dtot_drescale;
          if (cur_tot.Value() > 0) factor = Weight(-0.01);  // enough to approximate derivative.
          else factor = Weight(0.01);
        } else {
          factor = Weight(-cur_tot.Value() / dtot_drescale);  // want to get the value to zero.
          std::cerr << "Factor is " << factor.Value()<<'\n';
        }
      }
    }

    last_rescale = cur_rescale;
    last_tot = cur_tot;

    if (have_lower && have_upper) {
      float oldval = cur_rescale.Value(),
          newval = oldval+factor.Value();
      assert(upper_bound > lower_bound);
      assert(oldval >= lower_bound && oldval <= upper_bound);
      float margin = 0.1;  // limit must be 0.0 < margin < 0.5.
      float new_lower = lower_bound + (oldval-lower_bound)*margin;
      float new_upper = upper_bound + (oldval-upper_bound)*margin;
      if (newval > new_upper) newval = new_upper;
      if (newval < new_lower) newval = new_lower;
      factor = Weight(newval - cur_rescale.Value());
      cur_rescale = Weight(newval);
    } else {
      cur_rescale = Times(cur_rescale, factor);
    }
    Rescale(fst, factor);

  }
}



} // namespace fst.

#endif
