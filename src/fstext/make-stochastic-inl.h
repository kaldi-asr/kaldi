// fstext/make-stochastic-inl.h

// Copyright 2009-2011  Microsoft Corporation;  Jan Silovsky

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

#ifndef KALDI_FSTEXT_MAKE_STOCHASTIC_INL_H_
#define KALDI_FSTEXT_MAKE_STOCHASTIC_INL_H_
#include <cstring>
#include "base/kaldi-common.h"
#include "util/stl-utils.h"
#include "fstext/fstext-utils.h"

namespace fst {

/* Do not include this file directly.  It is an implementation file included by MakeStochastic.h */


/* Outline of MakeStochasticFst function:

   (A)

      (i) If no symbol table exists, work out a value "symbol_counter" that is one
           plus the highest numbered symbol currently in the FST.  Warn if this
           is excessively large.

     (ii) Initialize a map "label_map" from int to Label, which maps from the
           quantized weights to a symbol id.  The map is initially empty.

   (B)
      (i) Initialize a map m, from states to states.  Can use a vector.  If m(s) is
          defined, it gives a new state-id where arcs that previously went to state s,
          should go.   We will re-map the arcs in phase (iii).

      (ii)
          for each state s:
            let p(s) be the sum (in the semiring) of the out-arc probabilities of s,
              plus the final weight.

            if p(s) == 0 in the semiring:
               Do nothing.   We can't renormalize these states.
            if p(s) == 1 in the semiring [to within a tolerance defined by the precision]
              Do nothing.
            else:
              (a) work out the new output label o: let the integer i be
                    i = p(s).Value() / opts.delta
                  [note that p(s).Value() is really the log of p(s),
                  in the log semiring].
                  if label_map[i] is defined,
                     let o = label_map[i].
                  else:
                     allocate a new symbol (using the symbol table if
                     available, and otherwise symbol_counter), and let o = that
                     symbol.  if the symbol table is used, the string representation
                     will be the prefix plus the string representation of
                     p(s).Value().

                     if leftover_probs array provided, set relevent element
                     so the caller knows the floating-point value for that symbol.
              (b) create a new state t, and let m(s) = t.
              (c) create an arc from t to s, with unit weight, input label
                  epsilon, and output label o .
              (c) for all arcs leaving state s, divide their weight by p(s).
              (d) Do the same for the final-weight of s, if present.

         (iii)
            For each arc a in the fst:
            Write the destination state of arc a as n[a].  If m(n[a]) is defined,
            and the initial state p[a] of arc a is not equal to m(n[a]), then
            set n[a] to m(n[a]).

          (iv)
            Set the initial state to m(initial state), if this is defined.
*/



template<class Arc>
void MakeStochasticFst(const MakeStochasticOptions &opts,
                       MutableFst<Arc> *fst,
                       vector<float> *leftover_probs,
                       int *num_symbols_added) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  if (leftover_probs) leftover_probs->clear();
  if (num_symbols_added) *num_symbols_added = 0;
  assert(fst != NULL);
  assert(opts.delta > 0 && opts.delta >= 1.0e-10);  // we can't quantize as multiple of zero.
  // 1.0e-10 is just a ridiculously small value for delta; it's a slightly arbitrary limit.
  if (fst->Start() < 0) return;  // Empty FST, nothing to do.
  if (leftover_probs == NULL)
    KALDI_ERR << "StochasticFst: error: leftover_probs not provided";

  // part (A)(i): we have to work out the highest
  // numbered symbol in the FST so we can allocate new symbols.
  int64 symbol_counter = HighestNumberedOutputSymbol(*fst) + 1;
  if (symbol_counter > 1000000)
    KALDI_WARN << "MakeStochasticFst: FST has very high-numbered symols, " << symbol_counter;

  // (A)(ii) initialize label_map.
  std::map<int, Label> label_map;  // maps from quantized weights to labels.

  //(B)(i) initialize the map m.
  vector<StateId> m;  // We will resize this as necessary.
  // If the StateId's are non-consecutive there might be
  // an efficiency problem.

  //(B)(ii): for each state s...
  for (StateIterator<MutableFst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    Weight sum = fst->Final(s);
    for (ArcIterator<MutableFst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      sum = Plus(sum, arc.weight);
    }
    sum = sum.Quantize(opts.delta);
    if (sum == Weight::Zero()) continue;  // can't renormalize zero.

    if (sum == Weight::One()) continue;  // no need to renormalize one.

    int64 integer_weight_64 = (int64)(sum.Value()/opts.delta);
    int integer_weight = (int) integer_weight_64;
    if (integer_weight != integer_weight_64) {
      KALDI_ERR << "MakeStochasticFst: weight cannot be encoded, delta is possibly too small ["<<(opts.delta)<<"], weight = "<<(sum.Value());
    }
    assert(integer_weight != 0);  // or sum == Weight::One() should have been true.

    // (B)(ii)(a), work out output symbol.
    Label sym;
    typename std::map<int, Label>::iterator iter = label_map.find(integer_weight);
    if (iter != label_map.end()) {  // seen before.
      sym = iter->second;
    } else {
      // Allocate a new symbol.
      if (num_symbols_added) (*num_symbols_added)++;
      sym = symbol_counter++;
      if (leftover_probs != NULL) {
        if (static_cast<Label>(leftover_probs->size()) < sym+1) leftover_probs->resize(sym+1, 0.0);
        (*leftover_probs)[sym] = sum.Value();
      }
      label_map[integer_weight] = sym;
    }

    // (B)(ii)(b): get new state t.
    StateId t = fst->AddState();

    if (static_cast<StateId>(m.size()) <= s)  m.resize(std::max(s, t)+1, kNoStateId);  // This call
    // should happen just once, because AddState will give us a state past
    // the end of any of the original states.

    // (B)(ii)(b): add arc from t to s.
    m[s] = t;
    Arc arc;
    arc.ilabel = 0;  // epsilon.
    arc.olabel = sym;
    arc.weight = Weight::One();
    arc.nextstate = s;
    fst->AddArc(t, arc);

    // (B)(ii)(c), divide weights of arcs leaving state s by p(s) [=sum.value()]
    Weight inverse = Divide(Weight::One(), sum);
    assert(inverse.Value() == inverse.Value());

    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, s);
        !aiter.Done();
        aiter.Next()) {
      Arc arc = aiter.Value();
      arc.weight = Times(arc.weight, inverse);  // modify the weight(
      aiter.SetValue(arc);
    }

    // (B)(ii)(d) Do the same for the final-weight of s, if present.
    if (fst->Final(s) != Weight::Zero()) {
      fst->SetFinal(s, Times(fst->Final(s), inverse));
    }
  }

  // (B)(iii), re-map destination states of arcs.
  for (StateIterator<MutableFst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, s);
        !aiter.Done();
        aiter.Next()) {
      const Arc &arcref = aiter.Value();
      if (arcref.nextstate < (StateId)m.size() && m[arcref.nextstate] != kNoStateId
         && m[arcref.nextstate] != s) {
        Arc arc = arcref;
        arc.nextstate = m[arcref.nextstate];
        aiter.SetValue(arc);
      }
    }
  }

  // (B)(iv), set the initial state.
  if (fst->Start() < (StateId)m.size() && m[fst->Start()] != kNoStateId)
    fst->SetStart(m[fst->Start()]);

} // end function MakeStochasticFst.

// Declare an override of the template below.
template<>
inline bool IsStochasticFst(const Fst<LogArc> &fst,
                            float delta,
                            LogArc::Weight *min_sum,
                            LogArc::Weight *max_sum);

// Will override this for LogArc where NaturalLess will not work.
template<class Arc>
bool IsStochasticFst(const Fst<Arc> &fst,
                     float delta,
                     typename Arc::Weight *min_sum,
                     typename Arc::Weight *max_sum) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  NaturalLess<Weight> nl;
  bool first_time = true;
  bool ans = true;
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    Weight sum = fst.Final(s);
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      sum = Plus(sum, arc.weight);
    }
    if (!ApproxEqual(Weight::One(), sum, delta)) ans = false;
    if (first_time) {
      first_time = false;
      if (max_sum) *max_sum = sum;
      if (min_sum) *min_sum = sum;
    } else {
      if (max_sum && nl(*max_sum, sum)) *max_sum = sum;
      if (min_sum && nl(sum, *min_sum)) *min_sum = sum;
    }
  }
  if (first_time) {  // just avoid NaNs if FST was empty.
    if (max_sum) *max_sum = Weight::One();
    if (min_sum) *min_sum = Weight::One();
  }
  return ans;
}


// Overriding template for LogArc as NaturalLess does not work there.
template<>
bool IsStochasticFst(const Fst<LogArc> &fst,
                     float delta,
                     LogArc::Weight *min_sum,
                     LogArc::Weight *max_sum) {
  typedef LogArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  bool first_time = true;
  bool ans = true;
  for (StateIterator<Fst<Arc> > siter(fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    Weight sum = fst.Final(s);
    for (ArcIterator<Fst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      sum = Plus(sum, arc.weight);
    }
    if (!ApproxEqual(Weight::One(), sum, delta)) ans = false;
    if (first_time) {
      first_time = false;
      if (max_sum) *max_sum = sum;
      if (min_sum) *min_sum = sum;
    } else {
      // note that max and min are reversed from their normal
      // meanings here (max and min w.r.t. the underlying probabilities).
      if (max_sum && sum.Value() < max_sum->Value()) *max_sum = sum;
      if (min_sum && sum.Value() > min_sum->Value()) *min_sum = sum;
    }
  }
  if (first_time) {  // just avoid NaNs if FST was empty.
    if (max_sum) *max_sum = Weight::One();
    if (min_sum) *min_sum = Weight::One();
  }
  return ans;
}

bool IsStochasticFstInLog(const VectorFst<StdArc> &fst,
                          float delta,
                          StdArc::Weight *min_sum,
                          StdArc::Weight *max_sum) {
  VectorFst<LogArc> logfst;
  Cast(fst, &logfst);
  LogArc::Weight log_min, log_max;
  bool ans = IsStochasticFst(logfst, delta, &log_min, &log_max);
  if (min_sum) *min_sum = StdArc::Weight(log_min.Value());
  if (max_sum) *max_sum = StdArc::Weight(log_max.Value());
  return ans;
}



template<class Arc>
void ReverseMakeStochasticFst(const MakeStochasticOptions &opts,
                              const vector<float> &leftover_probs,
                              MutableFst<Arc> *fst,
                              int *num_syms_removed) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  assert(fst != NULL);
  if (num_syms_removed != NULL) (*num_syms_removed) = 0;

  int leftover_probs_size = leftover_probs.size();

  vector<bool> already_removed;  // for keeping track of "num_syms_removed".
  if (num_syms_removed != NULL)
    already_removed.resize(leftover_probs_size, false);

  for (StateIterator<MutableFst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (MutableArcIterator<MutableFst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      Label olabel = arc.olabel;  // copy it as arc will change on SetValue()
      assert(olabel>=0);
      float prob = 0.0;  // prob of special symbol.
      // Have to decide whether arc.olabel is a "special symbol".
      if (olabel < leftover_probs_size)
        prob = leftover_probs[olabel];
      // else leave it zero.
      if (prob != 0) {  // Have to remove the special symbol.
        Arc new_arc = arc;
        new_arc.olabel = 0;  // Set it to epsilon.
        new_arc.weight = Times(new_arc.weight, (Weight)prob);
        aiter.SetValue(new_arc);
        if (num_syms_removed != NULL) {  // this code is to work out number of unique
          // symbols removed.
          if (olabel >= static_cast<Label>(already_removed.size()))
            already_removed.resize(olabel+1, false);
          if (! already_removed[olabel]) {
            already_removed[olabel] = true;
            (*num_syms_removed)++;
          }
        }
      }
    }
  }
}

} // namespace fst.

#endif
