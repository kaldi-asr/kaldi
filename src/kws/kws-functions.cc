// kws/kws-functions.cc

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)

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

#include <algorithm>

#include "lat/lattice-functions.h"
#include "kws/kws-functions.h"
#include "fstext/determinize-star.h"
#include "fstext/epsilon-property.h"

// note: this .cc file does not include everything declared in kws-functions.h;
// the remainder are defined in kws-functions2.cc (for compilation speed and
// to avoid generating too-large object files on cygwin).

namespace kaldi {

bool CompareInterval(const Interval &i1,
                     const Interval &i2) {
  return (i1.Start() < i2.Start() ? true :
          i1.Start() > i2.Start() ? false:
          i1.End() < i2.End() ? true: false);
}

bool ClusterLattice(CompactLattice *clat,
                    const std::vector<int32> &state_times) {
  using namespace fst;
  typedef CompactLattice::StateId StateId;

  // Hashmap to store the cluster heads.
  unordered_map<StateId, std::vector<Interval> > head;

  // Step 1: Iterate over the lattice to get the arcs
  StateId max_id = 0;
  for (StateIterator<CompactLattice> siter(*clat); !siter.Done();
                                                    siter.Next()) {
    StateId state_id = siter.Value();
    for (ArcIterator<CompactLattice> aiter(*clat, state_id); !aiter.Done();
                                                             aiter.Next()) {
      CompactLatticeArc arc = aiter.Value();
      if (state_id >= state_times.size() || arc.nextstate >= state_times.size())
        return false;
      if (state_id > max_id)
        max_id = state_id;
      if (arc.nextstate > max_id)
        max_id = arc.nextstate;
      head[arc.ilabel].push_back(Interval(state_times[state_id],
                                          state_times[arc.nextstate]));
    }
  }
  // Check if alignments and the states match
  if (state_times.size() != max_id+1)
    return false;

  // Step 2: Iterates over the hashmap to get the cluster heads.
  //   We sort all the words on their start-time, and the process for getting
  //   the cluster heads is to take the first one as a cluster head; then go
  //   till we find the next one that doesn't overlap in time with the current
  //   cluster head, and so on.
  unordered_map<StateId, std::vector<Interval> >::iterator iter;
  for (iter = head.begin(); iter != head.end(); ++iter) {
    // For this ilabel, sort all the arcs on time, from first to last.
    sort(iter->second.begin(), iter->second.end(), CompareInterval);
    std::vector<Interval> tmp;
    tmp.push_back(iter->second[0]);
    for (int32 i = 1; i < iter->second.size(); i++) {
      if (tmp.back().End() <= iter->second[i].Start())
        tmp.push_back(iter->second[i]);
    }
    iter->second = tmp;
  }

  // Step 3: Cluster arcs according to the maximum overlap: attach
  //   each arc to the cluster-head (as identified in Step 2) which
  //   has the most temporal overlap with the current arc.
  for (StateIterator<CompactLattice> siter(*clat); !siter.Done();
                                                   siter.Next()) {
    CompactLatticeArc::StateId state_id = siter.Value();
    for (MutableArcIterator<CompactLattice> aiter(clat, state_id);
                                            !aiter.Done(); aiter.Next()) {
      CompactLatticeArc arc = aiter.Value();
      // We don't cluster the epsilon arcs
      if (arc.ilabel == 0)
        continue;
      // We cluster the non-epsilon arcs
      Interval interval(state_times[state_id], state_times[arc.nextstate]);
      int32 max_overlap = 0;
      size_t olabel = 1;
      for (int32 i = 0; i < head[arc.ilabel].size(); i++) {
        int32 overlap = interval.Overlap(head[arc.ilabel][i]);
        if (overlap > max_overlap) {
          max_overlap = overlap;
          olabel = i + 1;  // need non-epsilon label.
        }
      }
      arc.olabel = olabel;
      aiter.SetValue(arc);
    }
  }

  return true;
}


class CompactLatticeToKwsProductFstMapper {
 public:
  typedef CompactLatticeArc FromArc;
  typedef CompactLatticeWeight FromWeight;
  typedef KwsProductArc ToArc;
  typedef KwsProductWeight ToWeight;

  CompactLatticeToKwsProductFstMapper() {}

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel,
                 arc.olabel,
                 (arc.weight == FromWeight::Zero() ?
                  ToWeight::Zero() :
                  ToWeight(arc.weight.Weight().Value1()
                           +arc.weight.Weight().Value2(),
                           (arc.weight.Weight() == LatticeWeight::Zero() ?
                            StdXStdprimeWeight::Zero() :
                            StdXStdprimeWeight::One()))),
                 arc.nextstate);
  }

  fst::MapFinalAction FinalAction() const {
    return fst::MAP_NO_SUPERFINAL;
  }

  fst::MapSymbolsAction InputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  fst::MapSymbolsAction OutputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const {
    return props;
  }
};


bool CreateFactorTransducer(const CompactLattice &clat,
                            const std::vector<int32> &state_times,
                            int32 utterance_id,
                            KwsProductFst *factor_transducer) {
  using namespace fst;
  typedef KwsProductArc::StateId StateId;

  // We first compute the alphas and betas
  bool success = false;
  std::vector<double> alpha;
  std::vector<double> beta;
  success = ComputeCompactLatticeAlphas(clat, &alpha);
  success = success && ComputeCompactLatticeBetas(clat, &beta);
  if (!success)
    return false;

  // Now we map the CompactLattice to VectorFst<KwsProductArc>. We drop the
  // alignment information and only keep the negated log-probs
  Map(clat, factor_transducer, CompactLatticeToKwsProductFstMapper());

  // Now do the weight pushing manually on the CompactLattice format. Note that
  // the alphas and betas in Kaldi are stored as the log-probs, not the negated
  // log-probs, so the equation for weight pushing is a little different from
  // the original algorithm (pay attention to the sign). We push the weight to
  // initial and remove the total weight, i.e., the sum of all the outgoing
  // transitions and final weight at any state is equal to One() (push only the
  // negated log-prob, not the alignments)
  for (StateIterator<KwsProductFst>
       siter(*factor_transducer); !siter.Done(); siter.Next()) {
    KwsProductArc::StateId state_id = siter.Value();
    for (MutableArcIterator<KwsProductFst>
         aiter(factor_transducer, state_id); !aiter.Done(); aiter.Next()) {
      KwsProductArc arc = aiter.Value();
      BaseFloat w = arc.weight.Value1().Value();
      w += beta[state_id] - beta[arc.nextstate];
      KwsProductWeight weight(w, arc.weight.Value2());
      arc.weight = weight;
      aiter.SetValue(arc);
    }
    // Weight of final state
    if (factor_transducer->Final(state_id) != KwsProductWeight::Zero()) {
      BaseFloat w = factor_transducer->Final(state_id).Value1().Value();
      w += beta[state_id];
      KwsProductWeight weight(w, factor_transducer->Final(state_id).Value2());
      factor_transducer->SetFinal(state_id, weight);
    }
  }

  // Modify the alphas and set betas to zero. After that, we get the alphas and
  // betas for the pushed FST. Since I will not use beta anymore, here I don't
  // set them to zero. This can be derived from the weight pushing formula.
  for (int32 s = 0; s < alpha.size(); s++) {
    alpha[s] += beta[s] - beta[0];

    if (alpha[s] > 0.1) {
      KALDI_WARN << "Positive alpha " << alpha[s];
    }
  }

  // to understand the next part, look at the comment in
  // ../kwsbin/lattice-to-kws-index.cc just above the call to
  // EnsureEpsilonProperty().  We use the bool has_epsilon_property mainly to
  // handle the case when someone comments out that call.  It should always be
  // true in the normal case.
  std::vector<char> state_properties;
  ComputeStateInfo(*factor_transducer, &state_properties);
  bool has_epsilon_property = true;
  for (size_t i = 0; i < state_properties.size(); i++) {
    char c = state_properties[i];
    if ((c & kStateHasEpsilonArcsEntering) != 0 &&
        (c & kStateHasNonEpsilonArcsEntering) != 0)
      has_epsilon_property = false;
    if ((c & kStateHasEpsilonArcsLeaving) != 0 &&
        (c & kStateHasNonEpsilonArcsLeaving) != 0)
      has_epsilon_property = false;
  }
  if (!has_epsilon_property) {
    KALDI_WARN << "Epsilon property does not hold, reverting to old behavior.";
  }

  // OK, after the above preparation, we finally come to the factor generation
  // step.
  StateId ns = factor_transducer->NumStates();
  StateId ss = factor_transducer->AddState();
  StateId fs = factor_transducer->AddState();
  factor_transducer->SetStart(ss);
  factor_transducer->SetFinal(fs, KwsProductWeight::One());

  for (StateId s = 0; s < ns; s++) {
    // Add arcs from initial state to current state
    if (!has_epsilon_property ||
        (state_properties[s] & kStateHasNonEpsilonArcsLeaving))
      factor_transducer->AddArc(ss, KwsProductArc(0, 0, KwsProductWeight(-alpha[s], StdXStdprimeWeight(state_times[s], ArcticWeight::One())), s));
    // Add arcs from current state to final state
    if (!has_epsilon_property ||
        (state_properties[s] & kStateHasNonEpsilonArcsEntering))
      factor_transducer->AddArc(s, KwsProductArc(0, utterance_id, KwsProductWeight(0, StdXStdprimeWeight(TropicalWeight::One(), state_times[s])), fs));
    // The old final state is not final any more
    if (factor_transducer->Final(s) != KwsProductWeight::Zero())
      factor_transducer->SetFinal(s, KwsProductWeight::Zero());
  }

  return true;
}

void RemoveLongSilences(int32 max_silence_frames,
                        const std::vector<int32> &state_times,
                        KwsProductFst *factor_transducer) {
  using namespace fst;
  typedef KwsProductArc::StateId StateId;

  StateId ns = factor_transducer->NumStates();
  StateId ss = factor_transducer->Start();
  StateId bad_state = factor_transducer->AddState();
  for (StateId s = 0; s < ns; s++) {
    // Skip arcs start from the initial state
    if (s == ss)
      continue;
    for (MutableArcIterator<KwsProductFst>
         aiter(factor_transducer, s); !aiter.Done(); aiter.Next()) {
      KwsProductArc arc = aiter.Value();
      // Skip arcs end with the final state
      if (factor_transducer->Final(arc.nextstate) != KwsProductWeight::Zero())
        continue;
      // Non-silence arcs
      if (arc.ilabel != 0)
        continue;
      // Short silence arcs
      if (state_times[arc.nextstate]-state_times[s] <= max_silence_frames)
        continue;
      // The rest are the long silence arcs, we point their nextstate to
      // bad_state
      arc.nextstate = bad_state;
      aiter.SetValue(arc);
    }
  }

  // Trim the unsuccessful paths
  Connect(factor_transducer);
}


template<class Arc>
static void DifferenceWrapper(const fst::VectorFst<Arc> &fst1,
                              const fst::VectorFst<Arc> &fst2,
                              fst::VectorFst<Arc> *difference) {
  using namespace fst;
  if (!fst2.Properties(kAcceptor, true)) {
    // make it an acceptor by encoding the weights.
    EncodeMapper<Arc> encoder(kEncodeLabels, ENCODE);
    VectorFst<Arc> fst1_copy(fst1);
    VectorFst<Arc> fst2_copy(fst2);
    Encode(&fst1_copy, &encoder);
    Encode(&fst2_copy, &encoder);
    DifferenceWrapper(fst1_copy, fst2_copy, difference);
    Decode(difference, encoder);
  } else {
    VectorFst<Arc> fst2_copy(fst2);
    RmEpsilon(&fst2_copy);  // or Difference will crash.
    RemoveWeights(&fst2_copy);  // or Difference will crash.
    Difference(fst1, fst2_copy, difference);
  }
}


void MaybeDoSanityCheck(const KwsLexicographicFst &index_transducer) {
  typedef KwsLexicographicFst::Arc::Label Label;
  if (GetVerboseLevel() < 2) return;
  KwsLexicographicFst temp_transducer;
  ShortestPath(index_transducer, &temp_transducer);
  std::vector<Label> isymbols, osymbols;
  KwsLexicographicWeight weight;
  GetLinearSymbolSequence(temp_transducer, &isymbols, &osymbols, &weight);
  std::ostringstream os;
  for (size_t i = 0; i < isymbols.size(); i++)
    os << isymbols[i] << ' ';
  BaseFloat best_cost = weight.Value1().Value();
  KALDI_VLOG(3) << "Best path: " << isymbols.size() << " isymbols " << ", "
                << osymbols.size() << " osymbols, isymbols are " << os.str()
                << ", best cost is " << best_cost;

  // Now get second-best path.  This will exclude the best path, which
  // will generally correspond to the empty word sequence (there will
  // be isymbols and osymbols anyway though, because of the utterance-id
  // having been encoded as an osymbol (and later, the EncodeFst turning it
  // into a transducer).
  KwsLexicographicFst difference_transducer;
  DifferenceWrapper(index_transducer, temp_transducer, &difference_transducer);
  ShortestPath(difference_transducer, &temp_transducer);

  GetLinearSymbolSequence(temp_transducer, &isymbols, &osymbols, &weight);
  std::ostringstream os2;
  for (size_t i = 0; i < isymbols.size(); i++)
    os2 << isymbols[i] << ' ';
  BaseFloat second_best_cost = weight.Value1().Value();
  KALDI_VLOG(3) << "Second-best path: " << isymbols.size()
                << " isymbols " << ", "
                << osymbols.size() << " osymbols, isymbols are " << os2.str()
                << ", second-best cost is " << second_best_cost;
  if (second_best_cost < -0.01) {
    KALDI_WARN << "Negative second-best cost found " << second_best_cost;
  }
}


void MaybeDoSanityCheck(const KwsProductFst &product_transducer) {
  if (GetVerboseLevel() < 2) return;
  KwsLexicographicFst index_transducer;

  Map(product_transducer,
      &index_transducer,
      KwsProductFstToKwsLexicographicFstMapper());

  MaybeDoSanityCheck(index_transducer);
}

}  // end namespace kaldi
