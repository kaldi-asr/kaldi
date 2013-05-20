// lat/kws-functions.cc

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)

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


#include "lat/kws-functions.h"
#include "fstext/determinize-star.h"

namespace kaldi {

int32 Interval::overlap(Interval interval) {
  int32 start = interval.start_;
  int32 end = interval.end_;
  return ((start_ <= start && end_ >= end) ? end-start :
          (start <= start_ && end >= end_) ? end_-start_ :
          (start_ <= start && end_ >= start) ? end_-start :
          (start <= start_ && end >= start_) ? end-start_ : 0);
}

bool CompareInterval(const Interval &i1,
                     const Interval &i2) {
  return (i1.Start() < i2.Start() ? true :
          i1.Start() > i2.Start() ? false:
          i1.End() < i2.End() ? true: false);

}

bool ClusterLattice(CompactLattice *clat, 
                    const vector<int32> &state_times) {
  using namespace fst;
  using std::tr1::unordered_map;

  // Hashmap to store the cluster heads
  unordered_map<size_t, vector<Interval> > Head;

  // Step1: Iterate over the lattice to get the arcs
  CompactLattice::StateId max_id = 0;
  for (StateIterator<CompactLattice> siter(*clat); !siter.Done(); siter.Next()) {
    CompactLattice::StateId state_id = siter.Value();
    for (ArcIterator<CompactLattice> aiter(*clat, state_id); !aiter.Done(); aiter.Next()) {
      CompactLatticeArc arc = aiter.Value();
      if (state_id >= state_times.size() || arc.nextstate >= state_times.size())
        return false;
      if (state_id > max_id)
        max_id = state_id;
      if (arc.nextstate > max_id)
        max_id = arc.nextstate;
      Head[arc.ilabel].push_back(Interval(state_times[state_id], state_times[arc.nextstate]));
    }
  }
  // Check if alignments and the states match
  if (state_times.size() != max_id+1)
    return false;

  // Step2: Iterates over the hashmap to get the cluster head
  unordered_map<size_t, vector<Interval> >::iterator iter;
  for (iter = Head.begin(); iter != Head.end(); iter++) {
    sort(iter->second.begin(), iter->second.end(), CompareInterval);
    vector<Interval> tmp;
    tmp.push_back(iter->second[0]);
    for (int32 i = 1; i < iter->second.size(); i++) {
      if ((*(tmp.end() - 1)).End() <= iter->second[i].Start())
        tmp.push_back(iter->second[i]);
    }
    iter->second = tmp;
  }

  // Step3: Cluster arcs according to the maximum overlap
  for (StateIterator<CompactLattice> siter(*clat); !siter.Done(); siter.Next()) {
    CompactLatticeArc::StateId state_id = siter.Value();
    for (MutableArcIterator<CompactLattice> aiter(clat, state_id); !aiter.Done(); aiter.Next()) {
      CompactLatticeArc arc = aiter.Value();
      // We don't cluster the epsilon arcs
      if (arc.ilabel == 0)
        continue;
      // We cluster the non-epsilon arcs
      Interval interval(state_times[state_id], state_times[arc.nextstate]);
      int32 max_overlap = 0;
      size_t olabel = 1;
      for (int32 i = 0; i < Head[arc.ilabel].size(); i++) {
        int32 overlap = interval.overlap(Head[arc.ilabel][i]);
        if (overlap > max_overlap) {
          max_overlap = overlap;
          olabel = i+1;
        }
      }
      arc.olabel = olabel;
      aiter.SetValue(arc);
    }
  }

  return true;
}

bool ComputeCompactLatticeAlphas(const CompactLattice &clat,
                                 vector<double> *alpha) {
  using namespace fst;

  // typedef the arc, weight types
  typedef CompactLattice::Arc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;

  //Make sure the lattice is topologically sorted.
  if (clat.Properties(fst::kTopSorted, true) == 0) {
    KALDI_WARN << "Input lattice must be topologically sorted.";
    return false;
  }
  if (clat.Start() != 0) {
    KALDI_WARN << "Input lattice must start from state 0.";
    return false;
  }

  int32 num_states = clat.NumStates();
  (*alpha).resize(0);
  (*alpha).resize(num_states, kLogZeroDouble);

  // Now propagate alphas forward. Note that we don't acount the weight of the
  // final state to alpha[final_state] -- we acount it to beta[final_state];
  (*alpha)[0] = 0.0;
  for (StateId s = 0; s < num_states; s++) {
    double this_alpha =(*alpha)[s];
    for (ArcIterator<CompactLattice> aiter(clat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -(arc.weight.Weight().Value1() + arc.weight.Weight().Value2());
      (*alpha)[arc.nextstate] = LogAdd((*alpha)[arc.nextstate], this_alpha + arc_like);
    }
  }

  return true;
}

bool ComputeCompactLatticeBetas(const CompactLattice &clat,
                                vector<double> *beta) {
  using namespace fst;

  // typedef the arc, weight types
  typedef CompactLattice::Arc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;

  // Make sure the lattice is topologically sorted.
  if (clat.Properties(fst::kTopSorted, true) == 0) {
    KALDI_WARN << "Input lattice must be topologically sorted.";
    return false;
  }
  if (clat.Start() != 0) {
    KALDI_WARN << "Input lattice must start from state 0.";
    return false;
  }

  int32 num_states = clat.NumStates();
  (*beta).resize(0);
  (*beta).resize(num_states, kLogZeroDouble);

  // Now propagate betas backward. Note that beta[final_state] contains the
  // weight of the final state in the lattice -- compare that with alpha.
  for (StateId s = num_states-1; s >= 0; s--) {
    Weight f = clat.Final(s);
    double this_beta = -(f.Weight().Value1()+f.Weight().Value2());
    for (ArcIterator<CompactLattice> aiter(clat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -(arc.weight.Weight().Value1()+arc.weight.Weight().Value2());
      double arc_beta = (*beta)[arc.nextstate] + arc_like;
      this_beta = LogAdd(this_beta, arc_beta);
    }
    (*beta)[s] = this_beta;
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

  fst::MapFinalAction FinalAction() const { return fst::MAP_NO_SUPERFINAL; }

  fst::MapSymbolsAction InputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }

  fst::MapSymbolsAction OutputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS;}

  uint64 Properties(uint64 props) const { return props; }
};

class KwsProductFstToKwsLexicographicFstMapper {
 public:
  typedef KwsProductArc FromArc;
  typedef KwsProductWeight FromWeight;
  typedef KwsLexicographicArc ToArc;
  typedef KwsLexicographicWeight ToWeight;

  KwsProductFstToKwsLexicographicFstMapper() {}

  ToArc operator()(const FromArc &arc) const {
    /*return ToArc(arc.ilabel, 
                 arc.olabel, 
                 (arc.weight == FromWeight::Zero() ?
                  ToWeight::Zero() :
                  ToWeight(arc.weight.Value1().Value(), 
                           StdLStdWeight(arc.weight.Value2().Value1().Value(),
                                         (arc.weight.Value2().Value2().Value() 
                                          == ArcticWeight::Zero() ?
                                          TropicalWeight::Zero() :
                                          arc.weight.Value2().Value2().Value())))),
                 arc.nextstate);*/
    return ToArc(arc.ilabel, 
                 arc.olabel, 
                 (arc.weight == FromWeight::Zero() ?
                  ToWeight::Zero() :
                  ToWeight(arc.weight.Value1().Value(), 
                           StdLStdWeight(arc.weight.Value2().Value1().Value(),
                                         arc.weight.Value2().Value2().Value()))),
                 arc.nextstate);
  }

  fst::MapFinalAction FinalAction() const { return fst::MAP_NO_SUPERFINAL; }

  fst::MapSymbolsAction InputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }

  fst::MapSymbolsAction OutputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS;}

  uint64 Properties(uint64 props) const { return props; }
};


bool CreateFactorTransducer(const CompactLattice &clat,
                            const vector<int32> &state_times,
                            int32 utterance_id, 
                            KwsProductFst *factor_transducer) {
  using namespace fst;
  typedef KwsProductArc::StateId StateId;

  // We first compute the alphas and betas
  bool success = false;
  vector<double> alpha;
  vector<double> beta;
  success = ComputeCompactLatticeAlphas(clat, &alpha);
  success = success && ComputeCompactLatticeBetas(clat, &beta);
  if (!success)
    return false;

  // Now we map the CompactLattice to VectorFst<KwsProductArc>. We drop the
  // alignment information and only keep the negated log-probs
  Map(clat, factor_transducer, CompactLatticeToKwsProductFstMapper());

  // Now do the weight pushing manually on the CompactLattice format. Note that
  // the alphas and betas in Kaldi are stored as the log-probs, not the negated
  // log-probs, so the question for weight pushing is a little different from
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
      float w = arc.weight.Value1().Value();
      w += beta[state_id] - beta[arc.nextstate];
      KwsProductWeight weight(w, arc.weight.Value2());
      arc.weight = weight;
      aiter.SetValue(arc);
    }
    // Weight of final state
    if (factor_transducer->Final(state_id) != KwsProductWeight::Zero()) {
      float w = factor_transducer->Final(state_id).Value1().Value();
      w += beta[state_id];
      KwsProductWeight weight(w, factor_transducer->Final(state_id).Value2());
      factor_transducer->SetFinal(state_id, weight); 
    }
  }

  // Modify the alphas and set betas to zero. After that, we get the alphas and
  // betas for the pushed FST. Since I will not use beta anymore, here I don't
  // set them to zero. This can be derived from the weight pushing formula.
  for (int32 s = 0; s < alpha.size(); s++)
    alpha[s] += beta[s] - beta[0];

  // OK, after the above preparation, we finally come to the factor generation
  // step. 
  StateId ns = factor_transducer->NumStates(); 
  StateId ss = factor_transducer->AddState(); 
  StateId fs = factor_transducer->AddState();
  factor_transducer->SetStart(ss);
  factor_transducer->SetFinal(fs, KwsProductWeight::One());

  for (StateId s = 0; s < ns; s++) {
    // Add arcs from initial state to current state
    factor_transducer->AddArc(ss, KwsProductArc(0, 0, KwsProductWeight(-alpha[s], StdXStdprimeWeight(state_times[s], ArcticWeight::One())), s));
    // Add arcs from current state to final state
    factor_transducer->AddArc(s, KwsProductArc(0, utterance_id, KwsProductWeight(0, StdXStdprimeWeight(TropicalWeight::One(), state_times[s])), fs));
    // The old final state is not final any more
    if (factor_transducer->Final(s) != KwsProductWeight::Zero())
      factor_transducer->SetFinal(s, KwsProductWeight::Zero());
  }

  return true;
}

void RemoveLongSilences(int32 max_silence_frames,
                        const vector<int32> &state_times,
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

void DoFactorMerging(KwsProductFst factor_transducer,
                     KwsLexicographicFst *index_transducer) {
  using namespace fst;

  // Encode the transducer first
  EncodeMapper<KwsProductArc> encoder(kEncodeLabels, ENCODE);
  Encode(&factor_transducer, &encoder);

  // Use DeterminizeStar
  KALDI_VLOG(2) << "DoFactorMerging: determinization...";
  KwsProductFst dest_transducer;
  DeterminizeStar(factor_transducer, &dest_transducer);

  KALDI_VLOG(2) << "DoFactorMerging: minimization...";
  Minimize(&dest_transducer);

  Decode(&dest_transducer, encoder);

  Map(dest_transducer, index_transducer, KwsProductFstToKwsLexicographicFstMapper());
}

void DoFactorDisambiguation(KwsLexicographicFst *index_transducer) {
  using namespace fst;
  typedef KwsLexicographicArc::StateId StateId;

  StateId ns = index_transducer->NumStates();
  for (StateId s = 0; s < ns; s++) {
    for (MutableArcIterator<KwsLexicographicFst> 
         aiter(index_transducer, s); !aiter.Done(); aiter.Next()) {
      KwsLexicographicArc arc = aiter.Value();
      if (index_transducer->Final(arc.nextstate) != KwsLexicographicWeight::Zero())
        arc.ilabel = s;
      else
        arc.olabel = 0;
      aiter.SetValue(arc);
    }
  }
}

void OptimizeFactorTransducer(KwsLexicographicFst *index_transducer) {
  using namespace fst;
  KwsLexicographicFst ifst = *index_transducer;
  EncodeMapper<KwsLexicographicArc> encoder(kEncodeLabels, ENCODE);
  Encode(&ifst, &encoder);
  KALDI_VLOG(2) << "OptimizeFactorTransducer: determinization...";
  Determinize(ifst, index_transducer);
  KALDI_VLOG(2) << "OptimizeFactorTransducer: minimization...";
  Minimize(index_transducer);
  Decode(index_transducer, encoder);
}

} // end namespace kaldi
