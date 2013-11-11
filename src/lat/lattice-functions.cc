// lat/lattice-functions.cc

// Copyright 2009-2011  Saarland University (Author: Arnab Ghoshal)
//           2012-2013  Johns Hopkins University (Author: Daniel Povey);  Chao Weng;
//                      Bagher BabaAli
//                2013  Cisco Systems (author: Neha Agrawal) [code modified
//                      from original code in ../gmmbin/gmm-rescore-lattice.cc]

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


#include "lat/lattice-functions.h"
#include "hmm/transition-model.h"
#include "util/stl-utils.h"

namespace kaldi {
using std::map;
using std::vector;

int32 LatticeStateTimes(const Lattice &lat, vector<int32> *times) {
  kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";
  KALDI_ASSERT(lat.Start() == 0);
  int32 num_states = lat.NumStates();
  times->clear();
  times->resize(num_states, -1);
  (*times)[0] = 0;
  for (int32 state = 0; state < num_states; state++) { 
    int32 cur_time = (*times)[state];
    for (fst::ArcIterator<Lattice> aiter(lat, state); !aiter.Done();
        aiter.Next()) {
      const LatticeArc &arc = aiter.Value();

      if (arc.ilabel != 0) {  // Non-epsilon input label on arc
        // next time instance
        if ((*times)[arc.nextstate] == -1) {
          (*times)[arc.nextstate] = cur_time + 1;
        } else {
          KALDI_ASSERT((*times)[arc.nextstate] == cur_time + 1);
        }
      } else {  // epsilon input label on arc
        // Same time instance
        if ((*times)[arc.nextstate] == -1)
          (*times)[arc.nextstate] = cur_time;
        else
          KALDI_ASSERT((*times)[arc.nextstate] == cur_time);
      }
    }
  }
  return (*std::max_element(times->begin(), times->end()));
}

int32 CompactLatticeStateTimes(const CompactLattice &lat, vector<int32> *times) {
  kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";
  KALDI_ASSERT(lat.Start() == 0);
  int32 num_states = lat.NumStates();
  times->clear();
  times->resize(num_states, -1);
  (*times)[0] = 0;
  int32 utt_len = -1;
  for (int32 state = 0; state < num_states; state++) {
    int32 cur_time = (*times)[state];
    for (fst::ArcIterator<CompactLattice> aiter(lat, state); !aiter.Done();
        aiter.Next()) {
      const CompactLatticeArc &arc = aiter.Value();
      int32 arc_len = static_cast<int32>(arc.weight.String().size());
      if ((*times)[arc.nextstate] == -1)
        (*times)[arc.nextstate] = cur_time + arc_len;
      else
        KALDI_ASSERT((*times)[arc.nextstate] == cur_time + arc_len);
    }
    if (lat.Final(state) != CompactLatticeWeight::Zero()) {
      int32 this_utt_len = (*times)[state] + lat.Final(state).String().size();
      if (utt_len == -1) utt_len = this_utt_len;
      else {
        if (this_utt_len != utt_len) {
          KALDI_WARN << "Utterance does not "
              "seem to have a consistent length.";
          utt_len = std::max(utt_len, this_utt_len);
        }
      }
    }        
  }
  if (utt_len == -1) {
    KALDI_WARN << "Utterance does not have a final-state.";
    return 0;
  }
  return utt_len;
}

template<class LatType> // could be Lattice or CompactLattice
bool PruneLattice(BaseFloat beam, LatType *lat) {
  typedef typename LatType::Arc Arc;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  
  KALDI_ASSERT(beam > 0.0);
  kaldi::uint64 props = lat->Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted)) {
    if (fst::TopSort(lat) == false) {
      KALDI_WARN << "Cycles detected in lattice";
      return false;
    }
  }
  KALDI_ASSERT(lat->Start() == 0); // since top sorted.
  int32 num_states = lat->NumStates();
  if (num_states == 0) return false;
  std::vector<double> forward_cost(num_states,
                                   std::numeric_limits<double>::infinity()); // viterbi forward.
  forward_cost[0] = 0.0; // lattice can't have cycles so couldn't be
  // less than this.
  double best_final_cost = std::numeric_limits<double>::infinity();
  // Update the forward probs.
  // Thanks to Jing Zheng for finding a bug here.
  for (int32 state = 0; state < num_states; state++) {
    double this_forward_cost = forward_cost[state];
    for (fst::ArcIterator<LatType> aiter(*lat, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc(aiter.Value());
      StateId nextstate = arc.nextstate;
      KALDI_ASSERT(nextstate > state && nextstate < num_states);
      double next_forward_cost = this_forward_cost +
          ConvertToCost(arc.weight);
      if (forward_cost[nextstate] > next_forward_cost)
        forward_cost[nextstate] = next_forward_cost;
    }
    Weight final_weight = lat->Final(state);
    double this_final_cost = this_forward_cost +
        ConvertToCost(final_weight);
    if (this_final_cost < best_final_cost)
      best_final_cost = this_final_cost;
  }
  int32 bad_state = lat->AddState(); // this state is not final.
  double cutoff = best_final_cost + beam;
  
  // Go backwards updating the backward probs (which share memory with the
  // forward probs), and pruning arcs and deleting final-probs.  We prune arcs
  // by making them point to the non-final state "bad_state".  We'll then use
  // Trim() to remove unnecessary arcs and states.  [this is just easier than
  // doing it ourselves.]
  std::vector<double> &backward_cost(forward_cost);
  for (int32 state = num_states - 1; state >= 0; state--) {
    double this_forward_cost = forward_cost[state];
    double this_backward_cost = ConvertToCost(lat->Final(state));
    if (this_backward_cost + this_forward_cost > cutoff
        && this_backward_cost != std::numeric_limits<double>::infinity())
      lat->SetFinal(state, Weight::Zero());
    for (fst::MutableArcIterator<LatType> aiter(lat, state);
         !aiter.Done();
         aiter.Next()) {
      Arc arc(aiter.Value());
      StateId nextstate = arc.nextstate;
      KALDI_ASSERT(nextstate > state && nextstate < num_states);
      double arc_cost = ConvertToCost(arc.weight),
          arc_backward_cost = arc_cost + backward_cost[nextstate],
          this_fb_cost = this_forward_cost + arc_backward_cost;
      if (arc_backward_cost < this_backward_cost)
        this_backward_cost = arc_backward_cost;
      if (this_fb_cost > cutoff) { // Prune the arc.
        arc.nextstate = bad_state;
        aiter.SetValue(arc);
      }
    }
    backward_cost[state] = this_backward_cost;
  }
  fst::Connect(lat);
  return (lat->NumStates() > 0);
}

// instantiate the template for lattice and CompactLattice.
template bool PruneLattice(BaseFloat beam, Lattice *lat);
template bool PruneLattice(BaseFloat beam, CompactLattice *lat);


BaseFloat LatticeForwardBackward(const Lattice &lat, Posterior *arc_post,
                                 double *acoustic_like_sum) {
  // Note, Posterior is defined as follows:  Indexed [frame], then a list
  // of (transition-id, posterior-probability) pairs.
  // typedef std::vector<std::vector<std::pair<int32, BaseFloat> > > Posterior;
  using namespace fst;
  typedef Lattice::Arc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  
  if (acoustic_like_sum) *acoustic_like_sum = 0.0;

  // Make sure the lattice is topologically sorted.
  if (lat.Properties(fst::kTopSorted, true) == 0)
    KALDI_ERR << "Input lattice must be topologically sorted.";
  KALDI_ASSERT(lat.Start() == 0);

  int32 num_states = lat.NumStates();
  vector<int32> state_times;
  int32 max_time = LatticeStateTimes(lat, &state_times);
  std::vector<double> alpha(num_states, kLogZeroDouble);
  std::vector<double> &beta(alpha); // we re-use the same memory for
  // this, but it's semantically distinct so we name it differently.
  double tot_forward_prob = kLogZeroDouble;

  arc_post->clear();
  arc_post->resize(max_time);
  
  alpha[0] = 0.0;
  // Propagate alphas forward.
  for (StateId s = 0; s < num_states; s++) {
    double this_alpha = alpha[s];
    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight);
      alpha[arc.nextstate] = LogAdd(alpha[arc.nextstate], this_alpha + arc_like);
    }
    Weight f = lat.Final(s);
    if (f != Weight::Zero()) {
      double final_like = this_alpha - (f.Value1() + f.Value2());
      tot_forward_prob = LogAdd(tot_forward_prob, final_like);
      KALDI_ASSERT(state_times[s] == max_time &&
                   "Lattice is inconsistent (final-prob not at max_time)");
    }
  }
  for (StateId s = num_states-1; s >= 0; s--) {
    Weight f = lat.Final(s);
    double this_beta = -(f.Value1() + f.Value2());
    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight),
          arc_beta = beta[arc.nextstate] + arc_like;
      this_beta = LogAdd(this_beta, arc_beta);
      int32 transition_id = arc.ilabel;

      // The following "if" is an optimization to avoid un-needed exp().
      if (transition_id != 0 || acoustic_like_sum != NULL) {
        double posterior = exp(alpha[s] + arc_beta - tot_forward_prob);

        if (transition_id != 0) // Arc has a transition-id on it [not epsilon]
          (*arc_post)[state_times[s]].push_back(std::make_pair(transition_id,
                                                               posterior));
        if (acoustic_like_sum != NULL)
          *acoustic_like_sum -= posterior * arc.weight.Value2();
      }
    }
    if (acoustic_like_sum != NULL && f != Weight::Zero()) {
      double final_logprob = - ConvertToCost(f),
          posterior = exp(alpha[s] + final_logprob - tot_forward_prob);
      *acoustic_like_sum -= posterior * f.Value2();
    }
    beta[s] = this_beta;
  }
  double tot_backward_prob = beta[0];
  if (!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-8)) {
    KALDI_WARN << "Total forward probability over lattice = " << tot_forward_prob
              << ", while total backward probability = " << tot_backward_prob;
  }
  // Now combine any posteriors with the same transition-id.
  for (int32 t = 0; t < max_time; t++)
    MergePairVectorSumming(&((*arc_post)[t]));
  return tot_backward_prob;
}
  

void LatticeActivePhones(const Lattice &lat, const TransitionModel &trans,
                         const vector<int32> &silence_phones,
                         vector< std::set<int32> > *active_phones) {
  KALDI_ASSERT(IsSortedAndUniq(silence_phones));
  vector<int32> state_times;
  int32 num_states = lat.NumStates();
  int32 max_time = LatticeStateTimes(lat, &state_times);
  active_phones->clear();
  active_phones->resize(max_time);
  for (int32 state = 0; state < num_states; state++) {
    int32 cur_time = state_times[state];
    for (fst::ArcIterator<Lattice> aiter(lat, state); !aiter.Done();
        aiter.Next()) {
      const LatticeArc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // Non-epsilon arc
        int32 phone = trans.TransitionIdToPhone(arc.ilabel);
        if (!std::binary_search(silence_phones.begin(),
                                silence_phones.end(), phone))
          (*active_phones)[cur_time].insert(phone);
      }
    }  // end looping over arcs
  }  // end looping over states
}

void ConvertLatticeToPhones(const TransitionModel &trans,
                            Lattice *lat) {
  typedef LatticeArc Arc;
  int32 num_states = lat->NumStates();
  for (int32 state = 0; state < num_states; state++) {
    for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
        aiter.Next()) {
      Arc arc(aiter.Value());
      arc.olabel = 0; // remove any word.
      if ((arc.ilabel != 0) // has a transition-id on input..
          && (trans.TransitionIdToHmmState(arc.ilabel) == 0)
          && (!trans.IsSelfLoop(arc.ilabel)))
         // && trans.IsFinal(arc.ilabel)) // there is one of these per phone...
        arc.olabel = trans.TransitionIdToPhone(arc.ilabel);
      aiter.SetValue(arc);
    }  // end looping over arcs
  }  // end looping over states
}


static inline double LogAddOrMax(bool viterbi, double a, double b) {
  if (viterbi)
    return std::max(a, b);
  else
    return LogAdd(a, b);
}

// Computes (normal or Viterbi) alphas and betas; returns (total-prob, or
// best-path negated cost) Note: in either case, the alphas and betas are
// negated costs.  Requires that lat be topologically sorted.  This code
// will work for either CompactLattice or Latice.
template<typename LatticeType>
static double ComputeLatticeAlphasAndBetas(const LatticeType &lat,
                                           bool viterbi,
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
      (*alpha)[arc.nextstate] = LogAddOrMax(viterbi, (*alpha)[arc.nextstate],
                                                this_alpha + arc_like);
    }
    Weight f = lat.Final(s);
    if (f != Weight::Zero()) {
      double final_like = this_alpha - ConvertToCost(f);
      tot_forward_prob = LogAddOrMax(viterbi, tot_forward_prob, final_like);
    }
  }
  for (StateId s = num_states-1; s >= 0; s--) { // it's guaranteed signed.
    double this_beta = -ConvertToCost(lat.Final(s));
    for (fst::ArcIterator<LatticeType> aiter(lat, s); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight),
          arc_beta = (*beta)[arc.nextstate] + arc_like;
      this_beta = LogAddOrMax(viterbi, this_beta, arc_beta);
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



/// This is used in CompactLatticeLimitDepth.
struct LatticeArcRecord {
  BaseFloat logprob; // logprob <= 0 is the best Viterbi logprob of this arc,
                     // minus the overall best-cost of the lattice.
  CompactLatticeArc::StateId state; // state in the lattice.
  size_t arc; // arc index within the state.
  bool operator < (const LatticeArcRecord &other) const {
    return logprob < other.logprob;
  }
};

void CompactLatticeLimitDepth(int32 max_depth_per_frame,
                              CompactLattice *clat) {
  typedef CompactLatticeArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;

  if (clat->Start() == fst::kNoStateId) {
    KALDI_WARN << "Limiting depth of empty lattice.";
    return;
  }
  if (clat->Properties(fst::kTopSorted, true) == 0) {
    if (!TopSort(clat))
      KALDI_ERR << "Topological sorting of lattice failed.";
  }
  
  vector<int32> state_times;
  int32 T = CompactLatticeStateTimes(*clat, &state_times);

  // The alpha and beta quantities here are "viterbi" alphas and beta.
  std::vector<double> alpha;
  std::vector<double> beta;
  bool viterbi = true;
  double best_prob = ComputeLatticeAlphasAndBetas(*clat, viterbi,
                                                  &alpha, &beta);

  std::vector<std::vector<LatticeArcRecord> > arc_records(T);

  StateId num_states = clat->NumStates();
  for (StateId s = 0; s < num_states; s++) {
    for (fst::ArcIterator<CompactLattice> aiter(*clat, s); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      LatticeArcRecord arc_record;
      arc_record.state = s;
      arc_record.arc = aiter.Position();
      arc_record.logprob =
          (alpha[s] + beta[arc.nextstate] - ConvertToCost(arc.weight))
           - best_prob;
      KALDI_ASSERT(arc_record.logprob < 0.1); // Should be zero or negative.
      int32 num_frames = arc.weight.String().size(), start_t = state_times[s];
      for (int32 t = start_t; t < start_t + num_frames; t++) {
        KALDI_ASSERT(t < T);
        arc_records[t].push_back(arc_record);
      }
    }
  }
  StateId dead_state = clat->AddState(); // A non-coaccesible state which we use
                                         // to remove arcs (make them end
                                         // there).
  size_t max_depth = max_depth_per_frame;
  for (int32 t = 0; t < T; t++) {
    size_t size = arc_records[t].size();
    if (size > max_depth) {
      // we sort from worst to best, so we keep the later-numbered ones,
      // and delete the lower-numbered ones.
      size_t cutoff = size - max_depth;
      std::nth_element(arc_records[t].begin(),
                       arc_records[t].begin() + cutoff,
                       arc_records[t].end());
      for (size_t index = 0; index < cutoff; index++) {
        LatticeArcRecord record(arc_records[t][index]);
        fst::MutableArcIterator<CompactLattice> aiter(clat, record.state);
        aiter.Seek(record.arc);
        Arc arc = aiter.Value();
        if (arc.nextstate != dead_state) { // not already killed.
          arc.nextstate = dead_state;
          aiter.SetValue(arc);
        }
      }
    }
  }
  Connect(clat);
  TopSortCompactLatticeIfNeeded(clat);
}


void TopSortCompactLatticeIfNeeded(CompactLattice *clat) {
  if (clat->Properties(fst::kTopSorted, true) == 0) {
    if (fst::TopSort(clat) == false) {
      KALDI_ERR << "Topological sorting failed";
    }
  }
}

void TopSortLatticeIfNeeded(Lattice *lat) {
  if (lat->Properties(fst::kTopSorted, true) == 0) {
    if (fst::TopSort(lat) == false) {
      KALDI_ERR << "Topological sorting failed";
    }
  }
}


/// Returns the depth of the lattice, defined as the average number of
/// arcs crossing any given frame.  Returns 1 for empty lattices.
/// Requires that input is topologically sorted.
BaseFloat CompactLatticeDepth(const CompactLattice &clat,
                              int32 *num_frames) {
  typedef CompactLattice::Arc::StateId StateId;
  if (clat.Properties(fst::kTopSorted, true) == 0) {
    KALDI_ERR << "Lattice input to CompactLatticeDepth was not topologically "
              << "sorted.";
  }
  if (clat.Start() == fst::kNoStateId) {
    *num_frames = 0;
    return 1.0;
  }
  size_t num_arc_frames = 0;
  int32 t;
  {
    vector<int32> state_times;
    t = CompactLatticeStateTimes(clat, &state_times);
  }
  if (num_frames != NULL)
    *num_frames = t;
  for (StateId s = 0; s < clat.NumStates(); s++) {
    for (fst::ArcIterator<CompactLattice> aiter(clat, s); !aiter.Done();
         aiter.Next()) {
      const CompactLatticeArc &arc = aiter.Value();
      num_arc_frames += arc.weight.String().size();
    }
    num_arc_frames += clat.Final(s).String().size();
  }
  return num_arc_frames / static_cast<BaseFloat>(t);
}


void CompactLatticeDepthPerFrame(const CompactLattice &clat,
                                 std::vector<int32> *depth_per_frame) {
  typedef CompactLattice::Arc::StateId StateId;
  if (clat.Properties(fst::kTopSorted, true) == 0) {
    KALDI_ERR << "Lattice input to CompactLatticeDepthPerFrame was not "
              << "topologically sorted.";
  }
  if (clat.Start() == fst::kNoStateId) {
    depth_per_frame->clear();
    return;
  }
  vector<int32> state_times;
  int32 T = CompactLatticeStateTimes(clat, &state_times);

  depth_per_frame->clear();
  if (T <= 0) {
    return;
  } else {
    depth_per_frame->resize(T, 0);
    for (StateId s = 0; s < clat.NumStates(); s++) {
      int32 start_time = state_times[s];
      for (fst::ArcIterator<CompactLattice> aiter(clat, s); !aiter.Done();
           aiter.Next()) {
        const CompactLatticeArc &arc = aiter.Value();
        int32 len = arc.weight.String().size();
        for (int32 t = start_time; t < start_time + len; t++) {
          KALDI_ASSERT(t < T);
          (*depth_per_frame)[t]++;
        }
      }
      int32 final_len = clat.Final(s).String().size();
      for (int32 t = start_time; t < start_time + final_len; t++) {
        KALDI_ASSERT(t < T);
        (*depth_per_frame)[t]++;
      }
    }
  }
}



void ConvertCompactLatticeToPhones(const TransitionModel &trans,
                                   CompactLattice *clat) {
  typedef CompactLatticeArc Arc;
  typedef Arc::Weight Weight;
  int32 num_states = clat->NumStates();
  for (int32 state = 0; state < num_states; state++) {
    for (fst::MutableArcIterator<CompactLattice> aiter(clat, state);
         !aiter.Done();
         aiter.Next()) {
      Arc arc(aiter.Value());
      std::vector<int32> phone_seq;
      const std::vector<int32> &tid_seq = arc.weight.String();
      for (std::vector<int32>::const_iterator iter = tid_seq.begin();
           iter != tid_seq.end(); ++iter) {
        if (trans.IsFinal(*iter))// note: there is one of these per phone...
          phone_seq.push_back(trans.TransitionIdToPhone(*iter));
      }
      arc.weight.SetString(phone_seq);
      aiter.SetValue(arc);
    } // end looping over arcs
    Weight f = clat->Final(state);
    if (f != Weight::Zero()) {
      std::vector<int32> phone_seq;
      const std::vector<int32> &tid_seq = f.String();
      for (std::vector<int32>::const_iterator iter = tid_seq.begin();
           iter != tid_seq.end(); ++iter) {
        if (trans.IsFinal(*iter))// note: there is one of these per phone...
          phone_seq.push_back(trans.TransitionIdToPhone(*iter));
      }
      f.SetString(phone_seq);
      clat->SetFinal(state, f);
    }
  }  // end looping over states
}

bool LatticeBoost(const TransitionModel &trans,
                  const std::vector<std::set<int32> > &active_phones,
                  const std::vector<int32> &silence_phones,
                  BaseFloat b,
                  BaseFloat max_silence_error,
                  Lattice *lat) {

  TopSortLatticeIfNeeded(lat);
  
  KALDI_ASSERT(IsSortedAndUniq(silence_phones));
  KALDI_ASSERT(max_silence_error >= 0.0 && max_silence_error <= 1.0);
  vector<int32> state_times;
  int32 num_states = lat->NumStates();
  LatticeStateTimes(*lat, &state_times);
  for (int32 state = 0; state < num_states; state++) {
    int32 cur_time = state_times[state];
    if (cur_time < 0 || cur_time > active_phones.size()) {
      KALDI_WARN << "Lattice is too long for active_phones: mismatched den and num lattices/alignments?";
      return false;
    }
    for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
         aiter.Next()) {
      LatticeArc arc = aiter.Value();
      if (arc.ilabel != 0) {  // Non-epsilon arc
        if (arc.ilabel < 0 || arc.ilabel > trans.NumTransitionIds()) {
          KALDI_WARN << "Lattice has out-of-range transition-ids: lattice/model mismatch?";
          return false;
        }
        int32 phone = trans.TransitionIdToPhone(arc.ilabel);
        BaseFloat frame_error;
        if (active_phones[cur_time].count(phone) == 1) {
          frame_error = 0.0;
        } else { // an error...
          if (std::binary_search(silence_phones.begin(), silence_phones.end(), phone))
            frame_error = max_silence_error;
          else
            frame_error = 1.0;
        }
        BaseFloat delta_cost = -b * frame_error; // negative cost if
        // frame is wrong, to boost likelihood of arcs with errors on them.
        // Add this cost to the graph part.        
        arc.weight.SetValue1(arc.weight.Value1() + delta_cost);
        aiter.SetValue(arc);
      }
    }
  }
  return true;
}



int32 LatticePhoneFrameAccuracy(const Lattice &hyp, const TransitionModel &trans,
                                const vector< map<int32, int32> > &ref_phones,
                                vector< map<int32, char> > *arc_accs,
                                vector<int32> *state_times) {
  vector<int32> state_times_hyp;
  int32 max_time_hyp = LatticeStateTimes(hyp, &state_times_hyp),
      max_time_ref = ref_phones.size();
  if (max_time_ref != max_time_hyp) {
    KALDI_ERR << "Reference and hypothesis lattices must have same numbers of "
              << "frames. Found " << max_time_ref << " in ref and "
              << max_time_hyp  << " in hyp.";
  }

  int32 num_states_hyp = hyp.NumStates();
  for (int32 state = 0; state < num_states_hyp; state++) {
    int32 cur_time = state_times_hyp[state];
    for (fst::ArcIterator<Lattice> aiter(hyp, state); !aiter.Done();
        aiter.Next()) {
      const LatticeArc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // Non-epsilon arc
        int32 phone = trans.TransitionIdToPhone(arc.ilabel);
        (*arc_accs)[cur_time][phone] =
            (ref_phones[cur_time].find(phone) == ref_phones[cur_time].end())?
                0 : 1;
      }
    }  // end looping over arcs
  }  // end looping over states
  if (state_times != NULL)
    (*state_times) = state_times_hyp;
  return max_time_hyp;
}


BaseFloat LatticeForwardBackwardMpe(const Lattice &lat,
                                    const TransitionModel &trans,
                                    const vector< map<int32, char> > &arc_accs,
                                    Posterior *arc_post,
                                    const std::vector<int32> &silence_phones) {
  using namespace fst;
  typedef Lattice::Arc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
   
  if (lat.Properties(fst::kTopSorted, true) == 0)
    KALDI_ERR << "Input lattice must be topologically sorted.";
  KALDI_ASSERT(lat.Start() == 0);
  
  int32 num_states = lat.NumStates();
  vector<int32> state_times;
  int32 max_time = LatticeStateTimes(lat, &state_times);
  std::vector<double> alpha(num_states, kLogZeroDouble),
      alpha_mpe(num_states, 0), //forward variable for mpe
      beta(num_states, kLogZeroDouble),
      beta_mpe(num_states, 0); //backward variable for mpe
  
  double tot_forward_prob = kLogZeroDouble;
  double tot_forward_score = 0;

  arc_post->clear();
  arc_post->resize(max_time);

  alpha[0] = 0.0;
  //First Pass Forward, 
  for (StateId s = 0; s < num_states; s++) {
    double this_alpha = alpha[s];
    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight);
      alpha[arc.nextstate] = LogAdd(alpha[arc.nextstate], this_alpha + arc_like);
    }
    Weight f = lat.Final(s);
    if (f != Weight::Zero()) {
      double final_like = this_alpha - (f.Value1() + f.Value2());
      tot_forward_prob = LogAdd(tot_forward_prob, final_like);
      KALDI_ASSERT(state_times[s] == max_time &&
                   "Lattice is inconsistent (final-prob not at max_time)");
    }
  } 
  //First Pass Backward, 
  for (StateId s = num_states-1; s >= 0; s--) {
    Weight f = lat.Final(s);    
    double this_beta = -(f.Value1() + f.Value2());
    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight),
          arc_beta = beta[arc.nextstate] + arc_like;
      this_beta = LogAdd(this_beta, arc_beta);
    }
    beta[s] = this_beta;
  }
  //First Pass Forward-Backward Check
  double tot_backward_prob = beta[0];
  if (!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-8)) {
    KALDI_ERR << "Total forward probability over lattice = " << tot_forward_prob
              << ", while total backward probability = " << tot_backward_prob;
  }
 
  alpha_mpe[0] = 0.0;
  //Second Pass Forward, calculate forward for MPE,
  for (StateId s = 0; s < num_states; s++) {
    double this_alpha = alpha[s];
    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight);
      double frame_acc = 0.0;
      if (arc.ilabel != 0) {
      int32 cur_time = state_times[s];
      int32 phone = trans.TransitionIdToPhone(arc.ilabel);
      frame_acc = (std::binary_search(silence_phones.begin(), silence_phones.end(), phone))? 
          0.0 : ((arc_accs[cur_time].find(phone) == arc_accs[cur_time].end())?
          0.0 : 1.0);
      }
      double arc_scale = std::exp(alpha[s] + arc_like - alpha[arc.nextstate]);
      alpha_mpe[arc.nextstate] += arc_scale * (alpha_mpe[s] + frame_acc);
    }
    Weight f = lat.Final(s);
    if (f != Weight::Zero()) {
      double final_like = this_alpha - (f.Value1() + f.Value2());
      double arc_scale = std::exp(final_like - tot_forward_prob);
      tot_forward_score += arc_scale * alpha_mpe[s];   
      KALDI_ASSERT(state_times[s] == max_time &&
                   "Lattice is inconsistent (final-prob not at max_time)");
    }
  }
  //Second Pass Backward, collect Mpe style posteriors
  for (StateId s = num_states-1; s >= 0; s--) {
    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight),
          arc_beta = beta[arc.nextstate] + arc_like;
      double frame_acc = 0.0;
      int32 transition_id = arc.ilabel;
      if (arc.ilabel != 0) {
      int32 cur_time = state_times[s];
      int32 phone = trans.TransitionIdToPhone(arc.ilabel);
      frame_acc = (std::binary_search(silence_phones.begin(), silence_phones.end(), phone))? 
          0.0 : ((arc_accs[cur_time].find(phone) == arc_accs[cur_time].end())?
          0.0 : 1.0);
      }
      double arc_scale = std::exp(beta[arc.nextstate] + arc_like - beta[s]);
      // check arc_scale NAN, 
      // this is to prevent partial paths in Lattices
      // i.e., paths dont survive to the final state 
      if (KALDI_ISNAN(arc_scale)) arc_scale = 0; 
      beta_mpe[s] += arc_scale * (beta_mpe[arc.nextstate] + frame_acc);

      if (transition_id != 0) { // Arc has a transition-id on it [not epsilon]
        double posterior = exp(alpha[s] + arc_beta - tot_forward_prob);
        double acc_diff = alpha_mpe[s] + frame_acc + beta_mpe[arc.nextstate]
                               - tot_forward_score;
        double posterior_mpe = posterior * acc_diff; 
        (*arc_post)[state_times[s]].push_back(std::make_pair(transition_id,
                                                             posterior_mpe));
      }
    }
  }  
 
  //Second Pass Forward Backward check
  double tot_backward_score = beta_mpe[0];  // Initial state id == 0
  // may loose the condition somehow here 1e-5/1e-4
  if (!ApproxEqual(tot_forward_score, tot_backward_score, 1e-4)) {
    KALDI_ERR << "Total forward score over lattice = " << tot_forward_score
              << ", while total backward score = " << tot_backward_score;
  } 

  // Output the computed posteriors
  for (int32 t = 0; t < max_time; t++)
    MergePairVectorSumming(&((*arc_post)[t]));
  return tot_forward_score;
}


BaseFloat LatticeForwardBackwardSmbr(const Lattice &lat,
                                     const TransitionModel &trans,
                                     const vector< map<int32, char> > &arc_accs,
                                     const std::vector<int32> &silence_phones,
                                     Posterior *arc_post) {
  using namespace fst;
  typedef Lattice::Arc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;

  if (lat.Properties(fst::kTopSorted, true) == 0)
    KALDI_ERR << "Input lattice must be topologically sorted.";
  KALDI_ASSERT(lat.Start() == 0);

  int32 num_states = lat.NumStates();
  vector<int32> state_times;
  int32 max_time = LatticeStateTimes(lat, &state_times);
  std::vector<double> alpha(num_states, kLogZeroDouble),
      alpha_smbr(num_states, 0), //forward variable for sMBR
      beta(num_states, kLogZeroDouble),
      beta_smbr(num_states, 0); //backward variable for sMBR

  double tot_forward_prob = kLogZeroDouble;
  double tot_forward_score = 0;

  arc_post->clear();
  arc_post->resize(max_time);

  alpha[0] = 0.0;
  //First Pass Forward,
  for (StateId s = 0; s < num_states; s++) {
    double this_alpha = alpha[s];
    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight);
      alpha[arc.nextstate] = LogAdd(alpha[arc.nextstate], this_alpha + arc_like);
    }
    Weight f = lat.Final(s);
    if (f != Weight::Zero()) {
      double final_like = this_alpha - (f.Value1() + f.Value2());
      tot_forward_prob = LogAdd(tot_forward_prob, final_like);
      KALDI_ASSERT(state_times[s] == max_time &&
                   "Lattice is inconsistent (final-prob not at max_time)");
    }
  }
  //First Pass Backward,
  for (StateId s = num_states-1; s >= 0; s--) {
    Weight f = lat.Final(s);
    double this_beta = -(f.Value1() + f.Value2());
    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight),
          arc_beta = beta[arc.nextstate] + arc_like;
      this_beta = LogAdd(this_beta, arc_beta);
    }
    beta[s] = this_beta;
  }
  //First Pass Forward-Backward Check
  double tot_backward_prob = beta[0];
  if (!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-8)) {
    KALDI_ERR << "Total forward probability over lattice = " << tot_forward_prob
              << ", while total backward probability = " << tot_backward_prob;
  }

  alpha_smbr[0] = 0.0;
  //Second Pass Forward, calculate forward for MPE,
  for (StateId s = 0; s < num_states; s++) {
    double this_alpha = alpha[s];
    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight);
      double frame_acc = 0.0;
      if (arc.ilabel != 0) {
      int32 cur_time = state_times[s];
      int32 phone = trans.TransitionIdToPhone(arc.ilabel),
          pdf = trans.TransitionIdToPdf(arc.ilabel);
      frame_acc = (std::binary_search(silence_phones.begin(), silence_phones.end(), phone))?
          0.0 : ((arc_accs[cur_time].find(pdf) == arc_accs[cur_time].end())?
          0.0 : 1.0);
      }
      double arc_scale = std::exp(alpha[s] + arc_like - alpha[arc.nextstate]);
      alpha_smbr[arc.nextstate] += arc_scale * (alpha_smbr[s] + frame_acc);
    }
    Weight f = lat.Final(s);
    if (f != Weight::Zero()) {
      double final_like = this_alpha - (f.Value1() + f.Value2());
      double arc_scale = std::exp(final_like - tot_forward_prob);
      tot_forward_score += arc_scale * alpha_smbr[s];
      KALDI_ASSERT(state_times[s] == max_time &&
                   "Lattice is inconsistent (final-prob not at max_time)");
    }
  }
  //Second Pass Backward, collect Mpe style posteriors
  for (StateId s = num_states-1; s >= 0; s--) {
    for (ArcIterator<Lattice> aiter(lat, s); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_like = -ConvertToCost(arc.weight),
          arc_beta = beta[arc.nextstate] + arc_like;
      double frame_acc = 0.0;
      int32 transition_id = arc.ilabel;
      if (arc.ilabel != 0) {
      int32 cur_time = state_times[s];
      int32 phone = trans.TransitionIdToPhone(arc.ilabel),
          pdf = trans.TransitionIdToPdf(arc.ilabel);
      frame_acc = (std::binary_search(silence_phones.begin(), silence_phones.end(), phone))?
          0.0 : ((arc_accs[cur_time].find(pdf) == arc_accs[cur_time].end())?
          0.0 : 1.0);
      }
      double arc_scale = std::exp(beta[arc.nextstate] + arc_like - beta[s]);
      // check arc_scale NAN,
      // this is to prevent partial paths in Lattices
      // i.e., paths dont survive to the final state
      if (KALDI_ISNAN(arc_scale)) arc_scale = 0;
      beta_smbr[s] += arc_scale * (beta_smbr[arc.nextstate] + frame_acc);

      if (transition_id != 0) { // Arc has a transition-id on it [not epsilon]
        double posterior = exp(alpha[s] + arc_beta - tot_forward_prob);
        double acc_diff = alpha_smbr[s] + frame_acc + beta_smbr[arc.nextstate]
                               - tot_forward_score;
        double posterior_smbr = posterior * acc_diff;
        (*arc_post)[state_times[s]].push_back(std::make_pair(transition_id,
                                                             posterior_smbr));
      }
    }
  }

  //Second Pass Forward Backward check
  double tot_backward_score = beta_smbr[0];  // Initial state id == 0
  // may loose the condition somehow here 1e-5/1e-4
  if (!ApproxEqual(tot_forward_score, tot_backward_score, 1e-4)) {
    KALDI_ERR << "Total forward score over lattice = " << tot_forward_score
              << ", while total backward score = " << tot_backward_score;
  }

  // Output the computed posteriors
  for (int32 t = 0; t < max_time; t++)
    MergePairVectorSumming(&((*arc_post)[t]));
  return tot_forward_score;
}


bool CompactLatticeToWordAlignment(const CompactLattice &clat,
                                   std::vector<int32> *words,
                                   std::vector<int32> *begin_times,
                                   std::vector<int32> *lengths) {
  words->clear();
  begin_times->clear();
  lengths->clear();
  typedef CompactLattice::Arc Arc;
  typedef Arc::Label Label;
  typedef CompactLattice::StateId StateId;
  typedef CompactLattice::Weight Weight;
  using namespace fst;
  StateId state = clat.Start();
  int32 cur_time = 0;
  if (state == kNoStateId) {
    KALDI_WARN << "Empty lattice.";
    return false;
  }
  while (1) {
    Weight final = clat.Final(state);
    size_t num_arcs = clat.NumArcs(state);
    if (final != Weight::Zero()) {
      if (num_arcs != 0) {
        KALDI_WARN << "Lattice is not linear.";
        return false;
      }
      if (! final.String().empty()) {
        KALDI_WARN << "Lattice has alignments on final-weight: probably "
            "was not word-aligned (alignments will be approximate)";
      }
      return true;
    } else {
      if (num_arcs != 1) {
        KALDI_WARN << "Lattice is not linear: num-arcs = " << num_arcs;
        return false;
      }
      fst::ArcIterator<CompactLattice> aiter(clat, state);
      const Arc &arc = aiter.Value();
      Label word_id = arc.ilabel; // Note: ilabel==olabel, since acceptor.
      // Also note: word_id may be zero; we output it anyway.
      int32 length = arc.weight.String().size();
      words->push_back(word_id);
      begin_times->push_back(cur_time);
      lengths->push_back(length);
      cur_time += length;
      state = arc.nextstate;
    }
  }
}


void CompactLatticeShortestPath(const CompactLattice &clat,
                                CompactLattice *shortest_path) {
  using namespace fst;
  if (clat.Properties(fst::kTopSorted, true) == 0) {
    CompactLattice clat_copy(clat);
    if (!TopSort(&clat_copy))
      KALDI_ERR << "Was not able to topologically sort lattice (cycles found?)";
    CompactLatticeShortestPath(clat_copy, shortest_path);
    return;
  }
  // Now we can assume it's topologically sorted.
  shortest_path->DeleteStates();
  if (clat.Start() == kNoStateId) return;
  KALDI_ASSERT(clat.Start() == 0); // since top-sorted.
  typedef CompactLatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef CompactLatticeWeight Weight;
  vector<std::pair<double, StateId> > best_cost_and_pred(clat.NumStates() + 1);
  StateId superfinal = clat.NumStates();
  for (StateId s = 0; s <= clat.NumStates(); s++) {
    best_cost_and_pred[s].first = numeric_limits<double>::infinity();
    best_cost_and_pred[s].second = fst::kNoStateId;
  }
  best_cost_and_pred[0].first = 0;
  for (StateId s = 0; s < clat.NumStates(); s++) {
    double my_cost = best_cost_and_pred[s].first;
    for (ArcIterator<CompactLattice> aiter(clat, s);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      double arc_cost = ConvertToCost(arc.weight),
          next_cost = my_cost + arc_cost;
      if (next_cost < best_cost_and_pred[arc.nextstate].first) {
        best_cost_and_pred[arc.nextstate].first = next_cost;
        best_cost_and_pred[arc.nextstate].second = s;
      }
    }
    double final_cost = ConvertToCost(clat.Final(s)),
        tot_final = my_cost + final_cost;
    if (tot_final < best_cost_and_pred[superfinal].first) {
      best_cost_and_pred[superfinal].first = tot_final;
      best_cost_and_pred[superfinal].second = s;
    }
  }
  std::vector<StateId> states; // states on best path.
  StateId cur_state = superfinal;
  while (cur_state != 0) {
    StateId prev_state = best_cost_and_pred[cur_state].second;
    if (prev_state == kNoStateId) {
      KALDI_WARN << "Failure in best-path algorithm for lattice (infinite costs?)";
      return; // return empty best-path.
    }
    states.push_back(prev_state);
    KALDI_ASSERT(cur_state != prev_state && "Lattice with cycles");
    cur_state = prev_state;
  }
  std::reverse(states.begin(), states.end());
  for (size_t i = 0; i < states.size(); i++)
    shortest_path->AddState();
  for (StateId s = 0; static_cast<size_t>(s) < states.size(); s++) {
    if (s == 0) shortest_path->SetStart(s);
    if (static_cast<size_t>(s + 1) < states.size()) { // transition to next state.
      bool have_arc = false;
      Arc cur_arc;
      for (ArcIterator<CompactLattice> aiter(clat, states[s]);
           !aiter.Done();
           aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.nextstate == states[s+1]) {
          if (!have_arc ||
              ConvertToCost(arc.weight) < ConvertToCost(cur_arc.weight)) {
            cur_arc = arc;
            have_arc = true;
          }
        }
      }
      KALDI_ASSERT(have_arc && "Code error.");
      shortest_path->AddArc(s, Arc(cur_arc.ilabel, cur_arc.olabel,
                                   cur_arc.weight, s+1));
    } else { // final-prob.
      shortest_path->SetFinal(s, clat.Final(states[s]));
    }
  }
}

void AddWordInsPenToCompactLattice(BaseFloat word_ins_penalty, 
                                   CompactLattice *clat) {
  typedef CompactLatticeArc Arc;
  int32 num_states = clat->NumStates();

  //scan the lattice
  for (int32 state = 0; state < num_states; state++) {
    for (fst::MutableArcIterator<CompactLattice> aiter(clat, state);
         !aiter.Done(); aiter.Next()) {
      
      Arc arc(aiter.Value());
      
      if (arc.ilabel != 0) { // if there is a word on this arc
        LatticeWeight weight = arc.weight.Weight();
        // add word insertion penalty to lattice
        weight.SetValue1( weight.Value1() + word_ins_penalty);    
        arc.weight.SetWeight(weight);
        aiter.SetValue(arc);
      } 
    } // end looping over arcs
  }  // end looping over states  
} 

struct Tuple {
  Tuple(int32 state, int32 arc, int32 offset):
    state_id(state), arc_id(arc), trans_offset(offset) {}
  int32 state_id;
  int32 arc_id;
  int32 trans_offset;
};

bool RescoreCompactLattice(DecodableInterface *decodable,
                           CompactLattice *clat) {
  if (clat->NumStates() == 0) {
    KALDI_WARN << "Rescoring empty lattice";
    return false;
  }
  uint64 props = clat->Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted)) {
    if (fst::TopSort(clat) == false) {
      KALDI_WARN << "Cycles detected in lattice.";
      return false;
    }
  }
  std::vector<int32> state_times;
  int32 utt_len = kaldi::CompactLatticeStateTimes(*clat, &state_times);
  
  std::vector<std::vector<Tuple> > time_to_state(utt_len);

  int32 num_states = clat->NumStates();
  KALDI_ASSERT(num_states == state_times.size());
  for (size_t state = 0; state < num_states; state++) {
    KALDI_ASSERT(state_times[state] >= 0);
    int32 t = state_times[state];
    int32 arc_id = 0;
    for (fst::MutableArcIterator<CompactLattice> aiter(clat, state);
         !aiter.Done(); aiter.Next(), arc_id++) {
      CompactLatticeArc arc = aiter.Value();
      std::vector<int32> arc_string = arc.weight.String();
      
      for (size_t offset = 0; offset < arc_string.size(); offset++) {
        if (t < utt_len) { // end state may be past this..
          time_to_state[t+offset].push_back(Tuple(state, arc_id, offset));
        } else {
          if (t != utt_len) {
            KALDI_WARN << "There appears to be lattice/feature mismatch, "
                       << "aborting.";
            return false;
          }
        }
      }
    }
    if (clat->Final(state) != CompactLatticeWeight::Zero()) {
      arc_id = -1;
      std::vector<int32> arc_string = clat->Final(state).String();
      for (size_t offset = 0; offset < arc_string.size(); offset++) {
        KALDI_ASSERT(t + offset < utt_len); // already checked in
        // CompactLatticeStateTimes, so would be code error.
        time_to_state[t+offset].push_back(Tuple(state, arc_id, offset));
      }
    }
  }

  for (int32 t = 0; t < utt_len; t++) {
    if ((t < utt_len - 1) == decodable->IsLastFrame(t)) {
      // this if-statement compares two boolean values.
      KALDI_WARN << "Mismatch in lattice and feature length";
      return false;
    }
    for (size_t i = 0; i < time_to_state[t].size(); i++) {
      int32 state = time_to_state[t][i].state_id;
      int32 arc_id = time_to_state[t][i].arc_id;
      int32 offset = time_to_state[t][i].trans_offset;

      if (arc_id == -1) { // Final state
        // Access the trans_id
        CompactLatticeWeight curr_clat_weight = clat->Final(state);
        int32 trans_id = curr_clat_weight.String()[offset];
        
        // Calculate likelihood
        BaseFloat log_like = decodable->LogLikelihood(t, trans_id);
        // update weight
        CompactLatticeWeight new_clat_weight = curr_clat_weight;
        LatticeWeight new_lat_weight = new_clat_weight.Weight();
        new_lat_weight.SetValue2(-log_like + curr_clat_weight.Weight().Value2());
        new_clat_weight.SetWeight(new_lat_weight);
        clat->SetFinal(state, new_clat_weight);
      } else {
        fst::MutableArcIterator<CompactLattice> aiter(clat, state);

        // Access the trans_id
        aiter.Seek(arc_id);
        CompactLatticeArc arc = aiter.Value();
        int32 trans_id = arc.weight.String()[offset];

        // Calculate likelihood
        BaseFloat log_like = decodable->LogLikelihood(t, trans_id);
        // update weight
        LatticeWeight new_weight = arc.weight.Weight();
        new_weight.SetValue2(-log_like + arc.weight.Weight().Value2());
        arc.weight.SetWeight(new_weight);
        aiter.SetValue(arc);
      }
    }
  }
  return true;
}

bool RescoreLattice(DecodableInterface *decodable,
                    Lattice *lat) {
  if (lat->NumStates() == 0) {
    KALDI_WARN << "Rescoring empty lattice";
    return false;
  }
  uint64 props = lat->Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted)) {
    if (fst::TopSort(lat) == false) {
      KALDI_WARN << "Cycles detected in lattice.";
      return false;
    }
  }
  std::vector<int32> state_times;
  int32 utt_len = kaldi::LatticeStateTimes(*lat, &state_times);
  
  std::vector<std::vector<int32> > time_to_state(utt_len );
  
  int32 num_states = lat->NumStates();
  KALDI_ASSERT(num_states == state_times.size());
  for (size_t state = 0; state < num_states; state++) {
    int32 t = state_times[state];
    KALDI_ASSERT(t >= 0 && t <= utt_len);
    if (t < utt_len)
      time_to_state[t].push_back(state);
  }


  for (int32 t = 0; t < utt_len; t++) {
    if ((t < utt_len - 1) == decodable->IsLastFrame(t)) {
      // this if-statement compares two boolean values.
      KALDI_WARN << "Mismatch in lattice and feature length";
      return false;
    }
    for (size_t i = 0; i < time_to_state[t].size(); i++) {
      int32 state = time_to_state[t][i];
      for (fst::MutableArcIterator<Lattice> aiter(lat, state);
           !aiter.Done(); aiter.Next()) {
        LatticeArc arc = aiter.Value();
        if (arc.ilabel != 0) {
          int32 trans_id = arc.ilabel; // Note: it doesn't necessarily
          // have to be a transition-id, just whatever the Decodable
          // object is expecting, but it's normally a transition-id.

          BaseFloat log_like = decodable->LogLikelihood(t, trans_id);
          arc.weight.SetValue2(-log_like + arc.weight.Value2());
          aiter.SetValue(arc);
        }
      }
    }
  }
  return true;
}



}  // namespace kaldi
