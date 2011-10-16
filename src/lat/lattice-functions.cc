// lat/lattice-functions.cc

// Copyright 2009-2011   Saarland University
// Author: Arnab Ghoshal

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
using std::pair;
#include <map>
using std::map;
#include <vector>
using std::vector;
#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
using std::tr1::unordered_map;



#include "lat/lattice-functions.h"
#include "hmm/transition-model.h"
#include "util/stl-utils.h"

namespace kaldi {

int32 LatticeStateTimes(const Lattice &lat, vector<int32> *times) {
  kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";
  KALDI_ASSERT(lat.Start() == 0);
  int32 num_states = lat.NumStates();
  times->clear();
  times->resize(num_states, -1);
  (*times)[0] = 0;
  for (int32 state = 0; state < num_states; ++state) {
    int32 cur_time = (*times)[state];
    for (fst::ArcIterator<Lattice> aiter(lat, state); !aiter.Done();
        aiter.Next()) {
      const LatticeArc& arc = aiter.Value();

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


// Helper functions for lattice forward-backward
static void ForwardNode(const Lattice &lat, int32 state,
                        vector<double> *state_alphas);
static void BackwardNode(const Lattice &lat, int32 state, int32 cur_time,
                         double tot_forward_prob,
                         const vector< vector<int32> > &active_states,
                         const vector<double> &state_alphas,
                         vector<double> *state_betas,
                         map<int32, double> *post);


BaseFloat LatticeForwardBackward(const Lattice &lat, Posterior *arc_post) {
  // Make sure the lattice is topologically sorted.
  kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";
  KALDI_ASSERT(lat.Start() == 0);

  int32 num_states = lat.NumStates();
  vector<int32> state_times;
  int32 max_time = LatticeStateTimes(lat, &state_times);  
  vector< vector<int32> > active_states(max_time + 1);
  // the +1 is needed since time is indexed from 0.
  
  vector<double> state_alphas(num_states, kLogZeroDouble),
      state_betas(num_states, kLogZeroDouble);
  state_alphas[0] = 0.0;
  double tot_forward_prob = kLogZeroDouble;

  // Forward pass
  for (int32 state = 0; state < num_states; ++state) {
    int32 cur_time = state_times[state];
    active_states[cur_time].push_back(state);

    if (lat.Final(state) != LatticeWeight::Zero()) {  // Check if final state.
      BaseFloat final_loglike = -(lat.Final(state).Value1() + lat.Final(state).Value2());
      state_betas[state] = final_loglike;
      tot_forward_prob = LogAdd(tot_forward_prob, state_alphas[state] + final_loglike);
    } else {
      ForwardNode(lat, state, &state_alphas);
    }
  }

  // Backward pass and collect posteriors
  vector< map<int32, double> > tmp_arc_post(max_time);
  for (int32 state = num_states -1; state > 0; --state) {
    int32 cur_time = state_times[state];
    BackwardNode(lat, state, cur_time, tot_forward_prob, active_states,
                 state_alphas, &state_betas, &tmp_arc_post[cur_time - 1]);
  }
  double tot_backward_prob = state_betas[0];  // Initial state id == 0
  if (!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-9)) {
    KALDI_ERR << "Total forward probability over lattice = " << tot_forward_prob
              << ", while total backward probability = " << tot_backward_prob;
  }

  // Output the computed posteriors
  arc_post->resize(max_time);
  for (int32 cur_time = 0; cur_time < max_time; ++cur_time) {
    map<int32, double>::const_iterator post_itr =
        tmp_arc_post[cur_time].begin();
    for (; post_itr != tmp_arc_post[cur_time].end(); ++post_itr) {
      (*arc_post)[cur_time].push_back(std::make_pair(post_itr->first,
                                                     post_itr->second));
    }
  }

  return tot_forward_prob;
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
  for (int32 state = 0; state < num_states; ++state) {
    int32 cur_time = state_times[state];
    for (fst::ArcIterator<Lattice> aiter(lat, state); !aiter.Done();
        aiter.Next()) {
      const LatticeArc& arc = aiter.Value();
      if (arc.ilabel != 0) {  // Non-epsilon arc
        int32 phone = trans.TransitionIdToPhone(arc.ilabel);
        if (!std::binary_search(silence_phones.begin(),
                                silence_phones.end(), phone))
          (*active_phones)[cur_time].insert(phone);
      }
    }  // end looping over arcs
  }  // end looping over states
}

bool LatticeBoost(const TransitionModel &trans,
                  const std::vector<std::set<int32> > &active_phones,
                  const std::vector<int32> &silence_phones,
                  BaseFloat b,
                  BaseFloat max_silence_error,
                  Lattice *lat) {

  kaldi::uint64 props = lat->Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted)) {
    if (fst::TopSort(lat) == false) {
      KALDI_WARN << "Cycles detected in lattice";
      return false;
    }
  }
  
  KALDI_ASSERT(IsSortedAndUniq(silence_phones));
  KALDI_ASSERT(max_silence_error >= 0.0 && max_silence_error <= 1.0);
  vector<int32> state_times;
  int32 num_states = lat->NumStates();
  LatticeStateTimes(*lat, &state_times);
  for (int32 state = 0; state < num_states; ++state) {
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
  for (int32 state = 0; state < num_states_hyp; ++state) {
    int32 cur_time = state_times_hyp[state];
    for (fst::ArcIterator<Lattice> aiter(hyp, state); !aiter.Done();
        aiter.Next()) {
      const LatticeArc& arc = aiter.Value();
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


// Helper functions for MPE lattice forward-backward
static void ForwardNodeMpe(const Lattice &lat, const TransitionModel &trans,
                           int32 state, int32 cur_time,
                           const vector< map<int32, char> > &arc_accs,
                           vector< pair<double, double> > *state_alphas);
static void BackwardNodeMpe(const Lattice &lat, const TransitionModel &trans,
                            int32 state, int32 cur_time,
                            pair<double, double> tot_forward_score,
                            const vector< vector<int32> > &active_states,
                            const vector< pair<double, double> > &state_alphas,
                            vector< pair<double, double> > *state_betas);

BaseFloat LatticeForwardBackwardMpe(const Lattice &lat,
                                    const TransitionModel &trans,
                                    const vector< map<int32, char> > &arc_accs,
                                    Posterior *arc_post) {
  // Make sure the lattice is topologically sorted.
  kaldi::uint64 props = lat.Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";
  KALDI_ASSERT(lat.Start() == 0);

  int32 num_states = lat.NumStates();
  vector<int32> state_times;
  int32 max_time = LatticeStateTimes(lat, &state_times);
  vector< vector<int32> > active_states(max_time + 1);
  // the +1 is needed since time is indexed from 0

  vector< pair<double, double> > state_alphas(num_states,
                                              std::make_pair(kLogZeroDouble, 0)),
      state_betas(num_states, std::make_pair(kLogZeroDouble, 0));
  state_alphas[0].first = 0.0;
  pair<double, double> tot_forward_score = std::make_pair(kLogZeroDouble, 0.0);

  // Forward pass
  for (int32 state = 0; state < num_states; ++state) {
    int32 cur_time = state_times[state];
    active_states[cur_time].push_back(state);

    if (lat.Final(state) != LatticeWeight::Zero()) {  // Check if final state.
      state_betas[state] = std::make_pair(state_alphas[state].first, 0.0);
      tot_forward_score.first = LogAdd(tot_forward_score.first,
                                       state_alphas[state].first);
      tot_forward_score.second += state_alphas[state].second;
    } else {
      ForwardNodeMpe(lat, trans, state, cur_time, arc_accs, &state_alphas);
    }
  }

  // Backward pass and collect posteriors
  vector< map<int32, double> > tmp_arc_post_pos(max_time),
      tmp_arc_post_neg(max_time);
  for (int32 state = num_states -1; state > 0; --state) {
    int32 cur_time = state_times[state];
    BackwardNodeMpe(lat, trans, state, cur_time, tot_forward_score,
                    active_states, state_alphas, &state_betas);
  }
//  double tot_backward_prob = state_betas[0];  // Initial state id == 0
//  if (!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-9)) {
//    KALDI_ERR << "Total forward probability over lattice = " << tot_forward_prob
//              << ", while total backward probability = " << tot_backward_prob;
//  }

//  // Output the computed posteriors
//  arc_post->resize(max_time);
//  for (int32 cur_time = 0; cur_time < max_time; ++cur_time) {
//    map<int32, double>::const_iterator post_itr =
//        tmp_arc_post[cur_time].begin();
//    for (; post_itr != tmp_arc_post[cur_time].end(); ++post_itr) {
//      (*arc_post)[cur_time].push_back(std::make_pair(post_itr->first,
//                                                     post_itr->second));
//    }
//  }

  return tot_forward_score.second;
}


// ----------------------- Helper function definitions -----------------------

// static
void ForwardNode(const Lattice &lat, int32 state,
                        vector<double> *state_alphas) {
  for (fst::ArcIterator<Lattice> aiter(lat, state); !aiter.Done();
      aiter.Next()) {
    const LatticeArc& arc = aiter.Value();
    double graph_score = arc.weight.Value1(),
        am_score = arc.weight.Value2(),
        arc_loglike = (*state_alphas)[state] - am_score - graph_score;
    (*state_alphas)[arc.nextstate] = LogAdd((*state_alphas)[arc.nextstate],
                                            arc_loglike);
  }
}

// static
void BackwardNode(const Lattice &lat, int32 state, int32 cur_time,
                         double tot_forward_prob,
                         const vector< vector<int32> > &active_states,
                         const vector<double> &state_alphas,
                         vector<double> *state_betas,
                         map<int32, double> *post) {
  // Epsilon arcs leading into the state
  for (vector<int32>::const_iterator st_it = active_states[cur_time].begin();
      st_it != active_states[cur_time].end(); ++st_it) {
    if ((*st_it) < state) {
      for (fst::ArcIterator<Lattice> aiter(lat, (*st_it)); !aiter.Done();
            aiter.Next()) {
        const LatticeArc& arc = aiter.Value();
        if (arc.nextstate == state) {
          KALDI_ASSERT(arc.ilabel == 0);
          double arc_loglike = (*state_betas)[state] - arc.weight.Value1()
              - arc.weight.Value2();
          (*state_betas)[(*st_it)] = LogAdd((*state_betas)[(*st_it)],
                                            arc_loglike);
        }
      }
    }
  }

  if (cur_time == 0) return;

  // Non-epsilon arcs leading into the state
  int32 prev_time = cur_time - 1;
  for (vector<int32>::const_iterator st_it = active_states[prev_time].begin();
      st_it != active_states[prev_time].end(); ++st_it) {
    for (fst::ArcIterator<Lattice> aiter(lat, (*st_it)); !aiter.Done();
        aiter.Next()) {
      const LatticeArc& arc = aiter.Value();
      if (arc.nextstate == state) {
        int32 key = arc.ilabel;
        KALDI_ASSERT(key != 0);
        double graph_score = arc.weight.Value1(),
            am_score = arc.weight.Value2(),
            arc_loglike = (*state_betas)[state] - graph_score - am_score;
        (*state_betas)[(*st_it)] = LogAdd((*state_betas)[(*st_it)],
                                          arc_loglike);
        double gamma = std::exp(state_alphas[(*st_it)] - graph_score - am_score
                                + (*state_betas)[state] - tot_forward_prob);
        if (post->find(key) == post->end())  // New label found at prev_time
          (*post)[key] = gamma;
        else  // Arc label already seen at this time
          (*post)[key] += gamma;
      }
    }
  }
}


// static
void ForwardNodeMpe(const Lattice &lat, const TransitionModel &tr,
                    int32 state, int32 cur_time,
                    const vector< map<int32, char> > &arc_accs,
                    vector< pair<double, double> > *state_alphas) {
  for (fst::ArcIterator<Lattice> aiter(lat, state); !aiter.Done();
      aiter.Next()) {
    const LatticeArc& arc = aiter.Value();
    double graph_score = arc.weight.Value1(),
        am_score = arc.weight.Value2(),
        arc_loglike = (*state_alphas)[state].first - am_score - graph_score;
    (*state_alphas)[arc.nextstate].first =
        LogAdd((*state_alphas)[arc.nextstate].first, arc_loglike);
    double frame_acc = 0.0;
    if (arc.ilabel != 0) {
      int32 phone = tr.TransitionIdToPhone(arc.ilabel);
      frame_acc = (arc_accs[cur_time].find(phone) == arc_accs[cur_time].end())?
          0.0 : 1.0;
    }
    (*state_alphas)[arc.nextstate].second += ((*state_alphas)[state].second
                                                + frame_acc);
  }
}


//static
void BackwardNodeMpe(const Lattice &lat, const TransitionModel &trans,
                     int32 state, int32 cur_time,
                     pair<double, double> tot_forward_score,
                     const vector< vector<int32> > &active_states,
                     const vector< pair<double, double> > &state_alphas,
                     vector< pair<double, double> > *state_betas) {

}


}  // namespace kaldi
