// latbin/lattice-prune.cc

// Copyright 2009-2011  Saarland University

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
#include <vector>
using std::vector;
#include <tr1/unordered_map>
using std::tr1::unordered_map;

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

int32 CalculateStateTimes(const Lattice &lat, vector<int32> *times) {
  uint64 props = lat.Properties(fst::kFstProperties, false);
  KALDI_ASSERT(props & fst::kTopSorted);

  int32 num_states = lat.NumStates(),
      max_time = 0;
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
          max_time = cur_time + 1;
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
  return max_time;
}


void ForwardNode(const Lattice &lat, int32 state,
                 vector<double> *state_alphas,
                 unordered_map<int32, double> *arc_alphas) {
  for (fst::ArcIterator<Lattice> aiter(lat, state); !aiter.Done();
      aiter.Next()) {
    const LatticeArc& arc = aiter.Value();
    double graph_score = arc.weight.Value1(),
        am_score = arc.weight.Value2(),
        arc_score = (*state_alphas)[state] + am_score + graph_score;

    if (arc.ilabel != 0) {  // Non-epsilon input label on arc
      int32 key = arc.ilabel;
      unordered_map<int32, double>::iterator find_iter = arc_alphas->find(key);
      if (find_iter == arc_alphas->end()) {  // New label found at this time
        (*arc_alphas)[key] = arc_score;
      } else {  // Arc label already seen at this time
        (*arc_alphas)[key] = LogAdd((*arc_alphas)[key], arc_score);
      }
    }
    (*state_alphas)[arc.nextstate] = LogAdd((*state_alphas)[arc.nextstate],
                                            arc_score);
  }
}

void BackwardNode(const Lattice &lat, int32 state, int32 cur_time,
                  const vector< vector<int32> > &active_states,
                  vector<double> *state_betas,
                  unordered_map<int32, double> *arc_betas) {
  // Epsilon arcs leading into the state
  for (vector<int32>::const_iterator st_it = active_states[cur_time].begin();
      st_it != active_states[cur_time].end(); ++st_it) {
    if ((*st_it) < state) {
      for (fst::ArcIterator<Lattice> aiter(lat, (*st_it)); !aiter.Done();
            aiter.Next()) {
        const LatticeArc& arc = aiter.Value();
        if (arc.nextstate == state) {
          KALDI_ASSERT(arc.ilabel == 0);
          double arc_score = (*state_betas)[state] + arc.weight.Value1()
              + arc.weight.Value2();
          (*state_betas)[(*st_it)] = LogAdd((*state_betas)[(*st_it)],
                                            arc_score);
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
        KALDI_ASSERT(arc.ilabel != 0);
        double arc_score = (*state_betas)[state] + arc.weight.Value1()
            + arc.weight.Value2();
        (*state_betas)[(*st_it)] = LogAdd((*state_betas)[(*st_it)],
                                          arc_score);
        int32 key = arc.ilabel;
        unordered_map<int32, double>::iterator find_iter = arc_betas->find(key);
        if (find_iter == arc_betas->end()) {  // New label found at prev_time
          (*arc_betas)[key] = (*state_betas)[state];
        } else {  // Arc label already seen at this time
          (*arc_betas)[key] = LogAdd((*arc_betas)[key], (*state_betas)[state]);
        }
      }
    }
  }
}

BaseFloat LatticeForwardBackward(const Lattice &lat, Posterior *arc_post) {
  uint64 props = lat.Properties(fst::kFstProperties, false);
  KALDI_ASSERT(props & fst::kTopSorted);

  KALDI_ASSERT(lat.Start() == 0);
  int32 num_states = lat.NumStates();
  vector<int32> state_times;
  int32 max_time = CalculateStateTimes(lat, &state_times);
  vector< vector<int32> > active_states(max_time);

  vector<double> state_alphas(num_states, kLogZeroDouble),
      state_betas(num_states, kLogZeroDouble);
  state_alphas[0] = 0.0;

  vector< unordered_map<int32, double> > arc_alphas, arc_betas;
  double lat_forward_prob = kLogZeroDouble;

  // Forward pass
  for (int32 state = 0; state < num_states; ++state) {
    int32 cur_time = state_times[state];
    active_states[cur_time].push_back(state);

    if (lat.Final(state) != LatticeWeight::Zero()) {  // Check if final state.
      state_betas[state] = 0.0;
      lat_forward_prob = LogAdd(lat_forward_prob, state_alphas[state]);
    } else {
      ForwardNode(lat, state, &state_alphas, &arc_alphas[cur_time]);
    }
  }

  // Backward pass
  for (int32 state = num_states -1; state > 0; --state) {
    int32 cur_time = state_times[state];
    BackwardNode(lat, state, cur_time, active_states, &state_betas,
                 &arc_betas[cur_time]);
  }
  double lat_backward_prob = state_betas[0];  // Initial state id == 0
  if (!ApproxEqual(lat_forward_prob, lat_backward_prob, 1e-6)) {
    KALDI_ERR << "Total forward probability over lattice = " << lat_forward_prob
              << ", while total backward probability = " << lat_backward_prob;
  }

  // Compute posteriors
  arc_post->resize(max_time);
  for (int32 cur_time = 0; cur_time < max_time; ++cur_time) {
    size_t num_arcs = arc_alphas[cur_time].size();
    KALDI_ASSERT(arc_betas[cur_time].size() == num_arcs);
    (*arc_post)[cur_time].resize(num_arcs);
    Vector<double> post(num_arcs);
    unordered_map<int32, double>::const_iterator alpha_itr =
        arc_alphas[cur_time].begin();
    for (int32 d = 0; alpha_itr != arc_alphas[cur_time].end(); ++alpha_itr, ++d) {
      int32 key = alpha_itr->first;
      unordered_map<int32, double>::const_iterator beta_itr =
          arc_betas[cur_time].find(key);
      if (beta_itr == arc_betas[cur_time].end()) {
        KALDI_ERR << "Forward probabilities found for transition ID " << key
                  << " but no backward probabilities found.";
      }
      double gamma = alpha_itr->second + beta_itr->second - lat_forward_prob;
      (*arc_post)[cur_time].push_back(std::make_pair(key, static_cast<BaseFloat>(gamma)));
      post(d) = gamma;
    }

    BaseFloat norm = post.LogSumExp();  // Normalizer for computing posteriors
    vector< pair<int32, BaseFloat> >::iterator post_itr =
        (*arc_post)[cur_time].begin();
    for (; post_itr != (*arc_post)[cur_time].end(); ++post_itr) {
      post_itr->second = std::exp(post_itr->second - norm);
    }
  }
  return lat_forward_prob;
}


}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Do forward-backward and collect posteriors over lattices.\n"
        "Usage: lattice-to-post [options] lattice-rspecifier posterior-wspecifier\n"
        " e.g.: lattice-to-post --acoustic-scale=0.1 ark:1.lats ark:1.post\n";
      
    BaseFloat acoustic_scale = 1.0;
    ParseOptions po(usage);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    if (acoustic_scale == 0.0)
      KALDI_EXIT << "Do not use a zero acoustic scale (cannot be inverted)";

    std::string lats_rspecifier = po.GetArg(1),
        posteriors_wspecifier = po.GetArg(2);

    // Read as regular lattice
    SequentialLatticeReader lattice_reader(lats_rspecifier);

    PosteriorWriter posterior_writer(posteriors_wspecifier);

    int32 n_done = 0; // there is no failure mode, barring a crash.

    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      Lattice lat = lattice_reader.Value();
      lattice_reader.FreeCurrent();
      if (acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);

      uint64 props = lat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        KALDI_WARN << "Supplied lattice not topologically sorted. Sorting it.";
        if (fst::TopSort(&lat) == false)
          KALDI_ERR << "Cycles detected in lattice.";
      }

      Posterior post;
      LatticeForwardBackward(lat, &post);
      posterior_writer.Write(key, post);
      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
