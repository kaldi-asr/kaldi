// chain/chain-supervision-splitter.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)
//           2014-2015  Vimal Manohar
//                2017  Vimal Manohar

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

#ifndef KALDI_CHAIN_CHAIN_SUPERVISION_SPILTTER_H_
#define KALDI_CHAIN_CHAIN_SUPERVISION_SPILTTER_H_

#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace chain {

typedef fst::ArcTpl<LatticeWeight> LatticeArc;
typedef fst::VectorFst<LatticeArc> Lattice;

struct SupervisionLatticeSplitterOptions {
  BaseFloat acoustic_scale;
  bool normalize;
  bool add_phone_label_for_half_transition;

  SupervisionLatticeSplitterOptions(): 
    acoustic_scale(1.0), normalize(true),
    add_phone_label_for_half_transition(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Apply acoustic scale on the lattices before splitting.");
    opts->Register("normalize", &normalize,
                   "Normalize the initial and final scores added to split "
                   "lattices");
    opts->Register("add-phone-label-for-half-transition",
                   &add_phone_label_for_half_transition,
                   "Add a phone label to account for half phone transitions "
                   "in the split lattices");
  }
};

class SupervisionLatticeSplitter {
 public:
  SupervisionLatticeSplitter(const SupervisionLatticeSplitterOptions &opts,
                             const TransitionModel &trans_model,
                             const Lattice &lat);

  void GetFrameRange(int32 begin_frame, int32 frames_per_sequence,
                     Lattice *out_lat) const;

  // A structure used to store the forward and backward scores
  // and state times of a lattice
  struct LatticeInfo {
    // These values are stored in log.
    std::vector<double> alpha;
    std::vector<double> beta;
    std::vector<int32> state_times;

    void Check() const;
  };

 private:
  // Creates an output lattice covering frames begin_frame <= t < end_frame,
  // assuming that the corresponding state-range that we need to
  // include, begin_state <= s < end_state has been included.
  // (note: the output lattice will also have two special initial and final
  // states).
  void CreateRangeLattice(int32 begin_frame, int32 end_frame,
                          Lattice *out_lat) const;

  // Function to compute lattice scores for a lattice
  void ComputeLatticeScores();
  
  // Prepare lattice :
  // 1) Order states in breadth-first search order
  // 2) Compute states times, which must be a strictly non-decreasing vector
  // 3) Compute lattice alpha and beta scores
  void PrepareLattice();

  const SupervisionLatticeSplitterOptions &opts_;

  const TransitionModel &trans_model_;

  // LatticeInfo object for lattice.
  // This will be computed when PrepareLattice function is called.
  LatticeInfo lat_scores_;

  // Copy of the lattice. This is required because the lattice states
  // need to be ordered in breadth-first search order.
  Lattice lat_;
};

bool PhoneLatticeToSupervision(const fst::StdVectorFst &tolerance_fst,
                               const TransitionModel &trans_model,
                               const Lattice &lat,
                               chain::Supervision *supervision,
                               bool debug = false);

void MakeToleranceEnforcerFst(
    const SupervisionOptions &opts, const TransitionModel &trans_model,
    fst::StdVectorFst *fst);

}
}

#endif  // KALDI_CHAIN_CHAIN_SUPERVISION_SPLITTER_H_
