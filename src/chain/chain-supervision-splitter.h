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
  bool add_partial_phone_label_left;
  bool add_partial_phone_label_right;
  bool add_partial_unk_label_left;
  bool add_partial_unk_label_right;
  int32 unk_phone;
  bool add_tolerance_to_lat;
  bool debug;

  SupervisionLatticeSplitterOptions(): 
    acoustic_scale(1.0), normalize(true),
    add_partial_phone_label_left(false),
    add_partial_phone_label_right(false),
    add_partial_unk_label_left(false),
    add_partial_unk_label_right(false),
    unk_phone(0),
    add_tolerance_to_lat(true), debug(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Apply acoustic scale on the lattices before splitting.");
    opts->Register("normalize", &normalize,
                   "Normalize the initial and final scores added to split "
                   "lattices");
    opts->Register("add-partial-phone-label-left",
                   &add_partial_phone_label_left,
                   "Add a phone label to account for partial phone transitions "
                   "in the left split lattices");
    opts->Register("add-partial-phone-label-right",
                   &add_partial_phone_label_right,
                   "Add a phone label to account for partial phone transitions "
                   "in the right split lattices");
    opts->Register("add-partial-unk-label-left",
                   &add_partial_unk_label_left,
                   "Add an UNK phone to account for partial phone transitions "
                   "in the left split lattices");
    opts->Register("add-partial-unk-label-right",
                   &add_partial_unk_label_right,
                   "Add an UNK phone to account for partial phone transitions "
                   "in the right split lattices");
    opts->Register("unk-phone", &unk_phone,
                   "UNK phone is added at half transition");
    opts->Register("add-tolerance-to-lat", &add_tolerance_to_lat,
                   "If this is true, then the tolerance is directly added "
                   "to the lattice by inserting or deleting self-loop "
                   "transitions");
    opts->Register("debug", &debug,
                   "Run some debug test codes");
  }
};

class SupervisionLatticeSplitter {
 public:
  SupervisionLatticeSplitter(const SupervisionLatticeSplitterOptions &opts,
                             const SupervisionOptions &sup_opts,
                             const TransitionModel &trans_model);

  void LoadLattice(const Lattice &lat);

  bool GetFrameRangeSupervision(int32 begin_frame, int32 frames_per_sequence,
                                chain::Supervision *supervision,
                                Lattice *lat = NULL) const;

  bool GetFrameRangeProtoSupervision(
      const ContextDependencyInterface &ctx_dep, 
      const TransitionModel &trans_model,
      int32 begin_frame, int32 num_frames,
      ProtoSupervision *proto_supervision) const;
  
  int32 NumFrames() const { return lat_scores_.num_frames; }

  // A structure used to store the forward and backward scores
  // and state times of a lattice
  struct LatticeInfo {
    // These values are stored in log.
    std::vector<double> alpha;
    std::vector<double> beta;
    std::vector<int32> state_times;
    std::vector<std::vector<std::pair<int32, BaseFloat> > > post;
    int32 num_frames;

    void Reset() {
      alpha.clear(); 
      beta.clear(); 
      state_times.clear(); 
      post.clear();
    }

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

  void PostProcessLattice(Lattice *out_lat) const;

  bool GetSupervision(const Lattice &out_lat, Supervision *supervision) const;

  // Function to compute lattice scores for a lattice
  void ComputeLatticeScores();
  
  // Prepare lattice :
  // 1) Order states in breadth-first search order
  // 2) Compute states times, which must be a strictly non-decreasing vector
  // 3) Compute lattice alpha and beta scores
  void PrepareLattice();
  
  const SupervisionOptions &sup_opts_;
  
  const SupervisionLatticeSplitterOptions &opts_;

  const TransitionModel &trans_model_;

  fst::StdVectorFst tolerance_fst_;
  void MakeToleranceEnforcerFst();

  const int32 incomplete_phone_;  // Equal to trans_model_.NumPhones() + 1

  // Used to remove "incomplete phone" label
  // Applicable only when opts_.add_partial_unk_label_left is true.
  fst::StdVectorFst filter_fst_;  
  void MakeFilterFst();

  // Copy of the lattice loaded using LoadLattice(). 
  // This is required because the lattice states
  // need to be ordered in breadth-first search order.
  Lattice lat_;

  // LatticeInfo object for lattice.
  // This will be computed when PrepareLattice function is called.
  LatticeInfo lat_scores_;
};
  
void GetToleranceEnforcerFst(const SupervisionOptions &opts, const TransitionModel &trans_model, fst::StdVectorFst *tolerance_fst);

bool PhoneLatticeToSupervision(const fst::StdVectorFst &tolerance_fst,
                               const TransitionModel &trans_model,
                               const Lattice &lat,
                               chain::Supervision *supervision,
                               bool debug = false);

void FixLattice(const fst::StdVectorFst &lattice_fixer_fst,
                const Lattice &lat, CompactLattice *clat);

void MakeLatticeFixerFst(const TransitionModel &trans_model,
                         fst::StdVectorFst *fst);

/*
class LatticeFixerFst:
      public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  LatticeFixerFst(const TransitionModel &trans_model):
      trans_model_(trans_model) { }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return 0; }

  virtual Weight Final(StateId s) {
    return Weight::One();
  }

  // The ilabel is a transition-id; the state is interpreted as a frame-index.
  // The olabel on oarc will be a pdf-id.  The state-id is the time index 0 <= t
  // <= num_frames.  All transitions are to the next frame (but not all are
  // allowed).  The interface of GetArc requires ilabel to be nonzero (not
  // epsilon).
  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  const TransitionModel &trans_model_;
};
*/

}
}

#endif  // KALDI_CHAIN_CHAIN_SUPERVISION_SPLITTER_H_
