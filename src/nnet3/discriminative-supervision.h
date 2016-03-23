// nnet3/discriminative-supervision.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)
//           2014-2015  Vimal Manohar

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

#ifndef KALDI_NNET3_DISCRIMINATIVE_SUPERVISION_H
#define KALDI_NNET3_DISCRIMINATIVE_SUPERVISION_H

#include "util/table-types.h"
#include "hmm/posterior.h"
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {
namespace discriminative {

struct DiscriminativeSupervisionOptions {
  int32 frame_subsampling_factor;
  BaseFloat acoustic_scale;

  DiscriminativeSupervisionOptions(): frame_subsampling_factor(1), acoustic_scale(0.1) { }

  void Register(OptionsItf *opts) {
    opts->Register("frame-subsampling-factor", &frame_subsampling_factor, "Used "
                   "if the frame-rate for the model will be less than the "
                   "frame-rate of the original alignment.  Applied after "
                   "left-tolerance and right-tolerance are applied (so they are "
                   "in terms of the original num-frames.");
    opts->Register("acoustic-scale", &acoustic_scale,
                   "Scaling factor for acoustic likelihoods");
  }

  void Check() const;
};

struct SplitDiscriminativeSupervisionOptions {
  bool remove_output_symbols;
  bool collapse_transition_ids;
  bool remove_epsilons;
  bool determinize;
  bool minimize; // we'll push and minimize if this is true.
  DiscriminativeSupervisionOptions supervision_config;
  
  SplitDiscriminativeSupervisionOptions() :
    remove_output_symbols(false), collapse_transition_ids(false), 
    remove_epsilons(false), determinize(false),
    minimize(false) { }

  void Register(OptionsItf *opts) {
    opts->Register("collapse-transition-ids", &collapse_transition_ids,
                   "If true, modify the transition-ids on denominator lattice "
                   "so that on each frame, there is just one with any given "
                   "pdf-id. This allows us to determinize and minimize "
                   "more completely.");
    opts->Register("remove-output-symbols", &remove_output_symbols,
                   "Remove output symbols from lattice to convert it to an "
                   "acceptor and make it more determinizable");
    opts->Register("remove-epsilons", &remove_epsilons,
                   "Remove epsilons from the split lattices");
    opts->Register("determinize", &determinize, "If true, we determinize "
                   "lattices (as Lattice) after splitting and possibly minimize");
    opts->Register("minimize", &minimize, "If true, we push and "
                   "minimize lattices (as Lattice) after splitting");
    supervision_config.Register(opts);
  }
};

/*
  This file contains some declarations relating to the object we use to
  encode the supervision information for sequence training
*/

// struct DiscriminativeSupervision is the fully-processed information for
// a whole utterance or (after splitting) part of an utterance. 
struct DiscriminativeSupervision {
  // The weight we assign to this example;
  // this will typically be one, but we include it
  // for the sake of generality.  
  BaseFloat weight; 
  
  // num_sequences will be 1 if you create a DiscriminativeSupervision object from a single
  // lattice or alignment, but if you combine multiple DiscriminativeSupervision objects
  // the 'num_sequences' is the number of objects that were combined (the
  // lattices get appended).
  int32 num_sequences;

  // the number of frames in each sequence of appended objects.  num_frames *
  // num_sequences must equal the path length of any path in the lattices.
  // Technically this information is redundant with the lattices, but it's convenient
  // to have it separately.
  int32 frames_per_sequence;
  
  // The numerator alignment
  // Usually obtained by aligning the reference text with the seed neural
  // network model; can be the best path of generated lattice in the case of
  // semi-supervised training.
  std::vector<int32> num_ali;
  
  // Note: any acoustic
  // likelihoods in the lattices will be
  // recomputed at the time we train.
  
  // The denominator lattice.  
  Lattice den_lat; 
  
  DiscriminativeSupervision(): weight(1.0), num_sequences(1),
                               frames_per_sequence(-1) { }

  DiscriminativeSupervision(const DiscriminativeSupervision &other);


  // This function creates a supervision object from numerator alignment
  // and denominator lattice.  The supervision object is used for sequence
  // discriminative training.
  // Topologically sorts the lattice after copying to the supervision object.
  // Returns false when alignment or lattice is empty 
  bool Initialize(const std::vector<int32> &alignment,
                  const Lattice &lat,
                  BaseFloat weight);

  void Swap(DiscriminativeSupervision *other);

  bool operator == (const DiscriminativeSupervision &other) const;
  
  // This function checks that this supervision object satifsies some
  // of the properties we expect of it, and calls KALDI_ERR if not.
  void Check() const;
  
  inline int32 NumFrames() const { 
    return num_sequences * frames_per_sequence; 
  }

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};

// This class is used for splitting something of type
// DiscriminativeSupervision into
// multiple pieces corresponding to different frame-ranges.
class DiscriminativeSupervisionSplitter {
 public:
  typedef fst::ArcTpl<LatticeWeight> LatticeArc;
  typedef fst::VectorFst<LatticeArc> Lattice;
 
  DiscriminativeSupervisionSplitter(
      const SplitDiscriminativeSupervisionOptions &config,
      const TransitionModel &tmodel,
      const DiscriminativeSupervision &supervision);

  // A structure used to store the forward and backward scores 
  // and state times of a lattice
  struct LatticeInfo {
    // These values are stored in log. 
    std::vector<double> alpha;
    std::vector<double> beta;
    std::vector<int32> state_times;

    void Check() const;
  };
  
  // Extracts a frame range of the supervision into 'supervision'.  
  void GetFrameRange(int32 begin_frame, int32 frames_per_sequence,
                     bool normalize,
                     DiscriminativeSupervision *supervision) const;

  // Get the acoustic scaled denominator lattice out for debugging purposes
  inline const Lattice& DenLat() const { return den_lat_; }  

 private:

  // Creates an output lattice covering frames begin_frame <= t < end_frame,
  // assuming that the corresponding state-range that we need to
  // include, begin_state <= s < end_state has been included.
  // (note: the output lattice will also have two special initial and final
  // states).  
  // Also does post-processing (RmEpsilon, Determinize,
  // TopSort on the result).  See code for details.
  void CreateRangeLattice(const Lattice &in_lat,
                          const LatticeInfo &scores,
                          int32 begin_frame, int32 end_frame, bool normalize,
                          Lattice *out_lat) const;

  // Config options for splitting supervision object
  const SplitDiscriminativeSupervisionOptions &config_;

  // Transition model is used by the function
  // CollapseTransitionIds()
  const TransitionModel &tmodel_;
  
  // A reference to the supervision object that we will be splitting
  const DiscriminativeSupervision &supervision_;

  // LatticeInfo object for denominator lattice.
  // This will be computed when PrepareLattice function is called.
  LatticeInfo den_lat_scores_;

  // Copy of denominator lattice. This is required because the lattice states
  // need to be ordered in breadth-first search order.
  Lattice den_lat_;

  // Function to compute lattice scores for a lattice
  void ComputeLatticeScores(const Lattice &lat, LatticeInfo *scores) const;

  // Prepare lattice : 
  // 1) Order states in breadth-first search order
  // 2) Compute states times, which must be a strictly non-decreasing vector
  // 3) Compute lattice alpha and beta scores
  void PrepareLattice(Lattice *lat, LatticeInfo *scores) const;

  // Modifies the transition-ids on lat_ so that on each frame, there is just
  // one with any given pdf-id.  This allows us to determinize and minimize
  // more completely.
  void CollapseTransitionIds(const std::vector<int32> &state_times, 
                             Lattice *lat) const;

};

/// This function appends a list of supervision objects to create what will
/// usually be a single such object, but if the weights and num-frames are not
/// all the same it will only append Supervision objects where successive ones
/// have the same weight and num-frames, and if 'compactify' is true.  The
/// normal use-case for this is when you are combining neural-net examples for
/// training; appending them like this helps to simplify the training process.

void AppendSupervision(const std::vector<const DiscriminativeSupervision*> &input,
    bool compactify,
    std::vector<DiscriminativeSupervision> *output_supervision);

typedef TableWriter<KaldiObjectHolder<DiscriminativeSupervision> > DiscriminativeSupervisionWriter;
typedef SequentialTableReader<KaldiObjectHolder<DiscriminativeSupervision> > SequentialDiscriminativeSupervisionReader;
typedef RandomAccessTableReader<KaldiObjectHolder<DiscriminativeSupervision> > RandomAccessDiscriminativeSupervisionReader;

} // namespace discriminative
} // namespace kaldi

#endif // KALDI_NNET3_DISCRIMINATIVE_SUPERVISION_H
