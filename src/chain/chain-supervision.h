// chain/chain-supervision.h

// Copyright       2015  Johns Hopkins University (Author: Daniel Povey)


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


#ifndef KALDI_CHAIN_CHAIN_SUPERVISION_H_
#define KALDI_CHAIN_CHAIN_SUPERVISION_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "lat/kaldi-lattice.h"
#include "fstext/deterministic-fst.h"
#include "hmm/transition-model.h"

namespace kaldi {
namespace chain {

/*
  This file contains some declarations relating to the object we use to
  encode the supervision information for the 'chain' model.

  If we were training the model on whole utterances we could just use the
  reference phone sequence, but to make it easier to train on parts of
  utterances (and also for efficiency) we use the time-alignment information,
  extended by a user-specified margin, to limit the range of frames
  that the phones can appear at.
*/


struct SupervisionOptions {
  int32 left_tolerance;
  int32 right_tolerance;
  int32 frame_subsampling_factor;
  BaseFloat weight;
  BaseFloat lm_scale;
  bool convert_to_pdfs;

  SupervisionOptions(): left_tolerance(5),
                        right_tolerance(5),
                        frame_subsampling_factor(1),
                        weight(1.0),
                        lm_scale(0.0),
                        convert_to_pdfs(true) { }

  void Register(OptionsItf *opts) {
    opts->Register("left-tolerance", &left_tolerance, "Left tolerance for "
                   "shift in phone position relative to the alignment");
    opts->Register("right-tolerance", &right_tolerance, "Right tolerance for "
                   "shift in phone position relative to the alignment");
    opts->Register("frame-subsampling-factor", &frame_subsampling_factor, "Used "
                   "if the frame-rate for the chain model will be less than the "
                   "frame-rate of the original alignment.  Applied after "
                   "left-tolerance and right-tolerance are applied (so they are "
                   "in terms of the original num-frames.");
    opts->Register("weight", &weight,
                   "Use this to set the supervision weight for training. "
                   "This can be used to assign different weights to "
                   "different data sources.");
    opts->Register("lm-scale", &lm_scale, "The scale with which the graph/lm "
                   "weights from the phone lattice are included in the "
                   "supervision fst.");
    opts->Register("convert-to-pdfs", &convert_to_pdfs, "If true, convert "
                   "transition-ids to pdf-ids + 1 in supervision FSTs.");
  }
  void Check() const;
};


// This is the form that the supervision information for 'chain' models takes
// we compile it to Supervision.
//  The normal compilation sequence is:
// (AlignmentToProtoSupervision or PhoneLatticeToProtoSupervision)
// Then you would call ProtoSupervisionToSupervision.

struct ProtoSupervision {
  // a list of (sorted, unique) lists of phones that are allowed
  // on each frame.  number of frames is allowed_phones.size(), which
  // will equal the path length in 'fst'.
  std::vector<std::vector<int32> > allowed_phones;

  // The FST of phones; an epsilon-free acceptor.
  fst::StdVectorFst fst;

  bool operator == (const ProtoSupervision &other) const;

  // We have a Write but no Read function; this Write function is
  // only needed for debugging.
  void Write(std::ostream &os, bool binary) const;
};

/**  Creates a ProtoSupervision from a vector of phones and their durations,
     such as might be derived from a training-data alignment (see the function
     SplitToPhones()).  Note: this probably isn't the normal way you'll do it,
     it might be better to start with a phone-aligned lattice so you can capture
     the alternative pronunciations; see PhoneLatticeToProtoSupervision().
     Returns true on success (the only possible failure is that total duration <
     opts.subsampling_factor). */
bool AlignmentToProtoSupervision(const SupervisionOptions &opts,
                                 const std::vector<int32> &phones,
                                 const std::vector<int32> &durations,
                                 ProtoSupervision *proto_supervision);

/**   Creates a ProtoSupervision object from a vector of (phone, duration) pairs
      (see the function SplitToPhones()).  This does the same jobs as the other
      AlignmentToProtoSupervision, from different input.
 */
bool AlignmentToProtoSupervision(
    const SupervisionOptions &opts,
    const std::vector<std::pair<int32, int32> > &phones_durs,
    ProtoSupervision *proto_supervision);


/** Creates a proto-supervision from a phone-aligned phone lattice (i.e. a
    lattice with phones as the labels, and with the transition-ids aligned with
    the phones so you can compute the correct times.  The normal path to
    create such a lattice would be to generate a lattice containing multiple
    pronunciations of the transcript by using steps/align_fmllr_lats.sh or a
    similar script, followed by lattice-align-phones
    --replace-output-symbols=true.
    Returns true on success, and false on failure (the only failure modes are that
    the number of frames in the lattice is less than opts.frame_subsampling_factor,
    or there are epsilon phones in the lattice, or the final-probs have alignments
    on them.
*/
bool PhoneLatticeToProtoSupervision(const SupervisionOptions &opts,
                                    const CompactLattice &clat,
                                    ProtoSupervision *proto_supervision);


/** Modifies the duration information (start_time and end_time) of each phone
    instance by the left_tolerance and right_tolerance (being careful not to go
    over the edges of the utterance) and then applies frame-rate subsampling by
    dividing each frame index in start_times and end_times , and num_frames, by
    frame_subsampling_factor.  Requires that proto_supervision->num_frames >=
    options.frame_subsampling_factor.

*/
void ModifyProtoSupervisionTimes(const SupervisionOptions &options,
                                 ProtoSupervision *proto_supervision);



/**
   This class wraps the vector of allowed phones for each frame to create a
   DeterministicOnDemandFst that we can compose with the decoding-graph FST to
   limit the frames on which these phones are allowed to appear.  This FST also
   helps us convert the labels from transition-ids to (pdf-ids plus one), which
   is what we'll be using in the forward-backward (it avoids the need to
   keep the transition model around).

   Suppose the number of frames is T, then there will be T+1 states in
   this FST, numbered from 0 to T+1, where state 0 is initial and state
   T+1 is final.  A transition is only allowed from state t to state t+1
   with a particular transition-id as its ilabel, if the corresponding
   phone is listed in the 'allowed_phones' for that frame.  The olabels
   are pdf-ids plus one.
 */
class TimeEnforcerFst:
      public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  TimeEnforcerFst(const TransitionModel &trans_model,
                  bool convert_to_pdfs,
                  const std::vector<std::vector<int32> > &allowed_phones):
      trans_model_(trans_model),
      convert_to_pdfs_(convert_to_pdfs),
      allowed_phones_(allowed_phones) { }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return 0; }

  virtual Weight Final(StateId s) {
    return (s == allowed_phones_.size() ? Weight::One() : Weight::Zero());
  }

  // The ilabel is a transition-id; the state is interpreted as a frame-index.
  // The olabel on oarc will be a pdf-id.  The state-id is the time index 0 <= t
  // <= num_frames.  All transitions are to the next frame (but not all are
  // allowed).  The interface of GetArc requires ilabel to be nonzero (not
  // epsilon).
  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  const TransitionModel &trans_model_;
  // if convert_to_pdfs_ is true, this FST will map from transition-id (on the
  // input side) to pdf-id plus one (on the output side); if false, both sides'
  // labels will be transition-id.
  bool convert_to_pdfs_;
  const std::vector<std::vector<int32> > &allowed_phones_;
};


// struct Supervision is the fully-processed supervision information for
// a whole utterance or (after splitting) part of an utterance.  It contains the
// time limits on phones encoded into the FST.
struct Supervision {
  // The weight of this example (will usually be 1.0).
  BaseFloat weight;

  // num_sequences will be 1 if you create a Supervision object from a single
  // lattice or alignment, but if you combine multiple Supevision objects
  // the 'num_sequences' is the number of objects that were combined (the
  // FSTs get appended).
  int32 num_sequences;

  // the number of frames in each sequence of appended objects.  num_frames *
  // num_sequences must equal the path length of any path in the FST.
  // Technically this information is redundant with the FST, but it's convenient
  // to have it separately.
  int32 frames_per_sequence;

  // the maximum possible value of the labels in 'fst' (which go from 1 to
  // label_dim).  For fully-processed examples this will equal the NumPdfs() in the
  // TransitionModel object, but for newer-style "unconstrained" examples
  // that have been output by chain-get-supervision but not yet processed
  // by nnet3-chain-get-egs, it will be the NumTransitionIds() of the
  // TransitionModel object.
  int32 label_dim;

  // This is an epsilon-free unweighted acceptor that is sorted in increasing
  // order of frame index (this implies it's topologically sorted but it's a
  // stronger condition).  The labels will normally be pdf-ids plus one (to avoid epsilons,
  // since pdf-ids are zero-based), but for newer-style "unconstrained" examples
  // that have been output by chain-get-supervision but not yet processed
  // by nnet3-chain-get-egs, they will be transition-ids.
  // Each successful path in 'fst' has exactly 'frames_per_sequence *
  // num_sequences' arcs on it (first 'frames_per_sequence' arcs for the first
  // sequence; then 'frames_per_sequence' arcs for the second sequence, and so
  // on).
  fst::StdVectorFst fst;

  // 'e2e_fsts' may be set as an alternative to 'fst'.  These FSTs are used
  // when the numerator computation will be done with 'full forward_backward'
  // instead of constrained in time.  (The 'constrained in time' fsts are
  // how we described it in the original LF-MMI paper, where each phone can
  // only occur at the same time it occurred in the lattice, extended by
  // a tolerance).
  //
  // This 'e2e_fsts' is an array of FSTs, one per sequence, that are acceptors
  // with (pdf_id + 1) on the labels, just like 'fst', but which are cyclic FSTs.
  // Unlike with 'fst', it is not the case with 'e2e_fsts' that each arc
  // corresponds to a specific frame).
  //
  // There are two situations 'e2e_fsts' might be set.
  // The first is in 'end-to-end' training, where we train without a tree from
  // a flat start.  The function responsible for creating this object in that
  // case is TrainingGraphToSupervision(); to find out more about end-to-end
  // training, see chain-generic-numerator.h
  // The second situation is where we create the supervision from lattices,
  // and split them into chunks using the time marks in the lattice, but then
  // make a cyclic FST, and don't enforce the times on the lattice inside the
  // chunk.  [Code location TBD].
  std::vector<fst::StdVectorFst> e2e_fsts;


  // This member is only set to a nonempty value if we are creating 'unconstrained'
  // egs.  These are egs that are split into chunks using the lattice alignments,
  // but then within the chunks we remove the frame-level constraints on which
  // phones can appear when, and use the 'e2e_fsts' member.
  //
  // It is only required in order to accumulate the LDA stats using
  // `nnet3-chain-acc-lda-stats`, and it is not merged by nnet3-chain-merge-egs;
  // it will only be present for un-merged egs.
  std::vector<int32> alignment_pdfs;

  Supervision(): weight(1.0), num_sequences(1), frames_per_sequence(-1),
                 label_dim(-1) { }

  Supervision(const Supervision &other);

  void Swap(Supervision *other);

  bool operator == (const Supervision &other) const;

  // This function checks that this supervision object satifsies some
  // of the properties we expect of it, and calls KALDI_ERR if not.
  void Check(const TransitionModel &trans_model) const;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};


/** This function creates a Supervision object from a ProtoSupervision object.

    If convert_to_pdfs is true then the labels will be pdf-ids plus one and
    supervision->label_dim will be set to trans_model.NumPdfs(); otherwise, the
    labels will be transition-ids and supervision->label_dim will be
    trans_model.NumTransitionIds().

    It returns true on success, and false on failure; the only failure mode for
    which it might return false that would not be a bug, is when the FST is
    empty because there were too many phones for the number of frames.
*/
bool ProtoSupervisionToSupervision(
    const ContextDependencyInterface &ctx_dep,
    const TransitionModel &trans_model,
    const ProtoSupervision &proto_supervision,
    bool convert_to_pdfs,
    Supervision *supervision);

/** This function creates and initializes an end-to-end supervision object
    from a training FST (e.g. created using compile-train-graphs). It simply
    sets all the input and output labels to pdf_id+1 (i.e. converts the FST to
    an FSA) and stores the resulting FST in supervision->e2e_fsts[0].
    It returns true if there were no epsilon transitions, otherwise
    it would return false (the current implementation of forward-backward in
    chain-generic-numerator.cc does not support epsilon transitions).
    To find out more about end-to-end training, see chain-generic-numerator.h
 */
bool TrainingGraphToSupervisionE2e(
    const fst::StdVectorFst& training_graph,
    const TransitionModel& trans_model,
    int32 num_frames,
    Supervision *supervision);

/**
   This function sorts the states of the fst argument in an ordering
   corresponding with a breadth-first search order starting from the
   start state.  This gives us the sorting on frame index for the
   FSTs that appear in class Supervision (it relies on them being
   epsilon-free).
   This function requires that the input FST be connected (i.e. all states
   reachable from the start state).
   This function is called from ProtoSupervisionToSupervision().
*/
void SortBreadthFirstSearch(fst::StdVectorFst *fst);

// This class is used for splitting something of type Supervision into
// multiple pieces corresponding to different frame-ranges.
class SupervisionSplitter {
 public:
  SupervisionSplitter(const Supervision &supervision);

  // Extracts a frame range of the supervision into 'supervision'.  Note: the
  // supervision object should not be used for training before you do
  // 'AddWeightToSupervisionFst', which not only adds the weights from the
  // normalization graph (derived from the normalization FST), but also removes
  // epsilons and ensures the states are sorted on time.
  void GetFrameRange(int32 begin_frame, int32 frames_per_sequence,
                     Supervision *supervision) const;
 private:
  // Creates an output FST covering frames begin_frame <= t < end_frame,
  // assuming that the corresponding state-range that we need to
  // include, begin_state <= s < end_state has been included.
  // (note: the output FST will also have two special initial and final
  // states).  Does not do the post-processing (RmEpsilon, Determinize,
  // TopSort on the result).  See code for details.
  void CreateRangeFst(int32 begin_frame, int32 end_frame,
                      int32 begin_state, int32 end_state,
                      fst::StdVectorFst *fst) const;

  const Supervision &supervision_;
  // Indexed by the state-index of 'supervision_.fst', this is the frame-index,
  // which ranges from 0 to (supervision_.frames_per_sequence *
  // supervision_.num_sequences) - 1.  This will be monotonically increasing
  // (note that supervision_.fst is topologically sorted).
  std::vector<int32> frame_;
};


/// This function adds weights to the FST in the supervision object, by
/// composing with the 'normalization fst'.  It should be called directly after
/// GetFrameRange().  The 'normalization fst' is produced by the function
/// DenominatorGraph::GetNormalizationFst(); it's a slight modification of the
/// 'denominator fst'.  This function modifies the weights in the supervision
/// object- adding to each path, the weight that it gets in the normalization
/// fst, which is the same weight that it will get in the denominator
/// forward-backward computation.  This ensures that the (log) objective
/// function can never be positive (as the numerator graph will be a strict
/// subset of the denominator, with the same weights for the same paths).  This
/// function returns true on success, and false on the (hopefully) rare occasion
/// that the composition of the normalization fst with the supervision produced
/// an empty result (this shouldn't happen unless there were alignment errors in
/// the alignments used to train the phone language model leading to unseen
/// 3-grams that occur in the training sequences).
/// This function also removes epsilons and makes sure supervision->fst has the
/// required sorting of states.  Think of it as the final stage in preparation
/// of the supervision FST.
bool AddWeightToSupervisionFst(const fst::StdVectorFst &normalization_fst,
                               Supervision *supervision);

/// Assuming the 'fst' is epsilon-free, connected, and has the property that all
/// paths from the start-state are of the same length, output a vector
/// containing that length (from the start-state to the current state) to
/// 'state_times'.  The member 'fst' of struct Supervision has this property.
/// Returns the total number of frames.  This function is similar to
/// LatticeStateTimes() and CompactLatticeStateTimes() declared in
/// lat/lattice-functions.h, except that unlike LatticeStateTimes(), we don't
/// allow epsilons-- not because they are hard to handle but because in this
/// context we don't expect them.  This function also expects that the input fst
/// will have the property that the state times are in nondecreasing order (as
/// SortBreadthFirstSearch() will accomplish for FSTs satsifying the other
/// properties we mentioned).  This just happens to be something we enforce
/// while creating these FSTs.
///
/// @param fst[in] The input fst: should be epsilon-free; connected; nonempty;
///                should have the property that all paths to a given state (or
///                to a nonzero final-prob) should have the same number of arcs;
///                and its states should be sorted on this path length (e.g.
///                SortBreadthFirst will do this).
/// @param state_times[out]  The state times that we output; will be set to
///                a vector with the dimension fst.NumStates().
///
/// @return  Returns the path length
int32 ComputeFstStateTimes(const fst::StdVectorFst &fst,
                           std::vector<int32> *state_times);



/// This function merges a list of supervision objects, which must have the
/// same num-frames and label-dim.
void MergeSupervision(const std::vector<const Supervision*> &input,
                      Supervision *output_supervision);


/// This function helps you to pseudo-randomly split a sequence of length 'num_frames',
/// interpreted as frames 0 ... num_frames - 1, into pieces of length exactly
/// 'frames_per_range', to be used as examples for training.  Because frames_per_range
/// may not exactly divide 'num_frames', this function will leave either small gaps or
/// small overlaps in pseudo-random places.
/// The output 'range_starts' will be set to a list of the starts of ranges, the
/// output ranges are of the form
/// [ (*range_starts)[i] ... (*range_starts)[i] + frames_per_range - 1 ].
void SplitIntoRanges(int32 num_frames,
                     int32 frames_per_range,
                     std::vector<int32> *range_starts);


/// This utility function is not used directly in the 'chain' code.  It is used
/// to get weights for the derivatives, so that we don't doubly train on some
/// frames after splitting them up into overlapping ranges of frames.  The input
/// 'range_starts' will be obtained from 'SplitIntoRanges', but the
/// 'range_length', which is a length in frames, may be longer than the one
/// supplied to SplitIntoRanges, due the 'overlap'.  (see the calling code...
/// if we want overlapping ranges, we get it by 'faking' the input to
/// SplitIntoRanges).
///
/// The output vector 'weights' will be given the same dimension as
/// 'range_starts'.  By default the output weights in '*weights' will be vectors
/// of all ones, of length equal to 'range_length', and '(*weights)[i]' represents
/// the weights given to frames numbered
///   t = range_starts[i] ... range_starts[i] + range_length - 1.
/// If these ranges for two successive 'i' values overlap, then we
/// reduce the weights to ensure that no 't' value gets a total weight
/// greater than 1.  We do this by dividing the overlapped region
/// into three approximately equal parts, and giving the left part
/// to the left range; the right part to the right range; and
/// in between, interpolating linearly.
void GetWeightsForRanges(int32 range_length,
                         const std::vector<int32> &range_starts,
                         std::vector<Vector<BaseFloat> > *weights);


/// This function converts a 'Supervision' object that has a non-cyclic FST
/// as its 'fst' member, and converts it to one that has a cyclic FST in
/// its e2e_fsts[0], and has 'alignment_pdfs' set to a random path through
/// the original 'fst' (this used only in the binary nnet3-chain-acc-lda-stats).
/// This can be used to train without any constraints on the alignment of phones
/// internal to chunks, while still imposing constraints at chunk boundaries.
/// It returns true on success, and false if some kind of error happened
/// (this is not expected).
bool ConvertSupervisionToUnconstrained(
    const TransitionModel &trans_mdl,
    Supervision *supervision);


typedef TableWriter<KaldiObjectHolder<Supervision> > SupervisionWriter;
typedef SequentialTableReader<KaldiObjectHolder<Supervision> > SequentialSupervisionReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Supervision> > RandomAccessSupervisionReader;

}  // namespace chain
}  // namespace kaldi

#endif  // KALDI_CHAIN_CHAIN_SUPERVISION_H_
