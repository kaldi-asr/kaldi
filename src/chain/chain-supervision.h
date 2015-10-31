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


#ifndef KALDI_CTC_CCTC_SUPERVISION_H_
#define KALDI_CTC_CCTC_SUPERVISION_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "lat/kaldi-lattice.h"
#include "fstext/deterministic-fst.h"
#include "ctc/cctc-transition-model.h"

namespace kaldi {
namespace ctc {

/*
  This file contains some declarations relating to the object we use to
  encode the supervision information for the 'chain' model.

  If we were training the model on whole utterances we could just use the
  reference phone sequence, but to make it easier to train on parts of
  utterances (and also for efficiency) we use the time-alignment information,
  extended by a user-specified boundary, to limit the range of frames
  that the phones can appear at.
*/


struct SupervisionOptions {
  std::string silence_phones;
  int32 optional_silence_cutoff;
  int32 left_tolerance;
  int32 right_tolerance;
  int32 frame_subsampling_factor;

  SupervisionOptions(): optional_silence_cutoff(15),
                           left_tolerance(5),
                           right_tolerance(10),
                           frame_subsampling_factor(1) { }

  void Register(OptionsItf *opts) {
    opts->Register("silence-phones", &silence_phones, "Colon or comma-separated "
                   "list of integer ids of silence phones (relates to "
                   "--optional-silence-cutoff)");
    opts->Register("optional-silence-cutoff", &optional_silence_cutoff,
                   "Duration in frames such that --silence-phones shorter than "
                   "this will be made optional.");
    opts->Register("left-tolerance", &left_tolerance, "The furthest to the "
                   "left (in the original frame rate) that a label can appear "
                   "relative to the phone start in the alignment.");
    // TODO: maybe we should make this relative to where it began in the
    // alignment.
    opts->Register("right-tolerance", &right_tolerance, "The furthest to the "
                   "right (in the original frame rate) that a label can appear "
                   "relative to the phone end in the alignment.");
    opts->Register("frame-subsampling-factor", &frame_subsampling_factor, "Used "
                   "if the frame-rate in CTC will be less than the frame-rate "
                   "of the original alignment.  Applied after left-tolerance "
                   "and right-tolerance are applied (so they are in terms of "
                   "the original num-frames.");
  }
};


// struct PhoneInstance is an instance of a phone (or blank) in a graph of
// phones used for the 'chain' supervision.  We require the time information from a
// reference alignment; we don't intend to enforce it exactly (we will allow a
// margin), but it's necessary to enable training on split-up parts of
// utterances.
struct PhoneInstance {
  // the phone (or epsilon): 0 < phone_or_epsilon <= num-phones.
  int32 phone_or_epsilon;
  // begin time of the phone (e.g. based on an alignment., but we later
  // make this earlier by left_tolerance).
  int32 begin_frame;
  // last-frame-plus-one in the phone (e.g. based on the alignment, but we later
  // increase it by right_tolerance).
  int32 end_frame;
  PhoneInstance(int32 p, int32 b, int32 e):
      phone_or_epsilon(p), begin_frame(b), end_frame(e) { }
  PhoneInstance() { }
};

struct PhoneInstanceHasher {
  size_t operator() (const PhoneInstance &p) const {
    int32 p1 = 673, p2 = 1747, p3 = 2377;
    return p.phone_or_epsilon * p1 + p2 * p.begin_frame + p3 * p.end_frame;
  }
};

// This is the form of the supervision information for CTC that it takes before
// we compile it to CctcSupervision.  Caution: several different stages of
// compilation use this object, don't mix them up.
//  The normal compilation sequence is:
// (AlignmentToProtoSupervision or PhoneLatticeToProtoSupervision)
// MakeSilencesOptional
// ModifyProtoSupervisionTimes
// AddEpsilonsToProtoSupervision [calls MakeProtoSupervisionConvertor]
// Then you would call MakeCctcSupervisionNoContext, and then
//  AddContextToCctcSupervision.

struct ProtoSupervision {
  // the total number of frames in the utterance (will equal the sum of the
  // phone durations if this is directly derived from a simple alignment.
  int32 num_frames;

  // The FST of phone instances (may or may not be a linear sequence).
  // It's an acceptor; the nonzero labels are indexes into phone_instances.
  fst::StdVectorFst fst;

  // The vector of phone instances; the zeroth element is not used.
  std::vector<PhoneInstance> phone_instances;

  // We have a Write but no Read function; this Write function is
  // only needed for debugging.
  void Write(std::ostream &os, bool binary) const;
};

/**  Creates an ProtoSupervision from a vector of phones and their durations,
     such as might be derived from a training-data alignment (see the function
     SplitToPhones()).  Note: this probably isn't the normal way you'll do it, it
     might be better to start with a phone-aligned lattice so you can capture
     the alternative pronunciations; see PhoneLatticeToProtoSupervision(). */
void AlignmentToProtoSupervision(const std::vector<int32> &phones,
                                 const std::vector<int32> &durations,
                                 ProtoSupervision *proto_supervision);

/**   Creates a ProtoSupervision object from a vector of (phone, duration) pairs
      (see the function SplitToPhones()).  This does the same jobs as the other
      AlignmentToProtoSupervision, from different input.
 */
void AlignmentToProtoSupervision(
    const std::vector<std::pair<int32, int32> > &phones_durs,
    ProtoSupervision *proto_supervision);


/** Creates a proto-supervision from a phone-aligned phone lattice (i.e. a
    lattice with phones as the labels, and with the alignment information
    correctly aligned so you can compute the correct times.  The normal path to
    create such a lattice would be to generate a lattice containing multiple
    pronunciations of the transcript by using steps/align_fmllr_lats.sh or a
    similar script, followed by lattice-align-phones
    --replace-output-symbols=true. */
void PhoneLatticeToProtoSupervision(const CompactLattice &lat,
                                    ProtoSupervision *proto_supervision);

/** Modifies the FST by making all silence phones that you specify that have
    duration shorter than opts.optional_silence_cutoff, optional.  It does this
    by, for each silence arc from state a to state b with duration shorter than
    opts.optional_silence_cutoff, adding a epsilon arc with the same time
    limitations (begin_frame and end_frame), between the same states.
*/
void MakeSilencesOptional(const SupervisionOptions &opts,
                          ProtoSupervision *proto_supervision);


/** Modifies the duration information (start_time and end_time) of each phone
    instance by the left_tolerance and right_tolerance (being careful not to go
    over the edges of the utterance) and then applies frame-rate subsampling by
    dividing each frame index in start_times and end_times , and num_frames, by
    frame_subsampling_factor.
    Requires that proto_supervision->num_frames >=
    options.frame_subsampling_factor.
*/
void ModifyProtoSupervisionTimes(const SupervisionOptions &options,
                                 ProtoSupervision *proto_supervision);


/**  This function adds the optional epsilons.  For each arc from state s1 -> s2
     with label p[t1:t2], where p is not epsilon, adds a self-loop on state s1
     with label epsilon[t1:t2] and a self-loop on state s2 with label
     epsilon[t1:t2].  Note: x[t1:t2] will later be expanded to symbol x being
     allowed to appear at any time value from t1 to t2-1.
*/
void AddEpsilonsToProtoSupervision(ProtoSupervision *proto_supervision);



/**
   This class wraps a ProtoSupervision to create a DeterministicOnDemandFst
   that we can compose with the FST in the ProtoSupervision (with this
   FST on the right) and then project on the output, to get an FST representing
   the supervision, that enforces limits on the times when phones can appear.

   Suppose the number of frames is T, then there will be T+1 states in
   this FST, numbered from 0 to T+1, where state 0 is initial and state
   T+1 is final.  Suppose our proto_supervision contains a label, say a[10:20],
   meaning the index of a PhoneInstance with phone_or_epsilon == "a", begin_frame
   == 10 and end_frame == 20.  For states t = 10 through 19 of convertor_fst
   [note: 20 is interpreted as last-frame-plus one], we will have an arc from
   state t to t+1 with a[10:20] on the input and a+1 on the output.  The output-side
   symbols of this FST are phones-plus-one so that we don't have to use 0 for
   epsilon, which in this context we are treating as a real symbol meaning 'one copy
   of the most recently seen phone'.  The input-side symbols of this FST are
   the same as the symbols of the acceptor proto_supervision.fst, i.e. they are
   nonzero indexes into 'proto_supervision.phone_instances'.
 */
class TimeEnforcerFst:
      public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  TimeEnforcerFst(const ProtoSupervision &proto_supervision):
      proto_supervision_(proto_supervision) { }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return 0; }

  // Note: in the CCTC framework we don't really model final-probs in any non-trivial
  // way, since the probabibility of the end of the phone sequence is one if we saw
  // the end of the acoustic sequence, and zero otherwise.
  virtual Weight Final(StateId s) {
    return (s == proto_supervision_.num_frames ? Weight::One() : Weight::Zero());
  }

  // The ilabel is an index into proto_supervision_.instances.  The olabel on
  // oarc will be a phone-or-epsilon plus one.  (phone-or-epsilon) plus one.  The
  // state-id is the time index 0 <= t <= num_frames.  All transitions are to
  // the next frame (but not all are allowed).  The interface of GetArc requires
  // ilabel to be nonzero (not epsilon).
  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  const ProtoSupervision &proto_supervision_;
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
  int32 num_frames;

  // the maximum possible value of the labels in 'fst' (which go from 1 to
  // output_label_dim).  This should equal the NumGraphLabels() in the
  // TransitionModel object.  (Note: we don't mean the actual largest value
  // encountered in this example).
  int32 label_dim;


  // This is an epsilon-free unweighted acceptor that is sorted in increasing
  // order of frame index (this implies it's topologically sorted but it's a
  // stronger condition).  The labels are 'output-labels', which are indexes
  // into the nnet output plus one.  Each successful path in 'fst' has exactly
  // 'num_frames * num_sequences' arcs on it (first 'num_frames' arcs for the
  // first sequence, ; then 'num_frames' arcs for the second sequence, and so
  // on).
  fst::StdVectorFst fst;

  Supervision(): weight(1.0), num_sequences(1), num_frames(-1),
                 start_frame(0), label_dim(-1) { }

  Supervision(const Supervision &other);

  bool operator == (const Supervision &other) const;

  // This function checks that this supervision object satifsies some
  // of the properties we expect of it, and calls KALDI_ERR if not.
  void Check(const ContextDepTopology &ctx_dep_topo) const;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);
};


/** This function creates a Supervision object with phones-or-epsilon-plus-one
    as the labels (you should then further process it using
    AddContextToSupervision).  You should give it a fully processed
    ProtoSupervision object (i.e. after calling AddEpsilonsToProtoSupervision).
    It sets the weight of the output supervision object to 1.0; change it
    manually if that's not what you want.  It creates a Supervision object
    with phone-or-epsilon-plus-one as the labels.  It sets supervision->label_dim
    to num_phones + 1.  It does not ensure the correct sorting of the FST states;
    that is done later in AddContextToSupervision.

    It returns true on success, and false on failure; the only failure mode is
    that it might return false on that would not be a bug, is when the FST is
    empty because there were too many phones for the number of frames.
*/
bool MakeSupervisionNoContext(
    const ProtoSupervision &proto_supervision,
    int32 num_phones,
    Supervision *supervision);


/**
   This function sorts the states of the fst argument in an ordering
   corresponding with a breadth-first search order starting from the
   start state.  This gives us the sorting on frame index for the
   FSTs that appear in class Supervision (it relies on them being
   epsilon-free).
   This function requires that the input FST be connected (i.e. all states
   reachable from the start state).  */
void SortBreadthFirstSearch(fst::StdVectorFst *fst);

/** This function modifies a Supervision object with phones-or-epsilon-plus-one
    as the labels, as created by MakeSupervisionNoContext, and converts it to
    have 'graph labels' as the labels.

    This is done by composing on the left with the FST in 'context_dep_topology',
    which will be obtained by calling the function ContextDepTopology::GetAsFst(),
    making sure there are no output-epsilons (there shouldn't be, but best to check),
    and then adding one to the input labels.
 */
void AddContextToSupervision(
    const fst::VectorFst<fst::StdArc> &context_dep_topology,
    Supervision *supervision);

// This class is used for splitting something of type Supervision into
// multiple pieces corresponding to different frame-ranges.
class SupervisionSplitter {
 public:
  SupervisionSplitter(const Supervision &supervision);
  // Extracts a frame range of the supervision into "supervision".
  void GetFrameRange(int32 begin_frame, int32 num_frames,
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
  // which ranges from 0 to supervision_.num_frames - 1.  This will be
  // monotonically increasing (note that supervision_.fst is topologically
  // sorted).
  std::vector<int32> frame_;
};

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
///                a vector with the dimension fst.NumStates()
/// @return  Returns the path length
int32 ComputeFstStateTimes(const fst::StdVectorFst &fst,
                           std::vector<int32> *state_times);


/// This function appends a list of supervision objects to create what will
/// usually be a single such object, but if the weights are not all the same it
/// will only append Supervision objects where successive ones have the same
/// weight, and if 'compactify' is true.  The normal use-case for this is when
/// you are combining neural-net examples for training; appending them like
/// this helps to simplify the decoding process.
/// This function
/// will crash if the values of label_dim in the inputs are not all the same, or
///
void AppendCctcSupervision(const std::vector<const CctcSupervision*> &input,
                           bool compactify,
                           std::vector<CctcSupervision> *output_supervision);


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


typedef TableWriter<KaldiObjectHolder<Supervision> > SupervisionWriter;
typedef SequentialTableReader<KaldiObjectHolder<Supervision> > SequentialSupervisionReader;
typedef RandomAccessTableReader<KaldiObjectHolder<Supervision> > RandomAccessSupervisionReader;

}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CTC_CCTC_SUPERVISION_H_
