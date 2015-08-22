// lat/cctc-supervision.h

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


#ifndef KALDI_CTC_CTC_SUPERVISION_H_
#define KALDI_CTC_CTC_SUPERVISION_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "lat/kaldi-lattice.h"
#include "cctc/cctc-transition-model.h"
#include "fstext/deterministic-fst.h"

namespace kaldi {

// CTC means Connectionist Temporal Classification, see the paper by Graves et
// al.  CCTC means context-dependent CTC, which
//

//
// What we are implementing here is some things relating to computation of the
// CTC objective function.  The normal information required to compute the
// objective function is the reference phone sequence, and the computation is a
// forward-backward computation.  We plan to make possible both forward-backward
// and Viterbi versions of this.  However, we want to be able to train on parts
// of utterances e.g. a few seconds, instead of whole utterances.  For this we
// need a sensible way of splitting utterances, and for this we need the time
// information (obtained from a baseline system or frmo another CTC system).
// The framework for using the time information is that we will limit, in the
// forward-backward or Viterbi, each instance of a phone to be within the reference
// interval of the phone, extended by a provided left and right margin.  (this is
// something that Andrew Senior et al. did to reduce latency, in a paper they
// published).  This leads to a natural way to split utterances.
//
// The labels that we keep track of in CTC will be phone labels, but during the
// processing of the CTC "decoding-graph" there will be a stage where we add a
// specified phonetic context (using a specifiable context-width N and
// central-position P, as for the decoding-graph creation code).  This makes it
// possible to have available during the CTC alignment, a specifiable amount of
// phonetic context.  (When we add blanks, the user will also be able to control
// how much context the blank symbol gets; this may be useful for an extension I
// have in mind).

// Using a special namespace for this CTC stuff.

namespace ctc {


struct CtcSupervisionOptions {
  std::string silence_phones;
  int32 optional_silence_cutoff;
  int32 left_tolerance;
  int32 right_tolerance;
  int32 frame_subsampling_factor;

  CtcSupervisionOptions(): optional_silence_cutoff(15),
                           context_width(1),
                           central_position(0),
                           left_tolerance(10),
                           right_tolerance(20),
                           frame_subsampling_factor(1) { }

  void Register(OptionsItf *opts) {
    opts->Register("silence-phones", &silence_phones, "Colon or comma-separated "
                   "list of integer ids of silence phones (relates to "
                   "--optional-silence-cutoff)");
    opts->Register("optional-silence-cutoff", &optional_silence_cutoff, "Duration "
                   "in frames such that --silence-phones shorter than this will be "
                   "made optional.");
    opts->Register("context-width", &context_width, "The width of the context "
                   "window stored with the CTC supervision information (e.g. 3 "
                   "for triphone");
    opts->Register("central-position", &central_position, "The position in "
                   "the context window of the phone we are modeling "
                   "(zero-based); e.g. for a triphone system you'd have "
                   "--context-width=3, --central-position=1.");
    opts->Register("left-tolerance", &left_tolerance, "The furthest to the "
                   "left (in the original frame rate) that a label can appear "
                   "relative to the phone start in the alignment.");
    opts->Register("right-tolerance", &right_tolerance, "The furthest to the "
                   "right (in the original frame rate) that a label can appear "
                   "relative to the phone end in the alignment.");
    opts->Register("frame-subsampling-factor", &frame_subsampling_factor, "Used "
                   "if the frame-rate in CTC will be less than the frame-rate "
                   "of the original alignment");
  }
}


// struct PhoneInstance is an instance of a phone in a graph of phones used for
// the CTC supervision.  We require the time information from a reference
// alignment; we don't intend to enforce it exactly (we will allow a margin),
// but it's necessary to enable CTC training on split-up parts of utterances.
struct PhoneInstance {
  // the phone: 0 < phone <= num-phones.
  int32 phone;
  // begin time of the phone (e.g. based on an alignment., but we later
  // make this earlier by left_tolerance).
  int32 begin_frame;
  // last-frame-plus-one in the phone (e.g. based on the alignment, but we later
  // increase it by right_tolerance).
  int32 end_frame;
};

struct PhoneInstanceHasher {
  size_t operator() (const PhoneInstance &pinst) const {
    int32 p1 = 673, p2 = 1747, p3 = 2377;
    return phone * p1 + p2 * start_time + p3 * end_time;
  }
};

// This is the form of the supervision information for CTC that it takes before
// we compile it.  Caution: several different stages of compilation use this object,
// don't mix them up.
//  The normal compilation sequence is:
// (AlignmentToProtoSupervision or PhoneLatticeToProtoSupervision)
// MakeSilencesOptional
// ModifyProtoSupervisionTimes
// AddProtoSupervisionContext
// AddBlanksToProtoSupervision [calls MakeProtoSupervisionConvertor]

struct CtcProtoSupervision {
  // the total number of frames in the utterance (will equal the sum of the
  // phone durations if this is directly derived from a simple alignment.
  int32 num_frames;
  
  // The FST of phone instances (may or may not be a linear sequence).
  // It's an acceptor; the nonzero labels are indexes into phone_instances.
  Fst<StdArc> fst;

  // The vector of phone instances; the zeroth element is not used.
  std::vector<PhoneInstance> phone_instances;
};

// Creates an CtcProtoSupervision from a vector of phones and their durations,
// such as might be derived from a training-data alignment.  The phone_instances
// will at this point be monophone, with no context.
// Note: this probably isn't the normal way you'll do it, it's better to
// start with a phone-aligned lattice so you can capture the alternative pronunciations;
// see PhoneLatticeToProtoSupervision().
void AlignmentToProtoSupervision(const std::vector<int32> &phones,
                                 const std::vector<int32> &durations,
                                 CtcProtoSupervision *proto_supervision);


// This function creates a proto-supervision from a phone-aligned lattice.  The
// normal path would be to generate a lattice containing multiple pronunciations
// of the transcript by using steps/align_fmllr_lats.sh or a similar script,
// followed by lattice-align-phones --replace-output-symbols=true.
void PhoneLatticeToProtoSupervision(const CompactLattice &lat,
                                    CtcProtoSupervision *proto_supervision);

// Modifies the FST by making all silence phones that you specify that have
// duration shorter than opts.optional_silence_cutoff, optional.
void MakeSilencesOptional(const CtcSupervisionOptions &opts,
                          CtcProtoSupervision *proto_supervision);


/** Modifies the duration information (start_time and end_time) of each phone
    instance by the left_tolerance and right_tolerance (being careful not to go
    over the edges of the utterance) and then applies frame-rate subsampling by
    dividing each frame index in start_times and end_times by
    frame_subsampling_factor (and likewise num_frames).  */
void ModifyProtoSupervisionTimes(const CtcSupervisionOptions &options
                                 CtcProtoSupervision *proto_supervion);


/**  This function adds the optional blanks.  This is done by composing on the
     left with a special FST a bit like 'H' in the regular speech-recognition
     pipeline.  See the docs below for MakeProtoSupervisionConvertor() for
     details on this.  */
void AddBlanksToProtoSupervision(const CtcSupervisionOptions &options,
                                 CtcProtoSupervision *proto_supervion);

/** Called from AddBlanksToProtoSupervision, this function creates the FST,
   similar in spirit to the H FST that it used in speech recognition recipes
   (search for hbka.pdf), that is used by AddBlanksToProtoSupervision to add the
   optional blanks and (if options.allow_symbol_repeats) the optional repeats of
   phones.

     What we want for each real phone symbol is,

      - Exactly one copy of that phone symbol. 

      - Respectively (before and after) each real phone symbol mentioned above,
        optional repeats of the blank symbol (symbol 0).  The time information
        for these instances of the blank symbol will be the same as the phone
        symbol we generated it from.
         
     This fst will have a state 0 which is both initial and final.  Let's
     suppose we have a phone-in-context a.  first an arc from state 0 to a new
     state (say, state 1), with <eps> on the input and a on the output.  Then a
     loop on state 1 with an instance of symbol 0 (blank) on the input (note it
     won't be a literal zero in the FST, it will be an index into the
     phone-instances vector); and epsilon on the output.  Then a transition to a
     new state (say, state 2) with the phone symbol a on the output.  Then a
     loop on state 2 with the same instance of symbol 0 (blank) on the input and
     <eps> on the output.  Then an <eps>/<eps> transition from state 2 back to
     state 0.

     Below, the "proto_supervision" argument is both an input and output,
     because aside from reading it we may need to add (blank) symbols to its
     'phone_instances' member.  We will compose with the 'convertor_fst' on the
     left, and then project on the input.  */
void MakeProtoSupervisionConvertor(
    const CtcSupervisionOptions &options,
    CtcProtoSupervision *proto_supervision,
    VectorFst<StdArc> *convertor_fst);

/**
   This function creates an FST that we will use to enforce the time constraints
   in the PhoneInstance arguments.  We'll compose with 'enforcer_fst' on the
   left and the phone-plus-blanks-graph FST (the output of
   AddBlanksToProtoSupervision) on the right.
   
   Suppose the number of frames is T, then there are T+1 states in this,
   numbered from 0 to T+1, where state 0 is initial and state T+1 is final.
   Suppose our proto_supervision contains a label, say a.  For states
   t = 10 through 19 of convertor_fst [note: 20 is interpreted as
   last-frame-plus one], we will have an arc from state t to t+1 with a+1 on
   the input and a[10:20] on the output.  The input symbols are phones-plus-one
   (this is so that we don't have to use 0 for the blank phone, phone 0);
  and the output indexes are indexes into proto_supervision.phone_insetances.
   
   This function is called by MakeCtcSupervision.
 */
void MakeProtoSupervisionTimeEnforcer(
    const CtcProtoSupervision &proto_supervision,
    VectorFst<StdArc> *enforcer_fst);

/**
   This class wraps a CctcTransitionModel as a DeterministicOnDemandFst.
   Note, DeterministicOnDemandFst does not inherit from class Fst.
   You compose with this to convert a graph labeled with phones-plus-one
   (the plus-one is so that the blank symbol is not mapped to epsilon),
   to a graph with cctc "graph-labels" on it.
 */
class CctcDeterministicOnDemandFst:
      public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  CctcDeterministicOnDemandFst(const CctcTransitionModel &trans_model);

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return start_state_; }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual Weight Final(StateId s);

  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  typedef unordered_map<std::vector<Label>,
                        StateId, VectorHasher<Label> > MapType;
  StateId start_state_;
  MapType wseq_to_state_;
  std::vector<std::vector<Label> > state_to_wseq_;
  const ConstArpaLm& lm_;
};




// This function creates a CtcSupervision from a fully processed
// CtcProtoSupervision object (after calling AddBlanksToProtoSupervision).
// Internally it calls MakeProtoSupevisionTimeEnforcer and composes.
// the 'opts' is only needed for the context_width and central_position.
// Normally the weight will be 1.0.
void MakeCtcSupervision(
    const CtcSupervisionOptions &opts,
    const CtcProtoSupervision &proto_supervision,
    BaseFloat weight,
    CtcSupervision *supervision);

// struct CtcSupervision is the fully-processed CTC supervision information for
// a whole utterance or (after splitting) part of an utterance.  It contains the
// time limits on phones encoded into the FST.
struct CtcSupervision {
  // The weight of this example (will usually be 1.0).
  BaseFloat weight;

  int32 num_frames;

  // This is an epsilon-free unweighted acceptor that is topologically sorted;
  // the labels are indexes into the "phones" vector (zero is reserved for
  // <eps>, and thus should not appear as a label because it's epsilon free).
  // Each successful path in 'fst' has exactly 'num_frames' arcs on it; this
  // topology is less compact but is useful to enforce that phones appear within
  // a specified temporal window.
  Fst<StdArc> fst;

  // context_width is the width of context window, i.e. the length of the
  // vectors in phones_in_context.
  int32 context_width;
  // 0 <= central_position < context_width is the position of the central phone
  // in the context window.  In the experimental mode, this will equal
  // context_width - 1 as it supports only left-context.
  int32 central_position;
  
  // vector of phones-in-context; index zero is not used (it's reserved for <eps>)
  // Each element is a phone in context.  However, blank symbols with context also
  // appear here (this is needed for an extension of CTC).
  std::vector<std::vector<int32> > phones_in_context;

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);  
};

// This class is used for splitting something of type CtcSupervision into
// multiple pieces corresponding to different frame-ranges.
class CtcSupervisionSplitter {
 public:
  CtcSupervisionSplitter(const CtcSupervision &supervision);
  // Extracts a frame range of the supervision into "supervision".
  void GetFrameRange(int32 start_frame, int32 num_frames,
                     CtcSupervision *supervision);
 private:
  const CtcSupervision &supervision_;
  // Indexed by the state-index of 'supervision_.fst', this is the frame-index,
  // which ranges from 0 to supervision_.num_frames - 1.  This will be
  // monotonically increasing (note that supervision_.fst is topologically
  // sorted).
  std::vector<int32> frame_;  
};


}  // namespace ctc
}  // namespace kaldi

#endif  // KALDI_CTC_CTC_SUPERVISION_H_
