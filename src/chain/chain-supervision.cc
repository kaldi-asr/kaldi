// chain/chain-supervision.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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

#include "chain/chain-supervision.h"
#include "lat/lattice-functions.h"
#include "util/text-utils.h"
#include "hmm/hmm-utils.h"
#include <numeric>

namespace kaldi {
namespace chain {

const int kSupervisionMaxStates = 200000;  // we can later make this
                                           // configurable if needed.

// attempts determinization (with limited max-states) and minimization;
// returns true on success
bool TryDeterminizeMinimize(int32 supervision_max_states,
                            fst::StdVectorFst *supervision_fst) {
  if (supervision_fst->NumStates() >= supervision_max_states) {
    KALDI_WARN << "Not attempting determinization as number of states "
               << "is too large " << supervision_fst->NumStates();
    return false;
  }
  fst::DeterminizeOptions<fst::StdArc> opts;
  opts.state_threshold = supervision_max_states;
  fst::StdVectorFst fst_copy = *supervision_fst;
  fst::Determinize(fst_copy, supervision_fst, opts);
  // the - 1 here is just because I'm not sure if it stops just before the
  // threshold.
  if (supervision_fst->NumStates() >= opts.state_threshold - 1) {
    KALDI_WARN << "Determinization stopped early after reaching "
               << supervision_fst->NumStates() << " states.  Likely "
               << "this utterance has a very strange transcription.";
    return false;
  }
  fst::Minimize(supervision_fst);
  return true;
}

void ProtoSupervision::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ProtoSupervision>");
  if (!binary) os << "\n";
  int32 num_frames = allowed_phones.size();
  WriteToken(os, binary, "<NumFrames>");
  WriteBasicType(os, binary, num_frames);
  if (!binary) os << "\n";
  WriteToken(os, binary, "<AllowedPhones>");
  if (!binary) os << "\n";
  for (int32 i = 0; i < num_frames; i++)
    WriteIntegerVector(os, binary, allowed_phones[i]);
  if (!binary) os << "\n";
  WriteFstKaldi(os, binary, fst);
  WriteToken(os, binary, "</ProtoSupervision>");
  if (!binary) os << "\n";
}

void SupervisionOptions::Check() const {
  KALDI_ASSERT(left_tolerance >= 0 && right_tolerance >= 0 &&
               frame_subsampling_factor > 0 &&
               left_tolerance + right_tolerance >= frame_subsampling_factor);
}

bool AlignmentToProtoSupervision(const SupervisionOptions &opts,
                                 const std::vector<int32> &phones,
                                 const std::vector<int32> &durations,
                                 ProtoSupervision *proto_supervision) {
  opts.Check();
  KALDI_ASSERT(phones.size() > 0 && phones.size() == durations.size());
  std::vector<int32> labels(phones.size());
  int32 num_frames = std::accumulate(durations.begin(), durations.end(), 0),
      factor = opts.frame_subsampling_factor,
      num_frames_subsampled = (num_frames + factor - 1) / factor;
  proto_supervision->allowed_phones.clear();
  proto_supervision->allowed_phones.resize(num_frames_subsampled);
  proto_supervision->fst.DeleteStates();
  if (num_frames_subsampled == 0)
    return false;

  int32 current_frame = 0, num_phones = phones.size();
  for (int32 i = 0; i < num_phones; i++) {
    int32 phone = phones[i], duration = durations[i];
    KALDI_ASSERT(phone > 0 && duration > 0);
    int32 t_start = std::max<int32>(0, (current_frame - opts.left_tolerance)),
            t_end = std::min<int32>(num_frames,
                                    (current_frame + duration + opts.right_tolerance)),
       t_start_subsampled = (t_start + factor - 1) / factor,
       t_end_subsampled = (t_end + factor - 1) / factor;

    // note: if opts.Check() passed, the following assert should pass too.
    KALDI_ASSERT(t_end_subsampled > t_start_subsampled &&
                 t_end_subsampled <= num_frames_subsampled);
    for (int32 t_subsampled = t_start_subsampled;
         t_subsampled < t_end_subsampled; t_subsampled++)
      proto_supervision->allowed_phones[t_subsampled].push_back(phone);
    current_frame += duration;
  }
  KALDI_ASSERT(current_frame == num_frames);
  for (int32 t_subsampled = 0; t_subsampled < num_frames_subsampled;
       t_subsampled++) {
    KALDI_ASSERT(!proto_supervision->allowed_phones[t_subsampled].empty());
    SortAndUniq(&(proto_supervision->allowed_phones[t_subsampled]));
  }
  fst::MakeLinearAcceptor(phones, &(proto_supervision->fst));
  return true;
}

bool AlignmentToProtoSupervision(
    const SupervisionOptions &opts,
    const std::vector<std::pair<int32, int32> > &phones_durations,
    ProtoSupervision *proto_supervision) {
  KALDI_ASSERT(phones_durations.size() > 0);
  std::vector<int32> phones(phones_durations.size()),
      durations(phones_durations.size());
  for (size_t size = phones_durations.size(), i = 0; i < size; i++) {
    phones[i] = phones_durations[i].first;
    durations[i] = phones_durations[i].second;
  }
  return AlignmentToProtoSupervision(opts, phones, durations,
                                     proto_supervision);
}


bool ProtoSupervision::operator == (const ProtoSupervision &other) const {
  return (allowed_phones == other.allowed_phones &&
          fst::Equal(fst, other.fst));
}

bool PhoneLatticeToProtoSupervision(const SupervisionOptions &opts,
                                    const CompactLattice &lat,
                                    ProtoSupervision *proto_supervision) {
  opts.Check();
  if (lat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice provided";
    return false;
  }
  int32 num_states = lat.NumStates();
  proto_supervision->fst.DeleteStates();
  proto_supervision->fst.ReserveStates(num_states);
  std::vector<int32> state_times;
  int32 num_frames = CompactLatticeStateTimes(lat, &state_times),
      factor = opts.frame_subsampling_factor,
    num_frames_subsampled = (num_frames + factor - 1) / factor;
  for (int32 state = 0; state < num_states; state++)
    proto_supervision->fst.AddState();
  proto_supervision->fst.SetStart(lat.Start());

  proto_supervision->allowed_phones.clear();
  proto_supervision->allowed_phones.resize(num_frames_subsampled);

  for (int32 state = 0; state < num_states; state++) {
    int32 state_time = state_times[state];
    for (fst::ArcIterator<CompactLattice> aiter(lat, state); !aiter.Done();
         aiter.Next()) {
      const CompactLatticeArc &lat_arc = aiter.Value();
      int32 next_state_time = state_time + lat_arc.weight.String().size();
      int32 phone = lat_arc.ilabel;  // It's an acceptor so ilabel == ollabel.
      if (phone == 0) {
        KALDI_WARN << "CompactLattice has epsilon arc.  Unexpected.";
        return false;
      }
      proto_supervision->fst.AddArc(state,
                                    fst::StdArc(phone, phone,
                                                fst::TropicalWeight::One(),
                                                lat_arc.nextstate));
      int32 t_begin = std::max<int32>(0, (state_time - opts.left_tolerance)),
              t_end = std::min<int32>(num_frames,
                                      (next_state_time + opts.right_tolerance)),
 t_begin_subsampled = (t_begin + factor - 1)/ factor,
   t_end_subsampled = (t_end + factor - 1)/ factor;
    for (int32 t_subsampled = t_begin_subsampled;
         t_subsampled < t_end_subsampled; t_subsampled++)
      proto_supervision->allowed_phones[t_subsampled].push_back(phone);
    }
    if (lat.Final(state) != CompactLatticeWeight::Zero()) {
      proto_supervision->fst.SetFinal(state, fst::TropicalWeight::One());
      if (state_times[state] != num_frames) {
        KALDI_WARN << "Time of final state " << state << " in lattice is "
                   << "not equal to number of frames " << num_frames
                   << ".  Are you sure the lattice is phone-aligned? "
                   << "Rejecting it.";
        return false;
      }
    }
  }
  for (int32 t_subsampled = 0; t_subsampled < num_frames_subsampled;
       t_subsampled++) {
    KALDI_ASSERT(!proto_supervision->allowed_phones[t_subsampled].empty());
    SortAndUniq(&(proto_supervision->allowed_phones[t_subsampled]));
  }
  return true;
}


bool TimeEnforcerFst::GetArc(StateId s, Label ilabel, fst::StdArc* oarc) {
  // the following call will do the range-check on 'ilabel'.
  int32 phone = trans_model_.TransitionIdToPhone(ilabel);
  KALDI_ASSERT(static_cast<size_t>(s) <= allowed_phones_.size());
  if (static_cast<size_t>(s) == allowed_phones_.size()) {
    // No arcs come from the final state.a
    return false;
  }
  if (std::binary_search(allowed_phones_[s].begin(),
                         allowed_phones_[s].end(), phone)) {
    // the olabel will be a pdf-id plus one, not a transition-id.
    int32 pdf_id = trans_model_.TransitionIdToPdf(ilabel);
    oarc->ilabel = ilabel;
    oarc->olabel = pdf_id + 1;
    oarc->weight = fst::TropicalWeight::One();
    oarc->nextstate = s + 1;
    return true;
  } else {
    return false;
  }
}


bool ProtoSupervisionToSupervision(
    const ContextDependencyInterface &ctx_dep,
    const TransitionModel &trans_model,
    const ProtoSupervision &proto_supervision,
    Supervision *supervision) {
  using fst::VectorFst;
  using fst::StdArc;
  VectorFst<StdArc> phone_fst(proto_supervision.fst);
  int32 subsequential_symbol = trans_model.GetPhones().back() + 1;
  if (ctx_dep.CentralPosition() != ctx_dep.ContextWidth() - 1) {
    // note: this function only adds the subseq symbol to the input of what was
    // previously an acceptor, so we project, i.e. copy the ilabels to the
    // olabels
    AddSubsequentialLoop(subsequential_symbol, &phone_fst);
    fst::Project(&phone_fst, fst::PROJECT_INPUT);
  }
  std::vector<int32> disambig_syms;  // empty list of diambiguation symbols.
  fst::ContextFst<StdArc> cfst(subsequential_symbol, trans_model.GetPhones(),
                               disambig_syms, ctx_dep.ContextWidth(),
                               ctx_dep.CentralPosition());
  VectorFst<StdArc> context_dep_fst;
  fst::ComposeContextFst(cfst, phone_fst, &context_dep_fst);
  // at this point, context_dep_fst will have indexes into 'ilabels' as its
  // input symbol (representing context-dependent phones), and phones on its
  // output.  We don't need the phones, so we'll project.
  fst::Project(&context_dep_fst, fst::PROJECT_INPUT);

  std::vector<int32> disambig_syms_h; // disambiguation symbols on input side
                                      // of H -- will be empty.

  HTransducerConfig h_cfg;

  // We don't want to add any transition probabilities as they will be added
  // when we compose with the denominator graph.
  h_cfg.transition_scale = 0.0;
  h_cfg.push_weights = false;  // there's nothing to push.


  VectorFst<StdArc> *h_fst = GetHTransducer(cfst.ILabelInfo(),
                                            ctx_dep,
                                            trans_model,
                                            h_cfg,
                                            &disambig_syms_h);
  KALDI_ASSERT(disambig_syms_h.empty());

  VectorFst<StdArc> transition_id_fst;
  TableCompose(*h_fst, context_dep_fst, &transition_id_fst);
  delete h_fst;

  // We don't want to add any transition probabilities as they will be added
  // when we compose with the denominator graph.
  BaseFloat self_loop_scale = 0.0;

  bool reorder = true;  // more efficient in general; won't affect results.
  // add self-loops to the FST with transition-ids as its labels.
  AddSelfLoops(trans_model, disambig_syms_h, self_loop_scale, reorder,
               &transition_id_fst);

  // at this point transition_id_fst will have transition-ids as its ilabels and
  // context-dependent phones (indexes into ILabelInfo()) as its olabels.
  // Discard the context-dependent phones by projecting on the input, keeping
  // only the transition-ids.
  fst::Project(&transition_id_fst, fst::PROJECT_INPUT);
  if (transition_id_fst.Properties(fst::kIEpsilons, true) != 0) {
    // remove epsilons, if there are any.
    fst::RmEpsilon(&transition_id_fst);
  }
  KALDI_ASSERT(transition_id_fst.NumStates() > 0);

  // The last step is to enforce that phones can only appear on the frames they
  // are 'allowed' to appear on.  This will also convert the FST to have pdf-ids
  // plus one as the labels
  TimeEnforcerFst enforcer_fst(trans_model, proto_supervision.allowed_phones);
  ComposeDeterministicOnDemand(transition_id_fst,
                               &enforcer_fst,
                               &(supervision->fst));
  fst::Connect(&(supervision->fst));
  // at this point supervision->fst will have pdf-ids plus one as the olabels,
  // but still transition-ids as the ilabels.  Copy olabels to ilabels.
  fst::Project(&(supervision->fst), fst::PROJECT_OUTPUT);

  KALDI_ASSERT(supervision->fst.Properties(fst::kIEpsilons, true) == 0);
  if (supervision->fst.NumStates() == 0) {
    KALDI_WARN << "Supervision FST is empty (too many phones for too few "
               << "frames?)";
    // possibly there were too many phones for too few frames.
    return false;
  }

  supervision->weight = 1.0;
  supervision->num_sequences = 1;
  supervision->frames_per_sequence = proto_supervision.allowed_phones.size();
  supervision->label_dim = trans_model.NumPdfs();
  SortBreadthFirstSearch(&(supervision->fst));
  return true;
}



SupervisionSplitter::SupervisionSplitter(
    const Supervision &supervision):
    supervision_(supervision),
    frame_(supervision_.fst.NumStates(), -1) {
  const fst::StdVectorFst &fst(supervision_.fst);
  // The fst in struct Supervision is supposed to be epsilon-free and
  // topologically sorted; this function relies on those properties to
  // set up the frame_ vector (which maps each state in the
  // FST to a frame-index 0 <= t < num_frames), and it checks them.
  if (supervision_.num_sequences != 1) {
    KALDI_WARN << "Splitting already-reattached sequence (only expected in "
               << "testing code)";
  }
  int32 num_states = fst.NumStates(),
      num_frames = supervision_.frames_per_sequence * supervision_.num_sequences;
  KALDI_ASSERT(num_states > 0);
  int32 start_state = fst.Start();
  // FST should be top-sorted and connected, so start-state must be 0.
  KALDI_ASSERT(start_state == 0 && "Expecting start-state to be 0");
  frame_[start_state] = 0;
  for (int32 state = 0; state < num_states; state++) {
    int32 cur_frame = frame_[state];
    if (cur_frame == -1) {
      // If this happens it means the Supervision does not have the required
      // properties, e.g. being top-sorted and connected.
      KALDI_ERR << "Error computing frame indexes for Supervision";
    }
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      // The FST is supposed to be an epsilon-free acceptor.
      KALDI_ASSERT(arc.ilabel == arc.olabel && arc.ilabel > 0);
      int32 nextstate = arc.nextstate;
      KALDI_ASSERT(nextstate >= 0 && nextstate < num_states);
      // all arcs go from some t to t + 1.
      int32 &next_frame = frame_[nextstate];
      if (next_frame == -1)
        next_frame = cur_frame + 1;
      else
        KALDI_ASSERT(next_frame == cur_frame + 1);
    }
  }
  // The following assert checks that the number of frames in the FST
  // matches the num_frames stored in the supervision object; it also relies
  // on the topological sorting and connectedness of the FST.
  KALDI_ASSERT(frame_.back() == num_frames);
  std::vector<int32>::iterator iter = frame_.begin(),
      end = iter + (frame_.size() - 1);
  // check that the frame-indexes of states are monotonically non-decreasing, as
  // they should be based on the top-sorting.  We rely on this property to
  // compute the frame ranges while splitting.
  while (iter != end) {
    int32 cur_t = *iter;
    ++iter;
    int32 next_t = *iter;
    KALDI_ASSERT(next_t >= cur_t);
  }
}

void SupervisionSplitter::GetFrameRange(int32 begin_frame, int32 num_frames,
                                        Supervision *out_supervision) const {
  int32 end_frame = begin_frame + num_frames;
  // Note: end_frame is not included in the range of frames that the
  // output supervision object covers; it's one past the end.
  KALDI_ASSERT(num_frames > 0 && begin_frame >= 0 &&
               begin_frame + num_frames <=
               supervision_.num_sequences * supervision_.frames_per_sequence);
  std::vector<int32>::const_iterator begin_iter =
      std::lower_bound(frame_.begin(), frame_.end(), begin_frame),
      end_iter = std::lower_bound(begin_iter, frame_.end(), end_frame);
  KALDI_ASSERT(*begin_iter == begin_frame &&
               (begin_iter == frame_.begin() || begin_iter[-1] < begin_frame));
  // even if end_frame == supervision_.num_frames, there should be a state with
  // that frame index.
  KALDI_ASSERT(end_iter[-1] < end_frame &&
               (end_iter < frame_.end() || *end_iter == end_frame));
  int32 begin_state = begin_iter - frame_.begin(),
      end_state = end_iter - frame_.begin();

  CreateRangeFst(begin_frame, end_frame,
                 begin_state, end_state, &(out_supervision->fst));

  KALDI_ASSERT(out_supervision->fst.NumStates() > 0);
  KALDI_ASSERT(supervision_.num_sequences == 1);
  out_supervision->num_sequences = 1;
  out_supervision->weight = supervision_.weight;
  out_supervision->frames_per_sequence = num_frames;
  out_supervision->label_dim = supervision_.label_dim;
}

void SupervisionSplitter::CreateRangeFst(
    int32 begin_frame, int32 end_frame,
    int32 begin_state, int32 end_state,
    fst::StdVectorFst *fst) const {
  // There will be a special pre-start state that has epsilon transitions to all
  // states whose frame equals begin_frame; we'll later do RmEpsilon to remove
  // these.  Next we will include all states begin_state <= s < end_state in the
  // output FST, plus (if end_frame != supervision_.num_frames) a special final
  // state.  All transitions to states >= end_state will be turned into
  // a transition to the special final state.  There should be no final-probs
  // on the states begin_state <= s < end_state.
  KALDI_ASSERT(end_state > begin_state);
  fst->DeleteStates();
  fst->ReserveStates(end_state - begin_state + 2);
  int32 start_state = fst->AddState();
  fst->SetStart(start_state);
  for (int32 i = begin_state; i < end_state; i++)
    fst->AddState();
  // Add the special final-state.
  int32 final_state = fst->AddState();
  fst->SetFinal(final_state, fst::TropicalWeight::One());
  for (int32 state = begin_state; state < end_state; state++) {
    int32 output_state = state - begin_state + 1;
    if (frame_[state] == begin_frame) {
      // we'd like to make this an initial state, but OpenFst doesn't allow
      // multiple initial states.  Instead we add an epsilon transition to it
      // from our actual initial state; we'll later do RmEpsilon and
      // determinize.
      fst->AddArc(start_state,
                  fst::StdArc(0, 0, fst::TropicalWeight::One(),
                              output_state));
    } else {
      KALDI_ASSERT(frame_[state] < end_frame);
    }
    typedef fst::ArcIterator<fst::StdVectorFst> IterType;
    for (IterType aiter(supervision_.fst, state); !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc(aiter.Value());
      int32 nextstate = arc.nextstate;
      if (nextstate >= end_state) {
        // A transition to any state outside the range becomes a transition to
        // our special final-state.
        fst->AddArc(output_state,
                    fst::StdArc(arc.ilabel, arc.olabel,
                                arc.weight, final_state));
      } else {
        int32 output_nextstate = arc.nextstate - begin_state + 1;
        // note: arc.ilabel should equal arc.olabel and arc.weight should equal
        // fst::TropicalWeight::One().
        fst->AddArc(output_state,
                    fst::StdArc(arc.ilabel, arc.olabel,
                                arc.weight, output_nextstate));
      }
    }
  }
}


// I couldn't figure out how to do this with OpenFST's native 'visitor' and
// queue mechanisms so I'm just coding this myself.
void SortBreadthFirstSearch(fst::StdVectorFst *fst) {
  std::vector<int32> state_order(fst->NumStates(), -1);
  std::vector<bool> seen(fst->NumStates(), false);
  int32 start_state = fst->Start();
  KALDI_ASSERT(start_state >= 0);
  std::deque<int32> queue;
  queue.push_back(start_state);
  seen[start_state] = true;
  int32 num_output = 0;
  while (!queue.empty()) {
    int32 state = queue.front();
    state_order[state] = num_output++;
    queue.pop_front();
    for (fst::ArcIterator<fst::StdVectorFst> aiter(*fst, state);
         !aiter.Done(); aiter.Next()) {
      int32 nextstate = aiter.Value().nextstate;
      if (!seen[nextstate]) {
        seen[nextstate] = true;
        queue.push_back(nextstate);
      }
    }
  }
  if (num_output != fst->NumStates())
    KALDI_ERR << "Input to SortBreadthFirstSearch must be connected.";
  fst::StateSort(fst, state_order);
}



void Supervision::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Supervision>");
  WriteToken(os, binary, "<Weight>");
  WriteBasicType(os, binary, weight);
  WriteToken(os, binary, "<NumSequences>");
  WriteBasicType(os, binary, num_sequences);
  WriteToken(os, binary, "<FramesPerSeq>");
  WriteBasicType(os, binary, frames_per_sequence);
  WriteToken(os, binary, "<LabelDim>");
  WriteBasicType(os, binary, label_dim);
  KALDI_ASSERT(frames_per_sequence > 0 && label_dim > 0 &&
               num_sequences > 0);
  if (binary == false) {
    // In text mode, write the FST without any compactification.
    WriteFstKaldi(os, binary, fst);
  } else {
    // Write using StdAcceptorCompactFst, making use of the fact that it's an
    // acceptor.
    fst::FstWriteOptions write_options("<unknown>");
    fst::StdCompactAcceptorFst::WriteFst(
        fst, fst::AcceptorCompactor<fst::StdArc>(), os,
        write_options);
  }
  if(num_sequences > 1) {
    KALDI_ASSERT(fsts.size() == num_sequences);
    WriteToken(os, binary, "<UnmergedFSTs>");
    for (int i = 0; i < num_sequences; i++) {
      if (binary == false) {
        // In text mode, write the FST without any compactification.
        WriteFstKaldi(os, binary, fsts[i]);
      } else {
        // Write using StdAcceptorCompactFst, making use of the fact that it's an
        // acceptor.
        fst::FstWriteOptions write_options("<unknown>");
        fst::StdCompactAcceptorFst::WriteFst(
            fsts[i], fst::AcceptorCompactor<fst::StdArc>(), os,
            write_options);
      }
    }
    WriteToken(os, binary, "</UnmergedFSTs>");
  }
  WriteToken(os, binary, "</Supervision>");
}

void Supervision::Swap(Supervision *other) {
  std::swap(weight, other->weight);
  std::swap(num_sequences, other->num_sequences);
  std::swap(frames_per_sequence, other->frames_per_sequence);
  std::swap(label_dim, other->label_dim);
  std::swap(fst, other->fst);
  std::swap(fsts, other->fsts);
}

void Supervision::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Supervision>");
  ExpectToken(is, binary, "<Weight>");
  ReadBasicType(is, binary, &weight);
  ExpectToken(is, binary, "<NumSequences>");
  ReadBasicType(is, binary, &num_sequences);
  ExpectToken(is, binary, "<FramesPerSeq>");
  ReadBasicType(is, binary, &frames_per_sequence);
  ExpectToken(is, binary, "<LabelDim>");
  ReadBasicType(is, binary, &label_dim);
  if (!binary) {
    ReadFstKaldi(is, binary, &fst);
  } else {
    fst::StdCompactAcceptorFst *compact_fst =
        fst::StdCompactAcceptorFst::Read(
            is, fst::FstReadOptions(std::string("[unknown]")));
    if (compact_fst == NULL)
      KALDI_ERR << "Error reading compact FST from disk";
    fst = *compact_fst;
    delete compact_fst;
  }
  if(num_sequences > 1) {
    fsts.resize(num_sequences);
    ExpectToken(is, binary, "<UnmergedFSTs>");
    for (int i = 0; i < num_sequences; i++) {
      if (!binary) {
        ReadFstKaldi(is, binary, &fsts[i]);
      } else {
        fst::StdCompactAcceptorFst *compact_fst =
            fst::StdCompactAcceptorFst::Read(
                is, fst::FstReadOptions(std::string("[unknown]")));
        if (compact_fst == NULL)
          KALDI_ERR << "Error reading compact FST from disk";
        fsts[i] = *compact_fst;
        delete compact_fst;
      }
    }
    ExpectToken(is, binary, "</UnmergedFSTs>");
  }
    // ReadFstKaldi will work even though we wrote using a compact format.
  ExpectToken(is, binary, "</Supervision>");
}

int32 ComputeFstStateTimes(const fst::StdVectorFst &fst,
                           std::vector<int32> *state_times) {
  if (fst.Start() != 0)  // this is implied by our properties.
    KALDI_ERR << "Expecting input FST start state to be zero";
  int32 num_states = fst.NumStates();
  int32 total_length = -1;
  state_times->clear();
  state_times->resize(num_states, -1);
  (*state_times)[0] = 0;
  for (int32 state = 0; state < num_states; state++) {
    int32 next_state_time = (*state_times)[state] + 1;
    if (next_state_time <= 0)  // i.e. (*state_times)[state] < 0
      KALDI_ERR << "Input FST does not have required properties.";
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      KALDI_ASSERT(arc.ilabel != 0);
      int32 &next_state_ref = (*state_times)[arc.nextstate];
      if (next_state_ref == -1)
        next_state_ref = next_state_time;
      else if (next_state_ref != next_state_time)
        KALDI_ERR << "Input FST does not have required properties.";
    }
    if (fst.Final(state) != fst::TropicalWeight::Zero()) {
      if (total_length == -1)
        total_length = next_state_time - 1;
      else if (total_length != next_state_time - 1)
        KALDI_ERR << "Input FST does not have required properties.";
    }
  }
  if (total_length < 0)
    KALDI_ERR << "Input FST does not have required properties.";
  return total_length;
}

Supervision::Supervision(const Supervision &other):
    weight(other.weight), num_sequences(other.num_sequences),
    frames_per_sequence(other.frames_per_sequence),
    label_dim(other.label_dim), fst(other.fst), fsts(other.fsts) { }

void AppendSupervision(const std::vector<const Supervision*> &input,
                       bool compactify,
                       std::vector<Supervision> *output_supervision) {
  KALDI_ASSERT(!input.empty());
  int32 label_dim = input[0]->label_dim,
      num_inputs = input.size();
  if (num_inputs == 1) {
    output_supervision->resize(1);
    (*output_supervision)[0] = *(input[0]);
    return;
  }
  std::vector<bool> output_was_merged;
  for (int32 i = 1; i < num_inputs; i++)
    KALDI_ASSERT(input[i]->label_dim == label_dim &&
                 "Trying to append incompatible Supervision objects");
  output_supervision->clear();
  output_supervision->reserve(input.size());
  for (int32 i = 0; i < input.size(); i++) {
    const Supervision &src = *(input[i]);
    if (compactify && !output_supervision->empty() &&
        output_supervision->back().weight == src.weight &&
        output_supervision->back().frames_per_sequence ==
        src.frames_per_sequence) {
      // Combine with current output
      // append src.fst to output_supervision->fst.
      fst::Concat(&output_supervision->back().fst, src.fst);
      output_supervision->back().num_sequences++;
      output_was_merged.back() = true;
    } else {
      output_supervision->resize(output_supervision->size() + 1);
      output_supervision->back() = src;
      output_was_merged.push_back(false);
    }
  }
  for (int32 i = 0; i < input.size(); i++)
    (*output_supervision)[0].fsts.push_back(input[i]->fst);

  KALDI_ASSERT(output_was_merged.size() == output_supervision->size());
  for (size_t i = 0; i < output_supervision->size(); i++) {
    if (output_was_merged[i]) {
      fst::StdVectorFst &out_fst = (*output_supervision)[i].fst;
      // The process of concatenation will have introduced epsilons.
      fst::RmEpsilon(&out_fst);
      SortBreadthFirstSearch(&out_fst);
    }
  }
}

bool AddWeightToSupervisionFst(const fst::StdVectorFst &normalization_fst,
                               Supervision *supervision) {
  // remove epsilons before composing.  'normalization_fst' has noepsilons so
  // the composed result will be epsilon free.
  fst::StdVectorFst supervision_fst_noeps(supervision->fst);
  fst::RmEpsilon(&supervision_fst_noeps);
  if (!TryDeterminizeMinimize(kSupervisionMaxStates,
                              &supervision_fst_noeps))
    return false;

  // note: by default, 'Compose' will call 'Connect', so if the
  // resulting FST is not connected, it will end up empty.
  fst::StdVectorFst composed_fst;
  fst::Compose(supervision_fst_noeps, normalization_fst,
               &composed_fst);
  if (composed_fst.NumStates() == 0)
    return false;
  // projection should not be necessary, as both FSTs are acceptors.
  // determinize and minimize to make it as compact as possible.

  if (!TryDeterminizeMinimize(kSupervisionMaxStates,
                              &composed_fst))
    return false;
  supervision->fst = composed_fst;

  // Make sure the states are numbered in increasing order of time.
  SortBreadthFirstSearch(&(supervision->fst));
  KALDI_ASSERT(supervision->fst.Properties(fst::kAcceptor, true) == fst::kAcceptor);
  KALDI_ASSERT(supervision->fst.Properties(fst::kIEpsilons, true) == 0);
  return true;
}

void SplitIntoRanges(int32 num_frames,
                     int32 frames_per_range,
                     std::vector<int32> *range_starts) {
  if (frames_per_range > num_frames) {
    range_starts->clear();
    return;  // there is no room for even one range.
  }
  int32 num_ranges = num_frames  / frames_per_range,
      extra_frames = num_frames % frames_per_range;
  // this is a kind of heuristic.  If the number of frames we'd
  // be skipping is less than 1/4 of the frames_per_range, then
  // skip frames; otherwise, duplicate frames.
  // it's important that this is <=, not <, so that if
  // extra_frames == 0 and frames_per_range is < 4, we
  // don't insert an extra range.
  if (extra_frames <= frames_per_range / 4) {
    // skip frames.  we do this at start or end, or between ranges.
    std::vector<int32> num_skips(num_ranges + 1, 0);
    for (int32 i = 0; i < extra_frames; i++)
      num_skips[RandInt(0, num_ranges)]++;
    range_starts->resize(num_ranges);
    int32 cur_start = num_skips[0];
    for (int32 i = 0; i < num_ranges; i++) {
      (*range_starts)[i] = cur_start;
      cur_start += frames_per_range;
      cur_start += num_skips[i + 1];
    }
    KALDI_ASSERT(cur_start == num_frames);
  } else {
    // duplicate frames.
    num_ranges++;
    int32 num_duplicated_frames = frames_per_range - extra_frames;
    // the way we handle the 'extra_frames' frames of output is that we
    // backtrack zero or more frames between outputting each pair of ranges, and
    // the total of these backtracks equals 'extra_frames'.
    std::vector<int32> num_backtracks(num_ranges, 0);
    for (int32 i = 0; i < num_duplicated_frames; i++) {
      // num_ranges - 2 below is not a bug.  we only want to backtrack
      // between ranges, not past the end of the last range (i.e. at
      // position num_ranges - 1).  we make the vector one longer to
      // simplify the loop below.
      num_backtracks[RandInt(0, num_ranges - 2)]++;
    }
    range_starts->resize(num_ranges);
    int32 cur_start = 0;
    for (int32 i = 0; i < num_ranges; i++) {
      (*range_starts)[i] = cur_start;
      cur_start += frames_per_range;
      cur_start -= num_backtracks[i];
    }
    KALDI_ASSERT(cur_start == num_frames);
  }
}

bool Supervision::operator == (const Supervision &other) const {
  return weight == other.weight && num_sequences == other.num_sequences &&
      frames_per_sequence == other.frames_per_sequence &&
      label_dim == other.label_dim && fst::Equal(fst, other.fst);
}

void Supervision::Check(const TransitionModel &trans_mdl) const {
  if (weight <= 0.0)
    KALDI_ERR << "Weight should be positive.";
  if (frames_per_sequence <= 0)
    KALDI_ERR << "Invalid frames_per_sequence: " << frames_per_sequence;
  if (num_sequences <= 0)
    KALDI_ERR << "Invalid num_sequences: " << num_sequences;
  if (label_dim != trans_mdl.NumPdfs())
    KALDI_ERR << "Invalid label-dim: " << label_dim
              << ", expected " << trans_mdl.NumPdfs();
  std::vector<int32> state_times;
  if (frames_per_sequence * num_sequences !=
      ComputeFstStateTimes(fst, &state_times))
    KALDI_ERR << "Num-frames does not match fst.";
}

void GetWeightsForRanges(int32 range_length,
                         const std::vector<int32> &range_starts,
                         std::vector<Vector<BaseFloat> > *weights) {
  KALDI_ASSERT(range_length > 0);
  int32 num_ranges = range_starts.size();
  weights->resize(num_ranges);
  for (int32 i = 0; i < num_ranges; i++) {
    (*weights)[i].Resize(range_length);
    (*weights)[i].Set(1.0);
  }
  for (int32 i = 0; i + 1 < num_ranges; i++) {
    int32 j = i + 1;
    int32 i_start = range_starts[i], i_end = i_start + range_length,
          j_start = range_starts[j];
    KALDI_ASSERT(j_start > i_start);
    if (i_end > j_start) {
      Vector<BaseFloat> &i_weights = (*weights)[i], &j_weights = (*weights)[j];

      int32 overlap_length = i_end - j_start;
      // divide the overlapping piece of the 2 ranges into 3 regions of
      // approximately equal size, called the left, middle and right region.
      int32 left_length = overlap_length / 3,
          middle_length = (overlap_length - left_length) / 2,
           right_length = overlap_length - left_length - middle_length;
      KALDI_ASSERT(left_length >= 0 && middle_length >= 0 && right_length >= 0 &&
                   left_length + middle_length + right_length == overlap_length);
      // set the weight of the left region to be zero for the right (j) range.
      for (int32 k = 0; k < left_length; k++)
        j_weights(k) = 0.0;
      // set the weight of the right region to be zero for the left (i) range.
      for (int32 k = 0; k < right_length; k++)
        i_weights(range_length - 1 - k) = 0.0;
      // for the middle range, linearly interpolate between the 0's and 1's.
      // note: we multiply with existing weights instead of set in order to get
      // more accurate behavior in the unexpected case where things triply
      // overlap.
      for (int32 k = 0; k < middle_length; k++) {
        BaseFloat weight = (0.5 + k) / middle_length;
        j_weights(left_length + k) = weight;
        i_weights(range_length - 1 - right_length - k) = weight;
      }
    }
  }
}


void GetWeightsForRangesNew(int32 range_length,
                            int32 num_frames_zeroed,
                            const std::vector<int32> &range_starts,
                            std::vector<Vector<BaseFloat> > *weights) {
  KALDI_ASSERT(range_length > 0 && num_frames_zeroed * 2 < range_length);
  int32 num_ranges = range_starts.size();
  weights->resize(num_ranges);
  for (int32 i = 0; i < num_ranges; i++) {
    (*weights)[i].Resize(range_length);
    (*weights)[i].Set(1.0);
  }
  if (num_frames_zeroed == 0)
    return;
  for (int32 i = 1; i < num_ranges; i++)
    (*weights)[i].Range(0, num_frames_zeroed).Set(0.0);
  for (int32 i = 0; i + 1 < num_ranges; i++)
    (*weights)[i].Range(range_length - num_frames_zeroed,
                        num_frames_zeroed).Set(0.0);
}


}  // namespace chain
}  // namespace kaldi
