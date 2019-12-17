// decoder/grammar-fst.cc

// Copyright   2018  Johns Hopkins University (author: Daniel Povey)

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

#include "decoder/grammar-fst.h"
#include "fstext/grammar-context-fst.h"

namespace fst {


GrammarFst::GrammarFst(
    int32 nonterm_phones_offset,
    std::shared_ptr<const ConstFst<StdArc> > top_fst,
    const std::vector<std::pair<Label, std::shared_ptr<const ConstFst<StdArc> > > > &ifsts):
    nonterm_phones_offset_(nonterm_phones_offset),
    top_fst_(top_fst),
    ifsts_(ifsts) {
  Init();
}

void GrammarFst::Init() {
  KALDI_ASSERT(nonterm_phones_offset_ > 1);
  InitNonterminalMap();
  entry_arcs_.resize(ifsts_.size());
  if (!ifsts_.empty()) {
    // We call this mostly so that if something is wrong with the input FSTs, the
    // problem will be detected sooner rather than later.
    // There would be no problem if we were to call InitEntryArcs(i)
    // for all 0 <= i < ifsts_size(), but we choose to call it
    // lazily on demand, to save startup time if the number of nonterminals
    // is large.
    InitEntryArcs(0);
  }
  InitInstances();
}

GrammarFst::~GrammarFst() {
  Destroy();
}

void GrammarFst::Destroy() {
  for (size_t i = 0; i < instances_.size(); i++) {
    FstInstance &instance = instances_[i];
    std::unordered_map<BaseStateId, ExpandedState*>::const_iterator
        iter = instance.expanded_states.begin(),
        end = instance.expanded_states.end();
    for (; iter != end; ++iter) {
      ExpandedState *e = iter->second;
      delete e;
    }
  }
  top_fst_ = NULL;
  ifsts_.clear();
  nonterminal_map_.clear();
  entry_arcs_.clear();
  instances_.clear();
}


void GrammarFst::DecodeSymbol(Label label,
                              int32 *nonterminal_symbol,
                              int32 *left_context_phone) {
  // encoding_multiple will normally equal 1000 (but may be a multiple of 1000
  // if there are a lot of phones); kNontermBigNumber is 10000000.
  int32 big_number = static_cast<int32>(kNontermBigNumber),
      nonterm_phones_offset = nonterm_phones_offset_,
      encoding_multiple = GetEncodingMultiple(nonterm_phones_offset);
  // The following assertion should be optimized out as the condition is
  // statically known.
  KALDI_ASSERT(big_number % static_cast<int32>(kNontermMediumNumber) == 0);

  *nonterminal_symbol = (label - big_number) / encoding_multiple;
  *left_context_phone = label % encoding_multiple;
  if (*nonterminal_symbol <= nonterm_phones_offset ||
      *left_context_phone == 0 || *left_context_phone >
      nonterm_phones_offset + static_cast<int32>(kNontermBos))
    KALDI_ERR << "Decoding invalid label " << label
              << ": code error or invalid --nonterm-phones-offset?";

}

void GrammarFst::InitNonterminalMap() {
  nonterminal_map_.clear();
  for (size_t i = 0; i < ifsts_.size(); i++) {
    int32 nonterminal = ifsts_[i].first;
    if (nonterminal_map_.count(nonterminal))
      KALDI_ERR << "Nonterminal symbol " << nonterminal
                << " is paired with two FSTs.";
    if (nonterminal < GetPhoneSymbolFor(kNontermUserDefined))
      KALDI_ERR << "Nonterminal symbol " << nonterminal
                << " in input pairs, was expected to be >= "
                << GetPhoneSymbolFor(kNontermUserDefined);
    nonterminal_map_[nonterminal] = static_cast<int32>(i);
  }
}


bool GrammarFst::InitEntryArcs(int32 i) {
  KALDI_ASSERT(static_cast<size_t>(i) < ifsts_.size());
  const ConstFst<StdArc> &fst = *(ifsts_[i].second);
  if (fst.NumStates() == 0)
    return false;  /* this was the empty FST. */
  InitEntryOrReentryArcs(fst, fst.Start(),
                         GetPhoneSymbolFor(kNontermBegin),
                         &(entry_arcs_[i]));
  return true;
}

void GrammarFst::InitInstances() {
  KALDI_ASSERT(instances_.empty());
  instances_.resize(1);
  instances_[0].ifst_index = -1;
  instances_[0].fst = top_fst_.get();
  instances_[0].parent_instance = -1;
  instances_[0].parent_state = -1;
}

void GrammarFst::InitEntryOrReentryArcs(
    const ConstFst<StdArc> &fst,
    int32 entry_state,
    int32 expected_nonterminal_symbol,
    std::unordered_map<int32, int32> *phone_to_arc) {
  phone_to_arc->clear();
  ArcIterator<ConstFst<StdArc> > aiter(fst, entry_state);
  int32 arc_index = 0;
  for (; !aiter.Done(); aiter.Next(), ++arc_index) {
    const StdArc &arc = aiter.Value();
    int32 nonterminal, left_context_phone;
    if (arc.ilabel <= (int32)kNontermBigNumber) {
      if (entry_state == fst.Start()) {
        KALDI_ERR << "There is something wrong with the graph; did you forget to "
            "add #nonterm_begin and #nonterm_end to the non-top-level FSTs "
            "before compiling?";
      } else {
        KALDI_ERR << "There is something wrong with the graph; re-entry state is "
            "not as anticipated.";
      }
    }
    DecodeSymbol(arc.ilabel, &nonterminal, &left_context_phone);
    if (nonterminal != expected_nonterminal_symbol) {
      KALDI_ERR << "Expected arcs from this state to have nonterminal-symbol "
                << expected_nonterminal_symbol << ", but got "
                << nonterminal;
    }
    std::pair<int32, int32> p(left_context_phone, arc_index);
    if (!phone_to_arc->insert(p).second) {
      // If it was not successfully inserted in the phone_to_arc map, it means
      // there were two arcs with the same left-context phone, which does not
      // make sense; that's an error, likely a code error (or an error when the
      // input FSTs were generated).
      KALDI_ERR << "Two arcs had the same left-context phone.";
    }
  }
}

GrammarFst::ExpandedState *GrammarFst::ExpandState(
    int32 instance_id, BaseStateId state_id) {
  int32 big_number = kNontermBigNumber;
  const ConstFst<StdArc> &fst = *(instances_[instance_id].fst);
  ArcIterator<ConstFst<StdArc> > aiter(fst, state_id);
  KALDI_ASSERT(!aiter.Done() && aiter.Value().ilabel > big_number &&
               "Something is not right; did you call PrepareForGrammarFst()?");

  const StdArc &arc = aiter.Value();
  int32 encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_),
      nonterminal = (arc.ilabel - big_number) / encoding_multiple;
  if (nonterminal == GetPhoneSymbolFor(kNontermBegin) ||
      nonterminal == GetPhoneSymbolFor(kNontermReenter)) {
    KALDI_ERR << "Encountered unexpected type of nonterminal while "
        "expanding state.";
  } else if (nonterminal == GetPhoneSymbolFor(kNontermEnd)) {
    return ExpandStateEnd(instance_id, state_id);
  } else if (nonterminal >= GetPhoneSymbolFor(kNontermUserDefined)) {
    return ExpandStateUserDefined(instance_id, state_id);
  } else {
    KALDI_ERR << "Encountered unexpected type of nonterminal "
              << nonterminal << " while expanding state.";
  }
  return NULL;  // Suppress compiler warning
}


// static inline
void GrammarFst::CombineArcs(const StdArc &leaving_arc,
                             const StdArc &arriving_arc,
                             float cost_correction,
                             StdArc *arc) {
  // The following assertion shouldn't fail; we ensured this in
  // PrepareForGrammarFst(), search for 'olabel_problem'.
  KALDI_ASSERT(leaving_arc.olabel == 0);
  // 'leaving_arc' leaves one fst, and 'arriving_arcs', conceptually arrives in
  // another.  This code merges the information of the two arcs to make a
  // cross-FST arc.  The ilabel information is discarded as it was only intended
  // for the consumption of the GrammarFST code.
  arc->ilabel = 0;
  arc->olabel = arriving_arc.olabel;
  // conceptually, arc->weight =
  //  Times(Times(leaving_arc.weight, arriving_arc.weight), Weight(cost_correction)).
  // The below might be a bit faster, I hope-- avoiding checking.
  arc->weight = Weight(cost_correction + leaving_arc.weight.Value() +
                       arriving_arc.weight.Value());
  arc->nextstate = arriving_arc.nextstate;
}

GrammarFst::ExpandedState *GrammarFst::ExpandStateEnd(
    int32 instance_id, BaseStateId state_id) {
  if (instance_id == 0)
    KALDI_ERR << "Did not expect #nonterm_end symbol in FST-instance 0.";
  const FstInstance &instance = instances_[instance_id];
  int32 parent_instance_id = instance.parent_instance;
  const ConstFst<StdArc> &fst = *(instance.fst);
  const FstInstance &parent_instance = instances_[parent_instance_id];
  const ConstFst<StdArc> &parent_fst = *(parent_instance.fst);

  ExpandedState *ans = new ExpandedState;
  ans->dest_fst_instance = parent_instance_id;

  // parent_aiter is the arc-iterator in the state we return to.  We'll Seek()
  // to a different position 'parent_aiter' for each arc leaving this state.
  // (actually we expect just one arc to leave this state).
  ArcIterator<ConstFst<StdArc> > parent_aiter(parent_fst,
                                              instance.parent_state);

  // for explanation of cost_correction, see documentation for CombineArcs().
  float num_reentry_arcs = instances_[instance_id].parent_reentry_arcs.size(),
      cost_correction = -log(num_reentry_arcs);

  ArcIterator<ConstFst<StdArc> > aiter(fst, state_id);

  for (; !aiter.Done(); aiter.Next()) {
    const StdArc &leaving_arc = aiter.Value();
    int32 this_nonterminal, left_context_phone;
    DecodeSymbol(leaving_arc.ilabel, &this_nonterminal,
                 &left_context_phone);
    KALDI_ASSERT(this_nonterminal == GetPhoneSymbolFor(kNontermEnd) &&
                 ">1 nonterminals from a state; did you use "
                 "PrepareForGrammarFst()?");
    std::unordered_map<int32, int32>::const_iterator reentry_iter =
        instances_[instance_id].parent_reentry_arcs.find(left_context_phone),
        reentry_end = instances_[instance_id].parent_reentry_arcs.end();
    if (reentry_iter == reentry_end) {
      KALDI_ERR << "FST with index " << instance.ifst_index
                << " ends with left-context-phone " << left_context_phone
                << " but parent FST does not support that left-context "
          "at the return point.";
    }
    size_t parent_arc_index = static_cast<size_t>(reentry_iter->second);
    parent_aiter.Seek(parent_arc_index);
    const StdArc &arriving_arc = parent_aiter.Value();
    // 'arc' will combine the information on 'leaving_arc' and 'arriving_arc',
    // except that the ilabel will be set to zero.
    if (leaving_arc.olabel != 0) {
      // If the following fails it would maybe indicate you hadn't called
      // PrepareForGrammarFst(), or there was an error in that, because
      // we made sure the leaving arc does not have an olabel.  Search
      // in that code for 'olabel_problem' for more details.
      KALDI_ERR << "Leaving arc has zero olabel.";
    }
    StdArc arc;
    CombineArcs(leaving_arc, arriving_arc, cost_correction, &arc);
    ans->arcs.push_back(arc);
  }
  return ans;
}

int32 GrammarFst::GetChildInstanceId(int32 instance_id, int32 nonterminal,
                                     int32 state) {
  int64 encoded_pair = (static_cast<int64>(nonterminal) << 32) + state;
  // 'new_instance_id' is the instance-id we'd assign if we had to create a new one.
  // We try to add it at once, to avoid having to do an extra map lookup in case
  // it wasn't there and we did need to add it.
  int32 child_instance_id = instances_.size();
  {
    std::pair<int64, int32> p(encoded_pair, child_instance_id);
    std::pair<std::unordered_map<int64, int32>::const_iterator, bool> ans =
        instances_[instance_id].child_instances.insert(p);
    if (!ans.second) {
      // The pair was not inserted, which means the key 'encoded_pair' did exist in the
      // map.  Return the value in the map.
      child_instance_id = ans.first->second;
      return child_instance_id;
    }
  }
  // If we reached this point, we did successfully insert 'child_instance_id' into
  // the map, because the key didn't exist.  That means we have to actually create
  // the instance.
  instances_.resize(child_instance_id + 1);
  const FstInstance &parent_instance = instances_[instance_id];
  FstInstance &child_instance = instances_[child_instance_id];

  // Work out the ifst_index for this nonterminal.
  std::unordered_map<int32, int32>::const_iterator iter =
      nonterminal_map_.find(nonterminal);
  if (iter == nonterminal_map_.end()) {
    KALDI_ERR << "Nonterminal " << nonterminal << " was requested, but "
        "there is no FST for it.";
  }
  int32 ifst_index = iter->second;
  child_instance.ifst_index = ifst_index;
  child_instance.fst = ifsts_[ifst_index].second.get();
  child_instance.parent_instance = instance_id;
  child_instance.parent_state = state;
  InitEntryOrReentryArcs(*(parent_instance.fst), state,
                         GetPhoneSymbolFor(kNontermReenter),
                         &(child_instance.parent_reentry_arcs));
  return child_instance_id;
}

GrammarFst::ExpandedState *GrammarFst::ExpandStateUserDefined(
    int32 instance_id, BaseStateId state_id) {
  const ConstFst<StdArc> &fst = *(instances_[instance_id].fst);
  ArcIterator<ConstFst<StdArc> > aiter(fst, state_id);

  ExpandedState *ans = new ExpandedState;
  int32 dest_fst_instance = -1;  // We'll set it in the loop.
                                 // and->dest_fst_instance will be set to this.

  for (; !aiter.Done(); aiter.Next()) {
    const StdArc &leaving_arc = aiter.Value();
    int32 nonterminal, left_context_phone;
    DecodeSymbol(leaving_arc.ilabel, &nonterminal,
                 &left_context_phone);
    int32 child_instance_id = GetChildInstanceId(instance_id,
                                                 nonterminal,
                                                 leaving_arc.nextstate);
    if (dest_fst_instance < 0) {
      dest_fst_instance = child_instance_id;
    } else if (dest_fst_instance != child_instance_id) {
      KALDI_ERR << "Same state leaves to different FST instances "
          "(Did you use PrepareForGrammarFst()?)";
    }
    const FstInstance &child_instance = instances_[child_instance_id];
    const ConstFst<StdArc> &child_fst = *(child_instance.fst);
    int32 child_ifst_index = child_instance.ifst_index;
    std::unordered_map<int32, int32> &entry_arcs = entry_arcs_[child_ifst_index];
    if (entry_arcs.empty()) {
      if (!InitEntryArcs(child_ifst_index)) {
        // This child-FST was the empty FST.  There are no arcs to expand.
        continue;
      }
    }
    // for explanation of cost_correction, see documentation for CombineArcs().
    float num_entry_arcs = entry_arcs.size(),
        cost_correction = -log(num_entry_arcs);

    // Get the arc-index for the arc leaving the start-state of child FST that
    // corresponds to this phonetic context.
    std::unordered_map<int32, int32>::const_iterator entry_iter =
        entry_arcs.find(left_context_phone);
    if (entry_iter == entry_arcs.end()) {
      KALDI_ERR << "FST for nonterminal " << nonterminal
                << " does not have an entry point for left-context-phone "
                << left_context_phone;
    }
    int32 arc_index = entry_iter->second;
    ArcIterator<ConstFst<StdArc> > child_aiter(child_fst, child_fst.Start());
    child_aiter.Seek(arc_index);
    const StdArc &arriving_arc = child_aiter.Value();
    StdArc arc;
    CombineArcs(leaving_arc, arriving_arc, cost_correction, &arc);
    ans->arcs.push_back(arc);
  }
  ans->dest_fst_instance = dest_fst_instance;
  return ans;
}


void GrammarFst::Write(std::ostream &os, bool binary) const {
  using namespace kaldi;
  if (!binary)
    KALDI_ERR << "GrammarFst::Write only supports binary mode.";
  int32 format = 1,
      num_ifsts = ifsts_.size();
  WriteToken(os, binary, "<GrammarFst>");
  WriteBasicType(os, binary, format);
  WriteBasicType(os, binary, num_ifsts);
  WriteBasicType(os, binary, nonterm_phones_offset_);

  std::string stream_name("unknown");
  FstWriteOptions wopts(stream_name);
  top_fst_->Write(os, wopts);

  for (int32 i = 0; i < num_ifsts; i++) {
    int32 nonterminal = ifsts_[i].first;
    WriteBasicType(os, binary, nonterminal);
    ifsts_[i].second->Write(os, wopts);
  }
  WriteToken(os, binary, "</GrammarFst>");
}

static ConstFst<StdArc> *ReadConstFstFromStream(std::istream &is) {
  fst::FstHeader hdr;
  std::string stream_name("unknown");
  if (!hdr.Read(is, stream_name))
    KALDI_ERR << "Reading FST: error reading FST header";
  FstReadOptions ropts("<unspecified>", &hdr);
  ConstFst<StdArc> *ans = ConstFst<StdArc>::Read(is, ropts);
  if (!ans)
    KALDI_ERR << "Could not read ConstFst from stream.";
  return ans;
}



void GrammarFst::Read(std::istream &is, bool binary) {
  using namespace kaldi;
  if (!binary)
    KALDI_ERR << "GrammarFst::Read only supports binary mode.";
  if (top_fst_ != NULL)
    Destroy();
  int32 format = 1, num_ifsts;
  ExpectToken(is, binary, "<GrammarFst>");
  ReadBasicType(is, binary, &format);
  if (format != 1)
    KALDI_ERR << "This version of the code cannot read this GrammarFst, "
        "update your code.";
  ReadBasicType(is, binary, &num_ifsts);
  ReadBasicType(is, binary, &nonterm_phones_offset_);
  top_fst_ = std::shared_ptr<const ConstFst<StdArc> >(ReadConstFstFromStream(is));
  for (int32 i = 0; i < num_ifsts; i++) {
    int32 nonterminal;
    ReadBasicType(is, binary, &nonterminal);
    std::shared_ptr<const ConstFst<StdArc> >
        this_fst(ReadConstFstFromStream(is));
    ifsts_.push_back(std::pair<int32, std::shared_ptr<const ConstFst<StdArc> > >(
        nonterminal, this_fst));
  }
  Init();
}


/**
   This utility function input-determinizes a specified state s of the FST
   'fst'.   (This input-determinizes while treating epsilon as a real symbol,
   although for the application we expect to use it, there won't be epsilons).

   What this function does is: for any symbol i that appears as the ilabel of
   more than one arc leaving state s of FST 'fst', it creates an additional
   state, it creates a new state t with epsilon-input transitions leaving it for
   each of those multiple arcs leaving state s; it deletes the original arcs
   leaving state s; and it creates a single arc leaving state s to the newly
   created state with the ilabel i on it.  It sets the weights as necessary to
   preserve equivalence and also to ensure that if, prior to this modification,
   the FST was stochastic when cast to the log semiring (see
   IsStochasticInLog()), it still will be.  I.e. when interpreted as
   negative logprobs, the weight from state s to t would be the sum of
   the weights on the original arcs leaving state s.

   This is used as a very cheap solution when preparing FSTs for the grammar
   decoder, to ensure that there is only one entry-state to the sub-FST for each
   phonetic left-context; this keeps the grammar-FST code (i.e. the code that
   stitches them together) simple.  Of course it will tend to introduce
   unnecessary epsilons, and if we were careful we might be able to remove
   some of those, but this wouldn't have a substantial impact on overall
   decoder performance so we don't bother.
 */
static void InputDeterminizeSingleState(StdArc::StateId s,
                                        VectorFst<StdArc> *fst) {
  bool was_input_deterministic = true;
  typedef StdArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;
  typedef Arc::Weight Weight;

  struct InfoForIlabel {
    std::vector<size_t> arc_indexes;  // indexes of all arcs with this ilabel
    float tot_cost;  // total cost of all arcs leaving state s for this
                     // ilabel, summed as if they were negative log-probs.
    StateId new_state;  // state-id of new state, if any, that we have created
                        // to remove duplicate symbols with this ilabel.
    InfoForIlabel(): new_state(-1) { }
  };

  std::unordered_map<Label, InfoForIlabel> label_map;

  size_t arc_index = 0;
  for (ArcIterator<VectorFst<Arc> > aiter(*fst, s);
       !aiter.Done(); aiter.Next(), ++arc_index) {
    const Arc &arc = aiter.Value();
    InfoForIlabel &info = label_map[arc.ilabel];
    if (info.arc_indexes.empty()) {
      info.tot_cost = arc.weight.Value();
    } else {
      info.tot_cost = -kaldi::LogAdd(-info.tot_cost, -arc.weight.Value());
      was_input_deterministic = false;
    }
    info.arc_indexes.push_back(arc_index);
  }

  if (was_input_deterministic)
    return;  // Nothing to do.

  // 'new_arcs' will contain the modified list of arcs
  // leaving state s
  std::vector<Arc> new_arcs;
  new_arcs.reserve(arc_index);
  arc_index = 0;
  for (ArcIterator<VectorFst<Arc> > aiter(*fst, s);
       !aiter.Done(); aiter.Next(), ++arc_index) {
    const Arc &arc = aiter.Value();
    Label ilabel = arc.ilabel;
    InfoForIlabel &info = label_map[ilabel];
    if (info.arc_indexes.size() == 1) {
      new_arcs.push_back(arc);  // no changes needed
    } else {
      if (info.new_state < 0) {
        info.new_state = fst->AddState();
        // add arc from state 's' to newly created state.
        new_arcs.push_back(Arc(ilabel, 0, Weight(info.tot_cost),
                               info.new_state));
      }
      // add arc from new state to original destination of this arc.
      fst->AddArc(info.new_state, Arc(0, arc.olabel,
                                      Weight(arc.weight.Value() - info.tot_cost),
                                      arc.nextstate));
    }
  }
  fst->DeleteArcs(s);
  for (size_t i = 0; i < new_arcs.size(); i++)
    fst->AddArc(s, new_arcs[i]);
}


// This class contains the implementation of the function
// PrepareForGrammarFst(), which is declared in grammar-fst.h.
class GrammarFstPreparer {
 public:
  using FST = VectorFst<StdArc>;
  using Arc = StdArc;
  using StateId = Arc::StateId;
  using Label = Arc::Label;
  using Weight = Arc::Weight;

  GrammarFstPreparer(int32 nonterm_phones_offset,
                     VectorFst<StdArc> *fst):
      nonterm_phones_offset_(nonterm_phones_offset),
      fst_(fst), orig_num_states_(fst->NumStates()),
      simple_final_state_(kNoStateId) { }

  void Prepare() {
    if (fst_->Start() == kNoStateId) {
      KALDI_ERR << "FST has no states.";
    }
    for (StateId s = 0; s < fst_->NumStates(); s++) {
      if (IsSpecialState(s)) {
        if (NeedEpsilons(s)) {
          InsertEpsilonsForState(s);
          // This state won't be treated as a 'special' state any more;
          // all 'special' arcs (arcs with ilabels >= kNontermBigNumber)
          // have been moved and now leave from newly created states that
          // this state transitions to via epsilons arcs.
        } else {
          // OK, state s is a special state.
          FixArcsToFinalStates(s);
          MaybeAddFinalProbToState(s);
          // The following ensures that the start-state of sub-FSTs only has
          // a single arc per left-context phone (the graph-building recipe can
          // end up creating more than one if there were disambiguation symbols,
          // e.g. for langauge model backoff).
          if (s == fst_->Start() && IsEntryState(s))
            InputDeterminizeSingleState(s, fst_);
        }
      }
    }
    StateId num_new_states = fst_->NumStates() - orig_num_states_;
    KALDI_LOG << "Added " << num_new_states << " new states while "
        "preparing for grammar FST.";
  }

 private:

  // Returns true if state 's' has at least one arc coming out of it with a
  // special nonterminal-related ilabel on it (i.e. an ilabel >=
  // kNontermBigNumber), and false otherwise.
  bool IsSpecialState(StateId s) const;

  // This function verifies that state s does not currently have any
  // final-prob (crashes if that fails); then, if the arcs leaving s have
  // nonterminal symbols kNontermEnd or user-defined nonterminals (>=
  // kNontermUserDefined), it adds a final-prob with cost given by
  // KALDI_GRAMMAR_FST_SPECIAL_WEIGHT to the state.
  //
  // State s is required to be a 'special state', i.e. have special symbols on
  // arcs leaving it, and the function assumes (since it will already
  // have been checked) that the arcs leaving s, if there are more than
  // one, all correspond to the same nonterminal symbol.
  void MaybeAddFinalProbToState(StateId s);


  // This function does some checking for 'special states', that they have
  // certain expected properties, and also detects certain problematic
  // conditions that we need to fix.  It returns true if we need to
  // modify this state (by adding input-epsilon arcs), and false otherwise.
  bool NeedEpsilons(StateId s) const;

  // Returns true if state s (which is expected to be the start state, although we
  // don't check this) has arcs with nonterminal symbols #nonterm_begin.
  bool IsEntryState(StateId s) const;

  // Fixes any final-prob-related problems with this state.  The problem we aim
  // to fix is that there may be arcs with nonterminal symbol #nonterm_end which
  // transition from this state to a state with non-unit final prob.  This
  // function assimilates that final-prob into the arc leaving from this state,
  // by making the arc transition to a new state with unit final-prob, and
  // incorporating the original final-prob into the arc's weight.
  //
  // The purpose of this is to keep the GrammarFst code simple.
  //
  // It would have been more efficient to do this in CheckProperties(), but
  // doing it this way is clearer; and the extra time taken here will be tiny.
  void FixArcsToFinalStates(StateId s);


  // This struct represents a category of arcs that are allowed to leave from
  // the same 'special state'.  If a special state has arcs leaving it that
  // are in more than one category, it will need to be split up into
  // multiple states connected by epsilons.
  //
  // The 'nonterminal' and 'nextstate' have to do with ensuring that all
  // arcs leaving a particular FST state transition to the same FST instance
  // (which, in turn, helps to keep the ArcIterator code efficient).
  //
  // The 'olabel' has to do with ensuring that arcs with user-defined
  // nonterminals or kNontermEnd have no olabels on them.  This is a requirement
  // of the CombineArcs() function of GrammarFst, because it needs to combine
  // two olabels into one so we need to know that at least one of the olabels is
  // always epsilon.
  struct ArcCategory {
    int32 nonterminal;  //  The nonterminal symbol #nontermXXX encoded into the ilabel,
                        // or 0 if the ilabel was <kNontermBigNumber.
    StateId nextstate; //  If 'nonterminal' is a user-defined nonterminal like
                       //  #nonterm:foo,
                       // then the destination state of the arc, else kNoStateId (-1).
    Label olabel;  //  If 'nonterminal' is #nonterm_end or is a user-defined
                   // nonterminal (e.g. #nonterm:foo), then the olabel on the
                   // arc; else, 0.
    bool operator < (const ArcCategory &other) const {
      if (nonterminal < other.nonterminal) return true;
      else if (nonterminal > other.nonterminal) return false;
      if (nextstate < other.nextstate) return true;
      else if (nextstate > other.nextstate) return false;
      return olabel < other.olabel;
    }
  };

  // This function, which is used in CheckProperties() and
  // InsertEpsilonsForState(), works out the categrory of the arc; see
  // documentation of struct ArcCategory for more details.
  void GetCategoryOfArc(const Arc &arc,
                        ArcCategory *arc_category) const;


  // This will be called for 'special states' that need to be split up.
  // Non-special arcs leaving this state will stay here.  For each
  // category of special arcs (see ArcCategory for details), a new
  // state will be created and those arcs will leave from that state
  // instead; for each such state, an input-epsilon arc will leave this state
  // for that state.  For more details, see the code.
  void InsertEpsilonsForState(StateId s);

  inline int32 GetPhoneSymbolFor(enum NonterminalValues n) const {
    return nonterm_phones_offset_ + static_cast<int32>(n);
  }

  int32 nonterm_phones_offset_;
  VectorFst<StdArc> *fst_;
  StateId orig_num_states_;
  // If needed we may add a 'simple final state' to fst_, which has unit
  // final-prob.  This is used when we ensure that states with kNontermExit on
  // them transition to a state with unit final-prob, so we don't need to
  // look at the final-prob when expanding states.
  StateId simple_final_state_;
};

bool GrammarFstPreparer::IsSpecialState(StateId s) const {
  if (fst_->Final(s).Value() == KALDI_GRAMMAR_FST_SPECIAL_WEIGHT) {
    // TODO: find a way to detect if it was a coincidence, or not make it an
    // error, because in principle a user-defined grammar could contain this
    // special cost.
    KALDI_WARN << "It looks like you are calling PrepareForGrammarFst twice.";
  }
  for (ArcIterator<FST> aiter(*fst_, s ); !aiter.Done(); aiter.Next()) {
    const Arc &arc = aiter.Value();
    if (arc.ilabel >= kNontermBigNumber) // 1 million
      return true;
  }
  return false;
}

bool GrammarFstPreparer::IsEntryState(StateId s) const {
  int32 big_number = kNontermBigNumber,
      encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_);

  for (ArcIterator<FST> aiter(*fst_, s ); !aiter.Done(); aiter.Next()) {
    const Arc &arc = aiter.Value();
    int32 nonterminal = (arc.ilabel - big_number) /
        encoding_multiple;
    // we check that at least one has label with nonterminal equal to #nonterm_begin...
    // in fact they will all have this value if at least one does, and this was checked
    // in NeedEpsilons().
    if (nonterminal == GetPhoneSymbolFor(kNontermBegin))
      return true;
  }
  return false;
}


bool GrammarFstPreparer::NeedEpsilons(StateId s) const {

  // See the documentation for GetCategoryOfArc() for explanation of what these are.
  std::set<ArcCategory> categories;

  if (fst_->Final(s) != Weight::Zero()) {
    // A state having a final-prob is considered the same as it having
    // a non-nonterminal arc out of it.. this would be like a transition
    // within the same FST.
    ArcCategory category;
    category.nonterminal = 0;
    category.nextstate = kNoStateId;
    category.olabel = 0;
    categories.insert(category);
  }

  int32 big_number = kNontermBigNumber,
      encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_);

  for (ArcIterator<FST> aiter(*fst_, s ); !aiter.Done(); aiter.Next()) {
    const Arc &arc = aiter.Value();
    ArcCategory category;
    GetCategoryOfArc(arc, &category);
    categories.insert(category);

    // the rest of this block is just checking.
    int32 nonterminal = category.nonterminal;

    if (nonterminal >= GetPhoneSymbolFor(kNontermUserDefined)) {
      // Check that the destination state of this arc has arcs with
      // kNontermReenter on them.  We'll separately check that such states
      // don't have other types of arcs leaving them (search for
      // kNontermReenter below), so it's sufficient to check the first arc.
      ArcIterator<FST> next_aiter(*fst_, arc.nextstate);
      if (next_aiter.Done())
        KALDI_ERR << "Destination state of a user-defined nonterminal "
            "has no arcs leaving it.";
      const Arc &next_arc = next_aiter.Value();
      int32 next_nonterminal = (next_arc.ilabel - big_number) /
          encoding_multiple;
      if (next_nonterminal != GetPhoneSymbolFor(kNontermReenter)) {
        KALDI_ERR << "Expected arcs with user-defined nonterminals to be "
            "followed by arcs with kNontermReenter.";
      }
    }
    if (nonterminal == GetPhoneSymbolFor(kNontermBegin) &&
        s != fst_->Start()) {
      KALDI_ERR << "#nonterm_begin symbol is present but this is not the "
          "first state.  Did you do fstdeterminizestar while compiling?";
    }
    if (nonterminal == GetPhoneSymbolFor(kNontermEnd)) {
      if (fst_->NumArcs(arc.nextstate) != 0 ||
          fst_->Final(arc.nextstate) == Weight::Zero()) {
        KALDI_ERR << "Arc with kNontermEnd is not the final arc.";
      }
    }
  }
  if (categories.size() > 1) {
    // This state has arcs leading to multiple FST instances.
    // Do some checking to see that there is nothing really unexpected in
    // there.
    for (std::set<ArcCategory>::const_iterator
             iter = categories.begin();
         iter != categories.end(); ++iter) {
      int32 nonterminal = iter->nonterminal;
      if (nonterminal == nonterm_phones_offset_ + kNontermBegin ||
          nonterminal == nonterm_phones_offset_ + kNontermReenter)
        // we don't expect any state which has symbols like (kNontermBegin:p1)
        // on arcs coming out of it, to also have other types of symbol.  The
        // same goes for kNontermReenter.
        KALDI_ERR << "We do not expect states with arcs of type "
            "kNontermBegin/kNontermReenter coming out of them, to also have "
            "other types of arc.";
    }
  }
  // the first half of the || below relates to olabels on arcs with either
  // user-defined nonterminals or #nonterm_end (which would become 'leaving_arc'
  // in the CombineArcs() function of GrammarFst).  That function does not allow
  // nonzero olabels on 'leaving_arc', which would be a problem if the
  // 'arriving' arc had nonzero olabels, so we solve this by introducing
  // input-epsilon arcs and putting the olabels on them instead.
  bool need_epsilons = (categories.size() == 1 &&
                        categories.begin()->olabel != 0) ||
      categories.size() > 1;
  return need_epsilons;
}

void GrammarFstPreparer::FixArcsToFinalStates(StateId s) {
  int32 encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_),
      big_number = kNontermBigNumber;
  for (MutableArcIterator<FST> aiter(fst_, s ); !aiter.Done(); aiter.Next()) {
    Arc arc = aiter.Value();
    if (arc.ilabel < big_number)
      continue;
    int32 nonterminal = (arc.ilabel - big_number) / encoding_multiple;
    if (nonterminal ==  GetPhoneSymbolFor(kNontermEnd)) {
      KALDI_ASSERT(fst_->NumArcs(arc.nextstate) == 0 &&
                   fst_->Final(arc.nextstate) != Weight::Zero());
      if (fst_->Final(arc.nextstate) == Weight::One())
        continue;  // There is no problem to fix.
      if (simple_final_state_ == kNoStateId) {
        simple_final_state_ = fst_->AddState();
        fst_->SetFinal(simple_final_state_, Weight::One());
      }
      arc.weight = Times(arc.weight, fst_->Final(arc.nextstate));
      arc.nextstate = simple_final_state_;
      aiter.SetValue(arc);
    }
  }
}

void GrammarFstPreparer::MaybeAddFinalProbToState(StateId s) {
  if (fst_->Final(s) != Weight::Zero()) {
    // Something went wrong and it will require some debugging.  In Prepare(),
    // if we detected that the special state had a nonzero final-prob, we
    // would have inserted epsilons to remove it, so there may be a bug in
    // this class's code.
    KALDI_ERR << "State already final-prob.";
  }
  ArcIterator<FST> aiter(*fst_, s );
  KALDI_ASSERT(!aiter.Done());
  const Arc &arc = aiter.Value();
  int32 encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_),
      big_number = kNontermBigNumber,
      nonterminal = (arc.ilabel - big_number) / encoding_multiple;
  KALDI_ASSERT(nonterminal >= GetPhoneSymbolFor(kNontermBegin));
  if (nonterminal == GetPhoneSymbolFor(kNontermEnd) ||
      nonterminal >= GetPhoneSymbolFor(kNontermUserDefined)) {
    fst_->SetFinal(s, Weight(KALDI_GRAMMAR_FST_SPECIAL_WEIGHT));
  }
}

void GrammarFstPreparer::GetCategoryOfArc(
    const Arc &arc, ArcCategory *arc_category) const {
  int32 encoding_multiple = GetEncodingMultiple(nonterm_phones_offset_),
      big_number = kNontermBigNumber;

  int32 ilabel = arc.ilabel;
  if (ilabel < big_number) {
    arc_category->nonterminal = 0;
    arc_category->nextstate = kNoStateId;
    arc_category->olabel = 0;
  } else {
    int32 nonterminal = (ilabel - big_number) / encoding_multiple;
    arc_category->nonterminal = nonterminal;
    if (nonterminal <= nonterm_phones_offset_) {
      KALDI_ERR << "Problem decoding nonterminal symbol "
          "(wrong --nonterm-phones-offset option?), ilabel="
                << ilabel;
    }
    if (nonterminal >= GetPhoneSymbolFor(kNontermUserDefined)) {
      // This is a user-defined symbol.
      arc_category->nextstate = arc.nextstate;
      arc_category->olabel = arc.olabel;
    } else {
      arc_category->nextstate = kNoStateId;
      if (nonterminal == GetPhoneSymbolFor(kNontermEnd))
        arc_category->olabel = arc.olabel;
      else
        arc_category->olabel = 0;
    }
  }
}


void GrammarFstPreparer::InsertEpsilonsForState(StateId s) {
  // Maps from category of arc, to a pair:
  //  the StateId is the state corresponding to that category.
  //  the float is the cost on the arc leading to that state;
  //   we compute the value that corresponds to the sum of the probabilities
  //   of the leaving arcs, bearing in mind that p = exp(-cost).
  // We don't insert the arc-category whose 'nonterminal' is 0 here (i.e. the
  // category for normal arcs); those arcs stay at this state.
  std::map<ArcCategory, std::pair<StateId, float> > category_to_state;

  // This loop sets up 'category_to_state'.
  for (fst::ArcIterator<FST> aiter(*fst_, s); !aiter.Done(); aiter.Next()) {
    const Arc &arc = aiter.Value();
    ArcCategory category;
    GetCategoryOfArc(arc, &category);
    int32 nonterminal = category.nonterminal;
    if (nonterminal == 0)
      continue;
    if (nonterminal == GetPhoneSymbolFor(kNontermBegin) ||
        nonterminal == GetPhoneSymbolFor(kNontermReenter)) {
      KALDI_ERR << "Something went wrong; did not expect to insert epsilons "
          "for this type of state.";
    }
    auto iter = category_to_state.find(category);
    if (iter == category_to_state.end()) {
      StateId new_state = fst_->AddState();
      float cost = arc.weight.Value();
      category_to_state[category] = std::pair<StateId, float>(new_state, cost);
    } else {
      std::pair<StateId, float> &p = iter->second;
      p.second = -kaldi::LogAdd(-p.second, -arc.weight.Value());
    }
  }

  KALDI_ASSERT(!category_to_state.empty());  // would be a code error.

  // 'arcs_from_this_state' is a place to put arcs that will put on this state
  // after we delete all its existing arcs.
  std::vector<Arc> arcs_from_this_state;
  arcs_from_this_state.reserve(fst_->NumArcs(s) + category_to_state.size());

  // add arcs corresponding to transitions to the newly created states, to
  // 'arcs_from_this_state'
  for (std::map<ArcCategory, std::pair<StateId, float> >::const_iterator
           iter = category_to_state.begin(); iter != category_to_state.end();
       ++iter) {
    const ArcCategory &category = iter->first;
    StateId new_state = iter->second.first;
    float cost = iter->second.second;
    Arc arc;
    arc.ilabel = 0;
    arc.olabel = category.olabel;
    arc.weight = Weight(cost);
    arc.nextstate = new_state;
    arcs_from_this_state.push_back(arc);
  }

  // Now add to 'arcs_from_this_state', and to the newly created states,
  // arcs corresponding to each of the arcs that were originally leaving
  // this state.
  for (fst::ArcIterator<FST> aiter(*fst_, s); !aiter.Done(); aiter.Next()) {
    const Arc &arc = aiter.Value();
    ArcCategory category;
    GetCategoryOfArc(arc, &category);
    int32 nonterminal = category.nonterminal;
    if (nonterminal == 0) { // this arc remains unchanged; we'll put it back later.
      arcs_from_this_state.push_back(arc);
      continue;
    }
    auto iter = category_to_state.find(category);
    KALDI_ASSERT(iter != category_to_state.end());
    Arc new_arc;
    new_arc.ilabel = arc.ilabel;
    if (arc.olabel == category.olabel) {
      new_arc.olabel = 0;  // the olabel went on the epsilon-input arc.
    } else {
      KALDI_ASSERT(category.olabel == 0);
      new_arc.olabel = arc.olabel;
    }
    StateId new_state = iter->second.first;
    float epsilon_arc_cost = iter->second.second;
    new_arc.weight = Weight(arc.weight.Value() - epsilon_arc_cost);
    new_arc.nextstate = arc.nextstate;
    fst_->AddArc(new_state, new_arc);
  }

  fst_->DeleteArcs(s);
  for (size_t i = 0; i < arcs_from_this_state.size(); i++) {
    fst_->AddArc(s, arcs_from_this_state[i]);
  }
  // leave the final-prob on this state as it was before.
}


void PrepareForGrammarFst(int32 nonterm_phones_offset,
                          VectorFst<StdArc> *fst) {
  GrammarFstPreparer p(nonterm_phones_offset, fst);
  p.Prepare();
}

void CopyToVectorFst(GrammarFst *grammar_fst,
                     VectorFst<StdArc> *vector_fst) {
  typedef GrammarFstArc::StateId GrammarStateId;  // int64
  typedef StdArc::StateId StdStateId;  // int
  typedef StdArc::Label Label;
  typedef StdArc::Weight Weight;

  std::vector<std::pair<GrammarStateId, StdStateId> > queue;
  std::unordered_map<GrammarStateId, StdStateId> state_map;

  vector_fst->DeleteStates();
  state_map[grammar_fst->Start()] = vector_fst->AddState();  // state 0.
  vector_fst->SetStart(0);

  queue.push_back(
      std::pair<GrammarStateId, StdStateId>(grammar_fst->Start(), 0));

  while (!queue.empty()) {
    std::pair<GrammarStateId, StdStateId> p = queue.back();
    queue.pop_back();
    GrammarStateId grammar_state = p.first;
    StdStateId std_state = p.second;
    vector_fst->SetFinal(std_state, grammar_fst->Final(grammar_state));
    ArcIterator<GrammarFst> aiter(*grammar_fst, grammar_state);
    for (; !aiter.Done(); aiter.Next()) {
      const GrammarFstArc &grammar_arc = aiter.Value();
      StdArc std_arc;
      std_arc.ilabel = grammar_arc.ilabel;
      std_arc.olabel = grammar_arc.olabel;
      std_arc.weight = grammar_arc.weight;
      GrammarStateId next_grammar_state = grammar_arc.nextstate;
      StdStateId next_std_state;
      std::unordered_map<GrammarStateId, StdStateId>::const_iterator
          state_iter = state_map.find(next_grammar_state);
      if (state_iter == state_map.end()) {
        next_std_state = vector_fst->AddState();
        state_map[next_grammar_state] = next_std_state;
        queue.push_back(std::pair<GrammarStateId, StdStateId>(
            next_grammar_state, next_std_state));
      } else {
        next_std_state = state_iter->second;
      }
      std_arc.nextstate = next_std_state;
      vector_fst->AddArc(std_state, std_arc);
    }
  }
}



} // end namespace fst
