// hmm/hmm-utils.cc

// Copyright 2009-2011  Microsoft Corporation
//                2018  Johns Hopkins University (author: Daniel Povey)
//                2019  Daniel Galvez

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

#include <memory>
#include <vector>

#include "hmm/hmm-utils.h"
#include "fst/fstlib.h"
#include "fstext/fstext-lib.h"
#include "fstext/grammar-context-fst.h"

namespace kaldi {

std::shared_ptr<fst::ExpandedFst<fst::StdArc>> GetHmmAsFsa(
    std::vector<int32> phone_window,
    const ContextDependencyInterface &ctx_dep,
    const Transitions &trans_model,
    const HTransducerConfig &config,
    HmmCacheType *cache) {
  if (static_cast<int32>(phone_window.size()) != ctx_dep.ContextWidth())
    KALDI_ERR << "Context size mismatch, ilabel-info [from context FST is "
              << phone_window.size() << ", context-dependency object "
        "expects " << ctx_dep.ContextWidth();

  int P = ctx_dep.CentralPosition();
  int32 phone = phone_window[P];
  if (phone == 0)
    KALDI_ERR << "phone == 0.  Some mismatch happened, or there is "
          "a code error.";

  const Topology &topo = trans_model.GetTopo();

  // vector of the pdfs, indexed by pdf-class (pdf-classes must start from zero
  // and be contiguous).
  std::vector<int32> pdfs(topo.NumPdfClasses(phone));
  for (int32 pdf_class = 0;
       pdf_class < static_cast<int32>(pdfs.size());
       pdf_class++) {
    if (! ctx_dep.Compute(phone_window, pdf_class, &(pdfs[pdf_class])) ) {
      std::ostringstream ctx_ss;
      for (size_t i = 0; i < phone_window.size(); i++)
        ctx_ss << phone_window[i] << ' ';
      KALDI_ERR << "GetHmmAsFsa: context-dependency object could not produce "
                << "an answer: pdf-class = " << pdf_class << " ctx-window = "
                << ctx_ss.str() << ".  This probably points "
          "to either a coding error in some graph-building process, "
          "a mismatch of topology with context-dependency object, the "
          "wrong FST being passed on a command-line, or something of "
          " that general nature.";
    }
  }

  std::pair<int32, std::vector<int32> > cache_index(phone, pdfs);
  if (cache != NULL) {
    HmmCacheType::iterator iter = cache->find(cache_index);
    if (iter != cache->end())
      return iter->second;
  }

  using Arc = fst::StdArc;
  using MyEditFst = fst::EditFst<Arc>;
  using StateId = Arc::StateId;

  const fst::StdVectorFst &entry = topo.TopologyForPhone(phone);
  std::shared_ptr<MyEditFst> loopless_entry = std::make_shared<MyEditFst>(entry);

  for (fst::StateIterator<MyEditFst> siter(*loopless_entry);
       !siter.Done(); siter.Next()) {
    StateId state = siter.Value();
    std::vector<Arc> non_self_loops;
    BaseFloat non_self_loop_prob = 1.0;
    for (fst::ArcIterator<MyEditFst> aiter(*loopless_entry, state);
         !aiter.Done(); aiter.Next()) {
      const Arc& arc = aiter.Value();
      if (arc.nextstate != state) {
        non_self_loops.push_back(arc);
      } else {
        non_self_loop_prob -= exp(-arc.weight.Value());
      }
    }
    KALDI_ASSERT(non_self_loop_prob >= BaseFloat(0));
    if (non_self_loops.size() != loopless_entry->NumArcs(state)) {
      loopless_entry->DeleteArcs(state);
      for (Arc& arc: non_self_loops) {
        // Renormalize the remaining arcs to have an outgoing weight
        // of 1.0, so we maintain stochasticity
        arc.weight = Arc::Weight(-log(exp(-arc.weight.Value()) / non_self_loop_prob));
        loopless_entry->AddArc(state, arc);
      }
    }
  }

  ApplyProbabilityScale(config.transition_scale, loopless_entry.get());
  if (cache != NULL)
    (*cache)[cache_index] = loopless_entry;
  return loopless_entry;
}



const fst::VectorFst<fst::StdArc>&
GetHmmAsFsaSimple(std::vector<int32> phone_window,
                  const ContextDependencyInterface &ctx_dep,
                  const Transitions &trans_model,
                  BaseFloat prob_scale) {
  using namespace fst;

  if (static_cast<int32>(phone_window.size()) != ctx_dep.ContextWidth())
    KALDI_ERR <<"Context size mismatch, ilabel-info [from context FST is "
              <<(phone_window.size())<<", context-dependency object "
        "expects "<<(ctx_dep.ContextWidth());

  int P = ctx_dep.CentralPosition();
  int32 phone = phone_window[P];
  KALDI_ASSERT(phone != 0);

  const Topology &topo = trans_model.GetTopo();
  return topo.TopologyForPhone(phone);
}



/// This utility function, used in GetHTransducer(), creates an FSA (finite
/// state acceptor, i.e. an FST with ilabels equal to olabels) with a single
/// successful path, with a single label on it.
static inline std::unique_ptr<fst::VectorFst<fst::StdArc>>
MakeTrivialAcceptor(int32 label) {
  typedef fst::StdArc Arc;
  typedef Arc::Weight Weight;
  std::unique_ptr<fst::VectorFst<Arc>> ans(new fst::VectorFst<Arc>);
  ans->AddState();
  ans->AddState();
  ans->SetStart(0);
  ans->SetFinal(1, Weight::One());
  ans->AddArc(0, Arc(label, label, Weight::One(), 1));
  return ans;
}



// The H transducer has a separate outgoing arc for each of the symbols in ilabel_info.
std::unique_ptr<fst::VectorFst<fst::StdArc>>
GetHTransducer(const std::vector<std::vector<int32> > &ilabel_info,
               const ContextDependencyInterface &ctx_dep,
               const Transitions &trans_model,
               const HTransducerConfig &config,
               std::vector<int32> *disambig_syms_left) {
  KALDI_ASSERT(ilabel_info.size() >= 1 && ilabel_info[0].size() == 0);  // make sure that eps == eps.
  HmmCacheType cache;
  // "cache" is an optimization that prevents GetHmmAsFsa repeating work
  // unnecessarily.
  using namespace fst;
  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;

  // I would prefer to do this:
  // std::vector<std::unique_ptr<const ExpandedFst<Arc>>> fsts(ilabel_info.size(), std::unique_ptr(nullptr));
  // But the second arg of constructor (2) at https://en.cppreference.com/w/cpp/container/vector/vector
  // must be able to be turned into a const-reference, which std::unique_ptr cannot be.
  std::vector<std::unique_ptr<const ExpandedFst<Arc>>> fsts;
  for(std::size_t i = 0; i < ilabel_info.size(); ++i) {
    fsts.emplace_back(std::unique_ptr<const ExpandedFst<Arc>>(nullptr));
  }
  std::vector<int32> phones = trans_model.GetPhones();

  KALDI_ASSERT(disambig_syms_left != 0);
  disambig_syms_left->clear();

  int32 first_disambig_sym = trans_model.NumTransitionIds() + 1;  // First disambig symbol we can have on the input side.
  int32 next_disambig_sym = first_disambig_sym;

  if (ilabel_info.size() > 0)
    KALDI_ASSERT(ilabel_info[0].size() == 0);  // make sure epsilon is epsilon...

  for (int32 j = 1; j < static_cast<int32>(ilabel_info.size()); j++) {  // zero is eps.
    KALDI_ASSERT(!ilabel_info[j].empty());
    if (ilabel_info[j][0] < 0 ||
        (ilabel_info[j][0] == 0 && ilabel_info[j].size() == 1)) {
      // disambig symbol or special symbol for grammar FSTs.
      if (ilabel_info[j].size() == 1) {
        // disambiguation symbol.
        int32 disambig_sym_left = next_disambig_sym++;
        disambig_syms_left->push_back(disambig_sym_left);
        fsts[j] = MakeTrivialAcceptor(disambig_sym_left);
      } else if (ilabel_info[j].size() == 2) {
        if (config.nonterm_phones_offset <= 0) {
          KALDI_ERR << "ilabel-info seems to be for grammar-FST.  You need to "
              "supply the --nonterm-phones-offset option.";
        }
        int32 nonterm_phones_offset = config.nonterm_phones_offset,
            nonterminal = -ilabel_info[j][0],
            left_context_phone = ilabel_info[j][1];
        if (nonterminal <= nonterm_phones_offset ||
            left_context_phone <= 0 ||
            left_context_phone > nonterm_phones_offset) {
          KALDI_ERR << "Could not interpret this ilabel-info with "
              "--nonterm-phones-offset=" << nonterm_phones_offset
                    << ": nonterminal,left-context-phone="
                    << nonterminal << ',' << left_context_phone;
        }
        int32 big_number = static_cast<int32>(fst::kNontermBigNumber),
            encoding_multiple = fst::GetEncodingMultiple(nonterm_phones_offset);
        int32 encoded_symbol = big_number + nonterminal * encoding_multiple +
            left_context_phone;
        fsts[j] = MakeTrivialAcceptor(encoded_symbol);
      } else {
        KALDI_ERR << "Could not decode this ilabel_info entry.";
      }
    } else {  // Real phone-in-context.
      std::vector<int32> phone_window = ilabel_info[j];

      std::shared_ptr<ExpandedFst<Arc>> fst = GetHmmAsFsa(phone_window,
                                                          ctx_dep,
                                                          trans_model,
                                                          config,
                                                          &cache);
      std::unique_ptr<ExpandedFst<Arc>> u_fst(fst->Copy());
      fsts[j] = std::move(u_fst);
    }
  }

  std::unique_ptr<VectorFst<Arc>> ans = MakeLoopFst(fsts);
  return ans;
}


void GetIlabelMapping (const std::vector<std::vector<int32> > &ilabel_info_old,
                       const ContextDependencyInterface &ctx_dep,
                       const Transitions &trans_model,
                       std::vector<int32> *old2new_map) {
  KALDI_ASSERT(old2new_map != NULL);

  /// The next variable maps from the (central-phone, pdf-sequence) to
  /// the index in ilabel_info_old corresponding to the first phone-in-context
  /// that we saw for it.  We use this to work
  /// out the logical-to-physical mapping.  Each time we handle a phone
  /// in context, we see if its (central-phone, pdf-sequence) has already
  /// been seen; if yes, we map to the original phone-sequence, if no,
  /// we create a new "phyiscal-HMM" and there is no mapping.
  std::map<std::pair<int32, std::vector<int32> >, int32 >
      pair_to_physical;

  int32 N = ctx_dep.ContextWidth(),
      P = ctx_dep.CentralPosition();
  int32 num_syms_old = ilabel_info_old.size();

  /// old2old_map is a map from the old ilabels to themselves (but
  /// duplicates are mapped to one unique one.
  std::vector<int32> old2old_map(num_syms_old);
  old2old_map[0] = 0;
  for (int32 i = 1; i < num_syms_old; i++) {
    const std::vector<int32> &vec = ilabel_info_old[i];
    if (vec.size() == 1 && vec[0] <= 0) {  // disambig.
      old2old_map[i] = i;
    } else {
      KALDI_ASSERT(vec.size() == static_cast<size_t>(N));
      // work out the vector of context-dependent phone
      int32 central_phone = vec[P];
      int32 num_pdf_classes = trans_model.GetTopo().NumPdfClasses(central_phone);
      std::vector<int32> state_seq(num_pdf_classes);  // Indexed by pdf-class
      for (int32 pdf_class = 0; pdf_class < num_pdf_classes; pdf_class++) {
        if (!ctx_dep.Compute(vec, pdf_class, &(state_seq[pdf_class]))) {
          std::ostringstream ss;
          WriteIntegerVector(ss, false, vec);
          KALDI_ERR << "tree did not succeed in converting phone window "<<ss.str();
        }
      }
      std::pair<int32, std::vector<int32> > pr(central_phone, state_seq);
      std::map<std::pair<int32, std::vector<int32> >, int32 >::iterator iter
          = pair_to_physical.find(pr);
      if (iter == pair_to_physical.end()) {  // first time we saw something like this.
        pair_to_physical[pr] = i;
        old2old_map[i] = i;
      } else {  // seen it before.  look up in the map, the index we point to.
        old2old_map[i] = iter->second;
      }
    }
  }

  std::vector<bool> seen(num_syms_old, false);
  for (int32 i = 0; i < num_syms_old; i++)
    seen[old2old_map[i]] = true;

  // Now work out the elements of old2new_map corresponding to
  // things that are first in their equivalence class.  We're just
  // compacting the labels to those for which seen[i] == true.
  int32 cur_id = 0;
  old2new_map->resize(num_syms_old);
  for (int32 i = 0; i < num_syms_old; i++)
    if (seen[i])
      (*old2new_map)[i] = cur_id++;
  // fill in the other elements of old2new_map.
  for (int32 i = 0; i < num_syms_old; i++)
    (*old2new_map)[i] = (*old2new_map)[old2old_map[i]];
}



std::unique_ptr<fst::VectorFst<fst::StdArc>>
GetPdfToTransitionIdTransducer(const Transitions &trans_model) {
  using namespace fst;
  std::unique_ptr<VectorFst<StdArc>> ans(new VectorFst<StdArc>);
  typedef VectorFst<StdArc>::Weight Weight;
  typedef StdArc Arc;
  ans->AddState();
  ans->SetStart(0);
  ans->SetFinal(0, Weight::One());
  for (int32 tid = 1; tid <= trans_model.NumTransitionIds(); tid++) {
    int32 pdf = trans_model.TransitionIdToPdfFast(tid);
    ans->AddArc(0, Arc(pdf+1, tid, Weight::One(), 0));  // note the offset of 1 on the pdfs.
    // it's because 0 is a valid pdf.
  }
  return ans;
}

struct TransitionState {
public:
  TransitionState(const Transitions::TransitionIdInfo& info):
    info(info) { }

  bool operator==(const TransitionState& other) const {
    return info.phone == other.info.phone &&
      info.topo_state == other.info.topo_state &&
      info.pdf_id == other.info.pdf_id;
  }

  bool operator!=(const TransitionState& other) const {
    return !(*this == other);
  }

  TransitionState& operator=(TransitionState other) {
// TODO: Fix this bizarre error when I uncomment this:
    // this->info = other.info;
    KALDI_ASSERT(false);
// hmm-utils.cc: In member function ‘kaldi::TransitionState& kaldi::TransitionState::operator=(kaldi::TransitionState)’:
// hmm-utils.cc:351:24: error: passing ‘const kaldi::Transitions::TransitionIdInfo’ as ‘this’ argument discards qualifiers [-fpermissive]
//      this->info = other.info;
//                         ^~~~
// In file included from ../hmm/hmm-utils.h:27:0,
//                  from hmm-utils.cc:25:
// ../hmm/transitions.h:107:10: note:   in call to ‘kaldi::Transitions::TransitionIdInfo& kaldi::Transitions::TransitionIdInfo::operator=(const kaldi::Transitions::TransitionIdInfo&)’

    return *this;
  }

  bool operator<(const TransitionState& other) const {
    return info < other.info;
  }

  const Transitions::TransitionIdInfo& info;
};

class TidToTstateMapper {
public:
  // Function object used in MakePrecedingInputSymbolsSameClass and
  // MakeFollowingInputSymbolsSameClass (as called by AddSelfLoopsReorder and
  // AddSelfLoopsNoReorder).  It maps transition-ids to transition-states (and
  // -1 to -1, 0 to 0 and disambiguation symbols to 0).  If check_no_self_loops
  // == true, it also checks that there are no self-loops in the graph (i.e. in
  // the labels it is called with).  This is just a convenient place to put this
  // check.

  // This maps valid transition-ids to transition states, maps kNoLabel to -1, and
  // maps all other symbols (i.e. epsilon symbols, disambig symbols, and symbols
  // with values over 100000/kNontermBigNumber) to zero.
  // Its point is to provide an equivalence class on labels that's relevant to what
  // the self-loop will be on the following (or preceding) state.

  // TransitionState no longer exists. It's basically a
  // TransitionIdInfo without the arc_index field.

  TidToTstateMapper(const Transitions &trans_model,
                    const std::vector<int32> &disambig_syms,
                    bool check_no_self_loops):
      trans_model_(trans_model),
      disambig_syms_(disambig_syms),
      check_no_self_loops_(check_no_self_loops) {
    KALDI_ASSERT((*this)(fst::kNoLabel) == NoLabelClass());
    KALDI_ASSERT((*this)(0) == ZeroClass());
}

  typedef TransitionState Result;
  static const Result& NoLabelClass() {
    // Take advantage of the fact that phone must be greater than or
    // equal to 1 to create a TransitionIdInfo which in practice will
    // never be created normally.

    // Use -1 for all other fields so we can easily see when debugging
    // whether we are using one of these invalid TransitionIdInfo
    // classes.
    static auto *no_label =
      new Transitions::TransitionIdInfo{.phone = -1, .topo_state = -1,
                                        .arc_index = -1, .pdf_id = -1,
                                        .self_loop_pdf_id = -1};
    static auto *no_label_state = new TransitionState(*no_label);
    return *no_label_state;
  }

  static const Result& ZeroClass() {
    static auto *zero_label =
      new Transitions::TransitionIdInfo{.phone = 0, .topo_state = -1, .arc_index = -1,
                                        .pdf_id = -1, .self_loop_pdf_id = -1};
    static auto *zero_label_state = new TransitionState(*zero_label);
    return *zero_label_state;
  }

  Result operator() (int32 tid) const {
    if (tid == static_cast<int32>(fst::kNoLabel)) return NoLabelClass();  // -1 -> -1
    else if (tid >= 1 && tid <= trans_model_.NumTransitionIds()) {
      if (check_no_self_loops_ && trans_model_.InfoForTransitionId(tid).is_self_loop)
        KALDI_ERR << "AddSelfLoops: graph already has self-loops.";
      return TransitionState(trans_model_.InfoForTransitionId(tid));
    } else {  // 0 or (presumably) disambiguation symbol.  Map to zero
      int32 big_number = fst::kNontermBigNumber;  // 1000000
      if (tid != 0 && tid < big_number)
        KALDI_ASSERT(std::binary_search(disambig_syms_.begin(),
                                        disambig_syms_.end(),
                                        tid));  // or invalid tid
      return ZeroClass();
    }
  }

private:
  const Transitions &trans_model_;
  const std::vector<int32> &disambig_syms_;  // sorted.
  bool check_no_self_loops_;
};

// Returns true if the outgoing arcs of the state s sum to 1.0
template<typename FST>
static bool StateIsStochastic(FST fst, typename FST::StateId s) {
  using namespace fst;
  using Arc = typename FST::Arc;
  using Weight = typename Arc::Weight;
  Weight total_prob = Weight::Zero();
  for (MutableArcIterator<MutableFst<Arc> > aiter(fst, s);
       !aiter.Done();
       aiter.Next()) {
    total_prob = Plus(total_prob, aiter.Value());
  }
  return ApproxEqual(total_prob.Value(), Weight::One());
}

void AddSelfLoops(const Transitions &trans_model,
                  const std::vector<int32> &disambig_syms,
                  BaseFloat self_loop_scale,
                  bool check_no_self_loops,
                  fst::VectorFst<fst::StdArc> *fst) {
  KALDI_ASSERT(fst->Start() != fst::kNoStateId);
  using namespace fst;
  typedef StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  TidToTstateMapper f(trans_model, disambig_syms, check_no_self_loops);
  // Duplicate states as necessary so that each state will require at most one
  // self-loop to be added to it.  Approximately this means that if a
  // state has multiple different symbols on arcs entering it, it will be
  // duplicated, with one copy per incoming symbol.
  MakePrecedingInputSymbolsSameClass(fst, f);

  // use the following to keep track of the transition-state incoming
  // into each state. This works because each state now has only one
  // transition state coming into (because of
  // MakePrecedingInputSymbolsSameClass).
  std::vector<TransitionState> state_in(fst->NumStates(), f.NoLabelClass());

  // This first loop just works out the label into each state,
  // and converts the transitions in the graph from transition-states
  // to transition-ids.
  // state_in maps each state in the fst to its TransitionState

  for (StateIterator<VectorFst<Arc> > siter(*fst);
       !siter.Done();
       siter.Next()) {
    StateId s = siter.Value();
    for (MutableArcIterator<VectorFst<Arc> > aiter(fst, s);
         !aiter.Done();
         aiter.Next()) {
      const Arc& arc = aiter.Value();
      TransitionState trans_state = f(arc.ilabel);
      if (state_in[arc.nextstate] == f.NoLabelClass()) {
        state_in[arc.nextstate] = trans_state;
      } else {
        KALDI_ASSERT(state_in[arc.nextstate] == trans_state);
        // or probably an error in MakePrecedingInputSymbolsSame.
      }
    }
  }

  // The start state should have no incoming arcs (invariant of Topology)
  KALDI_ASSERT(state_in[fst->Start()] == f.ZeroClass());

  // The next loop looks at each graph state, adds the self-loop [if needed] and
  // multiples all the out-transitions' probs (and final-prob) by the
  // forward-prob for that state (which is one minus self-loop-prob).  We do it
  // like this to maintain stochasticity (i.e. rather than multiplying the arcs
  // with the corresponding labels on them by this probability).

  for (StateId s = 0; s < static_cast<StateId>(state_in.size()); s++) {
    const TransitionState& trans_state = state_in[s];
    if (trans_state != f.NoLabelClass() && trans_state != f.ZeroClass() &&
        trans_state.info.self_loop_pdf_id != -1) {
      // defined, and not eps or a disambiguation symbol or a
      // nonterminal-related symbol for grammar decoding, and has a
      // self-loop which needs to be added, while maintaining
      int32 self_loop_tid = trans_state.info.self_loop_transition_id;
      KALDI_ASSERT(self_loop_tid != 0 &&
                   "Can't have a self_loop_pdf_id without a self_loop_transition_id");
      // 1) Multiply all probabilities by "forward" probability.
      BaseFloat self_loop_log_prob =
        -trans_model.InfoForTransitionId(self_loop_tid).transition_cost;
      BaseFloat log_forward_prob = log(1.0 - exp(self_loop_log_prob));
      fst->SetFinal(s, Times(fst->Final(s), Weight(-log_forward_prob)));
      for (MutableArcIterator<MutableFst<Arc> > aiter(fst, s);
          !aiter.Done();
          aiter.Next()) {
        Arc arc = aiter.Value();
        arc.weight = Times(arc.weight, Weight(-log_forward_prob));
        aiter.SetValue(arc);
      }
      // 2) Add self-loop
      fst->AddArc(s, Arc(self_loop_tid, 0, Weight(-self_loop_log_prob), s));

    }
    KALDI_PARANOID_ASSERT(StateIsStochastic(fst, s));
  }
}

// SplitToPhonesInternal takes as input the "alignment" vector containing
// a sequence of transition-ids, and appends a single vector to
// "split_output" for each instance of a phone that occurs in the
// output.
// Returns true if the alignment passes some non-exhaustive consistency
// checks (if the input does not start at the start of a phone or does not
// end at the end of a phone, we should expect that false will be returned).

static bool SplitToPhonesInternal(const Transitions &trans_model,
                                  const std::vector<int32> &alignment,
                                  std::vector<std::vector<int32> > *split_output) {
  if (alignment.empty()) return true;  // nothing to split.
  std::vector<size_t> end_points;  // points at which phones end [in an
  // stl iterator sense, i.e. actually one past the last transition-id within
  // each phone]..

  bool was_ok = true;
  for (size_t i = 0; i < alignment.size(); i++) {
    int32 trans_id = alignment[i];
    if (trans_model.InfoForTransitionId(trans_id).is_final) {
      while (i+1 < alignment.size() &&
             trans_model.InfoForTransitionId(alignment[i+1]).is_self_loop) {
        KALDI_ASSERT(trans_model.InfoForTransitionId(alignment[i]) ==
                     trans_model.InfoForTransitionId(alignment[i+1]));
        i++;
      }
      end_points.push_back(i+1);
    } else if (i+1 == alignment.size()) {
      // need to have an end-point at the actual end.
      // but this is an error- should have been detected already.
      was_ok = false;
      end_points.push_back(i+1);
    } else {
      int32 this_phone = trans_model.InfoForTransitionId(trans_id).phone;
      int32 next_trans_id = alignment[i+1];
      int32 next_phone = trans_model.InfoForTransitionId(next_trans_id).phone;

      if (this_phone != next_phone){
        // The phone changed, but this is an error-- we should have detected this via the
        // is_final check.
        was_ok = false;
        end_points.push_back(i+1);
      }
    }
  }

  size_t cur_point = 0;
  for (int32 end_point: end_points) {
    split_output->push_back(std::vector<int32>());
    // The next if-statement checks if the initial trans-id at the
    // current end point is the initial-state of the current phone (a
    // cursory check that the alignment is plausible).
    int32 topo_state = trans_model.InfoForTransitionId(end_point).topo_state;
    if (topo_state != 0)
      was_ok = false;
    for (size_t j = cur_point; j < end_point; j++)
      split_output->back().push_back(alignment[j]);
    cur_point = end_point;
  }
  return was_ok;
}


bool SplitToPhones(const Transitions &trans_model,
                   const std::vector<int32> &alignment,
                   std::vector<std::vector<int32> > *split_alignment) {
  KALDI_ASSERT(split_alignment != NULL);
  split_alignment->clear();

  return SplitToPhonesInternal(trans_model, alignment, split_alignment);
}


/** This function is used internally inside ConvertAlignment;
    it converts the alignment for a single phone. 'new_phone_window'
    is the window of phones as required by the tree.
    The size of 'new_phone_alignment' is the length requested, which
    may not always equal 'old_phone_alignment' (in case the
    'subsample' value is not 1).
 */
static inline void ConvertAlignmentForPhone(
    const Transitions &old_trans_model,
    const Transitions &new_trans_model,
    const ContextDependencyInterface &new_ctx_dep,
    const std::vector<int32> &old_phone_alignment,
    const std::vector<int32> &new_phone_window,
    std::vector<int32> *new_phone_alignment) {
  KALDI_ASSERT(!old_phone_alignment.empty());
  int32 alignment_size = old_phone_alignment.size();
  static bool warned_topology = false;
  int32 P = new_ctx_dep.CentralPosition(),
      old_central_phone = old_trans_model.InfoForTransitionId(
          old_phone_alignment[0]).phone,
      new_central_phone = new_phone_window[P];
  const Topology &old_topo = old_trans_model.GetTopo(),
      &new_topo = new_trans_model.GetTopo();

  // TODO(galv): Do we need the transition costs to be the same? Right
  // now, I am assuming that we do, but it is unclear to me that we
  // really need this.
  bool topology_mismatch = !fst::Equal(old_topo.TopologyForPhone(old_central_phone),
                                       new_topo.TopologyForPhone(new_central_phone),
                                       0.0);
  if (topology_mismatch && !warned_topology) {
    warned_topology = true;
    KALDI_WARN << "Topology mismatch detected; automatically converting. "
               << "Won't warn again.";
  }
  bool length_mismatch =
      (new_phone_alignment->size() != old_phone_alignment.size());
  if (length_mismatch || topology_mismatch) {
    // We generate a random path from this FST, ignoring the
    // old alignment.
    GetRandomAlignmentForPhone(new_ctx_dep, new_trans_model,
                               new_phone_window, new_phone_alignment);
    return;
  }

  int32 new_num_pdf_classes = new_topo.NumPdfClasses(new_central_phone);
  std::vector<int32> pdf_ids(new_num_pdf_classes);  // Indexed by pdf-class
  for (int32 pdf_class = 0; pdf_class < new_num_pdf_classes; pdf_class++) {
    if (!new_ctx_dep.Compute(new_phone_window, pdf_class,
                             &(pdf_ids[pdf_class]))) {
      std::ostringstream ss;
      WriteIntegerVector(ss, false, new_phone_window);
      KALDI_ERR << "tree did not succeed in converting phone window "
                << ss.str();
    }
  }

  // the topologies and lengths match -> we can directly transfer
  // the alignment.
  for (int32 j = 0; j < alignment_size; j++) {
    int32 old_tid = old_phone_alignment[j];
    auto&& info = old_trans_model.InfoForTransitionId(old_tid);
    int32 old_forward_pdf_class = old_trans_model.PdfClassForTid(old_tid);
    int32 old_self_loop_pdf_class = old_trans_model.PdfClassForTid(info.self_loop_pdf_id);
    int32 new_forward_pdf_id = pdf_ids[old_forward_pdf_class];
    int32 new_self_loop_pdf_id = pdf_ids[old_self_loop_pdf_class];
    int32 new_tid =
      new_trans_model.TupleToTransitionId(new_central_phone, info.topo_state,
                                          info.arc_index, new_forward_pdf_id,
                                          new_self_loop_pdf_id);
    (*new_phone_alignment)[j] = new_tid;
  }
}



/**
   This function, called from ConvertAlignmentInternal(), works out suitable new
   lengths of phones in the case where subsample_factor != 1.  The input vectors
   'mapped_phones' and 'old_lengths' must be the same size-- the length of the
   phone sequence.  The 'topology' object and 'mapped_phones' are needed to
   work out the minimum length of each phone in the sequence.
   Returns false only if it could not assign lengths (because the topology was
   too long relative to the number of frames).

   @param topology [in]         The new phone lengths are computed with
                                regard to this topology
   @param mapped_phones [in]    The phones for which this function computes
                                new lengths
   @param old_lengths     [in]  The old lengths
   @param conversion_shift [in] This will normally equal subsample_factor - 1
                                but may be less than that if the 'repeat_frames'
                                option is true; it's used for generating
                                'frame-shifted' versions of alignments that
                                we will later interpolate. This helps us keep
                                the phone boundaries of the subsampled and
                                interpolated alignments the same as
                                the original alignment.
   @param subsample_factor [in] The frame subsampling factor... normally 1, but
                                might be > 1 if we're converting to a
                                reduced-frame-rate system.
   @param new_lengths [out]     The vector for storing new lengths.
*/
static bool ComputeNewPhoneLengths(const Topology &topology,
                                   const std::vector<int32> &mapped_phones,
                                   const std::vector<int32> &old_lengths,
                                   int32 conversion_shift,
                                   int32 subsample_factor,
                                   std::vector<int32> *new_lengths) {
  int32 phone_sequence_length = old_lengths.size();
  std::vector<int32> min_lengths(phone_sequence_length);
  new_lengths->resize(phone_sequence_length);
  for (int32 i = 0; i < phone_sequence_length; i++)
    min_lengths[i] = topology.MinLength(mapped_phones[i]);
  int32 cur_time_elapsed = 0;
  for (int32 i = 0; i < phone_sequence_length; i++) {
    // Note: the '+ subsample_factor - 1' here is needed so that
    // the subsampled alignments have the same length as features
    // subsampled with 'subsample-feats'.
    int32 subsampled_time =
        (cur_time_elapsed + conversion_shift) / subsample_factor;
    cur_time_elapsed += old_lengths[i];
    int32 next_subsampled_time =
        (cur_time_elapsed + conversion_shift) / subsample_factor;
    (*new_lengths)[i] = next_subsampled_time - subsampled_time;
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (int32 i = 0; i < phone_sequence_length; i++) {
      if ((*new_lengths)[i] < min_lengths[i]) {
        changed = true;
        // we need at least one extra frame.. just try to get one frame for now.
        // Get it from the left or the right, depending which one has the closest
        // availability of a 'spare' frame.
        int32 min_distance = std::numeric_limits<int32>::max(),
            best_other_phone_index = -1,
            cur_distance = 0;
        // first try to the left.
        for (int32 j = i - 1; j >= 0; j--) {
          if ((*new_lengths)[j] > min_lengths[j]) {
            min_distance = cur_distance;
            best_other_phone_index = j;
            break;
          } else {
            cur_distance += (*new_lengths)[j];
          }
        }
        // .. now to the right.
        cur_distance = 0;
        for (int32 j = i + 1; j < phone_sequence_length; j++) {
          if ((*new_lengths)[j] > min_lengths[j]) {
            if (cur_distance < min_distance) {
              min_distance = cur_distance;
              best_other_phone_index = j;
            }
            break;
          } else {
            cur_distance += (*new_lengths)[j];
          }
        }
        if (best_other_phone_index == -1)
          return false;
        // assign an extra frame to this phone...
        (*new_lengths)[i]++;
        // and borrow it from the place that we found.
        (*new_lengths)[best_other_phone_index]--;
      }
    }
  }
  return true;
}

/**
  This function is the same as 'ConvertAligment',
  but instead of the 'repeat_frames' option it supports the 'conversion_shift'
  option; see the documentation of ComputeNewPhoneLengths() for what
  'conversion_shift' is for.
*/

static bool ConvertAlignmentInternal(const Transitions &old_trans_model,
                      const Transitions &new_trans_model,
                      const ContextDependencyInterface &new_ctx_dep,
                      const std::vector<int32> &old_alignment,
                      int32 conversion_shift,
                      int32 subsample_factor,
                      const std::vector<int32> *phone_map,
                      std::vector<int32> *new_alignment) {
  KALDI_ASSERT(0 <= conversion_shift && conversion_shift < subsample_factor);
  KALDI_ASSERT(new_alignment != NULL);
  new_alignment->clear();
  new_alignment->reserve(old_alignment.size());
  std::vector<std::vector<int32> > old_split;  // split into phones.
  if (!SplitToPhones(old_trans_model, old_alignment, &old_split))
    return false;
  int32 phone_sequence_length = old_split.size();
  std::vector<int32> mapped_phones(phone_sequence_length);
  for (size_t i = 0; i < phone_sequence_length; i++) {
    KALDI_ASSERT(!old_split[i].empty());
    mapped_phones[i] = old_trans_model.InfoForTransitionId(old_split[i][0]).phone;
    if (phone_map != NULL) {  // Map the phone sequence.
      int32 sz = phone_map->size();
      if (mapped_phones[i] < 0 || mapped_phones[i] >= sz ||
          (*phone_map)[mapped_phones[i]] == -1)
        KALDI_ERR << "ConvertAlignment: could not map phone " << mapped_phones[i];
      mapped_phones[i] = (*phone_map)[mapped_phones[i]];
    }
  }

  // the sizes of each element of 'new_split' indicate the length of alignment
  // that we want for each phone in the new sequence.
  std::vector<std::vector<int32> > new_split(phone_sequence_length);
  if (subsample_factor == 1 &&
      old_trans_model.GetTopo() == new_trans_model.GetTopo()) {
    // we know the old phone lengths will be fine.
    for (size_t i = 0; i < phone_sequence_length; i++)
      new_split[i].resize(old_split[i].size());
  } else {
    // .. they may not be fine.
    std::vector<int32> old_lengths(phone_sequence_length), new_lengths;
    for (int32 i = 0; i < phone_sequence_length; i++)
      old_lengths[i] = old_split[i].size();
    if (!ComputeNewPhoneLengths(new_trans_model.GetTopo(),
                                mapped_phones, old_lengths, conversion_shift,
                                subsample_factor, &new_lengths)) {
      KALDI_WARN << "Failed to produce suitable phone lengths";
      return false;
    }
    for (int32 i = 0; i < phone_sequence_length; i++)
      new_split[i].resize(new_lengths[i]);
  }

  int32 N = new_ctx_dep.ContextWidth(),
      P = new_ctx_dep.CentralPosition();

  // by starting at -N and going to phone_sequence_length + N, we're
  // being generous and not bothering to work out the exact
  // array bounds.
  for (int32 win_start = -N;
       win_start < static_cast<int32>(phone_sequence_length + N);
       win_start++) {  // start of a context window.
    int32 central_pos = win_start + P;
    if (static_cast<size_t>(central_pos) < phone_sequence_length) {
      // i.e. if (central_pos >= 0 && central_pos < phone_sequence_length)
      std::vector<int32> new_phone_window(N, 0);
      for (int32 offset = 0; offset < N; offset++)
        if (static_cast<size_t>(win_start+offset) < phone_sequence_length)
          new_phone_window[offset] = mapped_phones[win_start+offset];
      const std::vector<int32> &old_alignment_for_phone = old_split[central_pos];
      std::vector<int32> &new_alignment_for_phone = new_split[central_pos];

      ConvertAlignmentForPhone(old_trans_model, new_trans_model, new_ctx_dep,
                               old_alignment_for_phone, new_phone_window,
                               &new_alignment_for_phone);
      new_alignment->insert(new_alignment->end(),
                            new_alignment_for_phone.begin(),
                            new_alignment_for_phone.end());
    }
  }
  KALDI_ASSERT(new_alignment->size() ==
               (old_alignment.size() + conversion_shift)/subsample_factor);
  return true;
}

bool ConvertAlignment(const Transitions &old_trans_model,
                      const Transitions &new_trans_model,
                      const ContextDependencyInterface &new_ctx_dep,
                      const std::vector<int32> &old_alignment,
                      int32 subsample_factor,
                      bool repeat_frames,
                      const std::vector<int32> *phone_map,
                      std::vector<int32> *new_alignment) {
  if (subsample_factor == 1) {
    if (repeat_frames) {
      KALDI_WARN << "repeat_frames being set to true has no effect when "
        "subsample_factor=1 (its default value)";
    }
    return ConvertAlignmentInternal(old_trans_model,
                                    new_trans_model,
                                    new_ctx_dep,
                                    old_alignment,
                                    subsample_factor - 1, // == 0
                                    subsample_factor,
                                    phone_map,
                                    new_alignment);
   // The value "subsample_factor - 1" for conversion_shift above ensures the
   // alignments have the same length as the output of 'subsample-feats'
  } else {
    // either repeat_frames or subsample_factor >= 2. But if repeat_frames == True
    // then and subsample_factor == 1, then it is the same as the above.
    std::vector<std::vector<int32> > shifted_alignments(subsample_factor);
    // We create alignments for all shifts from [subsample_factor -1
    // to 0], inclusive.
    for (int32 conversion_shift = subsample_factor - 1;
         conversion_shift >= 0; conversion_shift--) {
      if (!ConvertAlignmentInternal(old_trans_model,
                                    new_trans_model,
                                    new_ctx_dep,
                                    old_alignment,
                                    conversion_shift, // conversion_shift
                                    subsample_factor,
                                    phone_map,
                                    &shifted_alignments[conversion_shift]))
        return false;
    }
    KALDI_ASSERT(new_alignment != NULL);
    new_alignment->clear();
    new_alignment->reserve(old_alignment.size());
    int32 max_shifted_ali_length = (old_alignment.size() / subsample_factor)
                                   + (old_alignment.size() % subsample_factor);
    for (int32 i = 0; i < max_shifted_ali_length; i++)
      for (int32 conversion_shift = subsample_factor - 1;
           conversion_shift >= 0; conversion_shift--)
        if (i < static_cast<int32>(shifted_alignments[conversion_shift].size()))
          new_alignment->push_back(shifted_alignments[conversion_shift][i]);
  }
  KALDI_ASSERT(new_alignment->size() == old_alignment.size());
  return true;
}

void AddTransitionProbs(const Transitions &trans_model,
                        const std::vector<int32> &disambig_syms,  // may be empty
                        BaseFloat transition_scale,
                        fst::VectorFst<fst::StdArc> *fst) {
  using namespace fst;
  KALDI_ASSERT(IsSortedAndUniq(disambig_syms));
  int num_tids = trans_model.NumTransitionIds();
  for (StateIterator<VectorFst<StdArc> > siter(*fst);
      !siter.Done();
      siter.Next()) {
    for (MutableArcIterator<VectorFst<StdArc> > aiter(fst, siter.Value());
         !aiter.Done();
         aiter.Next()) {
      StdArc arc = aiter.Value();
      StdArc::Label l = arc.ilabel;
      if (l >= 1 && l <= num_tids) {  // a transition-id.
        BaseFloat scaled_log_prob =
          trans_model.InfoForTransitionId(l).transition_cost * transition_scale;
        arc.weight = Times(arc.weight, TropicalWeight(-scaled_log_prob));
      } else if (l != 0 && !std::binary_search(disambig_syms.begin(),
                                               disambig_syms.end(),l)) {
        KALDI_ERR << "AddTransitionProbs: invalid symbol " << arc.ilabel
                  << " on graph input side.";
      }
      aiter.SetValue(arc);
    }
  }
}

void AddTransitionProbs(const Transitions &trans_model,
                        BaseFloat transition_scale,
                        Lattice *lat) {
  using namespace fst;
  int num_tids = trans_model.NumTransitionIds();
  for (fst::StateIterator<Lattice> siter(*lat);
       !siter.Done();
       siter.Next()) {
    for (MutableArcIterator<Lattice> aiter(lat, siter.Value());
         !aiter.Done();
         aiter.Next()) {
      LatticeArc arc = aiter.Value();
      LatticeArc::Label l = arc.ilabel;
      if (l >= 1 && l <= num_tids) {  // a transition-id.
        BaseFloat scaled_log_prob =
          trans_model.InfoForTransitionId(l).transition_cost * transition_scale;
        // cost is negated log prob.
        arc.weight.SetValue1(arc.weight.Value1() - scaled_log_prob);
      } else if (l != 0) {
        KALDI_ERR << "AddTransitionProbs: invalid symbol " << arc.ilabel
                  << " on lattice input side.";
      }
      aiter.SetValue(arc);
    }
  }
}


// This function takes a phone-sequence with word-start and word-end
// tokens in it, and a word-sequence, and outputs the pronunciations
// "prons"... the format of "prons" is, each element is a vector,
// where the first element is the word (or zero meaning no word, e.g.
// for optional silence introduced by the lexicon), and the remaining
// elements are the phones in the word's pronunciation.
// It returns false if it encounters a problem of some kind, e.g.
// if the phone-sequence doesn't seem to have the right number of
// words in it.
bool ConvertPhnxToProns(const std::vector<int32> &phnx,
                        const std::vector<int32> &words,
                        int32 word_start_sym,
                        int32 word_end_sym,
                        std::vector<std::vector<int32> > *prons) {
  size_t i = 0, j = 0;

  while (i < phnx.size()) {
    if (phnx[i] == 0) return false; // zeros not valid here.
    if (phnx[i] == word_start_sym) { // start a word...
      std::vector<int32> pron;
      if (j >= words.size()) return false; // no word left..
      if (words[j] == 0) return false; // zero word disallowed.
      pron.push_back(words[j++]);
      i++;
      while (i < phnx.size()) {
        if (phnx[i] == 0) return false;
        if (phnx[i] == word_start_sym) return false; // error.
        if (phnx[i] == word_end_sym) { i++; break; }
        pron.push_back(phnx[i]);
        i++;
      }
      // check we did see the word-end symbol.
      if (!(i > 0 && phnx[i-1] == word_end_sym))
        return false;
      prons->push_back(pron);
    } else if (phnx[i] == word_end_sym) {
      return false;  // error.
    } else {
      // start a non-word sequence of phones (e.g. opt-sil).
      std::vector<int32> pron;
      pron.push_back(0); // 0 serves as the word-id.
      while (i < phnx.size()) {
        if (phnx[i] == 0) return false;
        if (phnx[i] == word_start_sym) break;
        if (phnx[i] == word_end_sym) return false; // error.
        pron.push_back(phnx[i]);
        i++;
      }
      prons->push_back(pron);
    }
  }
  return (j == words.size());
}


void GetRandomAlignmentForPhone(const ContextDependencyInterface &ctx_dep,
                                const Transitions &trans_model,
                                const std::vector<int32> &phone_window,
                                std::vector<int32> *alignment) {
  typedef fst::StdArc Arc;
  int32 length = alignment->size();
  BaseFloat prob_scale = 0.0;
  fst::VectorFst<Arc> fst = GetHmmAsFsaSimple(phone_window, ctx_dep,
                                              trans_model, prob_scale);
  fst::RmEpsilon(&fst);

  fst::VectorFst<Arc> length_constraint_fst;
  {  // set up length_constraint_fst.
    std::vector<int32> symbols;
    bool include_epsilon = false;
    // note: 'fst' is an acceptor so ilabels == olabels.
    GetInputSymbols(fst, include_epsilon, &symbols);
    int32 cur_state = length_constraint_fst.AddState();
    length_constraint_fst.SetStart(cur_state);
    for (int32 i = 0; i < length; i++) {
      int32 next_state = length_constraint_fst.AddState();
      for (size_t j = 0; j < symbols.size(); j++) {
        length_constraint_fst.AddArc(cur_state,
                                     Arc(symbols[j], symbols[j],
                                         fst::TropicalWeight::One(),
                                         next_state));
      }
      cur_state = next_state;
    }
    length_constraint_fst.SetFinal(cur_state, fst::TropicalWeight::One());
  }
  fst::VectorFst<Arc> composed_fst;
  fst::Compose(fst, length_constraint_fst, &composed_fst);
  fst::VectorFst<Arc> single_path_fst;
  {  // randomly generate a single path.
    fst::UniformArcSelector<Arc> selector;
    fst::RandGenOptions<fst::UniformArcSelector<Arc> > randgen_opts(selector);
    fst::RandGen(composed_fst, &single_path_fst, randgen_opts);
  }
  if (single_path_fst.NumStates() == 0) {
    KALDI_ERR << "Error generating random alignment (wrong length?): "
              << "requested length is " << length << " versus min-length "
              << trans_model.GetTopo().MinLength(
                  phone_window[ctx_dep.CentralPosition()]);
  }
  std::vector<int32> symbol_sequence;
  bool ans = fst::GetLinearSymbolSequence<Arc, int32>(
      single_path_fst, &symbol_sequence, NULL, NULL);
  KALDI_ASSERT(ans && symbol_sequence.size() == length);
  symbol_sequence.swap(*alignment);
}

} // namespace kaldi
