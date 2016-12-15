// chain/chain-den-graph.cc

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


#include "chain/chain-den-graph.h"
#include "hmm/hmm-utils.h"
#include "fstext/push-special.h"

namespace kaldi {
namespace chain {


DenominatorGraph::DenominatorGraph(const fst::StdVectorFst &fst,
                                   int32 num_pdfs):
    num_pdfs_(num_pdfs) {
  SetTransitions(fst, num_pdfs);
  SetInitialProbs(fst);
}

const Int32Pair* DenominatorGraph::BackwardTransitions() const {
  return backward_transitions_.Data();
}

const Int32Pair* DenominatorGraph::ForwardTransitions() const {
  return forward_transitions_.Data();
}

const DenominatorGraphTransition* DenominatorGraph::Transitions() const {
  return transitions_.Data();
}

const CuVector<BaseFloat>& DenominatorGraph::InitialProbs() const {
  return initial_probs_;
}

void DenominatorGraph::SetTransitions(const fst::StdVectorFst &fst,
                                      int32 num_pdfs) {
  int32 num_states = fst.NumStates();

  std::vector<std::vector<DenominatorGraphTransition> >
      transitions_out(num_states),
      transitions_in(num_states);
  for (int32 s = 0; s < num_states; s++) {
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, s); !aiter.Done();
         aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      DenominatorGraphTransition transition;
      transition.transition_prob = exp(-arc.weight.Value());
      transition.pdf_id = arc.ilabel - 1;
      transition.hmm_state = arc.nextstate;
      KALDI_ASSERT(transition.pdf_id >= 0 && transition.pdf_id < num_pdfs);
      transitions_out[s].push_back(transition);
      // now the reverse transition.
      transition.hmm_state = s;
      transitions_in[arc.nextstate].push_back(transition);
    }
  }

  std::vector<Int32Pair> forward_transitions(num_states);
  std::vector<Int32Pair> backward_transitions(num_states);
  std::vector<DenominatorGraphTransition> transitions;

  for (int32 s = 0; s < num_states; s++) {
    forward_transitions[s].first = static_cast<int32>(transitions.size());
    transitions.insert(transitions.end(), transitions_out[s].begin(),
                       transitions_out[s].end());
    forward_transitions[s].second = static_cast<int32>(transitions.size());
  }
  for (int32 s = 0; s < num_states; s++) {
    backward_transitions[s].first = static_cast<int32>(transitions.size());
    transitions.insert(transitions.end(), transitions_in[s].begin(),
                       transitions_in[s].end());
    backward_transitions[s].second = static_cast<int32>(transitions.size());
  }

  forward_transitions_ = forward_transitions;
  backward_transitions_ = backward_transitions;
  transitions_ = transitions;
}

void DenominatorGraph::SetInitialProbs(const fst::StdVectorFst &fst) {
  // we set only the start-state to have probability mass, and then 100
  // iterations of HMM propagation, over which we average the probabilities.
  // initial probs won't end up making a huge difference as we won't be using
  // derivatives from the first few frames, so this isn't 100% critical.
  int32 num_iters = 100;
  int32 num_states = fst.NumStates();

  // we normalize each state so that it sums to one (including
  // final-probs)... this is needed because the 'chain' code doesn't
  // have transition probabilities.
  Vector<double> normalizing_factor(num_states);
  for (int32 s = 0; s < num_states; s++) {
    double tot_prob = exp(-fst.Final(s).Value());
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, s); !aiter.Done();
         aiter.Next()) {
      tot_prob += exp(-aiter.Value().weight.Value());
    }
    KALDI_ASSERT(tot_prob > 0.0 && tot_prob < 100.0);
    normalizing_factor(s) = 1.0 / tot_prob;
  }

  Vector<double> cur_prob(num_states), next_prob(num_states),
      avg_prob(num_states);
  cur_prob(fst.Start()) = 1.0;
  for (int32 iter = 0; iter < num_iters; iter++) {
    avg_prob.AddVec(1.0 / num_iters, cur_prob);
    for (int32 s = 0; s < num_states; s++) {
      double prob = cur_prob(s) * normalizing_factor(s);

      for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, s); !aiter.Done();
           aiter.Next()) {
        const fst::StdArc &arc = aiter.Value();
        next_prob(arc.nextstate) += prob * exp(-arc.weight.Value());
      }
    }
    cur_prob.Swap(&next_prob);
    next_prob.SetZero();
    // Renormalize, beause the HMM won't sum to one even after the
    // previous normalization (due to final-probs).
    cur_prob.Scale(1.0 / cur_prob.Sum());
  }

  Vector<BaseFloat> avg_prob_float(avg_prob);
  initial_probs_ = avg_prob_float;
}

void DenominatorGraph::GetNormalizationFst(const fst::StdVectorFst &ifst,
                                           fst::StdVectorFst *ofst) {
  KALDI_ASSERT(ifst.NumStates() == initial_probs_.Dim());
  if (&ifst != ofst)
    *ofst = ifst;
  int32 new_initial_state = ofst->AddState();
  Vector<BaseFloat> initial_probs(initial_probs_);

  for (int32 s = 0; s < initial_probs_.Dim(); s++) {
    BaseFloat initial_prob = initial_probs(s);
    KALDI_ASSERT(initial_prob > 0.0);
    fst::StdArc arc(0, 0, fst::TropicalWeight(-log(initial_prob)), s);
    ofst->AddArc(new_initial_state, arc);
    ofst->SetFinal(s, fst::TropicalWeight::One());
  }
  ofst->SetStart(new_initial_state);
  fst::RmEpsilon(ofst);
  fst::ArcSort(ofst, fst::ILabelCompare<fst::StdArc>());
}


void MapFstToPdfIdsPlusOne(const TransitionModel &trans_model,
                           fst::StdVectorFst *fst) {
  int32 num_states = fst->NumStates();
  for (int32 s = 0; s < num_states; s++) {
    for (fst::MutableArcIterator<fst::StdVectorFst> aiter(fst, s);
         !aiter.Done(); aiter.Next()) {
      fst::StdArc arc = aiter.Value();
      KALDI_ASSERT(arc.ilabel == arc.olabel);
      if (arc.ilabel > 0) {
        arc.ilabel = trans_model.TransitionIdToPdf(arc.ilabel) + 1;
        arc.olabel = arc.ilabel;
        aiter.SetValue(arc);
      }
    }
  }
}

void MinimizeAcceptorNoPush(fst::StdVectorFst *fst) {
  BaseFloat delta = fst::kDelta * 10.0;  // use fairly loose delta for
                                         // aggressive minimimization.
  fst::ArcMap(fst, fst::QuantizeMapper<fst::StdArc>(delta));
  fst::EncodeMapper<fst::StdArc> encoder(fst::kEncodeLabels | fst::kEncodeWeights,
                                         fst::ENCODE);
  fst::Encode(fst, &encoder);
  fst::AcceptorMinimize(fst);
  fst::Decode(fst, encoder);
}

// This static function, used in CreateDenominatorFst, sorts an
// fst's states in decreasing order of number of transitions (into + out of)
// the state.  The aim is to have states that have a lot of transitions
// either into them or out of them, be numbered earlier, so hopefully
// they will be scheduled first and won't delay the computation
static void SortOnTransitionCount(fst::StdVectorFst *fst) {
  // negative_num_transitions[i] will contain (before sorting), the pair
  // ( -(num-transitions-into(i) + num-transition-out-of(i)), i)
  int32 num_states = fst->NumStates();
  std::vector<std::pair<int32, int32> > negative_num_transitions(num_states);
  for (int32 i = 0; i < num_states; i++) {
    negative_num_transitions[i].first = 0;
    negative_num_transitions[i].second = i;
  }
  for (int32 i = 0; i < num_states; i++) {
    for (fst::ArcIterator<fst::StdVectorFst> aiter(*fst, i); !aiter.Done();
         aiter.Next()) {
      negative_num_transitions[i].first--;
      negative_num_transitions[aiter.Value().nextstate].first--;
    }
  }
  std::sort(negative_num_transitions.begin(), negative_num_transitions.end());
  std::vector<fst::StdArc::StateId> order(num_states);
  for (int32 i = 0; i < num_states; i++)
    order[negative_num_transitions[i].second] = i;
  fst::StateSort(fst, order);
}

void DenGraphMinimizeWrapper(fst::StdVectorFst *fst) {
  for (int32 i = 1; i <= 3; i++) {
    fst::StdVectorFst fst_reversed;
    fst::Reverse(*fst, &fst_reversed);
    fst::PushSpecial(&fst_reversed, fst::kDelta * 0.01);
    MinimizeAcceptorNoPush(&fst_reversed);
    fst::Reverse(fst_reversed, fst);
    KALDI_LOG << "Number of states and arcs in transition-id FST after reversed "
              << "minimization is " << fst->NumStates() << " and "
              << NumArcs(*fst) << " (pass " << i << ")";
    fst::PushSpecial(fst, fst::kDelta * 0.01);
    MinimizeAcceptorNoPush(fst);
    KALDI_LOG << "Number of states and arcs in transition-id FST after regular "
              << "minimization is " << fst->NumStates() << " and "
              << NumArcs(*fst) << " (pass " << i << ")";
  }
  fst::RmEpsilon(fst);
  KALDI_LOG << "Number of states and arcs in transition-id FST after "
            << "removing any epsilons introduced by reversal is "
            << fst->NumStates() << " and "
            << NumArcs(*fst);
  fst::PushSpecial(fst, fst::kDelta * 0.01);
}


static void PrintDenGraphStats(const fst::StdVectorFst &den_graph) {
  int32 num_states = den_graph.NumStates();
  int32 degree_cutoff = 3;  // track states with <= transitions in/out.
  int32 num_states_low_degree_in = 0,
      num_states_low_degree_out = 0,
      tot_arcs = 0;
  std::vector<int32> num_in_arcs(num_states, 0);
  for (int32 s = 0; s < num_states; s++) {
    if (den_graph.NumArcs(s) <= degree_cutoff) {
      num_states_low_degree_out++;
    }
    tot_arcs += den_graph.NumArcs(s);
    for (fst::ArcIterator<fst::StdVectorFst> aiter(den_graph, s);
         !aiter.Done(); aiter.Next()) {
      int32 dest_state = aiter.Value().nextstate;
      num_in_arcs[dest_state]++;
    }
  }
  for (int32 s = 0; s < num_states; s++) {
    if (num_in_arcs[s] <= degree_cutoff) {
      num_states_low_degree_in++;
    }
  }
  KALDI_LOG << "Number of states is " << num_states << " and arcs "
            << tot_arcs << "; number of states with in-degree <= "
            << degree_cutoff << " is " << num_states_low_degree_in
            << " and with out-degree <= " << degree_cutoff
            << " is " << num_states_low_degree_out;
}


// Check that every pdf is seen, warn if some are not.
static void CheckDenominatorFst(int32 num_pdfs,
                                const fst::StdVectorFst &den_fst) {
  std::vector<bool> pdf_seen(num_pdfs);
  int32 num_states = den_fst.NumStates();
  for (int32 s = 0; s < num_states; s++) {
    for (fst::ArcIterator<fst::StdVectorFst> aiter(den_fst, s);
         !aiter.Done(); aiter.Next()) {
      int32 pdf_id = aiter.Value().ilabel - 1;
      KALDI_ASSERT(pdf_id >= 0 && pdf_id < num_pdfs);
      pdf_seen[pdf_id] = true;
    }
  }
  for (int32 pdf = 0; pdf < num_pdfs; pdf++) {
    if (!pdf_seen[pdf]) {
      KALDI_WARN << "Pdf-id " << pdf << " is not seen in denominator graph.";
    }
  }
}

void CreateDenominatorFst(const ContextDependency &ctx_dep,
                          const TransitionModel &trans_model,
                          const fst::StdVectorFst &phone_lm_in,
                          fst::StdVectorFst *den_fst) {
  using fst::StdVectorFst;
  using fst::StdArc;
  KALDI_ASSERT(phone_lm_in.NumStates() != 0);
  fst::StdVectorFst phone_lm(phone_lm_in);

  KALDI_LOG << "Number of states and arcs in phone-LM FST is "
            << phone_lm.NumStates() << " and " << NumArcs(phone_lm);

  int32 subsequential_symbol = trans_model.GetPhones().back() + 1;
  if (ctx_dep.CentralPosition() != ctx_dep.ContextWidth() - 1) {
    // note: this function only adds the subseq symbol to the input of what was
    // previously an acceptor, so we project, i.e. copy the ilabels to the
    // olabels
    AddSubsequentialLoop(subsequential_symbol, &phone_lm);
    fst::Project(&phone_lm, fst::PROJECT_INPUT);
  }
  std::vector<int32> disambig_syms;  // empty list of diambiguation symbols.
  fst::ContextFst<StdArc> cfst(subsequential_symbol, trans_model.GetPhones(),
                               disambig_syms, ctx_dep.ContextWidth(),
                               ctx_dep.CentralPosition());
  StdVectorFst context_dep_lm;
  fst::ComposeContextFst(cfst, phone_lm, &context_dep_lm);
  // at this point, context_dep_lm will have indexes into 'ilabels' as its
  // input symbol (representing context-dependent phones), and phones on its
  // output.  We don't need the phones, so we'll project.
  fst::Project(&context_dep_lm, fst::PROJECT_INPUT);

  KALDI_LOG << "Number of states and arcs in context-dependent LM FST is "
            << context_dep_lm.NumStates() << " and " << NumArcs(context_dep_lm);

  std::vector<int32> disambig_syms_h; // disambiguation symbols on input side
  // of H -- will be empty.
  HTransducerConfig h_config;
  // the default is 1, but just document that we want this to stay as one.
  // we'll use the same value in test time.  Consistency is the key here.
  h_config.transition_scale = 1.0;

  StdVectorFst *h_fst = GetHTransducer(cfst.ILabelInfo(),
                                       ctx_dep,
                                       trans_model,
                                       h_config,
                                       &disambig_syms_h);
  KALDI_ASSERT(disambig_syms_h.empty());
  StdVectorFst transition_id_fst;
  TableCompose(*h_fst, context_dep_lm, &transition_id_fst);
  delete h_fst;

  BaseFloat self_loop_scale = 1.0;  // We have to be careful to use the same
                                    // value in test time.
  bool reorder = true;
  // add self-loops to the FST with transition-ids as its labels.
  AddSelfLoops(trans_model, disambig_syms_h, self_loop_scale, reorder,
               &transition_id_fst);
  // at this point transition_id_fst will have transition-ids as its ilabels and
  // context-dependent phones (indexes into ILabelInfo()) as its olabels.
  // Discard the context-dependent phones by projecting on the input, keeping
  // only the transition-ids.
  fst::Project(&transition_id_fst, fst::PROJECT_INPUT);

  MapFstToPdfIdsPlusOne(trans_model, &transition_id_fst);
  KALDI_LOG << "Number of states and arcs in transition-id FST is "
            << transition_id_fst.NumStates() << " and "
            << NumArcs(transition_id_fst);

  // RemoveEpsLocal doesn't remove all epsilons, but it keeps the graph small.
  fst::RemoveEpsLocal(&transition_id_fst);
  // If there are remaining epsilons, remove them.
  fst::RmEpsilon(&transition_id_fst);
  KALDI_LOG << "Number of states and arcs in transition-id FST after "
            << "removing epsilons is "
            << transition_id_fst.NumStates() << " and "
            << NumArcs(transition_id_fst);

  DenGraphMinimizeWrapper(&transition_id_fst);

  SortOnTransitionCount(&transition_id_fst);

  *den_fst = transition_id_fst;
  CheckDenominatorFst(trans_model.NumPdfs(), *den_fst);
  PrintDenGraphStats(*den_fst);
}


int32 DenominatorGraph::NumStates() const {
  return forward_transitions_.Dim();
}
}  // namespace chain
}  // namespace kaldi
