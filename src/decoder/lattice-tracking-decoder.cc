// decoder/lattice-tracking-decoder.cc

// Copyright 2012  BUT (Author: Mirko Hannemann)
//                 Johns Hopkins University (Author: Daniel Povey)
//           2014  Guoguo Chen

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

#include "decoder/lattice-tracking-decoder.h"

namespace kaldi {

// instantiate this class once for each thing you have to decode.
LatticeTrackingDecoder::LatticeTrackingDecoder(const fst::Fst<fst::StdArc> &fst,
                                           const LatticeTrackingDecoderConfig &config):
    fst_(fst), config_(config), num_toks_(0) {
  config.Check();
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
}


// Returns true if any kind of traceback is available (not necessarily from
// a final state).
bool LatticeTrackingDecoder::Decode(DecodableInterface *decodable,
                                    const fst::StdVectorFst &arc_graph) {
  arc_graph_ = &arc_graph;
  // clean up from last time:
  ClearToks(toks_.Clear());
  cost_offsets_.clear();
  ClearActiveTokens();
  warned_ = false;
  final_active_ = false;
  final_costs_.clear();
  num_toks_ = 0;
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  active_toks_.resize(1);
  // the initial token will be tracked and starts the arc_graph
  Token *start_tok = new Token(0.0, 0.0, NULL, NULL, arc_graph.Start());
  active_toks_[0].toks = start_tok;
  toks_.Insert(start_state, start_tok);
  num_toks_++;
  ProcessNonemitting(0);
    
  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.
  for (int32 frame = 1; !decodable->IsLastFrame(frame-2); frame++) {
    active_toks_.resize(frame+1); // new column

    ProcessEmitting(decodable, frame);
      
    ProcessNonemitting(frame);

    if (decodable->IsLastFrame(frame-1))
      PruneActiveTokensFinal(frame);
    else if (frame % config_.prune_interval == 0)
      PruneActiveTokens(frame, config_.lattice_beam * 0.1); // use larger delta.        
  }
  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !final_costs_.empty();
}

// Outputs an FST corresponding to the single best path
// through the lattice.
bool LatticeTrackingDecoder::GetBestPath(fst::MutableFst<LatticeArc> *ofst) const {
  fst::VectorFst<LatticeArc> fst;
  if (!GetRawLattice(&fst)) return false;
  // std::cout << "Raw lattice is:\n";
  // fst::FstPrinter<LatticeArc> fstprinter(fst, NULL, NULL, NULL, false, true);
  // fstprinter.Print(&std::cout, "standard output");
  ShortestPath(fst, ofst);
  return true;
}

// Outputs an FST corresponding to the raw, state-level
// tracebacks.
bool LatticeTrackingDecoder::GetRawLattice(fst::MutableFst<LatticeArc> *ofst) const {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;
  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  int32 num_frames = active_toks_.size() - 1;
  KALDI_ASSERT(num_frames > 0);
  unordered_map<Token*, StateId> tok_map(num_toks_/2 + 3); // bucket count
  // First create all states.
  for (int32 f = 0; f <= num_frames; f++) {
    if (active_toks_[f].toks == NULL) {
      KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next)
      tok_map[tok] = ofst->AddState();
    // The next statement sets the start state of the output FST.
    // Because we always add new states to the head of the list
    // active_toks_[f].toks, and the start state was the first one
    // added, it will be the last one added to ofst.
    if (f == 0 && ofst->NumStates() > 0)
      ofst->SetStart(ofst->NumStates()-1);
  }
  KALDI_VLOG(3) << "init:" << num_toks_/2 + 3 << " buckets:"
                << tok_map.bucket_count() << " load:" << tok_map.load_factor()
                << " max:" << tok_map.max_load_factor();
  // Now create all arcs.
  StateId cur_state = 0; // we rely on the fact that we numbered these
  // consecutively (AddState() returns the numbers in order..)
  for (int32 f = 0; f <= num_frames; f++) {
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next,
             cur_state++) {
      for (ForwardLink *l = tok->links;
           l != NULL;
           l = l->next) {
        unordered_map<Token*, StateId>::const_iterator iter =
            tok_map.find(l->next_tok);
        StateId nextstate = iter->second;
        KALDI_ASSERT(iter != tok_map.end());
        BaseFloat cost_offset = 0.0;
        if (l->ilabel != 0) { // emitting..
          KALDI_ASSERT(f >= 0 && f < cost_offsets_.size());
          cost_offset = cost_offsets_[f];
        }
        Arc arc(l->ilabel, l->olabel,
                Weight(l->graph_cost, l->acoustic_cost - cost_offset),
                nextstate);
        ofst->AddArc(cur_state, arc);
      }
      if (f == num_frames) {
        std::map<Token*, BaseFloat>::const_iterator iter =
            final_costs_.find(tok);
        if (iter != final_costs_.end())
          ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
      }
    }
  }
  KALDI_ASSERT(cur_state == ofst->NumStates());
  return (cur_state != 0);
}

// This function is now deprecated, since now we do determinization from outside
// the LatticeTrackingDecoder class.
// Outputs an FST corresponding to the lattice-determinized
// lattice (one path per word sequence).
bool LatticeTrackingDecoder::GetLattice(fst::MutableFst<CompactLatticeArc> *ofst) const {
  Lattice raw_fst;
  if (!GetRawLattice(&raw_fst)) return false;
  Invert(&raw_fst); // make it so word labels are on the input.
  if (!TopSort(&raw_fst)) // topological sort makes lattice-determinization more efficient
    KALDI_WARN << "Topological sorting of state-level lattice failed "
        "(probably your lexicon has empty words or your LM has epsilon cycles; this "
        " is a bad idea.)";
  // (in phase where we get backward-costs).
  fst::ILabelCompare<LatticeArc> ilabel_comp;
  ArcSort(&raw_fst, ilabel_comp); // sort on ilabel; makes
  // lattice-determinization more efficient.
    
  fst::DeterminizeLatticePrunedOptions lat_opts;
  lat_opts.max_mem = config_.det_opts.max_mem;
    
  DeterminizeLatticePruned(raw_fst, config_.lattice_beam, ofst, lat_opts);
  raw_fst.DeleteStates(); // Free memory-- raw_fst no longer needed.
  Connect(ofst); // Remove unreachable states... there might be
  // a small number of these, in some cases.
  return true;
}

void LatticeTrackingDecoder::PossiblyResizeHash(size_t num_toks) {
  size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                      * config_.hash_ratio);
  if (new_sz > toks_.Size()) {
    toks_.SetSize(new_sz);
  }
}

// FindOrAddToken either locates a token in hash of toks_,
// or if necessary inserts a new, empty token (i.e. with no forward links)
// for the current frame.  [note: it's inserted if necessary into hash toks_
// and also into the singly linked list of tokens active on this frame
// (whose head is at active_toks_[frame]).
// Returns the Token pointer.  Sets "changed" (if non-NULL) to true
// if the token was newly created or the cost was changed,
// or when the token inherits the status "tracked"
// this will be needed when deciding whether to put it to the queue
// lat_state is the next state in the arc graph lattice
inline LatticeTrackingDecoder::Token *LatticeTrackingDecoder::FindOrAddToken(
    StateId state, StateId lat_state, int32 frame, BaseFloat tot_cost,
    bool *changed) { // "changed" also can be "newly_tracked"
    
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  KALDI_ASSERT(frame < active_toks_.size());
  Token *&toks = active_toks_[frame].toks;
  Elem *e_found = toks_.Find(state);
  if (e_found == NULL) { // no such token presently exists.
    const BaseFloat extra_cost = 0.0;
    // tokens on the currently final frame have zero extra_cost
    // as any of them could end up on the winning path.
    Token *new_tok = new Token (tot_cost, extra_cost, NULL, toks, lat_state);
    // NULL: no forward links yet
    toks = new_tok;
    num_toks_++;
    toks_.Insert(state, new_tok);
    //if (lat_state!=fst::kNoStateId) // newly tracked, but changed=true anyway
    if (changed) *changed = true; // new token means "changed"
    return new_tok;
  } else {
    Token *tok = e_found->val; // There is an existing Token for this state.
    if (tok->tot_cost > tot_cost) { // old cost was higher -> replace old token
      tok->tot_cost = tot_cost;
      // we don't allocate a new token, the old stays linked in active_toks_
      // we only replace the tot_cost
      // in the current frame, there are no forward links (and no extra_cost)
      // only in ProcessNonemitting we have to delete forward links
      // in case we visit a state for the second time
      // those forward links, that lead to this replaced token before:
      // they remain and will hopefully be pruned later (PruneForwardLinks...)
      if (changed) *changed = true;
      // we don't need to update the lat_state: if two "tracked" token meet
      // they meet at the same time in the same arc_graph state (lat_state)
      // all successor arcs will have the same HCLG state as ilabel
    } else {
      if (changed) *changed = false;
    }
    //          old new result
    // tracked? no  no  no
    //          no  yes yes (no matter which cost is better)
    //          yes no  yes (no matter which cost is better)
    //          yes yes yes
    // The "tracked" status can be turned on,
    // even if the token wasn't updated by the tracked token.
    // It can never be turned off, even if updated by a non-tracked token.
    if ((tok->lat_state == fst::kNoStateId) && // old token not yet "tracked"
        (lat_state != fst::kNoStateId)) {
      tok->lat_state = lat_state;
      if (changed) *changed = true; // newly tracked token -> put to queue
    }
    return tok;
  }
}
  
// prunes outgoing links for all tokens in active_toks_[frame]
// it's called by PruneActiveTokens
// all links, that have link_extra_cost > lattice_beam are pruned
void LatticeTrackingDecoder::PruneForwardLinks(
    int32 frame, bool *extra_costs_changed,
    bool *links_pruned, BaseFloat delta) {
  // delta is the amount by which the extra_costs must change
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned

  *extra_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame >= 0 && frame < active_toks_.size());
  if (active_toks_[frame].toks == NULL ) { // empty list; should not happen.
    if (!warned_) {
      KALDI_WARN << "No tokens alive [doing pruning].. warning first "
          "time only for each utterance\n";
      warned_ = true;
    }
  }
    
  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.
  bool changed = true; // difference new minus old extra cost >= delta ?
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link=NULL;
      // will recompute tok_extra_cost for tok.
      BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // tok_extra_cost is the best (min) of link_extra_cost of outgoing links
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost); // difference in brackets is >= 0
        // link_exta_cost is the difference in score between the best paths
        // through link source state and through link destination state
        KALDI_ASSERT(link_extra_cost == link_extra_cost); // check for NaN
        if (link_extra_cost > config_.lattice_beam) { // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
          *links_pruned = true;
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link; // move to next link
          link = link->next;
        }
      } // for all outgoing links
      if (fabs(tok_extra_cost - tok->extra_cost) > delta)
        changed = true;  // difference new minus old is bigger than delta
      tok->extra_cost = tok_extra_cost;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    } // for all Token on active_toks_[frame]
    if (changed) *extra_costs_changed = true;

    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  } // while changed
}

// PruneForwardLinksFinal is a version of PruneForwardLinks that we call
// on the final frame.  If there are final tokens active, it uses
// the final-probs for pruning, otherwise it treats all tokens as final.
void LatticeTrackingDecoder::PruneForwardLinksFinal(int32 frame) {
  KALDI_ASSERT(static_cast<size_t>(frame+1) == active_toks_.size());
  if (active_toks_[frame].toks == NULL ) // empty list; should not happen.
    KALDI_WARN << "No tokens alive at end of file";

  // First go through, working out the best token (do it in parallel
  // including final-probs and not including final-probs; we'll take
  // the one with final-probs if it's valid).
  const BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_cost_final = infinity,
    best_cost_nofinal = infinity;
    //worst_cost_final = - infinity,
    //best_tracked_final = infinity,
    //worst_tracked_final = -infinity,
  unordered_map<Token*, BaseFloat> tok_to_final_cost;
    
  Elem *cur_toks = toks_.Clear(); // swapping prev_toks_ / cur_toks_
  for (Elem *e = cur_toks; e != NULL;  e = e->tail) {
    StateId state = e->key;
    Token *tok = e->val;
    BaseFloat final_cost = fst_.Final(state).Value();
    // check if both final weights set: final weight and arc_graph final weight
    if ((tok->lat_state != fst::kNoStateId) && (final_cost != infinity)) {
      KALDI_ASSERT(arc_graph_->Final(tok->lat_state) != Weight::Zero());
      //best_tracked_final = std::min(best_tracked_final, tok->tot_cost + final_cost);
      //worst_tracked_final = std::max(worst_tracked_final, tok->tot_cost + final_cost);
    }
    best_cost_final = std::min(best_cost_final, tok->tot_cost + final_cost);
    //worst_cost_final = std::max(worst_cost_final, tok->tot_cost + final_cost);
    tok_to_final_cost[tok] = final_cost;
    best_cost_nofinal = std::min(best_cost_nofinal, tok->tot_cost);
  }
  final_active_ = (best_cost_final != infinity);
    
  // Now go through tokens on this frame, pruning forward links...  may have
  // to iterate a few times until there is no more change, because the list is
  // not in topological order.

  bool changed = true;
  BaseFloat delta = 1.0e-05;
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link=NULL;
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to infinity
      // below we set it to the difference between the (score+final_prob) of this token,
      // and the best such (score+final_prob).
      BaseFloat tok_extra_cost;
      if (final_active_) {
        BaseFloat final_cost = tok_to_final_cost[tok];
        tok_extra_cost = (tok->tot_cost + final_cost) - best_cost_final;
      } else 
        tok_extra_cost = tok->tot_cost - best_cost_nofinal;
      
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost)
             - next_tok->tot_cost);
        if (link_extra_cost > config_.lattice_beam) { // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link;
          link = link->next;
        }
      }
      // prune away tokens worse than lattice_beam above best path.  This step
      // was not necessary in the non-final case because then, this case
      // showed up as having no forward links.  Here, the tok_extra_cost has
      // an extra component relating to the final-prob.
      if (tok_extra_cost > config_.lattice_beam)
        tok_extra_cost = infinity;
      // to be pruned in PruneTokensForFrame

      if (!ApproxEqual(tok->extra_cost, tok_extra_cost, delta))
        changed = true;
      tok->extra_cost = tok_extra_cost; // will be +infinity or <= lattice_beam_.
    }
  } // while changed

  // Now put surviving Tokens in the final_costs_ hash, which is a class
  // member (unlike tok_to_final_costs).
  for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {    
    if (tok->extra_cost != infinity) {
      // If the token was not pruned away, 
      if (final_active_) {
        BaseFloat final_cost = tok_to_final_cost[tok];          
        if (final_cost != infinity)
          final_costs_[tok] = final_cost;
      } else {
        final_costs_[tok] = 0;
      }
    }
  }
}
  
// Prune away any tokens on this frame that have no forward links.
// [we don't do this in PruneForwardLinks because it would give us
// a problem with dangling pointers].
// It's called by PruneActiveTokens if any forward links have been pruned
void LatticeTrackingDecoder::PruneTokensForFrame(int32 frame) {
  KALDI_ASSERT(frame >= 0 && frame < active_toks_.size());
  Token *&toks = active_toks_[frame].toks;
  if (toks == NULL)
    KALDI_WARN << "No tokens alive [doing pruning]";
  Token *tok, *next_tok, *prev_tok = NULL;
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
    if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      if (prev_tok != NULL) prev_tok->next = tok->next;
      else toks = tok->next;
      delete tok;
      num_toks_--;
    } else { // fetch next Token
      prev_tok = tok;
    }
  }
}
  
// Go backwards through still-alive tokens, pruning them.  note: cur_frame is
// where hash toks_ are (so we do not want to mess with it because these tokens
// don't yet have forward pointers), but we do all previous frames, unless we
// know that we can safely ignore them because the frame after them was unchanged.
// delta controls when it considers a cost to have changed enough to continue
// going backward and propagating the change.
// for a larger delta, we will recurse less far back
void LatticeTrackingDecoder::PruneActiveTokens(int32 cur_frame, BaseFloat delta) {
  int32 num_toks_begin = num_toks_;
  for (int32 frame = cur_frame-1; frame >= 0; frame--) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them (new TokenList)
    // (2) we have not yet pruned the forward links to the next frame,
    // after any of those tokens have changed their extra_cost.
    if (active_toks_[frame].must_prune_forward_links) {
      bool extra_costs_changed = false, links_pruned = false;
      PruneForwardLinks(frame, &extra_costs_changed, &links_pruned, delta);
      if (extra_costs_changed && frame > 0) // any token has changed extra_cost
        active_toks_[frame-1].must_prune_forward_links = true;
      if (links_pruned) // any link was pruned
        active_toks_[frame].must_prune_tokens = true;
      active_toks_[frame].must_prune_forward_links = false; // job done
    }
    if (frame+1 < cur_frame &&      // except for last frame (no forward links)
        active_toks_[frame+1].must_prune_tokens) {
      PruneTokensForFrame(frame+1);
      active_toks_[frame+1].must_prune_tokens = false;
    }
  }
  KALDI_VLOG(3) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

// Version of PruneActiveTokens that we call on the final frame.
// Takes into account the final-prob of tokens.
void LatticeTrackingDecoder::PruneActiveTokensFinal(int32 cur_frame) {
  int32 num_toks_begin = num_toks_;
  PruneForwardLinksFinal(cur_frame); // prune final frame (with final-probs)
  // sets final_active_ and final_probs_
  for (int32 frame = cur_frame-1; frame >= 0; frame--) {
    bool b1, b2; // values not used.
    BaseFloat dontcare = 0.0; // delta of zero means we must always update
    PruneForwardLinks(frame, &b1, &b2, dontcare);
    PruneTokensForFrame(frame+1);
  }
  PruneTokensForFrame(0); 
  KALDI_VLOG(3) << "PruneActiveTokensFinal: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

/// Gets the weight cutoff.  Also counts the active tokens.
BaseFloat LatticeTrackingDecoder::GetCutoff(Elem *list_head, size_t *tok_count,
                                          BaseFloat *adaptive_beam, Elem **best_elem) {
  BaseFloat best_weight = std::numeric_limits<BaseFloat>::infinity();
  // positive == high cost == bad.
  BaseFloat worst_tracked = -std::numeric_limits<BaseFloat>::infinity();
  // this will contain the lower bound of the tracked tokens
  size_t count = 0;
  if (config_.max_active == std::numeric_limits<int32>::max()) {
  // no active tokens limit
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = static_cast<BaseFloat>(e->val->tot_cost);
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
      if (e->val->lat_state != fst::kNoStateId) { // tracked token
        worst_tracked = std::max(worst_tracked, static_cast<BaseFloat>(e->val->tot_cost));
      }
    }
    if (tok_count != NULL) *tok_count = count;
    BaseFloat cutoff = best_weight + config_.beam; // original beam
    BaseFloat extra_cutoff = std::min(worst_tracked + config_.extra_beam,
                                      best_weight + config_.max_beam);
    // the beam should at least include the worst tracked token
    // (plus a small extra beam) and should not exceed the max_beam
    if (extra_cutoff > cutoff) { // extending the original beam
      if (adaptive_beam != NULL) *adaptive_beam = extra_cutoff - best_weight;
      // this will be either the difference between the best token
      // and the worst tracked token or the max_beam
      KALDI_VLOG(2) << "increase beam:" << *adaptive_beam;
      return extra_cutoff; // use the extended beam
    } else { // using just original beam
      if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
      return cutoff;
    }
  } else { // using active tokens limit
    tmp_array_.clear(); // will contain all weights (sorted)
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = e->val->tot_cost;
      tmp_array_.push_back(w);
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
      if (e->val->lat_state != fst::kNoStateId) { // tracked token
        worst_tracked = std::max(worst_tracked, static_cast<BaseFloat>(e->val->tot_cost));
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (tmp_array_.size() <= static_cast<size_t>(config_.max_active)) {
      // no need to limit tokens (similar case as above)
      BaseFloat cutoff = best_weight + config_.beam;
      BaseFloat extra_cutoff = std::min(worst_tracked + config_.extra_beam,
                                        best_weight + config_.max_beam);
      // the beam should at least include the worst tracked token
      // (plus a small extra beam) and should not exceed the max_beam
      if (extra_cutoff > cutoff) { // extending the original beam
        if (adaptive_beam != NULL) *adaptive_beam = extra_cutoff - best_weight;
        KALDI_VLOG(2) << "increase beam:" << *adaptive_beam;
        return extra_cutoff;
      } else { // using just original beam
        if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
        return cutoff;
      }
    } else { // limit tokens
      // the lowest elements (lowest costs, highest likes)
      // will be put in the left part of tmp_array.
      std::vector<BaseFloat>::iterator nth_weight = 
                                    tmp_array_.begin() + config_.max_active;
      std::nth_element(tmp_array_.begin(), nth_weight, tmp_array_.end());

      BaseFloat cutoff = best_weight + config_.beam; // original beam
      BaseFloat extra_cutoff = std::min(worst_tracked + config_.extra_beam,
                                        best_weight + config_.max_beam);
      if (extra_cutoff > cutoff) { // extending the original beam
        cutoff = extra_cutoff;
        KALDI_VLOG(2) << "increase beam:" << extra_cutoff - best_weight;
      }
      // return the tighter of the two beams.
      BaseFloat ans = std::min(cutoff, *(nth_weight));
      if (adaptive_beam) 
        *adaptive_beam = std::min(cutoff - best_weight,
                                  ans - best_weight + config_.beam_delta);
      if ( *(nth_weight) < cutoff) {
        KALDI_VLOG(2) << "limit beam:" << *adaptive_beam;
      }
      return ans;
    }
  }
}

void LatticeTrackingDecoder::ProcessEmitting(DecodableInterface *decodable, int32 frame) {
  // Processes emitting arcs for one frame.  Propagates from prev_toks_ to cur_toks_.
  Elem *last_toks = toks_.Clear();
  // is analogous to swapping prev_toks_ / cur_toks_ in simple-decoder.h.
  Elem *best_elem = NULL;
  BaseFloat adaptive_beam;
  size_t tok_cnt;
  BaseFloat cur_cutoff = GetCutoff(last_toks, &tok_cnt, &adaptive_beam, &best_elem);
  PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.

  BaseFloat next_cutoff = std::numeric_limits<BaseFloat>::infinity();
  // pruning "online" before having seen all tokens
  BaseFloat next_beam = std::max(config_.beam, adaptive_beam);
  // used for updating next_cutoff
  // maybe the adaptive beam is bigger than the beam because of tracked tokens

  BaseFloat cost_offset = 0.0; // Used to keep probabilities in a good
  // dynamic range.

  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.  The only
  // products of the next block are "next_cutoff" and "cost_offset".
  if (best_elem) {
    StateId state = best_elem->key;
    Token *tok = best_elem->val;
    cost_offset = - tok->tot_cost;
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      if (arc.ilabel != 0) {  // propagate..
        arc.weight = Times(arc.weight,
                           Weight(cost_offset -
                                  decodable->LogLikelihood(frame-1, arc.ilabel)));
        BaseFloat new_weight = arc.weight.Value() + tok->tot_cost;
        if (new_weight + adaptive_beam < next_cutoff)
          next_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  // Store the offset on the acoustic likelihoods that we're applying.
  // Could just do cost_offsets_.push_back(cost_offset), but we
  // do it this way as it's more robust to future code changes.
  cost_offsets_.resize(frame, 0.0);
  cost_offsets_[frame-1] = cost_offset;

  // the tokens are now owned here, in last_toks, and the hash is empty.
  // 'owned' is a complex thing here; the point is we need to call DeleteElem
  // on each elem 'e' to let toks_ know we're done with them.
  for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) {
    // loop this way because we delete "e" as we go.
    StateId state = e->key;
    Token *tok = e->val;
    if ((tok->tot_cost <= cur_cutoff) || (tok->lat_state != fst::kNoStateId)) {
      // only prune tokens that are not tracked
      std::vector<std::pair<Label, StateId> > lat_arcs; // arc number and next state
      if (tok->lat_state != fst::kNoStateId) {
        // in case of tracked tokens, iterate all outgoing arcs from arc_graph_
        fst::MutableArcIterator<fst::StdVectorFst> lat_iter
          (const_cast<fst::StdVectorFst*>(arc_graph_), tok->lat_state);
        // do final states correspond? (lat_state and HCLG state should be final)
        if (arc_graph_->Final(tok->lat_state) != Weight::Zero()) {
          KALDI_ASSERT(fst_.Final(state) != Weight::Zero());
        }
        int32 last_arc_num = -1;
        for (; !lat_iter.Done(); lat_iter.Next()) {
          const Arc &lat_arc = lat_iter.Value();
          KALDI_ASSERT(lat_arc.ilabel == state); // ilabel contains HCLG state
          KALDI_ASSERT(lat_arc.olabel > last_arc_num);
          // assume arc_graph arcs are sorted by arc_num, each arc_num occurs once
          last_arc_num = lat_arc.olabel; // olabel contains HCLG arc number
          lat_arcs.push_back(std::make_pair(lat_arc.olabel, lat_arc.nextstate));
        }
      }
      int32 arc_num = 0, lat_arc_num = 0;
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state); // HCLG arcs
           !aiter.Done();
           aiter.Next()) {
        // match HCLG arcs with arc_graph arcs
        // not all HCLG arcs are in arc_graph
        StateId lat_nextstate = fst::kNoStateId;
        if (lat_arc_num < lat_arcs.size()) { // still graph arcs to process
          if (arc_num == lat_arcs[lat_arc_num].first) {
            lat_nextstate = lat_arcs[lat_arc_num].second;
            lat_arc_num++;
          }
        }
        arc_num++;
        // normal arc processing
        const Arc &arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          BaseFloat ac_cost = cost_offset -
              decodable->LogLikelihood(frame-1, arc.ilabel),
              graph_cost = arc.weight.Value(),
              cur_cost = tok->tot_cost,
              tot_cost = cur_cost + ac_cost + graph_cost;
          if ((tot_cost > next_cutoff) &&
               (lat_nextstate == fst::kNoStateId)) continue;
               // only prune tokens that are not tracked
          else if (tot_cost + next_beam < next_cutoff) // maybe new best token?
              next_cutoff = tot_cost + next_beam; // prune by best current token
          Token *next_tok = FindOrAddToken(arc.nextstate, lat_nextstate,
                                           frame, tot_cost, NULL);
          // NULL: no change indicator needed (no queue used)

          // Add ForwardLink from tok to next_tok (put on head of list tok->links)
          tok->links = new ForwardLink(next_tok, arc.ilabel, arc.olabel, 
                                       graph_cost, ac_cost, tok->links);
        }
      } // for all arcs
    }
    e_tail = e->tail;
    toks_.Delete(e); // delete Elem
  }
}

// TODO: could possibly add adaptive_beam back as an argument here (was
// returned from ProcessEmitting, in faster-decoder.h).
void LatticeTrackingDecoder::ProcessNonemitting(int32 frame) {
  // note: "frame" is the same as emitting states just processed.
    
  // Processes nonemitting arcs for one frame.  Propagates within toks_.
  // Note-- this queue structure is is not very optimal as
  // it may cause us to process states unnecessarily (e.g. more than once),
  // but in the baseline code, turning this vector into a set to fix this
  // problem did not improve overall speed.

  KALDI_ASSERT(queue_.empty());
  BaseFloat best_cost = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat worst_tracked = -std::numeric_limits<BaseFloat>::infinity();
  for (Elem *e = toks_.GetList(); e != NULL;  e = e->tail) {
    queue_.push_back(e->key);
    // for pruning with current best token
    best_cost = std::min(best_cost, static_cast<BaseFloat>(e->val->tot_cost));
    if (e->val->lat_state != fst::kNoStateId) {
      worst_tracked = std::max(worst_tracked,
                               static_cast<BaseFloat>(e->val->tot_cost));
    }
  }
  if (queue_.empty()) {
    if (!warned_) {
      KALDI_ERR << "Error in ProcessEmitting: no surviving tokens: frame is "
                << frame;
      warned_ = true;
    }
  }
  //KALDI_VLOG(2) << "nonemit:" << frame << ":" << best_cost;
  //if (frame > 0) KALDI_ASSERT(worst_tracked > 0.0); // track at least one token

  BaseFloat cutoff = best_cost + config_.beam; // original beam
  BaseFloat extra_cutoff = std::min(worst_tracked + config_.extra_beam,
                                    best_cost + config_.max_beam);
  if (extra_cutoff > cutoff) { // extending the beam
    KALDI_VLOG(2) << "increase beam:" << extra_cutoff - best_cost;
    cutoff = extra_cutoff;
  }
    
  while (!queue_.empty()) {
    StateId state = queue_.back();
    queue_.pop_back();

    Token *tok = toks_.Find(state)->val;  // would segfault if state not in toks_ but this can't happen.
    BaseFloat cur_cost = tok->tot_cost;
    if ((cur_cost > cutoff) && (tok->lat_state == fst::kNoStateId)) {
      // we should never kill a tracked token
      continue;  // Don't bother processing successors.
    }
    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder),
    // but since most states are emitting it's not a huge issue.
    tok->DeleteForwardLinks(); // necessary when re-visiting
    tok->links = NULL;
    //GetArcStateMap(&arc_map, state, tok->lat_state); // in case of tracked tokens, contains nextstates
    std::vector<std::pair<Label, StateId> > lat_arcs; // arc number and next state
    if (tok->lat_state != fst::kNoStateId) {
      // in case of tracked tokens, iterate all outgoing arcs from arc_graph_
      fst::MutableArcIterator<fst::StdVectorFst> lat_iter
        (const_cast<fst::StdVectorFst*>(arc_graph_), tok->lat_state);
      // do final states correspond? (lat_state and HCLG state should be final)
      if (arc_graph_->Final(tok->lat_state) != Weight::Zero()) {
        KALDI_ASSERT(fst_.Final(state) != Weight::Zero());
      }
      int32 last_arc_num = -1;
      for (; !lat_iter.Done(); lat_iter.Next()) {
        const Arc &lat_arc = lat_iter.Value();
        KALDI_ASSERT(lat_arc.ilabel == state); // ilabel contains HCLG state
        KALDI_ASSERT(lat_arc.olabel > last_arc_num);
        // assume arc_graph arcs are sorted by arc_num, each arc_num occurs once
        last_arc_num = lat_arc.olabel; // olabel contains HCLG arc number
        lat_arcs.push_back(std::make_pair(lat_arc.olabel, lat_arc.nextstate));
      }
    }
    int32 arc_num = 0, lat_arc_num = 0;
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      // match HCLG arcs with arc_graph arcs
      // not all HCLG arcs are in arc_graph
      StateId lat_nextstate = fst::kNoStateId;
      if (lat_arc_num < lat_arcs.size()) { // still graph arcs to process
        if (arc_num == lat_arcs[lat_arc_num].first) {
          lat_nextstate = lat_arcs[lat_arc_num].second;
          lat_arc_num++;
        }
      }
      arc_num++;
      // normal arc processing
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        BaseFloat graph_cost = arc.weight.Value(),
            tot_cost = cur_cost + graph_cost;
        if ((tot_cost < cutoff) || (lat_nextstate != fst::kNoStateId)) {
          bool changed;

          Token *new_tok = FindOrAddToken(arc.nextstate, lat_nextstate,
                                          frame, tot_cost, &changed);

          tok->links = new ForwardLink(new_tok, 0, arc.olabel,
                                       graph_cost, 0, tok->links);

          // "changed" tells us whether the new token has a different
          // cost from before, or is new [if so, add into queue].
          if (changed) queue_.push_back(arc.nextstate);
          // it is also true when the "tracked" status was changed
        }
      }
    } // for all arcs
  } // while queue not empty
}


void LatticeTrackingDecoder::ClearToks(Elem *list) {
  for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
    // Token::TokenDelete(e->val);
    e_tail = e->tail;
    toks_.Delete(e);
  }
  toks_.Clear();
}
  
void LatticeTrackingDecoder::ClearActiveTokens() { // a cleanup routine, at utt end/begin
  for (size_t i = 0; i < active_toks_.size(); i++) {
    // Delete all tokens alive on this frame, and any forward
    // links they may have.
    for (Token *tok = active_toks_[i].toks; tok != NULL; ) {
      tok->DeleteForwardLinks();
      Token *next_tok = tok->next;
      delete tok;
      num_toks_--;
      tok = next_tok;
    }
  }
  active_toks_.clear();
  KALDI_ASSERT(num_toks_ == 0);
}


// Takes care of output.  Returns true on success.
bool DecodeUtteranceLatticeTracking(
    LatticeTrackingDecoder &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const TransitionModel &trans_model,
    const fst::StdVectorFst &arc_graph, // contains graph arcs from forward pass lattice
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignment_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr) { // puts utterance's like in like_ptr on success.
  using fst::VectorFst;

  if (!decoder.Decode(&decodable, arc_graph)) {
    KALDI_WARN << "Failed to decode file " << utt;
    return false;
  }
  if (!decoder.ReachedFinal()) {
    if (allow_partial) {
      KALDI_WARN << "Outputting partial output for utterance " << utt
                 << " since no final-state reached\n";
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      return false;
    }
  }

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  { // First do some stuff with word-level traceback...
    VectorFst<LatticeArc> decoded;
    if (!decoder.GetBestPath(&decoded)) 
      // Shouldn't really reach this point as already checked success.
      KALDI_ERR << "Failed to get traceback for utterance " << utt;

    std::vector<int32> alignment;
    std::vector<int32> words;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    num_frames = alignment.size();
    if (words_writer->IsOpen())
      words_writer->Write(utt, words);
    if (alignment_writer->IsOpen())
      alignment_writer->Write(utt, alignment);
    if (word_syms != NULL) {
      std::cerr << utt << ' ';
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms->Find(words[i]);
        if (s == "")
          KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
        std::cerr << s << ' ';
      }
      std::cerr << '\n';
    }
    likelihood = -(weight.Value1() + weight.Value2());
  }

  // Get lattice, and do determinization if requested.
  Lattice lat;
  if (!decoder.GetRawLattice(&lat))
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt;
  fst::Connect(&lat);
  if (determinize) {
    CompactLattice clat;
    if (!DeterminizeLatticePhonePrunedWrapper(
            trans_model,
            &lat,
            decoder.GetOptions().lattice_beam,
            &clat,
            decoder.GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << utt;
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &clat);
    compact_lattice_writer->Write(utt, clat);
  } else {
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &lat);
    lattice_writer->Write(utt, lat);
  }
  KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
            << (likelihood / num_frames) << " over "
            << num_frames << " frames.";
  KALDI_VLOG(2) << "Cost for utterance " << utt << " is "
                << weight.Value1() << " + " << weight.Value2();
  *like_ptr = likelihood;
  return true;
}

} // end namespace kaldi.
