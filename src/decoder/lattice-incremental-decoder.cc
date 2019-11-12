// decoder/lattice-incremental-decoder.cc

// Copyright      2019  Zhehuai Chen

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

#include "decoder/lattice-incremental-decoder.h"
#include "lat/lattice-functions.h"
#include "base/timer.h"

namespace kaldi {

// instantiate this class once for each thing you have to decode.
template <typename FST, typename Token>
LatticeIncrementalDecoderTpl<FST, Token>::LatticeIncrementalDecoderTpl(
    const FST &fst, const TransitionModel &trans_model,
    const LatticeIncrementalDecoderConfig &config)
    : fst_(&fst),
      delete_fst_(false),
      config_(config),
      num_toks_(0),
      determinizer_(config, trans_model) {
  config.Check();
  toks_.SetSize(1000); // just so on the first frame we do something reasonable.
}

template <typename FST, typename Token>
LatticeIncrementalDecoderTpl<FST, Token>::LatticeIncrementalDecoderTpl(
    const LatticeIncrementalDecoderConfig &config, FST *fst,
    const TransitionModel &trans_model)
    : fst_(fst),
      delete_fst_(true),
      config_(config),
      num_toks_(0),
      determinizer_(config, trans_model) {
  config.Check();
  toks_.SetSize(1000); // just so on the first frame we do something reasonable.
}

template <typename FST, typename Token>
LatticeIncrementalDecoderTpl<FST, Token>::~LatticeIncrementalDecoderTpl() {
  DeleteElems(toks_.Clear());
  ClearActiveTokens();
  if (delete_fst_) delete fst_;
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::InitDecoding() {
  // clean up from last time:
  DeleteElems(toks_.Clear());
  cost_offsets_.clear();
  ClearActiveTokens();
  warned_ = false;
  num_toks_ = 0;
  decoding_finalized_ = false;
  final_costs_.clear();
  StateId start_state = fst_->Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  active_toks_.resize(1);
  Token *start_tok = new Token(0.0, 0.0, NULL, NULL, NULL);
  active_toks_[0].toks = start_tok;
  toks_.Insert(start_state, start_tok);
  num_toks_++;

  last_get_lattice_frame_ = 0;
  token_label_map_.clear();
  token_label_map_.reserve(std::min((int32)1e5, config_.max_active));
  token_label_available_idx_ = config_.max_word_id + 1;
  token_label_final_cost_.clear();
  determinizer_.Init();

  ProcessNonemitting(config_.beam);
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::DeterminizeLattice() {
  // We always incrementally determinize the lattice after lattice pruning in
  // PruneActiveTokens() since we need extra_cost as the weights
  // of final arcs to denote the "future" information of final states (Tokens)
  // Moreover, the delay on GetLattice to do determinization
  // make it process more skinny lattices which reduces the computation overheads.
  int32 frame_det_most = NumFramesDecoded() - config_.determinize_delay;
  // The minimum length of chunk is config_.determinize_period.
  if (frame_det_most % config_.determinize_period == 0) {
    int32 frame_det_least = last_get_lattice_frame_ + config_.determinize_period;
    // Incremental determinization:
    // To adaptively decide the length of chunk, we further compare the number of
    // tokens in each frame and a pre-defined threshold.
    // If the number of tokens in a certain frame is less than
    // config_.determinize_max_active, the lattice can be determinized up to this
    // frame. And we try to determinize as most frames as possible so we check
    // numbers from frame_det_most to frame_det_least
    for (int32 f = frame_det_most; f >= frame_det_least; f--) {
      if (config_.determinize_max_active == std::numeric_limits<int32>::max() ||
          GetNumToksForFrame(f) < config_.determinize_max_active) {
        KALDI_VLOG(2) << "Frame: " << NumFramesDecoded()
                      << " incremental determinization up to " << f;
        GetLattice(false, f);
        break;
      }
    }
  }
  return;
}
// Returns true if any kind of traceback is available (not necessarily from
// a final state).  It should only very rarely return false; this indicates
// an unusual search error.
template <typename FST, typename Token>
bool LatticeIncrementalDecoderTpl<FST, Token>::Decode(DecodableInterface *decodable) {
  InitDecoding();

  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.

  while (!decodable->IsLastFrame(NumFramesDecoded() - 1)) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    }

    DeterminizeLattice();

    BaseFloat cost_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(cost_cutoff);
  }
  Timer timer;
  FinalizeDecoding();
  GetLattice(true, NumFramesDecoded());
  KALDI_VLOG(2) << "Delay time during and after FinalizeDecoding()"
                << "(secs): " << timer.Elapsed();

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !active_toks_.empty() && active_toks_.back().toks != NULL;
}

// Outputs an FST corresponding to the single best path through the lattice.
template <typename FST, typename Token>
bool LatticeIncrementalDecoderTpl<FST, Token>::GetBestPath(Lattice *olat,
                                                           bool use_final_probs) {
  CompactLattice lat, slat;
  GetLattice(use_final_probs, NumFramesDecoded(), &lat);
  ShortestPath(lat, &slat);
  ConvertLattice(slat, olat);
  return (olat->NumStates() != 0);
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::PossiblyResizeHash(size_t num_toks) {
  size_t new_sz =
      static_cast<size_t>(static_cast<BaseFloat>(num_toks) * config_.hash_ratio);
  if (new_sz > toks_.Size()) {
    toks_.SetSize(new_sz);
  }
}

/*
  A note on the definition of extra_cost.

  extra_cost is used in pruning tokens, to save memory.

  Define the 'forward cost' of a token as zero for any token on the frame
  we're currently decoding; and for other frames, as the shortest-path cost
  between that token and a token on the frame we're currently decoding.
  (by "currently decoding" I mean the most recently processed frame).

  Then define the extra_cost of a token (always >= 0) as the forward-cost of
  the token minus the smallest forward-cost of any token on the same frame.

  We can use the extra_cost to accurately prune away tokens that we know will
  never appear in the lattice.  If the extra_cost is greater than the desired
  lattice beam, the token would provably never appear in the lattice, so we can
  prune away the token.

  The advantage of storing the extra_cost rather than the forward-cost, is that
  it is less costly to keep the extra_cost up-to-date when we process new frames.
  When we process a new frame, *all* the previous frames' forward-costs would change;
  but in general the extra_cost will change only for a finite number of frames.
  (Actually we don't update all the extra_costs every time we update a frame; we
  only do it every 'config_.prune_interval' frames).
 */

// FindOrAddToken either locates a token in hash of toks_,
// or if necessary inserts a new, empty token (i.e. with no forward links)
// for the current frame.  [note: it's inserted if necessary into hash toks_
// and also into the singly linked list of tokens active on this frame
// (whose head is at active_toks_[frame]).
template <typename FST, typename Token>
inline Token *LatticeIncrementalDecoderTpl<FST, Token>::FindOrAddToken(
    StateId state, int32 frame_plus_one, BaseFloat tot_cost, Token *backpointer,
    bool *changed) {
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  KALDI_ASSERT(frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;
  Elem *e_found = toks_.Find(state);
  if (e_found == NULL) { // no such token presently.
    const BaseFloat extra_cost = 0.0;
    // tokens on the currently final frame have zero extra_cost
    // as any of them could end up
    // on the winning path.
    Token *new_tok = new Token(tot_cost, extra_cost, NULL, toks, backpointer);
    // NULL: no forward links yet
    toks = new_tok;
    num_toks_++;
    toks_.Insert(state, new_tok);
    if (changed) *changed = true;
    return new_tok;
  } else {
    Token *tok = e_found->val;      // There is an existing Token for this state.
    if (tok->tot_cost > tot_cost) { // replace old token
      tok->tot_cost = tot_cost;
      // SetBackpointer() just does tok->backpointer = backpointer in
      // the case where Token == BackpointerToken, else nothing.
      tok->SetBackpointer(backpointer);
      // we don't allocate a new token, the old stays linked in active_toks_
      // we only replace the tot_cost
      // in the current frame, there are no forward links (and no extra_cost)
      // only in ProcessNonemitting we have to delete forward links
      // in case we visit a state for the second time
      // those forward links, that lead to this replaced token before:
      // they remain and will hopefully be pruned later (PruneForwardLinks...)
      if (changed) *changed = true;
    } else {
      if (changed) *changed = false;
    }
    return tok;
  }
}

// prunes outgoing links for all tokens in active_toks_[frame]
// it's called by PruneActiveTokens
// all links, that have link_extra_cost > lattice_beam are pruned
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::PruneForwardLinks(
    int32 frame_plus_one, bool *extra_costs_changed, bool *links_pruned,
    BaseFloat delta) {
  // delta is the amount by which the extra_costs must change
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned

  *extra_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  if (active_toks_[frame_plus_one].toks == NULL) { // empty list; should not happen.
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
    for (Token *tok = active_toks_[frame_plus_one].toks; tok != NULL;
         tok = tok->next) {
      ForwardLinkT *link, *prev_link = NULL;
      // will recompute tok_extra_cost for tok.
      BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // tok_extra_cost is the best (min) of link_extra_cost of outgoing links
      for (link = tok->links; link != NULL;) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost =
            next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost) -
             next_tok->tot_cost); // difference in brackets is >= 0
        // link_exta_cost is the difference in score between the best paths
        // through link source state and through link destination state
        KALDI_ASSERT(link_extra_cost == link_extra_cost); // check for NaN
        if (link_extra_cost > config_.lattice_beam) {     // excise link
          ForwardLinkT *next_link = link->next;
          if (prev_link != NULL)
            prev_link->next = next_link;
          else
            tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
          *links_pruned = true;
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost) tok_extra_cost = link_extra_cost;
          prev_link = link; // move to next link
          link = link->next;
        }
      } // for all outgoing links
      if (fabs(tok_extra_cost - tok->extra_cost) > delta)
        changed = true; // difference new minus old is bigger than delta
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
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::PruneForwardLinksFinal() {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame_plus_one = active_toks_.size() - 1;

  if (active_toks_[frame_plus_one].toks == NULL) // empty list; should not happen.
    KALDI_WARN << "No tokens alive at end of file";

  typedef typename unordered_map<Token *, BaseFloat>::const_iterator IterType;
  ComputeFinalCosts(&final_costs_, &final_relative_cost_, &final_best_cost_);
  decoding_finalized_ = true;
  // We call DeleteElems() as a nicety, not because it's really necessary;
  // otherwise there would be a time, after calling PruneTokensForFrame() on the
  // final frame, when toks_.GetList() or toks_.Clear() would contain pointers
  // to nonexistent tokens.
  DeleteElems(toks_.Clear());

  // Now go through tokens on this frame, pruning forward links...  may have to
  // iterate a few times until there is no more change, because the list is not
  // in topological order.  This is a modified version of the code in
  // PruneForwardLinks, but here we also take account of the final-probs.
  bool changed = true;
  BaseFloat delta = 1.0e-05;
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks; tok != NULL;
         tok = tok->next) {
      ForwardLinkT *link, *prev_link = NULL;
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to infinity
      // below we set it to the difference between the (score+final_prob) of this
      // token,
      // and the best such (score+final_prob).
      BaseFloat final_cost;
      if (final_costs_.empty()) {
        final_cost = 0.0;
      } else {
        IterType iter = final_costs_.find(tok);
        if (iter != final_costs_.end())
          final_cost = iter->second;
        else
          final_cost = std::numeric_limits<BaseFloat>::infinity();
      }
      BaseFloat tok_extra_cost = tok->tot_cost + final_cost - final_best_cost_;
      // tok_extra_cost will be a "min" over either directly being final, or
      // being indirectly final through other links, and the loop below may
      // decrease its value:
      for (link = tok->links; link != NULL;) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost =
            next_tok->extra_cost +
            ((tok->tot_cost + link->acoustic_cost + link->graph_cost) -
             next_tok->tot_cost);
        if (link_extra_cost > config_.lattice_beam) { // excise link
          ForwardLinkT *next_link = link->next;
          if (prev_link != NULL)
            prev_link->next = next_link;
          else
            tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
        } else {            // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost) tok_extra_cost = link_extra_cost;
          prev_link = link;
          link = link->next;
        }
      }
      // prune away tokens worse than lattice_beam above best path.  This step
      // was not necessary in the non-final case because then, this case
      // showed up as having no forward links.  Here, the tok_extra_cost has
      // an extra component relating to the final-prob.
      if (tok_extra_cost > config_.lattice_beam)
        tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // to be pruned in PruneTokensForFrame

      if (!ApproxEqual(tok->extra_cost, tok_extra_cost, delta)) changed = true;
      tok->extra_cost = tok_extra_cost; // will be +infinity or <= lattice_beam_.
    }
  } // while changed
}

template <typename FST, typename Token>
BaseFloat LatticeIncrementalDecoderTpl<FST, Token>::FinalRelativeCost() const {
  if (!decoding_finalized_) {
    BaseFloat relative_cost;
    ComputeFinalCosts(NULL, &relative_cost, NULL);
    return relative_cost;
  } else {
    // we're not allowed to call that function if FinalizeDecoding() has
    // been called; return a cached value.
    return final_relative_cost_;
  }
}

// Prune away any tokens on this frame that have no forward links.
// [we don't do this in PruneForwardLinks because it would give us
// a problem with dangling pointers].
// It's called by PruneActiveTokens if any forward links have been pruned
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::PruneTokensForFrame(
    int32 frame_plus_one) {
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;
  if (toks == NULL) KALDI_WARN << "No tokens alive [doing pruning]";
  Token *tok, *next_tok, *prev_tok = NULL;
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
    if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      if (prev_tok != NULL)
        prev_tok->next = tok->next;
      else
        toks = tok->next;
      delete tok;
      num_toks_--;
    } else { // fetch next Token
      prev_tok = tok;
    }
  }
}

// Go backwards through still-alive tokens, pruning them, starting not from
// the current frame (where we want to keep all tokens) but from the frame before
// that.  We go backwards through the frames and stop when we reach a point
// where the delta-costs are not changing (and the delta controls when we consider
// a cost to have "not changed").
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::PruneActiveTokens(BaseFloat delta) {
  int32 cur_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;
  // The index "f" below represents a "frame plus one", i.e. you'd have to subtract
  // one to get the corresponding index for the decodable object.
  for (int32 f = cur_frame_plus_one - 1; f >= 0; f--) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them (new TokenList)
    // (2) we have not yet pruned the forward links to the next f,
    // after any of those tokens have changed their extra_cost.
    if (active_toks_[f].must_prune_forward_links) {
      bool extra_costs_changed = false, links_pruned = false;
      PruneForwardLinks(f, &extra_costs_changed, &links_pruned, delta);
      if (extra_costs_changed && f > 0) // any token has changed extra_cost
        active_toks_[f - 1].must_prune_forward_links = true;
      if (links_pruned) // any link was pruned
        active_toks_[f].must_prune_tokens = true;
      active_toks_[f].must_prune_forward_links = false; // job done
    }
    if (f + 1 < cur_frame_plus_one && // except for last f (no forward links)
        active_toks_[f + 1].must_prune_tokens) {
      PruneTokensForFrame(f + 1);
      active_toks_[f + 1].must_prune_tokens = false;
    }
  }
  KALDI_VLOG(4) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::ComputeFinalCosts(
    unordered_map<Token *, BaseFloat> *final_costs, BaseFloat *final_relative_cost,
    BaseFloat *final_best_cost) const {
  KALDI_ASSERT(!decoding_finalized_);
  if (final_costs != NULL) final_costs->clear();
  const Elem *final_toks = toks_.GetList();
  BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_cost = infinity, best_cost_with_final = infinity;

  while (final_toks != NULL) {
    StateId state = final_toks->key;
    Token *tok = final_toks->val;
    const Elem *next = final_toks->tail;
    BaseFloat final_cost = fst_->Final(state).Value();
    BaseFloat cost = tok->tot_cost, cost_with_final = cost + final_cost;
    best_cost = std::min(cost, best_cost);
    best_cost_with_final = std::min(cost_with_final, best_cost_with_final);
    if (final_costs != NULL && final_cost != infinity)
      (*final_costs)[tok] = final_cost;
    final_toks = next;
  }
  if (final_relative_cost != NULL) {
    if (best_cost == infinity && best_cost_with_final == infinity) {
      // Likely this will only happen if there are no tokens surviving.
      // This seems the least bad way to handle it.
      *final_relative_cost = infinity;
    } else {
      *final_relative_cost = best_cost_with_final - best_cost;
    }
  }
  if (final_best_cost != NULL) {
    if (best_cost_with_final != infinity) { // final-state exists.
      *final_best_cost = best_cost_with_final;
    } else { // no final-state exists.
      *final_best_cost = best_cost;
    }
  }
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::AdvanceDecoding(
    DecodableInterface *decodable, int32 max_num_frames) {
  if (std::is_same<FST, fst::Fst<fst::StdArc> >::value) {
    // if the type 'FST' is the FST base-class, then see if the FST type of fst_
    // is actually VectorFst or ConstFst.  If so, call the AdvanceDecoding()
    // function after casting *this to the more specific type.
    if (fst_->Type() == "const") {
      LatticeIncrementalDecoderTpl<fst::ConstFst<fst::StdArc>, Token> *this_cast =
          reinterpret_cast<
              LatticeIncrementalDecoderTpl<fst::ConstFst<fst::StdArc>, Token> *>(
              this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    } else if (fst_->Type() == "vector") {
      LatticeIncrementalDecoderTpl<fst::VectorFst<fst::StdArc>, Token> *this_cast =
          reinterpret_cast<
              LatticeIncrementalDecoderTpl<fst::VectorFst<fst::StdArc>, Token> *>(
              this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    }
  }

  KALDI_ASSERT(!active_toks_.empty() && !decoding_finalized_ &&
               "You must call InitDecoding() before AdvanceDecoding");
  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_ASSERT(num_frames_ready >= NumFramesDecoded());
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded =
        std::min(target_frames_decoded, NumFramesDecoded() + max_num_frames);
  while (NumFramesDecoded() < target_frames_decoded) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    }

    DeterminizeLattice();

    BaseFloat cost_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(cost_cutoff);
  }
}

// FinalizeDecoding() is a version of PruneActiveTokens that we call
// (optionally) on the final frame.  Takes into account the final-prob of
// tokens.  This function used to be called PruneActiveTokensFinal().
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::FinalizeDecoding() {
  int32 final_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;
  // PruneForwardLinksFinal() prunes final frame (with final-probs), and
  // sets decoding_finalized_.
  PruneForwardLinksFinal();
  for (int32 f = final_frame_plus_one - 1; f >= 0; f--) {
    bool b1, b2;              // values not used.
    BaseFloat dontcare = 0.0; // delta of zero means we must always update
    PruneForwardLinks(f, &b1, &b2, dontcare);
    PruneTokensForFrame(f + 1);
  }
  PruneTokensForFrame(0);
  KALDI_VLOG(4) << "pruned tokens from " << num_toks_begin << " to " << num_toks_;
}

/// Gets the weight cutoff.  Also counts the active tokens.
template <typename FST, typename Token>
BaseFloat LatticeIncrementalDecoderTpl<FST, Token>::GetCutoff(
    Elem *list_head, size_t *tok_count, BaseFloat *adaptive_beam, Elem **best_elem) {
  BaseFloat best_weight = std::numeric_limits<BaseFloat>::infinity();
  // positive == high cost == bad.
  size_t count = 0;
  if (config_.max_active == std::numeric_limits<int32>::max() &&
      config_.min_active == 0) {
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = static_cast<BaseFloat>(e->val->tot_cost);
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
    return best_weight + config_.beam;
  } else {
    tmp_array_.clear();
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = e->val->tot_cost;
      tmp_array_.push_back(w);
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;

    BaseFloat beam_cutoff = best_weight + config_.beam,
              min_active_cutoff = std::numeric_limits<BaseFloat>::infinity(),
              max_active_cutoff = std::numeric_limits<BaseFloat>::infinity();

    KALDI_VLOG(6) << "Number of tokens active on frame " << NumFramesDecoded()
                  << " is " << tmp_array_.size();

    if (tmp_array_.size() > static_cast<size_t>(config_.max_active)) {
      std::nth_element(tmp_array_.begin(), tmp_array_.begin() + config_.max_active,
                       tmp_array_.end());
      max_active_cutoff = tmp_array_[config_.max_active];
    }
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_weight + config_.beam_delta;
      return max_active_cutoff;
    }
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      if (config_.min_active == 0)
        min_active_cutoff = best_weight;
      else {
        std::nth_element(tmp_array_.begin(), tmp_array_.begin() + config_.min_active,
                         tmp_array_.size() > static_cast<size_t>(config_.max_active)
                             ? tmp_array_.begin() + config_.max_active
                             : tmp_array_.end());
        min_active_cutoff = tmp_array_[config_.min_active];
      }
    }
    if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
      if (adaptive_beam)
        *adaptive_beam = min_active_cutoff - best_weight + config_.beam_delta;
      return min_active_cutoff;
    } else {
      *adaptive_beam = config_.beam;
      return beam_cutoff;
    }
  }
}

template <typename FST, typename Token>
BaseFloat LatticeIncrementalDecoderTpl<FST, Token>::ProcessEmitting(
    DecodableInterface *decodable) {
  KALDI_ASSERT(active_toks_.size() > 0);
  int32 frame = active_toks_.size() - 1; // frame is the frame-index
                                         // (zero-based) used to get likelihoods
                                         // from the decodable object.
  active_toks_.resize(active_toks_.size() + 1);

  Elem *final_toks = toks_.Clear(); // analogous to swapping prev_toks_ / cur_toks_
                                    // in simple-decoder.h.   Removes the Elems from
                                    // being indexed in the hash in toks_.
  Elem *best_elem = NULL;
  BaseFloat adaptive_beam;
  size_t tok_cnt;
  BaseFloat cur_cutoff = GetCutoff(final_toks, &tok_cnt, &adaptive_beam, &best_elem);
  KALDI_VLOG(6) << "Adaptive beam on frame " << NumFramesDecoded() << " is "
                << adaptive_beam;

  PossiblyResizeHash(tok_cnt); // This makes sure the hash is always big enough.

  BaseFloat next_cutoff = std::numeric_limits<BaseFloat>::infinity();
  // pruning "online" before having seen all tokens

  BaseFloat cost_offset = 0.0; // Used to keep probabilities in a good
                               // dynamic range.

  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.  The only
  // products of the next block are "next_cutoff" and "cost_offset".
  if (best_elem) {
    StateId state = best_elem->key;
    Token *tok = best_elem->val;
    cost_offset = -tok->tot_cost;
    for (fst::ArcIterator<FST> aiter(*fst_, state); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) { // propagate..
        BaseFloat new_weight = arc.weight.Value() + cost_offset -
                               decodable->LogLikelihood(frame, arc.ilabel) +
                               tok->tot_cost;
        if (new_weight + adaptive_beam < next_cutoff)
          next_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  // Store the offset on the acoustic likelihoods that we're applying.
  // Could just do cost_offsets_.push_back(cost_offset), but we
  // do it this way as it's more robust to future code changes.
  cost_offsets_.resize(frame + 1, 0.0);
  cost_offsets_[frame] = cost_offset;

  // the tokens are now owned here, in final_toks, and the hash is empty.
  // 'owned' is a complex thing here; the point is we need to call DeleteElem
  // on each elem 'e' to let toks_ know we're done with them.
  for (Elem *e = final_toks, *e_tail; e != NULL; e = e_tail) {
    // loop this way because we delete "e" as we go.
    StateId state = e->key;
    Token *tok = e->val;
    if (tok->tot_cost <= cur_cutoff) {
      for (fst::ArcIterator<FST> aiter(*fst_, state); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.ilabel != 0) { // propagate..
          BaseFloat ac_cost =
                        cost_offset - decodable->LogLikelihood(frame, arc.ilabel),
                    graph_cost = arc.weight.Value(), cur_cost = tok->tot_cost,
                    tot_cost = cur_cost + ac_cost + graph_cost;
          if (tot_cost > next_cutoff)
            continue;
          else if (tot_cost + adaptive_beam < next_cutoff)
            next_cutoff = tot_cost + adaptive_beam; // prune by best current token
          // Note: the frame indexes into active_toks_ are one-based,
          // hence the + 1.
          Token *next_tok =
              FindOrAddToken(arc.nextstate, frame + 1, tot_cost, tok, NULL);
          // NULL: no change indicator needed

          // Add ForwardLink from tok to next_tok (put on head of list tok->links)
          tok->links = new ForwardLinkT(next_tok, arc.ilabel, arc.olabel, graph_cost,
                                        ac_cost, tok->links);
        }
      } // for all arcs
    }
    e_tail = e->tail;
    toks_.Delete(e); // delete Elem
  }
  return next_cutoff;
}

// static inline
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::DeleteForwardLinks(Token *tok) {
  ForwardLinkT *l = tok->links, *m;
  while (l != NULL) {
    m = l->next;
    delete l;
    l = m;
  }
  tok->links = NULL;
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::ProcessNonemitting(BaseFloat cutoff) {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame = static_cast<int32>(active_toks_.size()) - 2;
  // Note: "frame" is the time-index we just processed, or -1 if
  // we are processing the nonemitting transitions before the
  // first frame (called from InitDecoding()).

  // Processes nonemitting arcs for one frame.  Propagates within toks_.
  // Note-- this queue structure is is not very optimal as
  // it may cause us to process states unnecessarily (e.g. more than once),
  // but in the baseline code, turning this vector into a set to fix this
  // problem did not improve overall speed.

  KALDI_ASSERT(queue_.empty());

  if (toks_.GetList() == NULL) {
    if (!warned_) {
      KALDI_WARN << "Error, no surviving tokens: frame is " << frame;
      warned_ = true;
    }
  }

  for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
    StateId state = e->key;
    if (fst_->NumInputEpsilons(state) != 0) queue_.push_back(state);
  }

  while (!queue_.empty()) {
    StateId state = queue_.back();
    queue_.pop_back();

    Token *tok =
        toks_.Find(state)
            ->val; // would segfault if state not in toks_ but this can't happen.
    BaseFloat cur_cost = tok->tot_cost;
    if (cur_cost > cutoff) // Don't bother processing successors.
      continue;
    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder),
    // but since most states are emitting it's not a huge issue.
    DeleteForwardLinks(tok); // necessary when re-visiting
    tok->links = NULL;
    for (fst::ArcIterator<FST> aiter(*fst_, state); !aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0) { // propagate nonemitting only...
        BaseFloat graph_cost = arc.weight.Value(), tot_cost = cur_cost + graph_cost;
        if (tot_cost < cutoff) {
          bool changed;

          Token *new_tok =
              FindOrAddToken(arc.nextstate, frame + 1, tot_cost, tok, &changed);

          tok->links =
              new ForwardLinkT(new_tok, 0, arc.olabel, graph_cost, 0, tok->links);

          // "changed" tells us whether the new token has a different
          // cost from before, or is new [if so, add into queue].
          if (changed && fst_->NumInputEpsilons(arc.nextstate) != 0)
            queue_.push_back(arc.nextstate);
        }
      }
    } // for all arcs
  }   // while queue not empty
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::DeleteElems(Elem *list) {
  for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
    e_tail = e->tail;
    toks_.Delete(e);
  }
}

template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<
    FST, Token>::ClearActiveTokens() { // a cleanup routine, at utt end/begin
  for (size_t i = 0; i < active_toks_.size(); i++) {
    // Delete all tokens alive on this frame, and any forward
    // links they may have.
    for (Token *tok = active_toks_[i].toks; tok != NULL;) {
      DeleteForwardLinks(tok);
      Token *next_tok = tok->next;
      delete tok;
      num_toks_--;
      tok = next_tok;
    }
  }
  active_toks_.clear();
  KALDI_ASSERT(num_toks_ == 0);
}

// static
template <typename FST, typename Token>
void LatticeIncrementalDecoderTpl<FST, Token>::TopSortTokens(
    Token *tok_list, std::vector<Token *> *topsorted_list) {
  unordered_map<Token *, int32> token2pos;
  typedef typename unordered_map<Token *, int32>::iterator IterType;
  int32 num_toks = 0;
  for (Token *tok = tok_list; tok != NULL; tok = tok->next) num_toks++;
  int32 cur_pos = 0;
  // We assign the tokens numbers num_toks - 1, ... , 2, 1, 0.
  // This is likely to be in closer to topological order than
  // if we had given them ascending order, because of the way
  // new tokens are put at the front of the list.
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    token2pos[tok] = num_toks - ++cur_pos;

  unordered_set<Token *> reprocess;

  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter) {
    Token *tok = iter->first;
    int32 pos = iter->second;
    for (ForwardLinkT *link = tok->links; link != NULL; link = link->next) {
      if (link->ilabel == 0) {
        // We only need to consider epsilon links, since non-epsilon links
        // transition between frames and this function only needs to sort a list
        // of tokens from a single frame.
        IterType following_iter = token2pos.find(link->next_tok);
        if (following_iter != token2pos.end()) { // another token on this frame,
                                                 // so must consider it.
          int32 next_pos = following_iter->second;
          if (next_pos < pos) { // reassign the position of the next Token.
            following_iter->second = cur_pos++;
            reprocess.insert(link->next_tok);
          }
        }
      }
    }
    // In case we had previously assigned this token to be reprocessed, we can
    // erase it from that set because it's "happy now" (we just processed it).
    reprocess.erase(tok);
  }

  size_t max_loop = 1000000, loop_count; // max_loop is to detect epsilon cycles.
  for (loop_count = 0; !reprocess.empty() && loop_count < max_loop; ++loop_count) {
    std::vector<Token *> reprocess_vec;
    for (typename unordered_set<Token *>::iterator iter = reprocess.begin();
         iter != reprocess.end(); ++iter)
      reprocess_vec.push_back(*iter);
    reprocess.clear();
    for (typename std::vector<Token *>::iterator iter = reprocess_vec.begin();
         iter != reprocess_vec.end(); ++iter) {
      Token *tok = *iter;
      int32 pos = token2pos[tok];
      // Repeat the processing we did above (for comments, see above).
      for (ForwardLinkT *link = tok->links; link != NULL; link = link->next) {
        if (link->ilabel == 0) {
          IterType following_iter = token2pos.find(link->next_tok);
          if (following_iter != token2pos.end()) {
            int32 next_pos = following_iter->second;
            if (next_pos < pos) {
              following_iter->second = cur_pos++;
              reprocess.insert(link->next_tok);
            }
          }
        }
      }
    }
  }
  KALDI_ASSERT(loop_count < max_loop &&
               "Epsilon loops exist in your decoding "
               "graph (this is not allowed!)");

  topsorted_list->clear();
  topsorted_list->resize(cur_pos, NULL); // create a list with NULLs in between.
  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter)
    (*topsorted_list)[iter->second] = iter->first;
}

template <typename FST, typename Token>
bool LatticeIncrementalDecoderTpl<FST, Token>::GetLattice(CompactLattice *olat) {
  return GetLattice(true, NumFramesDecoded(), olat);
}

template <typename FST, typename Token>
bool LatticeIncrementalDecoderTpl<FST, Token>::GetLattice(bool use_final_probs,
                                                          int32 last_frame_of_chunk,
                                                          CompactLattice *olat) {
  using namespace fst;
  bool not_first_chunk = last_get_lattice_frame_ != 0;
  bool ret = true;

  // last_get_lattice_frame_ is used to record the first frame of the chunk
  // last time we obtain from calling this function. If it reaches
  // last_frame_of_chunk
  // we cannot generate any more chunk
  if (last_get_lattice_frame_ < last_frame_of_chunk) {
    Lattice raw_fst;
    // step 1: Get lattice chunk with initial and final states
    // In this function, we do not create the initial state in
    // the first chunk, and we do not create the final state in the last chunk
    if (!GetIncrementalRawLattice(&raw_fst, use_final_probs, last_get_lattice_frame_,
                                  last_frame_of_chunk, not_first_chunk,
                                  !decoding_finalized_))
      KALDI_ERR << "Unexpected problem when getting lattice";
    // step 2-3
    ret = determinizer_.ProcessChunk(raw_fst, last_get_lattice_frame_,
                                     last_frame_of_chunk);
    last_get_lattice_frame_ = last_frame_of_chunk;
  } else if (last_get_lattice_frame_ > last_frame_of_chunk) {
    KALDI_WARN << "Call GetLattice up to frame: " << last_frame_of_chunk
               << " while the determinizer_ has already done up to frame: "
               << last_get_lattice_frame_;
  }

  // step 4
  if (decoding_finalized_) ret &= determinizer_.Finalize();
  if (olat) {
    *olat = determinizer_.GetDeterminizedLattice();
    ret &= (olat->NumStates() > 0);
  }
  if (!ret) {
    KALDI_WARN << "Last chunk processing failed."
               << " We will retry from frame 0.";
    // Reset determinizer_ and re-determinize from
    // frame 0 to last_frame_of_chunk
    last_get_lattice_frame_ = 0;
    determinizer_.Init();
  }

  return ret;
}

template <typename FST, typename Token>
bool LatticeIncrementalDecoderTpl<FST, Token>::GetIncrementalRawLattice(
    Lattice *ofst, bool use_final_probs, int32 frame_begin, int32 frame_end,
    bool create_initial_state, bool create_final_state) {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;

  if (decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "GetIncrementalRawLattice() with use_final_probs == false";

  unordered_map<Token *, BaseFloat> final_costs_local;

  const unordered_map<Token *, BaseFloat> &final_costs =
      (decoding_finalized_ ? final_costs_ : final_costs_local);
  if (!decoding_finalized_ && use_final_probs)
    ComputeFinalCosts(&final_costs_local, NULL, NULL);

  ofst->DeleteStates();
  unordered_multimap<int, StateId>
      token_label2last_state; // for GetInitialRawLattice
  // initial arcs for the chunk
  if (create_initial_state)
    determinizer_.GetInitialRawLattice(ofst, &token_label2last_state,
                                       token_label_final_cost_);
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  KALDI_ASSERT(frame_end > 0);
  const int32 bucket_count = num_toks_ / 2 + 3;
  unordered_map<Token *, StateId> tok_map(bucket_count);
  // First create all states.
  std::vector<Token *> token_list;
  for (int32 f = frame_begin; f <= frame_end; f++) {
    if (active_toks_[f].toks == NULL) {
      KALDI_WARN << "GetIncrementalRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
    TopSortTokens(active_toks_[f].toks, &token_list);
    for (size_t i = 0; i < token_list.size(); i++)
      if (token_list[i] != NULL) tok_map[token_list[i]] = ofst->AddState();
  }
  // The next statement sets the start state of the output FST.
  // No matter create_initial_state or not , state zero must be the start-state.
  StateId start_state = 0;
  ofst->SetStart(start_state);

  KALDI_VLOG(4) << "init:" << num_toks_ / 2 + 3
                << " buckets:" << tok_map.bucket_count()
                << " load:" << tok_map.load_factor()
                << " max:" << tok_map.max_load_factor();
  // step 1.1: create initial_arc for later appending with the previous chunk
  if (create_initial_state) {
    for (Token *tok = active_toks_[frame_begin].toks; tok != NULL; tok = tok->next) {
      StateId cur_state = tok_map[tok];
      // token_label_map_ is construct during create_final_state
      auto r = token_label_map_.find(tok);
      KALDI_ASSERT(r != token_label_map_.end()); // it should exist
      int32 token_label = r->second;
      auto range = token_label2last_state.equal_range(token_label);
      if (range.first == range.second) {
        KALDI_WARN
            << "The token in the first frame of this chunk does not "
               "exist in the last frame of previous chunk. It should seldom"
               " happen and would be caused by over-pruning in determinization,"
               "e.g. the lattice reaches --max-mem constrain.";
        continue;
      }
      for (auto it = range.first; it != range.second; ++it) {
        // the destination state of the last of the sequence of arcs w.r.t the token
        // label
        // here created by GetInitialRawLattice
        auto state_last_initial = it->second;
        // connect it to the state correponding to the token w.r.t the token label
        // here
        Arc arc(0, 0, Weight::One(), cur_state);
        ofst->AddArc(state_last_initial, arc);
      }
    }
  }
  // step 1.2: create all arcs as GetRawLattice() of LatticeFasterDecoder
  for (int32 f = frame_begin; f <= frame_end; f++) {
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next) {
      StateId cur_state = tok_map[tok];
      for (ForwardLinkT *l = tok->links; l != NULL; l = l->next) {
        // for the arcs outgoing from the last frame Token in this chunk, we will
        // create these arcs in the next chunk
        if (f == frame_end && l->ilabel > 0) continue;
        typename unordered_map<Token *, StateId>::const_iterator iter =
            tok_map.find(l->next_tok);
        KALDI_ASSERT(iter != tok_map.end());
        StateId nextstate = iter->second;
        BaseFloat cost_offset = 0.0;
        if (l->ilabel != 0) { // emitting..
          KALDI_ASSERT(f >= 0 && f < cost_offsets_.size());
          cost_offset = cost_offsets_[f];
        }
        Arc arc(l->ilabel, l->olabel,
                Weight(l->graph_cost, l->acoustic_cost - cost_offset), nextstate);
        ofst->AddArc(cur_state, arc);
      }
      // For the last frame in this chunk, we need to work out a
      // proper final weight for the corresponding state.
      // If use_final_probs == true, we will try to use the final cost we just
      // calculated
      // Otherwise, we use LatticeWeight::One(). We record these cost in the state
      // Later in the code, if create_final_state == true, we will create
      // a specific final state, and move the final costs to the cost of an arc
      // connecting to the final state
      if (f == frame_end) {
        LatticeWeight weight = LatticeWeight::One();
        if (use_final_probs && !final_costs.empty()) {
          typename unordered_map<Token *, BaseFloat>::const_iterator iter =
              final_costs.find(tok);
          if (iter != final_costs.end())
            weight = LatticeWeight(iter->second, 0);
          else
            weight = LatticeWeight::Zero();
        }
        ofst->SetFinal(cur_state, weight);
      }
    }
  }
  // step 1.3 create final_arc for later appending with the next chunk
  if (create_final_state) {
    StateId end_state = ofst->AddState(); // final-state for the chunk
    ofst->SetFinal(end_state, Weight::One());

    token_label_map_.clear();
    token_label_map_.reserve(std::min((int32)1e5, config_.max_active));
    for (Token *tok = active_toks_[frame_end].toks; tok != NULL; tok = tok->next) {
      StateId cur_state = tok_map[tok];
      // We assign an unique state label for each of the token in the last frame
      // of this chunk
      int32 id = token_label_available_idx_++;
      token_label_map_[tok] = id;
      // The final weight has been worked out in the previous for loop and
      // store in the states
      // Here, we create a specific final state, and move the final costs to
      // the cost of an arc connecting to the final state
      KALDI_ASSERT(ofst->Final(cur_state) != Weight::Zero());
      Weight final_weight = ofst->Final(cur_state);
      // Use cost_offsets to guide DeterminizeLatticePruned()
      // For now, we use extra_cost from the decoding stage , which has some
      // "future information", as
      // the final weights of this chunk
      BaseFloat cost_offset = tok->extra_cost - tok->tot_cost;
      // We record these cost_offset, and after we appending two chunks
      // we will cancel them out
      token_label_final_cost_[id] = cost_offset;
      Arc arc(0, id, Times(final_weight, Weight(0, cost_offset)), end_state);
      ofst->AddArc(cur_state, arc);
      ofst->SetFinal(cur_state, Weight::Zero());
    }
  }
  TopSortLatticeIfNeeded(ofst);
  return (ofst->NumStates() > 0);
}

template <typename FST, typename Token>
int32 LatticeIncrementalDecoderTpl<FST, Token>::GetNumToksForFrame(int32 frame) {
  int32 r = 0;
  for (Token *tok = active_toks_[frame].toks; tok; tok = tok->next) r++;
  return r;
}

template <typename FST>
LatticeIncrementalDeterminizer<FST>::LatticeIncrementalDeterminizer(
    const LatticeIncrementalDecoderConfig &config, const TransitionModel &trans_model)
    : config_(config), trans_model_(trans_model) {}

template <typename FST>
void LatticeIncrementalDeterminizer<FST>::Init() {
  final_arc_list_.clear();
  final_arc_list_prev_.clear();
  lat_.DeleteStates();
  determinization_finalized_ = false;
  forward_costs_.clear();
  state_last_initial_offset_ = 2 * config_.max_word_id;
  redeterminized_states_.clear();
  processed_prefinal_states_.clear();
}
template <typename FST>
bool LatticeIncrementalDeterminizer<FST>::AddRedeterminizedState(
    Lattice::StateId nextstate, Lattice *olat, Lattice::StateId *nextstate_copy) {
  using namespace fst;
  bool modified = false;
  StateId nextstate_insert = kNoStateId;
  auto r = redeterminized_states_.insert({nextstate, nextstate_insert});
  if (r.second) { // didn't exist, successfully insert here
    // create a new state w.r.t state
    nextstate_insert = olat->AddState();
    // map from arc.nextstate to nextstate_insert
    r.first->second = nextstate_insert;
    modified = true;
  } else { // else already exist
    // get nextstate_insert
    nextstate_insert = r.first->second;
    KALDI_ASSERT(nextstate_insert != kNoStateId);
    modified = false;
  }
  if (nextstate_copy) *nextstate_copy = nextstate_insert;
  return modified;
}

template <typename FST>
void LatticeIncrementalDeterminizer<FST>::GetRawLatticeForRedeterminizedStates(
    StateId start_state, StateId state,
    const unordered_map<int32, BaseFloat> &token_label_final_cost,
    unordered_multimap<int, LatticeArc::StateId> *token_label2last_state,
    Lattice *olat) {
  using namespace fst;
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;

  auto r = redeterminized_states_.find(state);
  KALDI_ASSERT(r != redeterminized_states_.end());
  auto state_copy = r->second;
  KALDI_ASSERT(state_copy != kNoStateId);
  ArcIterator<CompactLattice> aiter(lat_, state);

  // use state_label in initial arcs
  int state_label = state + state_last_initial_offset_;
  // Moreover, we need to use the forward coast (alpha) of this determinized and
  // appended state to guide the determinization later
  KALDI_ASSERT(state < forward_costs_.size());
  auto alpha_cost = forward_costs_[state];
  Arc arc_initial(0, state_label, LatticeWeight(0, alpha_cost), state_copy);
  if (alpha_cost != std::numeric_limits<BaseFloat>::infinity())
    olat->AddArc(start_state, arc_initial);

  for (; !aiter.Done(); aiter.Next()) {
    const auto &arc = aiter.Value();
    auto laststate_copy = kNoStateId;
    bool proc_nextstate = false;
    auto arc_weight = arc.weight;

    KALDI_ASSERT(arc.olabel == arc.ilabel);
    auto arc_olabel = arc.olabel;

    // the destination of the arc is the final state
    if (lat_.Final(arc.nextstate) != CompactLatticeWeight::Zero()) {
      KALDI_ASSERT(arc_olabel > config_.max_word_id &&
                   arc_olabel < state_last_initial_offset_); // token label
      // create a initial arc

      // Get arc weight here
      // We will include it in arc_last in the following
      CompactLatticeWeight weight_offset;
      // To cancel out the weight on the final arcs, which is (extra cost - forward
      // cost).
      // see token_label_final_cost for more details
      const auto r = token_label_final_cost.find(arc_olabel);
      KALDI_ASSERT(r != token_label_final_cost.end());
      auto cost_offset = r->second;
      weight_offset.SetWeight(LatticeWeight(0, -cost_offset));
      // The arc weight is a combination of original arc weight, above cost_offset
      // and the weights on the final state
      arc_weight = Times(Times(arc_weight, lat_.Final(arc.nextstate)), weight_offset);

      // We create a respective destination state for each final arc
      // later we will connect it to the state correponding to the token w.r.t
      // arc_olabel
      laststate_copy = olat->AddState();
      // the destination state of the last of the sequence of arcs will be recorded
      // and connected to the state corresponding to token w.r.t arc_olabel
      // Notably, we have multiple states for one token label after determinization,
      // hence we use multiset here
      token_label2last_state->insert(
          std::pair<int, StateId>(arc_olabel, laststate_copy));
      arc_olabel = 0; // remove token label
    } else {
      // the arc connects to a non-final state (redeterminized state)
      KALDI_ASSERT(arc_olabel < config_.max_word_id); // no token label
      KALDI_ASSERT(arc_olabel);
      // get the nextstate_copy w.r.t arc.nextstate
      StateId nextstate_copy = kNoStateId;
      proc_nextstate = AddRedeterminizedState(arc.nextstate, olat, &nextstate_copy);
      KALDI_ASSERT(nextstate_copy != kNoStateId);
      laststate_copy = nextstate_copy;
    }
    auto &state_seqs = arc_weight.String();
    // create new arcs w.r.t arc
    // the following is for a normal arc
    // We generate a linear sequence of arcs sufficient to contain all the
    // transition-ids on the string
    auto prev_state = state_copy; // from state_copy
    for (auto &j : state_seqs) {
      auto cur_state = olat->AddState();
      Arc arc(j, 0, LatticeWeight::One(), cur_state);
      olat->AddArc(prev_state, arc);
      prev_state = cur_state;
    }

    // connect previous sequence of arcs to the laststate_copy
    // the weight on the previous arc is stored in the arc to laststate_copy here
    Arc arc_last(0, arc_olabel, arc_weight.Weight(), laststate_copy);
    olat->AddArc(prev_state, arc_last);

    // not final state && previously didn't process this state
    if (proc_nextstate)
      GetRawLatticeForRedeterminizedStates(start_state, arc.nextstate,
                                           token_label_final_cost,
                                           token_label2last_state, olat);
  }
}
template <typename FST>
void LatticeIncrementalDeterminizer<FST>::GetRedeterminizedStates() {
  using namespace fst;
  processed_prefinal_states_.clear();
  // go over all prefinal state
  KALDI_ASSERT(final_arc_list_prev_.size());
  unordered_set<StateId> prefinal_states;

  for (auto &i : final_arc_list_prev_) {
    auto prefinal_state = i.first;
    ArcIterator<CompactLattice> aiter(lat_, prefinal_state);
    KALDI_ASSERT(lat_.NumArcs(prefinal_state) > i.second);
    aiter.Seek(i.second);
    auto final_arc = aiter.Value();
    auto final_weight = lat_.Final(final_arc.nextstate);
    KALDI_ASSERT(final_weight != CompactLatticeWeight::Zero());
    auto num_frames = Times(final_arc.weight, final_weight).String().size();
    // If the state is too far from the end of the current appended lattice,
    // we leave the non-final arcs unchanged and only redeterminize the final
    // arcs by the following procedure.
    // We also do above things once we prepare to redeterminize the start state.
    if (num_frames <= config_.redeterminize_max_frames && prefinal_state != 0)
      processed_prefinal_states_[prefinal_state] = prefinal_state;
    else {
      KALDI_VLOG(7) << "Impose a limit of " << config_.redeterminize_max_frames
                    << " on how far back in time we will redeterminize states. "
                    << num_frames << " frames in this arc. ";

      auto new_prefinal_state = lat_.AddState();
      forward_costs_.resize(new_prefinal_state + 1);
      forward_costs_[new_prefinal_state] = forward_costs_[prefinal_state];

      std::vector<CompactLatticeArc> arcs_remained;
      for (aiter.Reset(); !aiter.Done(); aiter.Next()) {
        auto arc = aiter.Value();
        bool remain_the_arc = true; // If we remain the arc, the state will not be
                                    // re-determinized, vice versa.
        if (arc.olabel > config_.max_word_id) { // final arc
          KALDI_ASSERT(arc.olabel < state_last_initial_offset_);
          KALDI_ASSERT(lat_.Final(arc.nextstate) != CompactLatticeWeight::Zero());
          remain_the_arc = false;
        } else {
          int num_frames_exclude_arc = num_frames - arc.weight.String().size();
          // destination-state of the arc is further than redeterminize_max_frames
          // from the most recent frame we are determinizing
          if (num_frames_exclude_arc > config_.redeterminize_max_frames)
            remain_the_arc = true;
          else {
            // destination-state of the arc is no further than
            // redeterminize_max_frames from the most recent frame we are
            // determinizing
            auto r = final_arc_list_prev_.find(arc.nextstate);
            // destination-state of the arc is not prefinal state
            if (r == final_arc_list_prev_.end()) remain_the_arc = true;
            // destination-state of the arc is prefinal state
            else
              remain_the_arc = false;
          }
        }

        if (remain_the_arc)
          arcs_remained.push_back(arc);
        else
          lat_.AddArc(new_prefinal_state, arc);
      }
      CompactLatticeArc arc_to_new(0, 0, CompactLatticeWeight::One(),
                                   new_prefinal_state);
      arcs_remained.push_back(arc_to_new);

      lat_.DeleteArcs(prefinal_state);
      for (auto &i : arcs_remained) lat_.AddArc(prefinal_state, i);
      processed_prefinal_states_[prefinal_state] = new_prefinal_state;
    }
  }
  KALDI_VLOG(8) << "states of the lattice after GetRedeterminizedStates: "
                << lat_.NumStates();
}

// This function is specifically designed to obtain the initial arcs for a chunk
// We have multiple states for one token label after determinization
template <typename FST>
void LatticeIncrementalDeterminizer<FST>::GetInitialRawLattice(
    Lattice *olat,
    unordered_multimap<int, LatticeArc::StateId> *token_label2last_state,
    const unordered_map<int32, BaseFloat> &token_label_final_cost) {
  using namespace fst;
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;

  GetRedeterminizedStates();

  olat->DeleteStates();
  token_label2last_state->clear();

  auto start_state = olat->AddState();
  olat->SetStart(start_state);
  // go over all prefinal states after preprocessing
  for (auto &i : processed_prefinal_states_) {
    auto prefinal_state = i.second;
    bool modified = AddRedeterminizedState(prefinal_state, olat);
    if (modified)
      GetRawLatticeForRedeterminizedStates(start_state, prefinal_state,
                                           token_label_final_cost,
                                           token_label2last_state, olat);
  }
}

template <typename FST>
bool LatticeIncrementalDeterminizer<FST>::ProcessChunk(Lattice &raw_fst,
                                                       int32 first_frame,
                                                       int32 last_frame) {
  bool not_first_chunk = first_frame != 0;
  bool ret = true;
  // step 2: Determinize the chunk
  CompactLattice clat;
  // We do determinization with beam pruning here
  // Only if we use a beam larger than (config_.beam+config_.lattice_beam) here, we
  // can guarantee no final or initial arcs in clat are pruned by this function.
  // These pruned final arcs can hurt oracle WER performance in the final lattice
  // (also result in less lattice density) but they seldom hurt 1-best WER.
  // Since pruning behaviors in DeterminizeLatticePhonePrunedWrapper and
  // PruneActiveTokens are not the same, to get similar lattice density as
  // LatticeFasterDecoder, we need to use a slightly larger beam here
  // than the lattice_beam used PruneActiveTokens. Hence the beam we use is
  // (0.1 + config_.lattice_beam)
  ret &= DeterminizeLatticePhonePrunedWrapper(
      trans_model_, &raw_fst, (config_.lattice_beam + 0.1), &clat, config_.det_opts);

  // step 3: Appending the new chunk in clat to the old one in lat_
  ret &= AppendLatticeChunks(clat, not_first_chunk);

  ret &= (lat_.NumStates() > 0);
  KALDI_VLOG(2) << "Frame: ( " << first_frame << " , " << last_frame << " )"
                << " states of the chunk: " << clat.NumStates()
                << " states of the lattice: " << lat_.NumStates();
  return ret;
}

template <typename FST>
bool LatticeIncrementalDeterminizer<FST>::AppendLatticeChunks(CompactLattice clat,
                                                              bool not_first_chunk) {
  using namespace fst;
  CompactLattice *olat = &lat_;

  // later we need to calculate forward_costs_ for clat
  TopSortCompactLatticeIfNeeded(&clat);

  // step 3.1: Appending new chunk to the old one
  int32 state_offset = olat->NumStates();
  if (not_first_chunk) {
    state_offset--; // since we do not append initial state in the first chunk
    // remove arcs from redeterminized_states_
    for (auto i : redeterminized_states_) {
      olat->DeleteArcs(i.first);
      olat->SetFinal(i.first, CompactLatticeWeight::Zero());
    }
    redeterminized_states_.clear();
  } else {
    forward_costs_.push_back(0); // for the first state
  }
  forward_costs_.resize(state_offset + clat.NumStates(),
                        std::numeric_limits<BaseFloat>::infinity());

  // Here we construct a map from the original prefinal state to the prefinal states
  // for later use
  unordered_map<StateId, StateId> invert_processed_prefinal_states;
  invert_processed_prefinal_states.reserve(processed_prefinal_states_.size());
  for (auto i : processed_prefinal_states_)
    invert_processed_prefinal_states[i.second] = i.first;
  for (StateIterator<CompactLattice> siter(clat); !siter.Done(); siter.Next()) {
    auto s = siter.Value();
    StateId state_appended = kNoStateId;
    // We do not copy initial state, which exists except the first chunk
    if (!not_first_chunk || s != 0) {
      state_appended = s + state_offset;
      auto r = olat->AddState();
      KALDI_ASSERT(state_appended == r);
      olat->SetFinal(state_appended, clat.Final(s));
    }

    for (ArcIterator<CompactLattice> aiter(clat, s); !aiter.Done(); aiter.Next()) {
      const auto &arc = aiter.Value();

      StateId source_state = kNoStateId;
      // We do not copy initial arcs, which exists except the first chunk.
      // These arcs will be taken care later in step 3.2
      CompactLatticeArc arc_appended(arc);
      arc_appended.nextstate += state_offset;
      // In the first chunk, there could be a final arc starting from state 0, and we
      // process it here
      // In the last chunk, there could be a initial arc ending in final state, and
      // we process it in "process initial arcs" in the following
      bool is_initial_state = (not_first_chunk && s == 0);
      if (!is_initial_state) {
        KALDI_ASSERT(state_appended != kNoStateId);
        KALDI_ASSERT(arc.olabel < state_last_initial_offset_);
        source_state = state_appended;
        // process final arcs
        if (arc.olabel > config_.max_word_id) {
          // record final_arc in this chunk for the step 3.2 in the next call
          KALDI_ASSERT(arc.olabel < state_last_initial_offset_);
          KALDI_ASSERT(clat.Final(arc.nextstate) != CompactLatticeWeight::Zero());
          // state_appended shouldn't be in invert_processed_prefinal_states
          // So we do not need to map it
          final_arc_list_.insert(
              pair<int32, size_t>(state_appended, aiter.Position()));
        }
        olat->AddArc(source_state, arc_appended);
      } else { // process initial arcs
        // a special olabel in the arc that corresponds to the identity of the
        // source-state of the last arc, we use its StateId and a offset here, called
        // state_label
        auto state_label = arc.olabel;
        KALDI_ASSERT(state_label > config_.max_word_id);
        KALDI_ASSERT(state_label >= state_last_initial_offset_);
        source_state = state_label - state_last_initial_offset_;
        arc_appended.olabel = 0;
        arc_appended.ilabel = 0;
        CompactLatticeWeight weight_offset;
        // remove alpha in weight
        weight_offset.SetWeight(LatticeWeight(0, -forward_costs_[source_state]));
        arc_appended.weight = Times(arc_appended.weight, weight_offset);

        // if it is an extra prefinal state, we should use its original prefinal
        // state
        int arc_offset = 0;
        auto r = invert_processed_prefinal_states.find(source_state);
        if (r != invert_processed_prefinal_states.end() && r->second != r->first) {
          source_state = r->second;
          arc_offset = olat->NumArcs(source_state);
        }

        if (clat.Final(arc.nextstate) != CompactLatticeWeight::Zero()) {
          // it should be the last chunk
          olat->SetFinal(source_state, 
              Times(arc_appended.weight, clat.Final(arc.nextstate)));
        } else {
          // append lattice chunk and remove Epsilon together
          for (ArcIterator<CompactLattice> aiter_postinitial(clat, arc.nextstate);
               !aiter_postinitial.Done(); aiter_postinitial.Next()) {
            auto arc_postinitial(aiter_postinitial.Value());
            arc_postinitial.weight =
                Times(arc_appended.weight, arc_postinitial.weight);
            arc_postinitial.nextstate += state_offset;
            olat->AddArc(source_state, arc_postinitial);
            if (arc_postinitial.olabel > config_.max_word_id) {
              KALDI_ASSERT(arc_postinitial.olabel < state_last_initial_offset_);
              final_arc_list_.insert(pair<int32, size_t>(
                  source_state, aiter_postinitial.Position() + arc_offset));
            }
          }
        }
      }
      // update forward_costs_ (alpha)
      KALDI_ASSERT(arc_appended.nextstate < forward_costs_.size());
      auto &alpha_nextstate = forward_costs_[arc_appended.nextstate];
      auto &weight = arc_appended.weight.Weight();
      alpha_nextstate =
          std::min(alpha_nextstate,
                   forward_costs_[source_state] + weight.Value1() + weight.Value2());
    }
  }
  KALDI_ASSERT(olat->NumStates() == clat.NumStates() + state_offset);
  KALDI_VLOG(8) << "states of the lattice: " << olat->NumStates();

  if (!not_first_chunk) {
    olat->SetStart(0); // Initialize the first chunk for olat
  } else {
    // The extra prefinal states generated by
    // GetRedeterminizedStates are removed here, while splicing
    // the compact lattices together
    for (auto &i : processed_prefinal_states_) {
      auto prefinal_state = i.first;
      auto new_prefinal_state = i.second;
      // It is without an extra prefinal state, hence do not need to process
      if (prefinal_state == new_prefinal_state) continue;
      for (ArcIterator<CompactLattice> aiter(*olat, new_prefinal_state);
           !aiter.Done(); aiter.Next())
        olat->AddArc(prefinal_state, aiter.Value());
      olat->DeleteArcs(new_prefinal_state);
      olat->SetFinal(new_prefinal_state, CompactLatticeWeight::Zero());
    }
  }

  final_arc_list_.swap(final_arc_list_prev_);
  final_arc_list_.clear();

  return true;
}

template <typename FST>
bool LatticeIncrementalDeterminizer<FST>::Finalize() {
  using namespace fst;
  auto *olat = &lat_;
  // The lattice determinization only needs to be finalized once
  if (determinization_finalized_) return true;
  // step 4: remove dead states
  if (config_.final_prune_after_determinize)
    PruneLattice(config_.lattice_beam, olat);
  else
    Connect(olat); // Remove unreachable states... there might be

  KALDI_VLOG(2) << "states of the lattice: " << olat->NumStates();
  determinization_finalized_ = true;

  return (olat->NumStates() > 0);
}

// Instantiate the template for the combination of token types and FST types
// that we'll need.
template class LatticeIncrementalDecoderTpl<fst::Fst<fst::StdArc>, decoder::StdToken>;
template class LatticeIncrementalDecoderTpl<fst::VectorFst<fst::StdArc>,
                                            decoder::StdToken>;
template class LatticeIncrementalDecoderTpl<fst::ConstFst<fst::StdArc>,
                                            decoder::StdToken>;
template class LatticeIncrementalDecoderTpl<fst::GrammarFst, decoder::StdToken>;

template class LatticeIncrementalDecoderTpl<fst::Fst<fst::StdArc>,
                                            decoder::BackpointerToken>;
template class LatticeIncrementalDecoderTpl<fst::VectorFst<fst::StdArc>,
                                            decoder::BackpointerToken>;
template class LatticeIncrementalDecoderTpl<fst::ConstFst<fst::StdArc>,
                                            decoder::BackpointerToken>;
template class LatticeIncrementalDecoderTpl<fst::GrammarFst,
                                            decoder::BackpointerToken>;

} // end namespace kaldi.
