// decoder/lattice-biglm-faster-decoder-combine.cc

// Copyright 2013-2019  Johns Hopkins University (Author: Daniel Povey)
//                2019  Zhehuai Chen
//                2019  Hang Lyu               

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

#include "decoder/lattice-biglm-faster-decoder-combine.h"
#include "lat/lattice-functions.h"

namespace kaldi {

template<typename Token>
BucketQueue<Token>::BucketQueue(BaseFloat cost_scale) :
    cost_scale_(cost_scale) {
  // NOTE: we reserve plenty of elements to avoid expensive reallocations
  // later on. Normally, the size is a little bigger than (adaptive_beam +
  // 15) * cost_scale.
  int32 bucket_size = (15 + 20) * cost_scale_;
  buckets_.resize(bucket_size);
  bucket_offset_ = 15 * cost_scale_;
  first_nonempty_bucket_index_ = bucket_size - 1;
  first_nonempty_bucket_ = &buckets_[first_nonempty_bucket_index_];
  bucket_size_tolerance_ = 1.2 * bucket_size;
}

template<typename Token>
void BucketQueue<Token>::Push(Token *tok) {
  size_t bucket_index = std::floor(tok->tot_cost * cost_scale_) +
                        bucket_offset_;
  if (bucket_index >= buckets_.size()) {
    int32 margin = 10;  // a margin which is used to reduce re-allocate
                        // space frequently
    if (static_cast<int32>(bucket_index) > 0) {
      buckets_.resize(bucket_index + margin);
    } else {  // less than 0
      int32 increase_size = - static_cast<int32>(bucket_index) + margin;
      buckets_.resize(buckets_.size() + increase_size);
      // translation
      for (size_t i = buckets_.size() - 1; i >= increase_size; i--) {
        buckets_[i].swap(buckets_[i - increase_size]);
      }
      bucket_offset_ = bucket_offset_ + increase_size;
      bucket_index += increase_size;
      first_nonempty_bucket_index_ = bucket_index;
    }
    first_nonempty_bucket_ = &buckets_[first_nonempty_bucket_index_];
  }
  tok->in_queue = true;
  buckets_[bucket_index].push_back(tok);
  if (bucket_index < first_nonempty_bucket_index_) {
    first_nonempty_bucket_index_ = bucket_index;
    first_nonempty_bucket_ = &buckets_[first_nonempty_bucket_index_];
  }
}

template<typename Token>
Token* BucketQueue<Token>::Pop() {
  while (true) {
    if (!first_nonempty_bucket_->empty()) {
      Token *ans = first_nonempty_bucket_->back();
      first_nonempty_bucket_->pop_back();
      if (ans->in_queue) {  // If ans->in_queue is false, this means it is a
                            // duplicate instance of this Token that was left
                            // over when a Token's best_cost changed, and the
                            // Token has already been processed(so conceptually,
                            // it is not in the queue).
        ans->in_queue = false;
        return ans;
      }
    }
    if (first_nonempty_bucket_->empty()) {
      for (; first_nonempty_bucket_index_ + 1 < buckets_.size();
           first_nonempty_bucket_index_++) {
        if (!buckets_[first_nonempty_bucket_index_].empty()) break;
      }
      first_nonempty_bucket_ = &buckets_[first_nonempty_bucket_index_];
      if (first_nonempty_bucket_->empty()) return NULL;
    }
  }
}

template<typename Token>
void BucketQueue<Token>::Clear() {
  for (size_t i = first_nonempty_bucket_index_; i < buckets_.size(); i++) {
    buckets_[i].clear();
  }
  if (buckets_.size() > bucket_size_tolerance_) {
    buckets_.resize(bucket_size_tolerance_);
    bucket_offset_ = 15 * cost_scale_;
  }
  first_nonempty_bucket_index_ = buckets_.size() - 1;
  first_nonempty_bucket_ = &buckets_[first_nonempty_bucket_index_];
}

// instantiate this class once for each thing you have to decode.
template <typename FST, typename Token>
LatticeBiglmFasterDecoderCombineTpl<FST, Token>::LatticeBiglmFasterDecoderCombineTpl(
    const FST &fst,
    const LatticeBiglmFasterDecoderCombineConfig &config,
    fst::DeterministicOnDemandFst<FST::Arc> *lm_diff_fst):
    fst_(&fst), delete_fst_(false), lm_diff_fst_(lm_diff_fst), config_(config),
    num_toks_(0), cur_queue_(config_.cost_scale) {
  config.Check();
  KALDI_ASSERT(fst_->Start() != fst::kNoStateId &&
               lm_diff_fst_->Start() != fst::kNoStateId);
}


template <typename FST, typename Token>
LatticeBiglmFasterDecoderCombineTpl<FST, Token>::LatticeBiglmFasterDecoderCombineTpl(
    const LatticeBiglmFasterDecoderCombineConfig &config, FST *fst,
    fst::DeterministicOnDemandFst<FST::ARC> *lm_diff_fst):
    fst_(fst), delete_fst_(true), lm_diff_fst_(lm_diff_fst), config_(config),
    num_toks_(0), cur_queue_(config_.cost_scale) {
  config.Check();
  KALDI_ASSERT(fst_->Start() != fst::kNoStateId &&
               lm_diff_fst_->Start() != fst::kNoStateId);
}


template <typename FST, typename Token>
LatticeBiglmFasterDecoderCombineTpl<FST, Token>::~LatticeBiglmFasterDecoderCombineTpl() {
  ClearActiveTokens();
  if (delete_fst_) delete fst_;
}

template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::InitDecoding() {
  // clean up from last time:
  cost_offsets_.clear();
  ClearActiveTokens();  // num_toks_ is set to 0

  warned_ = false;
  warned_noarc_ = false;
  decoding_finalized_ = false;
  complete_frame_ = -1;

  final_costs_.clear();
  adaptive_beam_ = config_.beam;

  for (size_t i = token_map_.size(); i >= 0; i--) {
    token_map_[i]->clear();
    delete token_map_[i];
    best_token_map_[i]->clear();
    delete best_token_map_[i];
  }
  token_map_.resize(1);
  best_token_map_.resize(1);
  best_token_.resize(1);

  // initialize
  token_map_[0] = new PairIdToTokenMap();
  best_token_map_[0] = new StateIdToTokenMap();

  StateId base_start_state = fst_->Start();
  StateId lm_start_state = lm_diff_fst->Start();
  PairId start_state = ConstructPair(base_start_state, lm_start_state);
  Token *start_tok = new Token(0.0, std::numeric_limits<BaseFloat>::infinity(),
      base_start_state, lm_start_state, NULL, NULL, NULL);

  active_toks_.resize(1);
  active_toks_[0].toks = start_tok;

  (*token_map_[0])[start_state] = start_tok;  // initialize current tokens map
  (*best_token_map_[0])[base_start_state] = start_tok;
  best_token_[0] = start_tok;

  num_toks_++;
  cost_offsets_.resize(1);
  cost_offsets_[0] = 0.0;
}

// Returns true if any kind of traceback is available (not necessarily from
// a final state).  It should only very rarely return false; this indicates
// an unusual search error.
template <typename FST, typename Token>
bool LatticeBiglmFasterDecoderCombineTpl<FST, Token>::Decode(
    DecodableInterface *decodable) {
  InitDecoding();

  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.
  while (!decodable->IsLastFrame(NumFramesDecoded() - 1)) {
    ProcessForFrame(decodable);
    if (NumFramesDecoded() % config_.backfill_interval == 0)
      complete_frame_ = DoBackfill();
    // Only the tokens and forwardlinks of the complete expanded frames will
    // be processed for saving memory.
    if (NumFramesDecoded() % config_.prune_interval == 0)
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
  }
  // A complete token list of the last frame will be generated in FinalizeDecoding()
  FinalizeDecoding();

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !active_toks_.empty() && active_toks_.back().toks != NULL;
}


// Outputs an FST corresponding to the single best path through the lattice.
template <typename FST, typename Token>
bool LatticeBiglmFasterDecoderCombineTpl<FST, Token>::GetBestPath(
    Lattice *olat,
    bool use_final_probs) {
  Lattice raw_lat;
  GetRawLattice(&raw_lat, use_final_probs);
  ShortestPath(raw_lat, olat);
  return (olat->NumStates() != 0);
}


// Outputs an FST corresponding to the raw, state-level lattice
template <typename FST, typename Token>
bool LatticeBiglmFasterDecoderCombineTpl<FST, Token>::GetRawLattice(
    Lattice *ofst,
    bool use_final_probs) {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;
  // Note: you can't use the old interface (Decode()) if you want to
  // get the lattice with use_final_probs = false.  You'd have to do
  // InitDecoding() and then AdvanceDecoding().
  if (decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "GetRawLattice() with use_final_probs == false";

  if (!decoding_finalized_ && use_final_probs) {
    // Process the non-emitting arcs for the unfinished last frame.
    ProcessNonemitting();
  }


  unordered_map<Token*, BaseFloat> final_costs_local;

  const unordered_map<Token*, BaseFloat> &final_costs =
      (decoding_finalized_ ? final_costs_ : final_costs_local);
  if (!decoding_finalized_ && use_final_probs)
    ComputeFinalCosts(&final_costs_local, NULL, NULL);

  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  int32 num_frames = active_toks_.size() - 1;
  KALDI_ASSERT(num_frames > 0);
  const int32 bucket_count = num_toks_/2 + 3;
  unordered_map<Token*, StateId> tok_map(bucket_count);
  // First create all states.
  std::vector<Token*> token_list;
  for (int32 f = 0; f <= num_frames; f++) {
    if (active_toks_[f].toks == NULL) {
      KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
    TopSortTokens(active_toks_[f].toks, &token_list);
    for (size_t i = 0; i < token_list.size(); i++)
      if (token_list[i] != NULL)
        tok_map[token_list[i]] = ofst->AddState();
  }
  // The next statement sets the start state of the output FST.  Because we
  // topologically sorted the tokens, state zero must be the start-state.
  ofst->SetStart(0);

  KALDI_VLOG(4) << "init:" << num_toks_/2 + 3 << " buckets:"
                << tok_map.bucket_count() << " load:" << tok_map.load_factor()
                << " max:" << tok_map.max_load_factor();
  // Now create all arcs.
  for (int32 f = 0; f <= num_frames; f++) {
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next) {
      StateId cur_state = tok_map[tok];
      for (ForwardLinkT *l = tok->links;
           l != NULL;
           l = l->next) {
        typename unordered_map<Token*, StateId>::const_iterator
            iter = tok_map.find(l->next_tok);
        StateId nextstate = iter->second;
        KALDI_ASSERT(iter != tok_map.end());
        BaseFloat cost_offset = 0.0;
        if (l->ilabel != 0) {  // emitting..
          KALDI_ASSERT(f >= 0 && f < cost_offsets_.size());
          cost_offset = cost_offsets_[f];
        }
        Arc arc(l->ilabel, l->olabel,
                Weight(l->graph_cost, l->acoustic_cost - cost_offset),
                nextstate);
        ofst->AddArc(cur_state, arc);
      }
      if (f == num_frames) {
        if (use_final_probs && !final_costs.empty()) {
          typename unordered_map<Token*, BaseFloat>::const_iterator
              iter = final_costs.find(tok);
          if (iter != final_costs.end())
            ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
        } else {
          ofst->SetFinal(cur_state, LatticeWeight::One());
        }
      }
    }
  }
  
  return (ofst->NumStates() > 0);
}

// This function is now deprecated, since now we do determinization from outside
// the LatticeFasterDecoder class.  Outputs an FST corresponding to the
// lattice-determinized lattice (one path per word sequence).
template <typename FST, typename Token>
bool LatticeBiglmFasterDecoderCombineTpl<FST, Token>::GetLattice(
    CompactLattice *ofst,
    bool use_final_probs) {
  Lattice raw_fst;
  GetRawLattice(&raw_fst, use_final_probs);
  Invert(&raw_fst);  // make it so word labels are on the input.
  // (in phase where we get backward-costs).
  fst::ILabelCompare<LatticeArc> ilabel_comp;
  ArcSort(&raw_fst, ilabel_comp);  // sort on ilabel; makes
  // lattice-determinization more efficient.

  fst::DeterminizeLatticePrunedOptions lat_opts;
  lat_opts.max_mem = config_.det_opts.max_mem;

  DeterminizeLatticePruned(raw_fst, config_.lattice_beam, ofst, lat_opts);
  raw_fst.DeleteStates();  // Free memory-- raw_fst no longer needed.
  Connect(ofst);  // Remove unreachable states... there might be
  // a small number of these, in some cases.
  // Note: if something went wrong and the raw lattice was empty,
  // we should still get to this point in the code without warnings or failures.
  return (ofst->NumStates() != 0);
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

// FindOrAddToken either locates a token in hash map "token_map", or if
// necessary inserts a new, empty token (i.e. with no forward links) for the
// current frame.
// [note: it's inserted if necessary into "token_map", "token_best_map" and
// also into the singly linked list of tokens active on this frame (whose head
// is at active_toks_[frame]).  The token_list_index argument is used to index
// into the active_toks_ array.
template <typename FST, typename Token>
inline Token* LatticeBiglmFasterDecoderCombineTpl<FST, Token>::FindOrAddToken(
    PairId state, int32 token_list_index, BaseFloat tot_cost,
    Token *backpointer, PairIdToTokenMap *token_map, bool *changed) {
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  KALDI_ASSERT(token_list_index < active_toks_.size());
  Token* &toks = active_toks_[token_list_index].toks;
  typename PairIdToTokenMap::iterator e_found = token_map->find(state);
  
  StateId base_state = PairToBaseState(state);
  StateId lm_state = PairToLmState(state);
  typename StateIdToTokenMap::iterator e_best_found =
    token_best_map->find(base_state);
  
  if (e_found == token_map->end()) {  // no such token presently.
    const BaseFloat extra_cost = 0.0;
    cosnt BaseFloat backward_cost = std::numeric_limits<BaseFloat>::infinity();
    // tokens on the currently final frame have zero extra_cost
    // as any of them could end up
    // on the winning path.
    Token *new_tok = new Token (tot_cost, backward_cost, base_state,
                                lm_state, NULL, toks, backpointer);
    // NULL: no forward links yet
    toks = new_tok;
    num_toks_++;
    // insert into the map
    (*token_map)[state] = new_tok;
    if (changed) *changed = true;
    // Check token_best_map
    return new_tok;
  } else {
    Token *tok = e_found->second;  // There is an existing Token for this state.
    if (tok->tot_cost > tot_cost) {  // replace old token
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
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::PruneForwardLinks(
    int32 frame_plus_one, bool *backward_costs_changed,
    bool *links_pruned, BaseFloat delta) {
  // delta is the amount by which the extra_costs must change
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned

  *backward_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  if (active_toks_[frame_plus_one].toks == NULL) {  // empty list; should not happen.
    if (!warned_) {
      KALDI_WARN << "No tokens alive [doing pruning].. warning first "
          "time only for each utterance\n";
      warned_ = true;
    }
  }

  Token* &best_token = best_token_[frame_plus_one];
  BaseFloat best_cost = best_token->tot_cost + best_token->backward_cost;
  BaseFloat threshold = best_cost + config_.lattice_beam;
  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.
  bool changed = true;  // difference new minus old extra cost >= delta ?
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLinkT *link, *prev_link = NULL;
      // will recompute tok_extra_cost for tok.
      BaseFloat tok_backward_cost = std::numeric_limits<BaseFloat>::infinity();
      // tok_extra_cost is the best (min) of link_extra_cost of outgoing links
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_backward_cost = link->acoustic_cost + link->graph_cost +
          next_tok->backward_cost;
        BaseFloat link_cost = link_backward_cost + tok->tot_cost;
        // link_exta_cost is the difference in score between the best paths
        // through link source state and through link destination state
        KALDI_ASSERT(link_cost == link_cost);  // check for NaN
        if (link_cost > threshold) {  // excise link
          ForwardLinkT *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link;  // advance link but leave prev_link the same.
          *links_pruned = true;
        } else {   // keep the link
          if (link_cost < best_cost) {  // this is just a precaution.
            if (link_cost < best_cost - 0.01)
              KALDI_WARN << "This link's cost is smaller than the best one."
            best_token = tok;
            best_cost = link_cost;
            threshold = best_cost + config_.lattice_beam;
          }
          if (link_backward_cost < tok_backward_cost)
            tok_backward_cost = link_backward_cost;
          prev_link = link;  // move to next link
          link = link->next;
        }
      }  // for all outgoing links
      if (fabs(tok_backward_cost - tok->backward_cost) > delta)
        changed = true;   // difference new minus old is bigger than delta
      tok->backward_cost = tok_backward_cost;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    }  // for all Token on active_toks_[frame]
    if (changed) *backward_costs_changed = true;

    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  } // while changed
}

// PruneForwardLinksFinal is a version of PruneForwardLinks that we call
// on the final frame.  If there are final tokens active, it uses
// the final-probs for pruning, otherwise it treats all tokens as final.
template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::PruneForwardLinksFinal() {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame_plus_one = active_toks_.size() - 1;

  if (active_toks_[frame_plus_one].toks == NULL)  // empty list; should not happen.
    KALDI_WARN << "No tokens alive at end of file";

  typedef typename unordered_map<Token*, BaseFloat>::const_iterator IterType;
  ComputeFinalCosts(&final_costs_, &final_relative_cost_, &final_best_cost_);
  decoding_finalized_ = true;

  // Initialize the backward_costs of the tokens on the final frame.
  // We will recompute tok_backward_cost.  It has a term in it that corresponds
  // to the "final-prob", so instead of initializing tok_backward_cost to -alpha
  // below we set it to the difference between the (score+final_prob) of this
  // token and the best such (score+final_prob) minus alpha.
  BaseFloat best_cost = std::numeric_limits<BaseFloat>::infinity();
  for (Token *tok = active_toks_[frame_plus_one].toks; tok != NULL;
       tok = tok->next) {
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
    tok->backward_cost = -tok->tot_cost + (tok->tot_cost + final_cost -
        final_best_cost_);  // difference in brackets is >= 0
    best_cost = min(best_cost, tok->tot_cost + tok->backward_cost);
  }

  BaseFloat threshold = best_cost + config_.lattice_beam;
  // Now go through tokens on this frame, pruning forward links...  may have to
  // iterate a few times until there is no more change, because the list is not
  // in topological order.  This is a modified version of the code in
  // PruneForwardLinks, but here we also take account of the final-probs.
  bool changed = true;
  BaseFloat delta = 1.0e-05;
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLinkT *link, *prev_link = NULL;
      // tok_backward_cost will be a "min" over either directly being final, or
      // being indirectly final through other links, and the loop below may
      // decrease its value:
      BaseFloat tok_backward_cost = std::numeric_limits<BaseFloat>::infinity();
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        BaseFloat link_backward_cost = next_tok->backward_cost + 
          link->acoustic_cost + link->graph_cost;
        BaseFloat link_cost = link_backward_cost + tok->tot_cost;
        if (link_cost > threshold) {  // excise link
          ForwardLinkT *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_cost < best_cost) { // this is just a precaution.
            if (link_cost < best_cost - 0.01)
              KALDI_WARN << "Negative extra_cost: " << link_cost;
            best_cost = link_cost;
            threshold = best_cost + config_.lattice_beam;
          }
          if (link_backward_cost < tok_backward_cost)
            tok_backward_cost = link_backward_cost;
          prev_link = link;
          link = link->next;
        }
      }
      // prune away tokens worse than lattice_beam above best path.  This step
      // was not necessary in the non-final case because then, this case
      // showed up as having no forward links.  Here, the tok_extra_cost has
      // an extra component relating to the final-prob.
      if (tok_backward_cost > threshold)
        tok_backward_cost = std::numeric_limits<BaseFloat>::infinity();
      // to be pruned in PruneTokensForFrame

      if (!ApproxEqual(tok->backward_cost, tok_backward_cost, delta))
        changed = true;
      tok->backward_cost = tok_backward_cost; // will be +infinity or <= lattice_beam_.
    }
  } // while changed
}


template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::PruneForwardLinksWithoutRecompute(
    int32 token_list_index) {
  KALDI_ASSERT(token_list_index >= 0 && token_list_index < active_toks_.size());
  if (active_toks_[token_list_index].toks == NULL) {  // empty list
                                                      // should not happen.
    if (!warned_) {
      KALDI_WARN << "No tokens alive [doing pruning].. warning first "
          "time only for each utterance\n";
      warned_ = true;
    }
  }

  for (Token *tok = active_toks_[token_list_index].toks; tok != NULL;
      tok = tok->next) {
    ForwardLinkT *link, *prev_link = NULL;
    for (link = tok->links; link != NULL; ) {
      // See if we need to excise this link...
      Token *next_tok = link->next_tok;
      if (next_tok->backward_cost ==
          std::numeric_limits<BaseFloat>::infinity()) {  // excise link
        ForwardLinkT *next_link = link->next;
        if (prev_link != NULL) prev_link->next = next_link;
        else tok->links = next_link;
        delete link;
        link = next_link;  // advance link but leave prev_link the same.
      } else {   // keep the link
        prev_link = link;  // move to next link
        link = link->next;
      }
    }
  }
}


template <typename FST, typename Token>
BaseFloat LatticeBiglmFasterDecoderCombineTpl<FST, Token>::FinalRelativeCost() const {
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
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::PruneTokensForFrame(
    int32 frame_plus_one) {
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;
  if (toks == NULL)
    KALDI_WARN << "No tokens alive [doing pruning]";
  Token *tok, *next_tok, *prev_tok = NULL;
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
    if (tok->backward_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      if (prev_tok != NULL) prev_tok->next = tok->next;
      else toks = tok->next;
      delete tok;
      num_toks_--;
    } else {  // fetch next Token
      prev_tok = tok;
    }
  }
}


template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::PruneTokensForFrameFromMap(
    int32 token_list_index) {
  KALDI_ASSERT(token_list_index >= 0 && token_list_index < active_toks_.size());
  Token *&toks = active_toks_[token_list_index].toks;
  if (toks == NULL)
    KALDI_WARN << "No tokens alive [doing pruning]";
  if (token_map_[token_list_index]->size() == 0 ||
      best_token_map_[token_list_index]->size() == 0) {
    KALDI_WARN << "No maps alive [doing pruning]";
    return;
  }
  PairIdToTokenMap* &token_map = token_map_[token_list_index];
  StateIdToTokenMap* &best_token_map = best_token_map_[token_list_index];

  Token *tok, *next_tok, *prev_tok = NULL;
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
    if (tok->backward_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      if (prev_tok != NULL) prev_tok->next = tok->next;
      else toks = tok->next;
      StateId base_state = tok->base_state;
      PairId state = ConstructPair(tok->base_state, tok->lm_state);
      if (best_token_map->find(base_state) != best_token_map->end() &&
          (*best_token_map)[base_state] == tok) {
        best_token_map->erase(base_state);
      }
      token_map->erase(state);
      delete tok;
      num_toks_--;
    } else {  // fetch next Token
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
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::PruneActiveTokens(
    BaseFloat delta) {
  int32 cur_frame_plus_one = complete_frame_;
  int32 num_toks_begin = num_toks_;
  // The index "f" below represents a "frame plus one", i.e. you'd have to subtract
  // one to get the corresponding index for the decodable object.

  // TODO: Maybe we needn't reach the begining. T-25 is OK.
  for (int32 f = cur_frame_plus_one - 1; f >= 0; f--) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them (new TokenList)
    // (2) we have not yet pruned the forward links to the next f,
    // after any of those tokens have changed their extra_cost.
    if (active_toks_[f].must_prune_forward_links) {
      bool backward_costs_changed = false, links_pruned = false;
      PruneForwardLinks(f, &backward_costs_changed, &links_pruned, delta);
      if (backward_costs_changed && f > 0) // any token has changed extra_cost
        active_toks_[f-1].must_prune_forward_links = true;
      if (links_pruned) // any link was pruned
        active_toks_[f].must_prune_tokens = true;
      active_toks_[f].must_prune_forward_links = false; // job done
    }
    if (f+1 < cur_frame_plus_one &&      // except for last f (no forward links)
        active_toks_[f+1].must_prune_tokens) {
      PruneTokensForFrame(f+1);
      active_toks_[f+1].must_prune_tokens = false;
    }
  }
  KALDI_VLOG(4) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::ComputeFinalCosts(
    unordered_map<Token*, BaseFloat> *final_costs,
    BaseFloat *final_relative_cost,
    BaseFloat *final_best_cost) const {
  KALDI_ASSERT(!decoding_finalized_);
  if (final_costs != NULL)
    final_costs->clear();
  BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_cost = infinity,
      best_cost_with_final = infinity;

  // The final tokens are recorded in active_toks_[last_frame]
  for (Token *tok = active_toks_[active_toks_.size() - 1].toks; tok != NULL;
       tok = tok->next) {
    StateId state = tok->base_state;
    BaseFloat final_cost = fst_->Final(state).Value();
    BaseFloat cost = tok->tot_cost,
        cost_with_final = cost + final_cost;
    best_cost = std::min(cost, best_cost);
    best_cost_with_final = std::min(cost_with_final, best_cost_with_final);
    if (final_costs != NULL && final_cost != infinity)
      (*final_costs)[tok] = final_cost;
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
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::AdvanceDecoding(
    DecodableInterface *decodable,
    int32 max_num_frames) {
  if (std::is_same<FST, fst::Fst<fst::StdArc> >::value) {
    // if the type 'FST' is the FST base-class, then see if the FST type of fst_
    // is actually VectorFst or ConstFst.  If so, call the AdvanceDecoding()
    // function after casting *this to the more specific type.
    if (fst_->Type() == "const") {
      LatticeBiglmFasterDecoderCombineTpl<fst::ConstFst<fst::StdArc>, Token>
        *this_cast = reinterpret_cast<LatticeBiglmFasterDecoderCombineTpl<
        fst::ConstFst<fst::StdArc>, Token>* >(this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    } else if (fst_->Type() == "vector") {
      LatticeBiglmFasterDecoderCombineTpl<fst::VectorFst<fst::StdArc>, Token>
        *this_cast = reinterpret_cast<LatticeBiglmFasterDecoderCombineTpl<
        fst::VectorFst<fst::StdArc>, Token>* >(this);
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
    target_frames_decoded = std::min(target_frames_decoded,
                                     NumFramesDecoded() + max_num_frames);
  while (NumFramesDecoded() < target_frames_decoded) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    }
    ProcessForFrame(decodable);
  }
}

// FinalizeDecoding() is a version of PruneActiveTokens that we call
// (optionally) on the final frame.  Takes into account the final-prob of
// tokens.  This function used to be called PruneActiveTokensFinal().
template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::FinalizeDecoding() {
  // Process the espilon arcs for the last frame
  ProcessNonemitting();

  // Do backfill for the last few frames
  int32 beta_end = std::max(0, active_toks_.size() - config_.beta_interval);
  InitBeta(NumFramesDecoded());
  for (int32 frame = NumFramesDecoded() - 1; frame >= beta_end; frame--) {
    ComputeBetas(frame, config_.lattice_beam * config_.prune_scale);
    PruneTokensForFrame(frame);
  }
  for (int32 frame = complete_frame_; frame <= NumFramesDecoded(); frame++) {
    ExpandForward(frame, true);
  }

  int32 final_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;
  // PruneForwardLinksFinal() prunes final frame (with final-probs), and
  // sets decoding_finalized_.
  PruneForwardLinksFinal();
  for (int32 f = final_frame_plus_one - 1; f >= 0; f--) {
    bool b1, b2; // values not used.
    BaseFloat dontcare = 0.0; // delta of zero means we must always update
    PruneForwardLinks(f, &b1, &b2, dontcare);
    PruneTokensForFrame(f + 1);
  }
  PruneTokensForFrame(0);
  KALDI_VLOG(4) << "pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}


template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::ProcessForFrame(
    DecodableInterface *decodable) {
  KALDI_ASSERT(active_toks_.size() > 0);
  int32 cur_frame = active_toks_.size() - 1, // frame is the frame-index (zero-
                                             // based) used to get likelihoods
                                             // from the decodable object.
        next_frame = cur_frame + 1;

  active_toks_.resize(active_toks_.size() + 1);
  token_map_.resize(active_toks_.size());
  best_token_map_.resize(active_toks_.size());
  best_token_.resize(active_toks_.size());

  token_map_[next_frame] = new PairIdTokenMap();
  best_token_map_[next_frame] = new StateIdTokenMap();

  PairIdToTokenMap* &cur_toks = token_map_[cur_frame];
  PairIdToTokenMap* &next_toks = token_map_[next_frame];
  StateIdToTokenMap* &cur_best_toks = best_token_map_[cur_frame];
  StateIdToTokenMap* &next_best_toks = best_token_map_[next_frame];

  if (cur_toks.empty()) {
    if (!warned_) {
      KALDI_WARN << "Error, no surviving tokens on frame " << cur_frame;
      warned_ = true;
    }
  }

  cur_queue_.Clear();
  // Add tokens to queue
  for (typename StateIdToTokenMap::const_iterator iter = cur_best_toks->begin();
       iter != cur_best_toks->end(); iter++) {
    cur_queue_.Push(iter->second);
  }

  // Declare a local variable so the compiler can put it in a register, since
  // C++ assumes other threads could be modifying class members.
  BaseFloat adaptive_beam = adaptive_beam_;
  // "cur_cutoff" will be kept to the best-seen-so-far token on this frame
  // + adaptive_beam
  BaseFloat cur_cutoff = std::numeric_limits<BaseFloat>::infinity();
  // "next_cutoff" is used to limit a new token in next frame should be handle
  // or not. It will be updated along with the further processing.
  // this will be kept updated to the best-seen-so-far token "on next frame"
  // + adaptive_beam
  BaseFloat next_cutoff = std::numeric_limits<BaseFloat>::infinity();
  // "cost_offset" contains the acoustic log-likelihoods on current frame in 
  // order to keep everything in a nice dynamic range. Reduce roundoff errors.
  BaseFloat cost_offset = cost_offsets_[cur_frame];

  // Iterator the "cur_queue_" to process non-emittion and emittion arcs in fst.
  Token *tok = NULL;
  int32 num_toks_processed = 0;
  int32 max_active = config_.max_active;
  for (; num_toks_processed < max_active && (tok = cur_queue_.Pop()) != NULL;
       num_toks_processed++) {
    BaseFloat cur_cost = tok->tot_cost;
    StateId base_state = tok->base_state,
            lm_state = tok->lm_state;
    if (cur_cost > cur_cutoff &&
        num_toks_processed > config_.min_active) { // Don't bother processing
                                                     // successors.
      break;  // This is a priority queue. The following tokens will be worse
    } else if (cur_cost + adaptive_beam < cur_cutoff) {
      cur_cutoff = cur_cost + adaptive_beam; // a tighter boundary
    }
    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder),
    DeleteForwardLinks(tok);  // necessary when re-visiting
    for (fst::ArcIterator<FST> aiter(*fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc_ref = aiter.Value();
      bool changed;
      Arc arc(arc_ref);
      BaseFloat graph_cost_ori = arc.weight.Value();
      StateId next_lm_state = PropagateLm(lm_state, &arc);

      if (arc.ilabel == 0) {  // propagate nonemitting
        BaseFloat graph_cost = arc.weight.Value();
        BaseFloat tot_cost = cur_cost + graph_cost;
        if (tot_cost < cur_cutoff) {
          PairId next_pair = ConstructPair(arc.nextstate, next_lm_state);
          Token *new_tok = FindOrAddToken(next_pair, cur_frame, tot_cost, tok,
                                          cur_toks, &changed);

          // Add ForwardLink from tok to new_tok. Put it on the head of
          // tok->link list
          tok->links = new ForwardLinkT(new_tok, 0, arc.olabel,
                                        graph_cost, 0, graph_cost_ori,
                                        tok->links);
          
          // "changed" tells us whether the new token has a different
          // cost from before, or is new.
          if (changed) {
            cur_queue_.Push(new_tok);
          }
        }
      } else {  // propagate emitting
        BaseFloat graph_cost = arc.weight.Value(),
                  ac_cost = cost_offset - decodable->LogLikelihood(cur_frame,
                                                                   arc.ilabel),
                  cur_cost = tok->tot_cost,
                  tot_cost = cur_cost + ac_cost + graph_cost;
        if (tot_cost > next_cutoff) continue;
        else if (tot_cost + adaptive_beam < next_cutoff) {
          next_cutoff = tot_cost + adaptive_beam;  // a tighter boundary for
                                                   // emitting
        }

        PairId next_pair = ConstructPair(arc.nextstate, next_lm_state);
        // no change flag is needed
        Token *next_tok = FindOrAddToken(next_pair, next_frame, tot_cost,
                                         tok, next_toks, NULL);
        // Add ForwardLink from tok to next_tok. Put it on the head of tok->link
        // list
        tok->links = new ForwardLinkT(next_tok, arc.ilabel, arc.olabel,
                                      graph_cost, ac_cost, graph_cost_ori,
                                      tok->links);
      }
    }  // for all arcs
  }  // end of while loop

  // Store the offset on the acoustic likelihoods that we're applying.
  // Could just do cost_offsets_.push_back(cost_offset), but we
  // do it this way as it's more robust to future code changes.
  // Set the cost_offset_ for next frame, it equals "- best_cost_on_next_frame".
  cost_offsets_.resize(cur_frame + 2, 0.0);
  cost_offsets_[next_frame] = adaptive_beam - next_cutoff;

  {  // This block updates adaptive_beam_
    BaseFloat beam_used_this_frame = adaptive_beam;
    Token *tok = cur_queue_.Pop();
    if (tok != NULL) {
      // We hit the max-active contraint, meaning we effectively pruned to a
      // beam tighter than 'beam'. Work out what this was, it will be used to
      // update 'adaptive_beam'.
      BaseFloat best_cost_this_frame = cur_cutoff - adaptive_beam;
      beam_used_this_frame = tok->tot_cost - best_cost_this_frame;
    }
    if (num_toks_processed <= config_.min_active) {
      // num-toks active is dangerously low, increase the beam even if it
      // already exceeds the user-specified beam.
      adaptive_beam_ = std::max<BaseFloat>(
          config_.beam, beam_used_this_frame + 2.0 * config_.beam_delta);
    } else {
      // have adaptive_beam_ approach beam_ in intervals of config_.beam_delta
      BaseFloat diff_from_beam = beam_used_this_frame - config_.beam;
      if (std::abs(diff_from_beam) < config_.beam_delta) {
        adaptive_beam_ = config_.beam;
      } else {
        // make it close to beam_
        adaptive_beam_ = beam_used_this_frame -
          config_.beam_delta * (diff_from_beam > 0 ? 1 : -1);
      }
    }
  }
}


template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::ProcessNonemitting() {
  int32 cur_frame = active_toks_.size() - 1;
  StateIdToTokenMap &cur_toks = next_toks_;

  cur_queue_.Clear();
  for (Token* tok = active_toks_[cur_frame].toks; tok != NULL; tok = tok->next)
    cur_queue_.Push(tok);

  // Declare a local variable so the compiler can put it in a register, since
  // C++ assumes other threads could be modifying class members.
  BaseFloat adaptive_beam = adaptive_beam_;
  // "cur_cutoff" will be kept to the best-seen-so-far token on this frame
  // + adaptive_beam
  BaseFloat cur_cutoff = std::numeric_limits<BaseFloat>::infinity();

  Token *tok = NULL;
  int32 num_toks_processed = 0;
  int32 max_active = config_.max_active;

  for (; num_toks_processed < max_active && (tok = cur_queue_.Pop()) != NULL;
       num_toks_processed++) {
    BaseFloat cur_cost = tok->tot_cost;
    StateId state = tok->base_state;
    if (cur_cost > cur_cutoff &&
        num_toks_processed > config_.min_active) { // Don't bother processing
                                                     // successors.
      break;  // This is a priority queue. The following tokens will be worse
    } else if (cur_cost + adaptive_beam < cur_cutoff) {
      cur_cutoff = cur_cost + adaptive_beam; // a tighter boundary
    }
    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder),
    DeleteForwardLinks(tok);  // necessary when re-visiting
    for (fst::ArcIterator<FST> aiter(*fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      bool changed;
      if (arc.ilabel == 0) {  // propagate nonemitting
        BaseFloat graph_cost = arc.weight.Value();
        BaseFloat tot_cost = cur_cost + graph_cost;
        if (tot_cost < cur_cutoff) {
          Token *new_tok = FindOrAddToken(arc.nextstate, cur_frame, tot_cost,
                                          tok, &cur_toks, &changed);

          // Add ForwardLink from tok to new_tok. Put it on the head of
          // tok->link list
          tok->links = new ForwardLinkT(new_tok, 0, arc.olabel,
                                        graph_cost, 0, tok->links);
          
          // "changed" tells us whether the new token has a different
          // cost from before, or is new.
          if (changed) {
            cur_queue_.Push(new_tok);
          }
        }
      }
    }  // end of for loop
  }  // end of while loop
  if (!decoding_finalized_) {
    // Update cost_offsets_, it equals "- best_cost".
    cost_offsets_[cur_frame] = adaptive_beam - cur_cutoff;
    // Needn't to update adaptive_beam_, since we still process this frame in
    // ProcessForFrame.
  }
}



// static inline
template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::DeleteForwardLinks(Token *tok) {
  ForwardLinkT *l = tok->links, *m;
  while (l != NULL) {
    m = l->next;
    delete l;
    l = m;
  }
  tok->links = NULL;
}


template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::ClearActiveTokens() {
  // a cleanup routine, at utt end/begin
  for (size_t i = 0; i < active_toks_.size(); i++) {
    // Delete all tokens alive on this frame, and any forward
    // links they may have.
    for (Token *tok = active_toks_[i].toks; tok != NULL; ) {
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
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::TopSortTokens(
    Token *tok_list, std::vector<Token*> *topsorted_list) {
  unordered_map<Token*, int32> token2pos;
  typedef typename unordered_map<Token*, int32>::iterator IterType;
  int32 num_toks = 0;
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    num_toks++;
  int32 cur_pos = 0;
  // We assign the tokens numbers num_toks - 1, ... , 2, 1, 0.
  // This is likely to be in closer to topological order than
  // if we had given them ascending order, because of the way
  // new tokens are put at the front of the list.
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    token2pos[tok] = num_toks - ++cur_pos;

  unordered_set<Token*> reprocess;

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
  for (loop_count = 0;
       !reprocess.empty() && loop_count < max_loop; ++loop_count) {
    std::vector<Token*> reprocess_vec;
    for (typename unordered_set<Token*>::iterator iter = reprocess.begin();
         iter != reprocess.end(); ++iter)
      reprocess_vec.push_back(*iter);
    reprocess.clear();
    for (typename std::vector<Token*>::iterator iter = reprocess_vec.begin();
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
  KALDI_ASSERT(loop_count < max_loop && "Epsilon loops exist in your decoding "
               "graph (this is not allowed!)");

  topsorted_list->clear();
  topsorted_list->resize(cur_pos, NULL);  // create a list with NULLs in between.
  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter)
    (*topsorted_list)[iter->second] = iter->first;
}


template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::InitBeta(int32 frame) {
  for (Token* tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    tok->backward_cost = -tok->tot_cost;
  }
}


template <typename FST, typename Token>
int32 LatticeBiglmFasterDecoderCombineTpl<FST, Token>::DoBackfill() {
  // Update Beta
  InitBeta(NumFramesDecoded());
  for (int32 frame = NumFramesDecoded() - 1; frame > complete_frame_; frame--) {
    ComputeBetas(frame, config_.lattice_beam * config_.prune_scale);
  }
  // Prune
  PruneForwardLinksWithoutRecompute(complete_frame_);
  for (int32 frame = complete_frame_ + 1; frame < NumFramesDecoded(); frame++) {
    PruneForwardLinksWithoutRecompute(frame);
    PruneTokensForFrame(frame);
  }

  int32 expand_best_only_start =
    NumFramesDecoded() - config_.expand_best_interval;
  for (int32 frame = complete_frame_ + 1; frame < expand_best_only_start;
      frame++) {
    ExpandForward(frame, true);
  }
  for (int32 frame = expand_best_only_start; frame < NumFramesDecode();
      frame++) {
    ExpandForward(frame, false);
  }
  return std::max(expand_best_only_start - 1, -1);
}


template <typename FST, typename Token>
int32 LatticeBiglmFasterDecoderCombineTpl<FST, Token>::ComputeBeta(int32 frame,
    BaseFloat delta) {
  // a. Update the expanded token's beta
  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.
  bool changed = true; // difference new minus old extra cost >= delta ?
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
      if (!tok->expanded) continue;
      ForwardLink *link;
      // will recompute tok_backward_cost for expanded tok.
      BaseFloat tok_backward_cost = std::numeric_limits<BaseFloat>::infinity();
      // tok_backward_cost is the best (min) of link_backward_cost of
      // outgoing links
      for (link = tok->links; link != NULL; link = link->next) {
        Token *next_tok = link->next_tok;
        BaseFloat link_backward_cost = 
          std::numeric_limits<BaseFloat>::infinity();
        link_backward_cost = next_tok->backward_cost + link->acoustic_cost +
                             link->graph_cost;
        KALDI_ASSERT(link_backward_cost == link_backward_cost); // check for NaN
        tok_backward_cost = std::min(tok_backward_cost, link_backward_cost);
      }
      if (fabs(tok_backward_cost - tok->backward_cost) > delta)
        changed = true;  // difference new minus old is bigger than delta
      tok->backward_cost = tok_backward_cost;
    } // for all expanded token
  }

  // b. find the best token for the particular frame
  Token* &best_tok = best_token_[frame];
  for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    if (tok->expanded) {  // Get the best token from expanded tokens
      if (best_tok == NULL || *tok < *best_tok) {
        best_tok = tok;
      }
    }   
  }

  // c. Build best_token_map with alpha + beta and prune tokens that fall below
  // the config_.beam
  best_token_map_[frame]->clear();
  StateIdToTokenMap* &best_token_map = best_token_map_[frame];
  for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    if (tok->expanded) {
      // Prune useless expanded token with beam
      if ((tok->tot_cost + tok->backward_cost) < (best_tok->tot_cost +
            best_tok->backward_cost) + config_.beam) {
        *best_token_map[tok->hclg_state] = tok;
      } else {  // the expanded token should be pruned.
        DeleteForwardLinks(tok);
        tok->backward_cost = std::numeric_limits<BaseFloat>::infinity();
      }
    }
  }

  // d. Update un-expanded tokens
  for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    if (!tok->expanded) {
      if (best_token_map.find(tok->hclg) != best_token_map.end()) {
        tok->backward_cost = best_token_map[tok->hclg];
        // Prune the worse unexpanded token
        if ((tok->tot_cost + tok->backward_cost) >=
            (best_tok->tot_cost + best_tok->backward_cost) + config_.beam) {
          DeleteForwardLinks(tok);
          tok->backward_cost = std::numeric_limits<BaseFloat>::infinity();
        }
      } else {
        // Prune the token whose shadowing token has been pruned.
        DeleteForwardLinks(tok);
        tok->backward_cost = std::numeric_limits<BaseFloat>::infinity();
      }
    }
  }
}

template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::ExpandForward(int32 frame,
    bool expand_not_best) {
  std::queue<Token*> queue;
  // Add the token which should be processed into "queue".
  // a. Process expanded tokens
  for (Token *tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    if (tok->expanded) {
      if (tok->update_alpha) {
        KALDI_ASSERT(*best_token_map_[frame].find(tok->hclg_state) !=
                     *best_token_map_[frame].end());
        KALDI_ASSERT(*tok < *(*best_token_map_[frame])[tok->hclg_state]);
        // Check the expanded token whose alpha is updated so that the alphas of
        // its successor tokens may need to be updated.
        queue.push(tok);
      }
    } else {  // For unexpanded token 
      if (expand_not_best) {  // expand all survived un-expanded tokens
        queue.push(tok);
      } else {  // only the better hclg token
        KALDI_ASSERT(best_token_map.find(tok->hclg_state) !=
                     best_token_map.end());
        if (*tok < *(*best_token_map[frame])[tok->hclg_state]) queue.push(tok);
      }
    }
  }

  while (!queue.empty()) {
    Token* tok = queue.front();
    queue.pop();
    if (tok->expanded) {
      tok->update_alpha = false;  // Set update_alpha flag
      for (ForwardLink* link = tok->links; link != NULL; link = link->next) {
        Token* next_tok = link->next_tok;
        if (link->ilabel == 0) {  // epsilon
          BaseFloat link_tot_cost = tok->tot_cost + link->graph_cost;
          if (link_tot_cost < next_tok->tot_cost) {  // the successor token's
                                                     // alpha need to be updated
            next_tok->tot_cost = link_tot_cost;
            // For un-expanded token, the "update_alpha" flag is meaningless as
            // it doesn't have successor tokens.
            if (next_tok->expanded) {
              next_tok->update_alpha = true;
              queue.push(next_tok);
              // next_tok is an expanded token. So, when its alpha is updated,
              // it may become the best-in-class token
              if (*next_tok <
                  *(*best_token_map_[frame])[next_tok->hclg_state]) {
                (*best_token_map_[frame])[next_tok->hclg_state] = next_tok;
                // may become the best token in this frame
                if (*next_tok < *best_token_[frame]) {
                  best_token_[frame] = next_tok;
                }
              }
            }
          }
        } else {  // non-epsilon. Take care of cost_offset_
          KALDI_ASSERT(active_toks_[frame + 1]);
          BaseFloat link_tot_cost = tok->tot_cost + link->graph_cost +
                                    link->acoustic_cost;
          if (link_tot_cost < next_tok->tot_cost) {  // update successor token's
                                                     // alpha
            next_tok->tot_cost = link_tot_cost;
            if (next_tok->expanded) {
              next_tok->update_alpha = true;
              // Needn't add it into queue. It will be processed in next frame.
              // next_tok is an expanded token. So, when its alpha is updated,
              // it may become the best-in-class token
              if (*next_tok <
                  *(*best_token_map_[frame])[next_tok->hclg_state]) {
                (*best_token_map_[frame])[next_tok->hclg_state] = next_tok;
                // may become the best token in this frame
                if (*next_tok < *best_token_[frame]) {
                  best_token_[frame] = next_tok;
                }
              }
            }       
          }
        }
      }  // all links
    } else {  // un-expanded token
      ExpandTokenBackfill(frame, tok);
    }
  }  // while loop
}

template <typename FST, typename Token>
void LatticeBiglmFasterDecoderCombineTpl<FST, Token>::ExpandTokenBackfill(
    int32 frame, Token* tok) {
  StateIdToTokenMap* &best_token_map = best_token_map_[frame];
  if (best_token_map->find(tok->hclg_state) == best_token_map->end()) {
    KALDI_WARN << "The token (" << tok->hclg_state << "," << tok->lm_state
               << ") doesn't have reference token. It's highly unexpected."
               << " Set the token's beta to infinity and prune later."
    tok->backward_cost = std::numeric_limits<BaseFloat>::infinity();
    return;
  }
  Token* &ref_tok = *best_token_map[tok->hclg_state];
  for (ForwardLink *ref_link = ref_tok->links; ref_link != NULL;
      ref_link = ref_link->next) {
    Token* &ref_next_tok = ref_link->next_tok;

    StateId new_hclg_state = ref_next_tok->hclg_state;
    Arc arc(ref_link->ilabel, ref_link->olabel, ref_link->graph_cost_ori, 0);
    StateId new_lm_state = PropagateLm(tok->lm_state, &arc);
    PairId new_pair_id = ConstructPair(new_hclg_state, new_lm_state);
    BaseFloat ac_cost = ref_link->acoustic_cost,
              graph_cost = arc.weight.Value(),
              cur_cost = tok->tot_cost,
              tot_cost = cur_cost + ac_cost + graph_cost,
              backward_cost = ref_next_tok->backward_cost,
    int32 new_frame_index = link->ilabel ? frame + 1 : frame;
    Token* &toks = link->ilabel ? active_toks_[frame + 1].toks :
                                  active_toks_[frame].toks;
    
    KALDI_ASSERT(token_map_.size() >= new_frame_index + 1);
    KALDI_ASSERT(best_token_map_.size() == token_map_.size() ==
        best_token_.size());

    // Process the new token
    PairIdToTokenMap* &token_map = token_map_[new_frame_index];
    Token* new_tok;
    if (token_map->find(new_pair_id) == token_map->end()) {  // a new token
      // check its alpha_beta is within the beam
      Token* &best_token = best_token_[new_frame_index];
      if (tot_cost + backward_cost >
          best_token->tot_cost + best_token->backward_cost + config_.beam)
        continue;
      // create a new token
      new_tok = new Token(tot_cost, backward_cost, new_hclg_state,
          new_lm_state, NULL, toks);
      (*token_map_[new_frame_index])[new_pair_id] = new_tok;
      if (ref_link->ilabel == 0)
        queue.push(new_tok);  // TODO
    } else {  // an existing token
      new_tok = token_map[new_pair_id];
      tok->links = new ForwardLink(new_tok, arc.ilabel, arc.olabel,
                                   graph_cost, ac_cost, tok->links,
                                   ref_link->graph_cost_ori);
      // For an existing token, the beta (backward cost) is generated from its 
      // successors. Use the beta directly. So compare the alpha (forward cost)
      // only.
      if (tot_cost < new_tok->tot_cost) {  // update
        new_tok->tot_cost = tot_cost;
        new_tok->update_alpha = true;  // indicate the token's tot_cost is
                                       // updated, update the successors.
        if (new_tok->expanded) {
          if (*new_tok <
              *(*best_token_map_[new_frame_index])[new_tok->hclg_state]) {
            (*best_token_map_[new_frame_index])[new_tok->hclg_state] = new_tok;
            if (*new_tok < *best_token_[new_frame_index]) {
              best_token_[new_frame_index] = new_tok;
            }
          }
        }
      }
    }  // end of the processing of one new token
  }  // end of the for loop
  tok->expanded = true;  // Set expanded property to true
  
  // Recompute the token's beta
  BaseFloat tok_backward_cost = std::numeric_limits<BaseFloat>::infinity();
  // tok_backward_cost is the best (min) of link_backward_cost of outgoing links
  for (ForwardLink *link = tok->links; link != NULL; link = link->next) {
    Token *next_tok = link->next_tok;
    BaseFloat link_backward_cost = std::numeric_limits<BaseFloat>::infinity();
    link_backward_cost = next_tok->backward_cost + link->acoustic_cost + 
      link->graph_cost;
    KALDI_ASSERT(link_backward_cost == link_backward_cost); // check for NaN
    tok_backward_cost = std::min(tok_backward_cost, link_backward_cost);
  }
  tok->backward_cost = tok_backward_cost;

  // Check best_token_map_ and best_token_
  if (*tok < *(*best_token_map_[frame])[tok->hclg_state]) {
    (*best_token_map_[frame])[tok->hclg_state] = tok;
    if (*tok < *(best_token_[frame]))
      best_token_[frame] = tok;
  }
}


// Instantiate the template for the combination of token types and FST types
// that we'll need.
template class LatticeBiglmFasterDecoderCombineTpl<fst::Fst<fst::StdArc>,
         biglmdecodercombine::StdToken<fst::Fst<fst::StdArc> > >;
template class LatticeBiglmFasterDecoderCombineTpl<fst::VectorFst<fst::StdArc>,
         biglmdecodercombine::StdToken<fst::VectorFst<fst::StdArc> > >;
template class LatticeBiglmFasterDecoderCombineTpl<fst::ConstFst<fst::StdArc>,
         biglmdecodercombine::StdToken<fst::ConstFst<fst::StdArc> > >;
template class LatticeBiglmFasterDecoderCombineTpl<fst::GrammarFst,
         biglmdecodercombine::StdToken<fst::GrammarFst> >;

template class LatticeBiglmFasterDecoderCombineTpl<fst::Fst<fst::StdArc> ,
         biglmdecodercombine::BackpointerToken<fst::Fst<fst::StdArc> > >;
template class LatticeBiglmFasterDecoderCombineTpl<fst::VectorFst<fst::StdArc>,
         biglmdecodercombine::BackpointerToken<fst::VectorFst<fst::StdArc> > >;
template class LatticeBiglmFasterDecoderCombineTpl<fst::ConstFst<fst::StdArc>,
         biglmdecodercombine::BackpointerToken<fst::ConstFst<fst::StdArc> > >;
template class LatticeBiglmFasterDecoderCombineTpl<fst::GrammarFst,
         biglmdecodercombine::BackpointerToken<fst::GrammarFst> >;


} // end namespace kaldi.
