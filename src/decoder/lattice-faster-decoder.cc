// decoder/lattice-faster-decoder.cc

// Copyright 2009-2012  Microsoft Corporation  Mirko Hannemann
//           2013-2018  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
//                2018  Zhehuai Chen

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

#include "decoder/lattice-faster-decoder.h"

#include "lat/lattice-functions.h"

namespace kaldi {

// instantiate this class once for each thing you have to decode.
template<typename FST, typename Token>
LatticeFasterDecoderTpl<FST, Token>::LatticeFasterDecoderTpl(
    const FST &fst, const LatticeFasterDecoderConfig &config):
    fst_(&fst), delete_fst_(false), config_(config), num_toks_(0) {
  config.Check();

  // just so on the first frame we do something reasonable.
  toks_.SetSize(1000);
}

template<typename FST, typename Token>
LatticeFasterDecoderTpl<FST, Token>::LatticeFasterDecoderTpl(
    const LatticeFasterDecoderConfig &config, FST *fst):
    fst_(fst), delete_fst_(true), config_(config), num_toks_(0) {
  config.Check();

  // just so on the first frame we do something reasonable.
  toks_.SetSize(1000);
}


template<typename FST, typename Token>
LatticeFasterDecoderTpl<FST, Token>::~LatticeFasterDecoderTpl() {
  DeleteElems(toks_.Clear());
  ClearActiveTokens();

  if (delete_fst_) {
    delete fst_;
  }
}

template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::InitDecoding() {
  DeleteElems(toks_.Clear());
  cost_offsets_.clear();
  ClearActiveTokens();
  warned_ = false;
  num_toks_ = 0;
  decoding_finalized_ = false;
  final_costs_.clear();
  const StateId start_state = fst_->Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  active_toks_.emplace_back();
  active_toks_[0].toks.emplace_front(0.0, 0.0, nullptr);
  toks_.Insert(start_state, &active_toks_[0].toks.front());
  ++num_toks_;
  ProcessNonemitting(config_.beam);
}

// Returns true if any kind of traceback is available (not necessarily from
// a final state).  It should only very rarely return false; this indicates
// an unusual search error.
template<typename FST, typename Token>
bool
LatticeFasterDecoderTpl<FST, Token>::Decode(DecodableInterface *decodable) {
  InitDecoding();

  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.

  while (!decodable->IsLastFrame(NumFramesDecoded() - 1)) {
    if (NumFramesDecoded() % config_.prune_interval == 0)
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    BaseFloat cost_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(cost_cutoff);
  }
  FinalizeDecoding();

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !active_toks_.empty() && !active_toks_.back().toks.empty();
}


// Outputs an FST corresponding to the single best path through the lattice.
template<typename FST, typename Token>
bool LatticeFasterDecoderTpl<FST, Token>::GetBestPath(
    Lattice *olat, bool use_final_probs) const {
  Lattice raw_lat;
  GetRawLattice(&raw_lat, use_final_probs);
  ShortestPath(raw_lat, olat);
  return olat->NumStates() != 0;
}


// Outputs an FST corresponding to the raw, state-level lattice
template<typename FST, typename Token>
bool LatticeFasterDecoderTpl<FST, Token>::GetRawLattice(
    Lattice *ofst, bool use_final_probs) const {
  // Note: you can't use the old interface (Decode()) if you want to
  // get the lattice with use_final_probs = false.  You'd have to do
  // InitDecoding() and then AdvanceDecoding().
  if (decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "GetRawLattice() with use_final_probs == false";

  unordered_map<const Token *, BaseFloat> final_costs_local;

  const unordered_map<const Token *, BaseFloat> &final_costs =
      (decoding_finalized_ ? final_costs_ : final_costs_local);
  if (!decoding_finalized_ && use_final_probs)
    ComputeFinalCosts(&final_costs_local, nullptr, nullptr);

  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  const int32 num_frames = static_cast<int32>(active_toks_.size()) - 1;
  KALDI_ASSERT(num_frames > 0);
  const unsigned bucket_count = num_toks_ / 2 + 3;
  unordered_map<const Token *, LatticeArc::StateId> tok_map(bucket_count);

  // First create all states.
  for (int32 f = 0; f <= num_frames; ++f) {
    if (active_toks_[f].toks.empty()) {
      KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }

    for (const auto &token_ptr : TopSortTokens(active_toks_[f].toks)) {
      if (token_ptr != nullptr) {
        tok_map[token_ptr] = ofst->AddState();
      }
    }
  }

  // The next statement sets the start state of the output FST.  Because we
  // topologically sorted the tokens, state zero must be the start-state.
  ofst->SetStart(0);

  KALDI_VLOG(4) << "init:" << num_toks_ / 2 + 3 << " buckets:"
                << tok_map.bucket_count() << " load:" << tok_map.load_factor()
                << " max:" << tok_map.max_load_factor();
  // Now create all arcs.
  for (int32 f = 0; f <= num_frames; ++f) {
    for (const auto &tok : active_toks_[f].toks) {
      const auto cur_state = tok_map[&tok];

      for (const auto &link : tok.links) {
        const auto iter = tok_map.find(link.next_tok);
        auto next_state = iter->second;
        KALDI_ASSERT(iter != tok_map.end());
        BaseFloat cost_offset = 0.0;

        if (link.i_label != 0) {
          KALDI_ASSERT(f < cost_offsets_.size());
          cost_offset = cost_offsets_[f];
        }

        LatticeArc arc(link.i_label, link.o_label,
                       {link.graph_cost, link.acoustic_cost - cost_offset},
                       next_state);

        ofst->AddArc(cur_state, arc);
      }

      if (f == num_frames) {
        if (use_final_probs && !final_costs.empty()) {
          auto iter = final_costs.find(&tok);

          if (iter != final_costs.end()) {
            ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
          }
        } else {
          ofst->SetFinal(cur_state, LatticeWeight::One());
        }
      }
    }
  }

  return ofst->NumStates() > 0;
}


// This function is now deprecated, since now we do determinization from outside
// the LatticeFasterDecoder class.  Outputs an FST corresponding to the
// lattice-determinized lattice (one path per word sequence).
template<typename FST, typename Token>
bool LatticeFasterDecoderTpl<FST, Token>::GetLattice(
    CompactLattice *ofst, bool use_final_probs) const {
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
  return ofst->NumStates() != 0;
}

template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::PossiblyResizeHash(
    size_t num_toks) noexcept {
  const auto new_sz = static_cast<size_t>(num_toks * config_.hash_ratio);

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
template<typename FST, typename Token>
inline Token *LatticeFasterDecoderTpl<FST, Token>::FindOrAddToken(
    StateId state, decltype(active_toks_.size()) frame_plus_one,
    BaseFloat tot_cost, Token *backpointer, bool *changed) noexcept {
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  KALDI_ASSERT(frame_plus_one < active_toks_.size());
  auto &toks = active_toks_[frame_plus_one].toks;
  const Elem *e_found = toks_.Find(state);

  if (e_found == nullptr) {  // no such token presently.
    toks.emplace_front(tot_cost, 0.0, backpointer);
    ++num_toks_;
    toks_.Insert(state, &toks.front());

    if (changed) {
      *changed = true;
    }

    return &toks.front();
  } else {
    Token *tok = e_found->val;  // There is an existing Token for this state.

    if (tok->tot_cost > tot_cost) {  // replace old token
      tok->tot_cost = tot_cost;
      // SetBackpointer() just does tok->backpointer = backpointer in
      // the case where Token == BackpointerToken, else nothing.
      tok->SetBackpointer(backpointer);
      // we don't allocate a new token, the old stays linked in active_toks_
      // we only replace the tot_cost in the current frame, there are no forward
      // links (and no extra_cost) only in ProcessNonemitting we have to delete
      // forward links in case we visit a state for the second time those
      // forward links, that lead to this replaced token before: they remain and
      // will hopefully be pruned later (PruneForwardLinks...)
      if (changed) {
        *changed = true;
      }
    } else if (changed) {
      *changed = false;
    }

    return tok;
  }
}

// prunes outgoing links for all tokens in active_toks_[frame] it's called by
// PruneActiveTokens all links, that have link_extra_cost > lattice_beam are
// pruned
template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::PruneForwardLinks(
    int32 frame_plus_one, bool *extra_costs_changed,
    bool *links_pruned, BaseFloat delta) noexcept {
  // delta is the amount by which the extra_costs must change If delta is
  // larger, we'll tend to go back less far toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned
  *extra_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());

  // empty list; should not happen.
  if (active_toks_[frame_plus_one].toks.empty() && !warned_) {
    KALDI_WARN << "No tokens alive [doing pruning].. warning first "
                  "time only for each utterance\n";
    warned_ = true;
  }

  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.
  bool changed = true;  // difference new minus old extra cost >= delta ?

  while (changed) {
    changed = false;

    for (auto &tok : active_toks_[frame_plus_one].toks) {
      // will recompute tok_extra_cost for tok.
      BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();

      // tok_extra_cost is the best (min) of link_extra_cost of outgoing links
      for (auto prev = tok.links.before_begin(), link = tok.links.begin();
           link != tok.links.end();) {
        // See if we need to excise this link.
        Token *next_tok = link->next_tok;

        BaseFloat link_extra_cost = next_tok->extra_cost +
                                    ((tok.tot_cost + link->acoustic_cost +
                                      link->graph_cost)
                                     -
                                     next_tok->tot_cost);  // difference in brackets is >= 0
        // link_exta_cost is the difference in score between the best paths
        // through link source state and through link destination state
        KALDI_ASSERT(link_extra_cost == link_extra_cost);  // check for NaN

        if (link_extra_cost > config_.lattice_beam) {  // excise link
          link = tok.links.erase_after(prev);
          *links_pruned = true;
        } else {  // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) {  // this is just a precaution.
            if (link_extra_cost < -0.01) {
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            }

            link_extra_cost = 0.0;
          }

          if (link_extra_cost < tok_extra_cost) {
            tok_extra_cost = link_extra_cost;
          }

          ++link;
          ++prev;
        }
      }

      if (std::fabs(tok_extra_cost - tok.extra_cost) > delta) {
        changed = true;   // difference new minus old is bigger than delta
      }

      tok.extra_cost = tok_extra_cost;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    }

    if (changed) {
      *extra_costs_changed = true;
    }
    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  } // while changed
}

// PruneForwardLinksFinal is a version of PruneForwardLinks that we call
// on the final frame.  If there are final tokens active, it uses
// the final-probs for pruning, otherwise it treats all tokens as final.
template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::PruneForwardLinksFinal() {
  KALDI_ASSERT(!active_toks_.empty());
  const int32 frame_plus_one = static_cast<int32>(active_toks_.size()) - 1;

  // empty list; should not happen.
  if (active_toks_[frame_plus_one].toks.empty()) {
    KALDI_WARN << "No tokens alive at end of file";
  }

  BaseFloat final_best_cost = 0;
  ComputeFinalCosts(&final_costs_, &final_relative_cost_, &final_best_cost);
  decoding_finalized_ = true;

  // Now go through tokens on this frame, pruning forward links...  may have to
  // iterate a few times until there is no more change, because the list is not
  // in topological order.  This is a modified version of the code in
  // PruneForwardLinks, but here we also take account of the final-probs.
  bool changed = true;
  const BaseFloat delta = 1.0e-05;

  while (changed) {
    changed = false;

    for (auto &tok : active_toks_[frame_plus_one].toks) {
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to infinity
      // below we set it to the difference between the (score+final_prob) of this token,
      // and the best such (score+final_prob).
      BaseFloat final_cost;

      if (final_costs_.empty()) {
        final_cost = 0.0;
      } else {
        auto iter = final_costs_.find(&tok);

        if (iter != final_costs_.end()) {
          final_cost = iter->second;
        } else {
          final_cost = std::numeric_limits<BaseFloat>::infinity();
        }
      }

      BaseFloat tok_extra_cost = tok.tot_cost + final_cost - final_best_cost;

      // tok_extra_cost will be a "min" over either directly being final, or
      // being indirectly final through other links, and the loop below may
      // decrease its value:
      for (auto prev = tok.links.before_begin(), link = tok.links.begin();
           link != tok.links.end();) {
        // See if we need to excise this link...
        const Token *next_tok = link->next_tok;
        BaseFloat link_extra_cost = next_tok->extra_cost +
                                    ((tok.tot_cost + link->acoustic_cost +
                                      link->graph_cost)
                                     - next_tok->tot_cost);

        if (link_extra_cost > config_.lattice_beam) {  // excise link
          link = tok.links.erase_after(prev);
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01) {
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            }

            link_extra_cost = 0.0;
          }

          if (link_extra_cost < tok_extra_cost) {
            tok_extra_cost = link_extra_cost;
          }

          ++link;
          ++prev;
        }
      }

      // prune away tokens worse than lattice_beam above best path.  This step
      // was not necessary in the non-final case because then, this case
      // showed up as having no forward links.  Here, the tok_extra_cost has
      // an extra component relating to the final-prob.
      if (tok_extra_cost > config_.lattice_beam)
        tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // to be pruned in PruneTokensForFrame

      if (!ApproxEqual(tok.extra_cost, tok_extra_cost, delta)) {
        changed = true;
      }

      tok.extra_cost = tok_extra_cost; // will be +infinity or <= lattice_beam_.
    }
  } // while changed
}

template<typename FST, typename Token>
BaseFloat LatticeFasterDecoderTpl<FST, Token>::FinalRelativeCost() const {
  if (!decoding_finalized_) {
    BaseFloat relative_cost;
    ComputeFinalCosts(nullptr, &relative_cost, nullptr);
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
template<typename FST, typename Token>
void
LatticeFasterDecoderTpl<FST, Token>::PruneTokensForFrame(
    int32 frame_plus_one) noexcept {
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());

  if (active_toks_[frame_plus_one].toks.empty()) {
    KALDI_WARN << "No tokens alive [doing pruning]";
  }

  for (auto tok = active_toks_[frame_plus_one].toks.begin();
       tok != active_toks_[frame_plus_one].toks.end();) {
    if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      tok = active_toks_[frame_plus_one].toks.erase(tok);
      --num_toks_;
    } else {  // fetch next Token
      ++tok;
    }
  }
}

// Go backwards through still-alive tokens, pruning them, starting not from
// the current frame (where we want to keep all tokens) but from the frame before
// that.  We go backwards through the frames and stop when we reach a point
// where the delta-costs are not changing (and the delta controls when we consider
// a cost to have "not changed").
template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::PruneActiveTokens(
    BaseFloat delta) noexcept {
  const int32 cur_frame_plus_one = NumFramesDecoded();
  const int32 num_toks_begin = num_toks_;

  // The index "f" below represents a "frame plus one", i.e. you'd have to
  // subtract one to get the corresponding index for the decodable object.
  for (int32 f = cur_frame_plus_one - 1; f >= 0; --f) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them (new TokenList)
    // (2) we have not yet pruned the forward links to the next f,
    // after any of those tokens have changed their extra_cost.
    if (active_toks_[f].must_prune_forward_links) {
      bool extra_costs_changed = false;
      bool links_pruned = false;
      PruneForwardLinks(f, &extra_costs_changed, &links_pruned, delta);

      if (extra_costs_changed && f > 0) {  // any token has changed extra_cost
        active_toks_[f - 1].must_prune_forward_links = true;
      }

      if (links_pruned) {  // any link was pruned
        active_toks_[f].must_prune_tokens = true;
      }

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

template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::ComputeFinalCosts(
    unordered_map<const Token *, BaseFloat> *final_costs,
    BaseFloat *final_relative_cost, BaseFloat *final_best_cost) const {
  KALDI_ASSERT(!decoding_finalized_);

  if (final_costs != nullptr) {
    final_costs->clear();
  }

  const BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_cost = infinity;
  BaseFloat best_cost_with_final = infinity;

  for (const Elem *final_toks = toks_.GetList();
       final_toks != nullptr; final_toks = final_toks->tail) {
    StateId state = final_toks->key;
    Token *tok = final_toks->val;
    const BaseFloat final_cost = fst_->Final(state).Value();
    BaseFloat cost = tok->tot_cost;
    BaseFloat cost_with_final = cost + final_cost;
    best_cost = std::min(cost, best_cost);
    best_cost_with_final = std::min(cost_with_final, best_cost_with_final);

    if (final_costs != nullptr && final_cost != infinity) {
      (*final_costs)[tok] = final_cost;
    }
  }

  if (final_relative_cost != nullptr) {
    if (best_cost == infinity && best_cost_with_final == infinity) {
      // Likely this will only happen if there are no tokens surviving.
      // This seems the least bad way to handle it.
      *final_relative_cost = infinity;
    } else {
      *final_relative_cost = best_cost_with_final - best_cost;
    }
  }

  if (final_best_cost != nullptr) {
    if (best_cost_with_final != infinity) { // final-state exists.
      *final_best_cost = best_cost_with_final;
    } else { // no final-state exists.
      *final_best_cost = best_cost;
    }
  }
}

template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::AdvanceDecoding(
    DecodableInterface *decodable, int32 max_num_frames) {
  if (std::is_same<FST, fst::Fst<fst::StdArc>>::value) {
    // if the type 'FST' is the FST base-class, then see if the FST type of fst_
    // is actually VectorFst or ConstFst.  If so, call the AdvanceDecoding()
    // function after casting *this to the more specific type.
    if (fst_->Type() == "const") {
      auto *this_cast = reinterpret_cast<LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>, Token> * >(this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    } else if (fst_->Type() == "vector") {
      auto *this_cast = reinterpret_cast<LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>, Token> * >(this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    }
  }

  KALDI_ASSERT(!active_toks_.empty() && !decoding_finalized_ &&
               "You must call InitDecoding() before AdvanceDecoding");

  const int32 num_frames_ready = decodable->NumFramesReady();
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

    ProcessNonemitting(ProcessEmitting(decodable));
  }
}

// FinalizeDecoding() is a version of PruneActiveTokens that we call
// (optionally) on the final frame.  Takes into account the final-prob of
// tokens.  This function used to be called PruneActiveTokensFinal().
template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::FinalizeDecoding() {
  int32 final_frame_plus_one = NumFramesDecoded();
  const int32 num_toks_begin = num_toks_;
  // PruneForwardLinksFinal() prunes final frame (with final-probs), and
  // sets decoding_finalized_.
  PruneForwardLinksFinal();

  for (int32 f = final_frame_plus_one - 1; f >= 0; --f) {
    bool b1, b2; // values not used.
    PruneForwardLinks(f, &b1, &b2, 0.0);
    PruneTokensForFrame(f + 1);
  }

  PruneTokensForFrame(0);
  KALDI_VLOG(4) << "pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

/// Gets the weight cutoff.  Also counts the active tokens.
template<typename FST, typename Token>
BaseFloat LatticeFasterDecoderTpl<FST, Token>::GetCutoff(
    Elem *list_head, size_t *tok_count, BaseFloat *adaptive_beam,
    Elem **best_elem) const noexcept {
  // positive == high cost == bad.
  BaseFloat best_weight = std::numeric_limits<BaseFloat>::infinity();

  if (config_.max_active == std::numeric_limits<int32>::max() &&
      config_.min_active == 0) {
    *tok_count = 0;

    for (Elem *e = list_head; e != nullptr; e = e->tail, ++(*tok_count)) {
      const BaseFloat w = e->val->tot_cost;

      if (w < best_weight) {
        best_weight = w;
        *best_elem = e;
      }
    }

    *adaptive_beam = config_.beam;
    return best_weight + config_.beam;
  } else {
    std::vector<BaseFloat> tmp_array;
    tmp_array.reserve(256);

    for (Elem *e = list_head; e != nullptr; e = e->tail) {
      const BaseFloat w = e->val->tot_cost;
      tmp_array.push_back(w);

      if (w < best_weight) {
        best_weight = w;
        *best_elem = e;
      }
    }

    *tok_count = tmp_array.size();
    const BaseFloat beam_cutoff = best_weight + config_.beam;
    BaseFloat min_active_cutoff = std::numeric_limits<BaseFloat>::infinity();

    KALDI_VLOG(6) << "Number of tokens active on frame " << NumFramesDecoded()
                  << " is " << tmp_array.size();

    if (tmp_array.size() > config_.max_active) {
      std::nth_element(tmp_array.begin(),
                       tmp_array.begin() + config_.max_active, tmp_array.end());

      const BaseFloat max_active_cutoff = tmp_array[config_.max_active];

      if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
        *adaptive_beam = max_active_cutoff - best_weight + config_.beam_delta;
        return max_active_cutoff;
      }
    }

    if (tmp_array.size() > config_.min_active) {
      if (config_.min_active == 0) {
        min_active_cutoff = best_weight;
      } else {
        std::nth_element(tmp_array.begin(),
                         tmp_array.begin() + config_.min_active,
                         tmp_array.size() > config_.max_active ?
                         tmp_array.begin() + config_.max_active :
                         tmp_array.end());

        min_active_cutoff = tmp_array[config_.min_active];
      }
    }

    if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
      *adaptive_beam = min_active_cutoff - best_weight + config_.beam_delta;
      return min_active_cutoff;
    } else {
      *adaptive_beam = config_.beam;
      return beam_cutoff;
    }
  }
}

template<typename FST, typename Token>
BaseFloat LatticeFasterDecoderTpl<FST, Token>::ProcessEmitting(
    DecodableInterface *decodable) {
  KALDI_ASSERT(!active_toks_.empty());

  // frame is the frame-index (zero-based) used to get likelihoods from the
  // decodable object.
  const auto frame = active_toks_.size() - 1;
  active_toks_.emplace_back();
  Elem *final_toks = toks_.Clear();
  Elem *best_elem = nullptr;
  BaseFloat adaptive_beam;
  size_t tok_cnt;
  auto cur_cutoff = GetCutoff(final_toks, &tok_cnt, &adaptive_beam, &best_elem);

  KALDI_VLOG(6) << "Adaptive beam on frame " << NumFramesDecoded() << " is "
                << adaptive_beam;

  // This makes sure the hash is always big enough.
  PossiblyResizeHash(tok_cnt);
  BaseFloat next_cutoff = std::numeric_limits<BaseFloat>::infinity();

  // pruning "online" before having seen all tokens
  // Used to keep probabilities in a good dynamic range.
  BaseFloat cost_offset = 0.0;

  // First process the best token to get a hopefully reasonably tight bound on
  // the next cutoff. The only products of the next block are "next_cutoff" and
  // "cost_offset".
  if (best_elem) {
    cost_offset = -best_elem->val->tot_cost;

    for (fst::ArcIterator<FST> aiter(*fst_, best_elem->key); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();

      if (arc.ilabel != 0) {  // propagate..
        const auto new_weight =
            arc.weight.Value() -
            decodable->LogLikelihood(static_cast<int32>(frame), arc.ilabel);

        const auto est_next_cutoff = new_weight + adaptive_beam;

        if (est_next_cutoff < next_cutoff) {
          next_cutoff = est_next_cutoff;
        }
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
  for (Elem *e = final_toks, *e_tail; e != nullptr; e = e_tail) {
    // loop this way because we delete "e" as we go.
    Token *tok = e->val;

    if (tok->tot_cost <= cur_cutoff) {
      for (fst::ArcIterator<FST> aiter(*fst_, e->key); !aiter.Done();
           aiter.Next()) {
        const Arc &arc = aiter.Value();

        if (arc.ilabel != 0) {  // propagate..
          const auto ac_cost =
              cost_offset -
              decodable->LogLikelihood(static_cast<int32>(frame), arc.ilabel),
              graph_cost = arc.weight.Value();

          BaseFloat tot_cost = tok->tot_cost + ac_cost + graph_cost;

          if (tot_cost > next_cutoff) continue;
          else if (tot_cost + adaptive_beam < next_cutoff) {
            // prune by best current token
            next_cutoff = tot_cost + adaptive_beam;
          }

          // Note: the frame indexes into active_toks_ are one-based,
          // hence the + 1.
          Token *next_tok = FindOrAddToken(arc.nextstate,
                                           frame + 1, tot_cost, tok, nullptr);
          // NULL: no change indicator needed

          // Add Link from tok to next_tok (put on head of list tok->links)
          tok->links.emplace_front(next_tok, arc.ilabel, arc.olabel, graph_cost,
                                   ac_cost);
        }
      } // for all arcs
    }

    e_tail = e->tail;
    toks_.Delete(e); // delete Elem
  }

  return next_cutoff;
}

template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::ProcessNonemitting(BaseFloat cutoff) {
  KALDI_ASSERT(!active_toks_.empty());

  // Note: "frame" is the time-index we just processed, or -1 if we are
  // processing the nonemitting transitions before the first frame(called from
  // InitDecoding()).
  const int32 frame = static_cast<int32>(active_toks_.size()) - 2;

  if (toks_.GetList() == nullptr && !warned_) {
    KALDI_WARN << "Error, no surviving tokens: frame is " << frame;
    warned_ = true;
  }

  // Processes nonemitting arcs for one frame.  Propagates within toks_.
  // Note-- this queue structure is is not very optimal as
  // it may cause us to process states unnecessarily (e.g. more than once),
  // but in the baseline code, turning this vector into a set to fix this
  // problem did not improve overall speed.
  std::stack<StateId> queue;

  for (const Elem *e = toks_.GetList(); e != nullptr; e = e->tail) {
    StateId state = e->key;

    if (fst_->NumInputEpsilons(state) != 0) {
      queue.push(state);
    }
  }

  while (!queue.empty()) {
    const StateId state = queue.top();
    queue.pop();

    // would segfault if state not in toks_ but this can't happen.
    Token *tok = toks_.Find(state)->val;

    const BaseFloat cur_cost = tok->tot_cost;

    if (cur_cost > cutoff) {  // Don't bother processing successors.
      continue;
    }

    // If "tok" has any existing forward links, delete them, because we're about
    // to regenerate them.  This is a kind of non-optimality, but since most
    // states are emitting it's not a huge issue.
    tok->links.clear();  // necessary when re-visiting

    for (fst::ArcIterator<FST> aiter(*fst_, state); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();

      // propagate non-emitting only.
      if (arc.ilabel == 0) {
        const BaseFloat graph_cost = arc.weight.Value();
        const BaseFloat tot_cost = cur_cost + graph_cost;

        if (tot_cost < cutoff) {
          bool changed;

          Token *new_tok = FindOrAddToken(arc.nextstate, frame + 1, tot_cost,
                                          tok, &changed);

          tok->links.emplace_front(new_tok, 0, arc.olabel, graph_cost, 0);

          // "changed" tells us whether the new token has a different
          // cost from before, or is new [if so, add into queue].
          if (changed && fst_->NumInputEpsilons(arc.nextstate) != 0) {
            queue.push(arc.nextstate);
          }
        }
      }
    } // for all arcs
  } // while queue not empty
}

template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::DeleteElems(Elem *list) noexcept {
  for (Elem *e = list, *e_tail; e != nullptr; e = e_tail) {
    e_tail = e->tail;
    toks_.Delete(e);
  }
}

template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::ClearActiveTokens() noexcept {
  active_toks_.clear();
  num_toks_ = 0;
}

// static
template<typename FST, typename Token>
void LatticeFasterDecoderTpl<FST, Token>::UpdateTokSetAndTok2Pos(
    const std::forward_list<decoder::Link<Token>> &links, int32 pos,
    unordered_set<const Token *> *process,
    unordered_map<const Token *, int32> *token2pos, int32 *cur_pos) noexcept {
  for (const auto &link : links) {
    // We only need to consider epsilon links, since non-epsilon links
    // transition between frames and this function only needs to sort a list
    // of tokens from a single frame.
    if (link.i_label == 0) {
      auto following_iter = token2pos->find(link.next_tok);

      // another token on this frame, so must consider it.
      if (following_iter != token2pos->end()) {
        int32 next_pos = following_iter->second;

        if (next_pos < pos) { // reassign the position of the next Token.
          following_iter->second = (*cur_pos)++;
          process->insert(link.next_tok);
        }
      }
    }
  }
}

// static
template<typename FST, typename Token>
std::vector<const Token *> LatticeFasterDecoderTpl<FST, Token>::TopSortTokens(
    const std::list<Token> &tok_list) noexcept {
  unordered_map<const Token *, int32> token2pos;
  int32 cur_pos = 0;

  // We assign the tokens numbers num_toks - 1, ... , 2, 1, 0.
  // This is likely to be in closer to topological order than
  // if we had given them ascending order, because of the way
  // new tokens are put at the front of the list.
  for (auto &tok : tok_list) {
    token2pos[&tok] = tok_list.size() - ++cur_pos;
  }

  unordered_set<const Token *> reprocess;

  for (const auto &item : token2pos) {
    const Token *tok = item.first;
    UpdateTokSetAndTok2Pos(tok->links, item.second, &reprocess, &token2pos,
                           &cur_pos);

    // In case we had previously assigned this token to be reprocessed, we can
    // erase it from that set because it's "happy now" (we just processed it).
    reprocess.erase(tok);
  }

  size_t max_loop = 1000000, loop_count; // max_loop is to detect epsilon cycles.

  for (loop_count = 0;
       !reprocess.empty() && loop_count < max_loop; ++loop_count) {
    std::vector<const Token *> reprocess_vec(reprocess.begin(),
                                             reprocess.end());

    reprocess.clear();

    for (const auto &tok : reprocess_vec) {
      int32 pos = token2pos[tok];
      UpdateTokSetAndTok2Pos(tok->links, pos, &reprocess, &token2pos, &cur_pos);
    }
  }

  KALDI_ASSERT(loop_count < max_loop && "Epsilon loops exist in your decoding "
                                        "graph (this is not allowed!)");

  std::vector<const Token *> top_sorted_list(static_cast<unsigned>(cur_pos));

  for (const auto &item : token2pos) {
    top_sorted_list[item.second] = item.first;
  }

  return top_sorted_list;
}

// Instantiate the template for the combination of token types and FST types
// that we'll need.
template
class LatticeFasterDecoderTpl<fst::Fst<fst::StdArc>, decoder::StdToken>;

template
class LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>, decoder::StdToken>;

template
class LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>, decoder::StdToken>;

template
class LatticeFasterDecoderTpl<fst::GrammarFst, decoder::StdToken>;

template
class LatticeFasterDecoderTpl<fst::Fst<fst::StdArc>, decoder::BackpointerToken>;

template
class LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>, decoder::BackpointerToken>;

template
class LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>, decoder::BackpointerToken>;

template
class LatticeFasterDecoderTpl<fst::GrammarFst, decoder::BackpointerToken>;


} // end namespace kaldi.
