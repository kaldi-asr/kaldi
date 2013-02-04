// decoder/faster-decoder.cc

// Copyright 2009-2012 Microsoft Corporation
//                     Johns Hopkins University (author: Daniel Povey)

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

#include "decoder/faster-decoder.h"

namespace kaldi {


FasterDecoder::FasterDecoder(const fst::Fst<fst::StdArc> &fst,
                             const FasterDecoderOptions &opts): fst_(fst), opts_(opts) {
  KALDI_ASSERT(opts_.hash_ratio >= 1.0);  // less doesn't make much sense.
  KALDI_ASSERT(opts_.max_active > 1);
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
}


void FasterDecoder::Decode(DecodableInterface *decodable) {
  // clean up from last time:
  ClearToks(toks_.Clear());
  StateId start_state = fst_.Start();
  KALDI_ASSERT(start_state != fst::kNoStateId);
  Arc dummy_arc(0, 0, Weight::One(), start_state);
  toks_.Insert(start_state, new Token(dummy_arc, NULL));
  ProcessNonemitting(std::numeric_limits<float>::max());
  for (int32 frame = 0; !decodable->IsLastFrame(frame-1); frame++) {
    BaseFloat weight_cutoff = ProcessEmitting(decodable, frame);
    ProcessNonemitting(weight_cutoff);
  }
}

bool FasterDecoder::ReachedFinal() {
  for (Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
    Weight this_weight = Times(e->val->weight_, fst_.Final(e->key));
    if (this_weight != Weight::Zero())
      return true;
  }
  return false;
}

bool FasterDecoder::GetBestPath(fst::MutableFst<LatticeArc> *fst_out) {
  // GetBestPath gets the decoding output.  If is_final == true, it limits itself
  // to final states; otherwise it gets the most likely token not taking into
  // account final-probs.  fst_out will be empty (Start() == kNoStateId) if
  // nothing was available.  It returns true if it got output (thus, fst_out
  // will be nonempty).
  fst_out->DeleteStates();
  Token *best_tok = NULL;
  bool is_final = ReachedFinal();
  if (!is_final) {
    for (Elem *e = toks_.GetList(); e != NULL; e = e->tail)
      if (best_tok == NULL || *best_tok < *(e->val) )
        best_tok = e->val;
  } else {
    Weight best_weight = Weight::Zero();
    for (Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
      Weight this_weight = Times(e->val->weight_, fst_.Final(e->key));
      if (this_weight != Weight::Zero() &&
          this_weight.Value() < best_weight.Value()) {
        best_weight = this_weight;
        best_tok = e->val;
      }
    }
  }
  if (best_tok == NULL) return false;  // No output.

  std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.

  for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
    BaseFloat tot_cost = tok->weight_.Value() -
        (tok->prev_ ? tok->prev_->weight_.Value() : 0.0),
        graph_cost = tok->arc_.weight.Value(),
        ac_cost = tot_cost - graph_cost;
    LatticeArc l_arc(tok->arc_.ilabel,
                     tok->arc_.olabel,
                     LatticeWeight(graph_cost, ac_cost),
                     tok->arc_.nextstate);
    arcs_reverse.push_back(l_arc);
  }
  KALDI_ASSERT(arcs_reverse.back().nextstate == fst_.Start());
  arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

  StateId cur_state = fst_out->AddState();
  fst_out->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    LatticeArc arc = arcs_reverse[i];
    arc.nextstate = fst_out->AddState();
    fst_out->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final) {
    Weight final_weight = fst_.Final(best_tok->arc_.nextstate);
    fst_out->SetFinal(cur_state, LatticeWeight(final_weight.Value(), 0.0));
  } else {
    fst_out->SetFinal(cur_state, LatticeWeight::One());
  }
  RemoveEpsLocal(fst_out);
  return true;
}


// Gets the weight cutoff.  Also counts the active tokens.
BaseFloat FasterDecoder::GetCutoff(Elem *list_head, size_t *tok_count,
                                   BaseFloat *adaptive_beam, Elem **best_elem) {
  BaseFloat best_weight = 1.0e+10;  // positive == high cost == bad.
  size_t count = 0;
  if (opts_.max_active == std::numeric_limits<int32>::max()) {
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = static_cast<BaseFloat>(e->val->weight_.Value());
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (adaptive_beam != NULL) *adaptive_beam = opts_.beam;
    return best_weight + opts_.beam;
  } else {
    tmp_array_.clear();
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = e->val->weight_.Value();
      tmp_array_.push_back(w);
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (tmp_array_.size() <= static_cast<size_t>(opts_.max_active)) {
      if (adaptive_beam) *adaptive_beam = opts_.beam;
      return best_weight + opts_.beam;
    } else {
      // the lowest elements (lowest costs, highest likes)
      // will be put in the left part of tmp_array.
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin()+opts_.max_active,
                       tmp_array_.end());
      // return the tighter of the two beams.
      BaseFloat ans = std::min(best_weight + opts_.beam,
                               *(tmp_array_.begin()+opts_.max_active));
      if (adaptive_beam)
        *adaptive_beam = std::min(opts_.beam,
                                  ans - best_weight + opts_.beam_delta);
      return ans;
    }
  }
}

void FasterDecoder::PossiblyResizeHash(size_t num_toks) {
  size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                      * opts_.hash_ratio);
  if (new_sz > toks_.Size()) {
    toks_.SetSize(new_sz);
  }
}

// ProcessEmitting returns the likelihood cutoff used.
BaseFloat FasterDecoder::ProcessEmitting(DecodableInterface *decodable, int frame) {
  Elem *last_toks = toks_.Clear();
  size_t tok_cnt;
  BaseFloat adaptive_beam;
  Elem *best_elem = NULL;
  BaseFloat weight_cutoff = GetCutoff(last_toks, &tok_cnt,
                                      &adaptive_beam, &best_elem);
  PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.
    
  // This is the cutoff we use after adding in the log-likes (i.e.
  // for the next frame).  This is a bound on the cutoff we will use
  // on the next frame.
  BaseFloat next_weight_cutoff = 1.0e+10;

  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.
  if (best_elem) {
    StateId state = best_elem->key;
    Token *tok = best_elem->val;
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // we'd propagate..
        BaseFloat ac_cost = - decodable->LogLikelihood(frame, arc.ilabel),
            new_weight = arc.weight.Value() + tok->weight_.Value() + ac_cost;
        if (new_weight + adaptive_beam < next_weight_cutoff)
          next_weight_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  // int32 n = 0, np = 0;

  // the tokens are now owned here, in last_toks, and the hash is empty.
  // 'owned' is a complex thing here; the point is we need to call TokenDelete
  // on each elem 'e' to let toks_ know we're done with them.
  for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) {  // loop this way
    // n++;
    // because we delete "e" as we go.
    StateId state = e->key;
    Token *tok = e->val;
    if (tok->weight_.Value() < weight_cutoff) {  // not pruned.
      // np++;
      KALDI_ASSERT(state == tok->arc_.nextstate);
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
           !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          Weight ac_weight(- decodable->LogLikelihood(frame, arc.ilabel));
          BaseFloat new_weight = arc.weight.Value() + tok->weight_.Value()
              + ac_weight.Value();
          if (new_weight < next_weight_cutoff) {  // not pruned..
            Token *new_tok = new Token(arc, ac_weight, tok);
            Elem *e_found = toks_.Find(arc.nextstate);
            if (new_weight + adaptive_beam < next_weight_cutoff)
              next_weight_cutoff = new_weight + adaptive_beam;
            if (e_found == NULL) {
              toks_.Insert(arc.nextstate, new_tok);
            } else {
              if ( *(e_found->val) < *new_tok ) {
                Token::TokenDelete(e_found->val);
                e_found->val = new_tok;
              } else {
                Token::TokenDelete(new_tok);
              }
            }
          }
        }
      }
    }
    e_tail = e->tail;
    Token::TokenDelete(e->val);
    toks_.Delete(e);
  }
  return next_weight_cutoff;
}

// TODO: first time we go through this, could avoid using the queue.
void FasterDecoder::ProcessNonemitting(BaseFloat cutoff) {
  // Processes nonemitting arcs for one frame. 
  KALDI_ASSERT(queue_.empty());
  for (Elem *e = toks_.GetList(); e != NULL;  e = e->tail)
    queue_.push_back(e->key);
  while (!queue_.empty()) {
    StateId state = queue_.back();
    queue_.pop_back();
    Token *tok = toks_.Find(state)->val;  // would segfault if state not
    // in toks_ but this can't happen.
    if (tok->weight_.Value() > cutoff) { // Don't bother processing successors.
      continue;
    }
    KALDI_ASSERT(tok != NULL && state == tok->arc_.nextstate);
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        Token *new_tok = new Token(arc, tok);
        if (new_tok->weight_.Value() > cutoff) {  // prune
          Token::TokenDelete(new_tok);
        } else {
          Elem *e_found = toks_.Find(arc.nextstate);
          if (e_found == NULL) {
            toks_.Insert(arc.nextstate, new_tok);
            queue_.push_back(arc.nextstate);
          } else {
            if ( *(e_found->val) < *new_tok ) {
              Token::TokenDelete(e_found->val);
              e_found->val = new_tok;
              queue_.push_back(arc.nextstate);
            } else {
              Token::TokenDelete(new_tok);
            }
          }
        }
      }
    }
  }
}

void FasterDecoder::ClearToks(Elem *list) {
  for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
    Token::TokenDelete(e->val);
    e_tail = e->tail;
    toks_.Delete(e);
  }
}

} // end namespace kaldi.
