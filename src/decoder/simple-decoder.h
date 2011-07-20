// decoder/simple-decoder.h

// Copyright 2009-2011  Microsoft Corporation;  Lukas Burget;
//                      Saarland University

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

#ifndef KALDI_DECODER_SIMPLE_DECODER_H_
#define KALDI_DECODER_SIMPLE_DECODER_H_


#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"

#include <algorithm>
#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
using std::tr1::unordered_map;

namespace kaldi {

/** Simplest possible decoder, included largely for didactic purposes and as a
    means to debug more highly optimized decoders.  See \ref decoders_simple
    for more information.
 */
class SimpleDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  // instantiate this class onece for each thing you have to decode.
  SimpleDecoder(const fst::Fst<fst::StdArc> &fst, BaseFloat beam): fst_(fst), beam_(beam) { }

  ~SimpleDecoder() {
    ClearToks(cur_toks_);
    ClearToks(prev_toks_);
  }

  void Decode(DecodableInterface *decodable) {
    // clean up from last time:
    ClearToks(cur_toks_);
    ClearToks(prev_toks_);
    StateId start_state = fst_.Start();
    KALDI_ASSERT(start_state != fst::kNoStateId);
    Arc dummy_arc(0, 0, Weight::One(), start_state);
    cur_toks_[start_state] = new Token(dummy_arc, NULL);
    ProcessNonemitting();
    for (int32 frame = 0; !decodable->IsLastFrame(frame-1); frame++) {
      ClearToks(prev_toks_);
      std::swap(cur_toks_, prev_toks_);
      ProcessEmitting(decodable, frame);
      ProcessNonemitting();
      PruneToks(beam_, &cur_toks_);
    }
  }

  bool GetOutput(bool is_final, fst::MutableFst<fst::StdArc> *fst_out) {
    // GetOutput gets the decoding output.  If is_final == true, it limits itself to final states;
    // otherwise it gets the most likely token not taking into account final-probs.
    // fst_out will be empty (Start() == kNoStateId) if nothing was available.
    // It returns true if it got output (thus, fst_out will be nonempty).
    fst_out->DeleteStates();
    Token *best_tok = NULL;
    if (!is_final) {
      for (unordered_map<StateId, Token*>::iterator iter = cur_toks_.begin();
          iter != cur_toks_.end();
          ++iter)
        if (best_tok == NULL || *best_tok < *(iter->second) )
          best_tok = iter->second;
    } else {
      Weight best_weight = Weight::Zero();
      for (unordered_map<StateId, Token*>::iterator iter = cur_toks_.begin();
          iter != cur_toks_.end();
          ++iter) {
        Weight this_weight = Times(iter->second->arc_.weight, fst_.Final(iter->first));
        if (this_weight != Weight::Zero() &&
           this_weight.Value() < best_weight.Value()) {
          best_weight = this_weight;
          best_tok = iter->second;
        }
      }
    }
    if (best_tok == NULL) return false;  // No output.

    std::vector<Arc> arcs_reverse;  // arcs in reverse order.
    for (Token *tok = best_tok; tok != NULL; tok = tok->prev_)
      arcs_reverse.push_back(tok->arc_);
    KALDI_ASSERT(arcs_reverse.back().nextstate == fst_.Start());
    arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

    StateId cur_state = fst_out->AddState();
    fst_out->SetStart(cur_state);
    for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
      Arc arc = arcs_reverse[i];
      arc.nextstate = fst_out->AddState();
      fst_out->AddArc(cur_state, arc);
      cur_state = arc.nextstate;
    }
    if (is_final)
      fst_out->SetFinal(cur_state, fst_.Final(best_tok->arc_.nextstate));
    else
      fst_out->SetFinal(cur_state, Weight::One());
    RemoveEpsLocal(fst_out);
    return true;
  }

 private:

  class Token {
   public:
    Arc arc_;
    Token *prev_;
    int32 ref_count_;
    Weight weight_; // accumulated weight up to this point.
    Token(const Arc &arc, Token *prev): arc_(arc), prev_(prev), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        weight_ = Times(prev->weight_, arc.weight);
      } else {
        weight_ = arc.weight;
      }
    }
    bool operator < (const Token &other) {
      return weight_.Value() > other.weight_.Value();
      // This makes sense for log + tropical semiring.
    }

    ~Token() {
      KALDI_ASSERT(ref_count_ == 1);
      if (prev_ != NULL) TokenDelete(prev_);
    }
    static void TokenDelete(Token *tok) {
      if (tok->ref_count_ == 1) delete tok;
      else
        tok->ref_count_--;
    }
  };

  void ProcessEmitting(DecodableInterface *decodable, int frame) {
    // Processes emitting arcs for one frame.  Propagates from
    // prev_toks_ to cur_toks_.
    for (unordered_map<StateId, Token*>::iterator iter = prev_toks_.begin();
        iter != prev_toks_.end();
        ++iter) {
      StateId state = iter->first;
      Token *tok = iter->second;
      KALDI_ASSERT(state == tok->arc_.nextstate);
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
          !aiter.Done();
          aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          arc.weight = Times(arc.weight,
                             Weight(- decodable->LogLikelihood(frame, arc.ilabel)));
          Token *new_tok = new Token(arc, tok);
          unordered_map<StateId, Token*>::iterator find_iter
              = cur_toks_.find(arc.nextstate);
          if (find_iter == cur_toks_.end()) {
            cur_toks_[arc.nextstate] = new_tok;
          } else {
            if ( *(find_iter->second) < *new_tok ) {
              Token::TokenDelete(find_iter->second);
              find_iter->second = new_tok;
            } else {
              Token::TokenDelete(new_tok);
            }
          }
        }
      }
    }
  }

  void ProcessNonemitting() {
    // Processes nonemitting arcs for one frame.  Propagates within
    // cur_toks_.
    std::vector<StateId> queue_;
    float best_weight = 1.0e+10;
    for (unordered_map<StateId, Token*>::iterator iter = cur_toks_.begin();
        iter != cur_toks_.end();
        ++iter) {
      queue_.push_back(iter->first);
      best_weight = std::min(best_weight, iter->second->arc_.weight.Value());
    }
    BaseFloat cutoff = best_weight + beam_;

    while (!queue_.empty()) {
      StateId state = queue_.back();
      queue_.pop_back();
      Token *tok = cur_toks_[state];
      KALDI_ASSERT(tok != NULL && state == tok->arc_.nextstate);
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
          !aiter.Done();
          aiter.Next()) {
        const Arc &arc = aiter.Value();
        if (arc.ilabel == 0) {  // propagate nonemitting only...
          Token *new_tok = new Token(arc, tok);
          if (new_tok->arc_.weight.Value() > cutoff) {
            Token::TokenDelete(new_tok);
          } else {
            unordered_map<StateId, Token*>::iterator find_iter
                = cur_toks_.find(arc.nextstate);
            if (find_iter == cur_toks_.end()) {
              cur_toks_[arc.nextstate] = new_tok;
              queue_.push_back(arc.nextstate);
            } else {
              if ( *(find_iter->second) < *new_tok ) {
                Token::TokenDelete(find_iter->second);
                find_iter->second = new_tok;
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

  unordered_map<StateId, Token*> cur_toks_;
  unordered_map<StateId, Token*> prev_toks_;
  const fst::Fst<fst::StdArc> &fst_;
  BaseFloat beam_;

  static void ClearToks(unordered_map<StateId, Token*> &toks) {
    for (unordered_map<StateId, Token*>::iterator iter = toks.begin();
        iter != toks.end(); ++iter) {
      Token::TokenDelete(iter->second);
    }
    toks.clear();
  }


  static void PruneToks(BaseFloat beam, unordered_map<StateId, Token*> *toks) {
    if (toks->empty()) {
      KALDI_VLOG(2) <<  "No tokens to prune.\n";
      return;
    }
    BaseFloat best_weight = 1.0e+10;  // positive == high cost == bad.
    for (unordered_map<StateId, Token*>::iterator iter = toks->begin();
        iter != toks->end(); ++iter) {
      best_weight =
          std::min(best_weight,
                   static_cast<BaseFloat>(iter->second->weight_.Value()));
    }
    std::vector<StateId> retained;
    BaseFloat cutoff = best_weight + beam;
    for (unordered_map<StateId, Token*>::iterator iter = toks->begin();
        iter != toks->end(); ++iter) {
      if (iter->second->weight_.Value() < cutoff)
        retained.push_back(iter->first);
      else
        Token::TokenDelete(iter->second);
    }
    unordered_map<StateId, Token*> tmp;
    for (size_t i = 0; i < retained.size(); i++) {
      tmp[retained[i]] = (*toks)[retained[i]];
    }
    KALDI_VLOG(2) <<  "Pruned to "<<(retained.size())<<" toks.\n";
    std::swap(tmp, *toks);
  }
};


} // end namespace kaldi.


#endif
