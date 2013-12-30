// decoder/nbest-decoder.h

// Copyright 2009-2011 Mirko Hannemann; Microsoft Corporation
//                     Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_DECODER_NBEST_DECODER_H_
#define KALDI_DECODER_NBEST_DECODER_H_

#ifdef _MSC_VER
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
#include "util/stl-utils.h"
#include "itf/options-itf.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "lat/kaldi-lattice.h" // for CompactLatticeArc

namespace kaldi {


struct NBestDecoderOptions {
  BaseFloat beam;
  int32 max_active;
  int32 n_best;
  BaseFloat beam_delta;
  BaseFloat hash_ratio;
  NBestDecoderOptions(): beam(16.0),
                          max_active(std::numeric_limits<int32>::max()),
                          n_best(1),
                          beam_delta(0.5), hash_ratio(2.0) { }
  void Register(OptionsItf *po, bool full) {  /// if "full", use obscure
    /// options too.
    /// Depends on program.
    po->Register("beam", &beam, "Decoder beam");
    po->Register("max-active", &max_active, "Decoder max active states.");
    po->Register("n-best", &n_best, "Decoder number of best tokens.");
    if (full) {
      po->Register("beam-delta", &beam_delta,
                   "Increment used in decoder [obscure setting]");
      po->Register("hash-ratio", &hash_ratio,
                   "Setting used in decoder to control hash behavior");
    }
  }
};

class NBestDecoder {
 public:
  // maybe use fst<LatticeArc>/fst<CompactLatticeArc>, as in lat/kaldi-lattice.h
  // to store information to get graph and acoustic scores separately
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  // instantiate this class once for each thing you have to decode.
  NBestDecoder(const fst::Fst<fst::StdArc> &fst,
                NBestDecoderOptions opts): fst_(fst), opts_(opts) {
    KALDI_ASSERT(opts_.hash_ratio >= 1.0);  // less doesn't make much sense.
    KALDI_ASSERT(opts_.max_active > 1);
    toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
    decodable_ = NULL;
  }

  void SetOptions(const NBestDecoderOptions &opts) { opts_ = opts; }

  ~NBestDecoder() {
    ClearToks(toks_.Clear());
    token_store_.Clear();
    decodable_ = NULL;
  }

  void Decode(DecodableInterface *decodable) {
    decodable_ = decodable;
    // clean up from last time:
    ClearToks(toks_.Clear());
    token_store_.Init(decodable, &toks_, opts_.n_best);
    StateId start_state = fst_.Start();
    KALDI_ASSERT(start_state != fst::kNoStateId);
    Token *tok = token_store_.CreateTok(0, NULL);
    tok->c = Weight::One();
    tok->ca = Weight::One();
    tok->I = NULL;
    toks_.Insert(start_state, tok);
    PropagateEpsilon(std::numeric_limits<float>::max());
    for (int32 frame = 0; !decodable_->IsLastFrame(frame-1); frame++) {
      BaseFloat adaptive_beam = PropagateEmitting(frame);
      PropagateEpsilon(adaptive_beam);
      // Prune();
      // if (frame==12) break;
    }
  }

  bool ReachedFinal() {
    for (Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
      Weight this_weight = Times(e->val->c, fst_.Final(e->key));
      if (this_weight != Weight::Zero()) {
        return true;
      }
    }
    return false;
  }

  bool GetNBestLattice(fst::MutableFst<CompactLatticeArc> *fst_out,
                        bool *was_final, int32 *nbest, BaseFloat *nbest_beam) {
    // GetBestPath gets the decoding output.  If is_final == true, it limits itself
    // to final states; otherwise it gets the most likely token not taking into
    // account final-probs.  fst_out will be empty (Start() == kNoStateId) if
    // nothing was available.  It returns number of paths if it got output
    // (thus, fst_out will be nonempty), otherwise zero.
    int n_paths = 0;
    BaseFloat worst_final = 0.0,
              best_final = std::numeric_limits<BaseFloat>::infinity();
    *was_final = ReachedFinal();
    Elem *last_toks = toks_.Clear(); // P <- C , C = {}
    Token *tok = token_store_.CreateTok(0, NULL);
    StateId end_state = 1E9; // some imaginary super end state
    tok->c = Weight::Zero();
    tok->I = NULL;
    toks_.Insert(end_state, tok);
    Elem *best_e = toks_.Find(end_state);

    if (!(*was_final)) { // only look for best tokens in this frame
      for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) {
        token_store_.CombineN(best_e, e->val);
        e_tail = e->tail;
        toks_.Delete(e);
      }
    } else { // find best tokens in final states
      for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) {
        Token *source = e->val;
        Weight fw = fst_.Final(e->key);
        if (fw != Weight::Zero()) {
          source->c = Times(source->c, fw);
          token_store_.CombineN(best_e, source);
        } else {
          token_store_.DeleteTok(source);
        }
        e_tail = e->tail;
        toks_.Delete(e);
      }
    }
    // start building output FST
    fst_out->DeleteStates();
    StateId start_state = fst_out->AddState();
    fst_out->SetStart(start_state);
    last_toks = toks_.Clear(); // just tokens in super end state this time
    if (last_toks->val == NULL) return false;  // No output.
    // go through tokens in imaginary super end state
    for (Elem *e = last_toks, *e_tail = NULL; e != NULL; e = e_tail) {
      Token *best_tok = e->val;
      BaseFloat amscore = best_tok->ca.Value(),
                lmscore = best_tok->c.Value() - amscore;
      if (KALDI_ISINF(amscore) || KALDI_ISINF(lmscore)) {
        KALDI_WARN << "infinity token! probably too narrow beam to retrieve n best";
        e_tail = e->tail;
        toks_.Delete(e);
        continue; // skip that token
      }
      LatticeWeight path_w(lmscore, amscore);
      CompactLatticeWeight path_weight(path_w, vector<int32>());

      std::vector<CompactLatticeArc*> arcs_reverse; // reverse order output arcs
      // outer loop for word tokens
      for (Token *tok = best_tok; tok != NULL; tok = tok->previous) {
        // inner loop for input label tokens
        std::vector<int32> str_rev, str;
        for (SeqToken *stok = tok->I; stok != NULL; stok = stok->previous) {
          str_rev.push_back(stok->i);
        }
        // reverse vector
        std::vector<int32>::reverse_iterator rit;
        for (rit = str_rev.rbegin(); rit < str_rev.rend(); ++rit)
          str.push_back(*rit);
        arcs_reverse.push_back(new CompactLatticeArc(
            tok->o, tok->o, CompactLatticeWeight(LatticeWeight::One(), str), 0));
        // no weight info (tok->c), no state info
      }
      token_store_.DeleteTok(best_tok);

      StateId cur_state = start_state;
      for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
        CompactLatticeArc *arc = arcs_reverse[i];
        arc->nextstate = fst_out->AddState();
        fst_out->AddArc(cur_state, *arc);
        cur_state = arc->nextstate;
        delete arc;
      }
      fst_out->SetFinal(cur_state, path_weight);
      e_tail = e->tail;
      toks_.Delete(e);
      n_paths++;
      BaseFloat p_score = amscore + lmscore;
      if (p_score > worst_final) worst_final = p_score;
      if (p_score < best_final) best_final = p_score;
    }
    if (nbest_beam) *nbest_beam = worst_final - best_final;
    if (nbest) *nbest = n_paths;

    RemoveEpsLocal(fst_out);
    return true;
  }
//  bool GetBestPath(fst::MutableFst<LatticeArc> *fst_out, bool *was_final) {
//    fst::VectorFst<CompactLatticeArc> fst, fst_one;
//    if (!GetNBestLattice(&fst, was_final)) return false;
    // std::cout << "n-best paths:\n";
    // fst::FstPrinter<CompactLatticeArc> fstprinter(fst, NULL, NULL, NULL, false, true);
    // fstprinter.Print(&std::cout, "standard output");
//    ShortestPath(fst, &fst_one);
//    ConvertLattice(fst_one, fst_out, true);
//    return true;
//  } 
  
 private:

  // TokenStore is a store of linked tokens with its own allocator
  class TokenStore {
    // the pointer *previous has a double function:
    // it's also used in linked lists to store allocated tokens
   public:
    struct SeqToken { // an incremental/relative token inside a full Token
      Label i;   // input label i
      SeqToken *previous;  // lattice backward pointer (also as linked list)
      int refs;       // reference counter (for memory management)
    };
    class Token {
     public:
      // here will be the c and I of 'full tokens'
      Weight c; // c (total weight)
      Weight ca; // acoustic part of c
      SeqToken *I; // sequence I
      Label o; // o
      Token *previous; // t'
      int32 refs; // reference counter (for memory management)
      unsigned hash; // hashing the output symbol sequence
      inline bool operator < (const Token &other) {
        return c.Value() > other.c.Value();
        // This makes sense for log + tropical semiring.
      }
      inline bool Equal(Token *other) { // compares output sequences of Tokens
        if (hash != other->hash) return false;
        Token *t1 = this, *t2 = other;
        while (t1 != NULL && t2 != NULL) {
          if (t1->o != t2->o) return false;
          t1 = t1->previous; t2 = t2->previous;
          if (t1 == t2) { return true; }
          if ((!t1) || (!t2)) { return false; }
        }
	KALDI_ASSERT(false); // should never reach this point
	return true;
      }
    };
    typedef HashList<StateId, Token*> TokenHash;
    typedef TokenHash::Elem Elem;
    void Init(DecodableInterface *decodable, TokenHash *toks, int32 n_best) {
      Clear();
      n_best_ = n_best;
      decodable_ = decodable;
      toks_ = toks;
    }

    inline void DeleteSeq(SeqToken *seq) {
      // delete seq token: either decrease reference count or put to linked list
      seq->refs--;
      if (seq->refs > 0) return;
      // really kill sequence token
      // clean up unused sequence tokens recursively backwards
      if (seq->previous != NULL) DeleteSeq(seq->previous);
      // save the unused sequence token in linked list (abusing *previous)
      seq->previous = free_st_head_;
      free_st_head_ = seq;
    }
    inline SeqToken *NewSeq() {
      // new seq token: either take from linked list or extend list
      SeqToken *tmp;
      if (free_st_head_) {  // take from free list
        tmp = free_st_head_;
        free_st_head_ = free_st_head_->previous;
      } else {  // make new entries in free list
        tmp = new SeqToken[allocate_block_size_];
        for (size_t i = 0; i + 1 < allocate_block_size_; i++) {
          tmp[i].previous = tmp + i + 1;
        }
        tmp[allocate_block_size_ - 1].previous = NULL;
        allocated_s_.push_back(tmp);
        free_st_head_ = tmp->previous;
      }
      // initialize new sequence token (append or start new sequence)
      tmp->refs = 1;
      return tmp;
    }
    inline SeqToken *CreateSeq(Label input, SeqToken *prev) {
      SeqToken *tmp = NULL;
      if (input > 0) {
        tmp = NewSeq();
        tmp->i = input;
        tmp->previous = prev;
        if (prev) {
          prev->refs++;
        }
      } else {
        if (prev) {
          tmp = prev;
          if (prev) { // prev->previous?
            prev->refs++;
          }
        }
      }
      return tmp;
    }

    inline void DeleteTok(Token *tok) {
      // delete token: either decrease reference count or put to linked list
      tok->refs--;
      if (tok->refs > 0) return;
      // really kill token
      // clean up unused tokens recursively backwards
      if (tok->previous != NULL) {
        DeleteTok(tok->previous);
      }
      if (tok->I != NULL) { // delete sequence I
        DeleteSeq(tok->I);
      }
      // save the unused token in linked list (abusing *previous)
      tok->previous = free_t_head_;
      free_t_head_ = tok;
    }
    inline Token *CreateTok(Label output, Token *prev) {
      // new token: either take from linked list or extend list
      Token *tmp;
      if (free_t_head_) {  // take from free list
        tmp = free_t_head_;
        free_t_head_ = free_t_head_->previous;
      } else {  // make new entries in free list
        tmp = new Token[allocate_block_size_];
        for (size_t i = 0; i + 1 < allocate_block_size_; i++) {
          tmp[i].previous = tmp + i + 1;
        }
        tmp[allocate_block_size_ - 1].previous = NULL;
        allocated_t_.push_back(tmp);
        free_t_head_ = tmp->previous;
      }
      // initialize data
      tmp->refs = 1;
      tmp->c = Weight::Zero();
      tmp->I = NULL;
      tmp->o = output;
      tmp->previous = prev;
      if (prev) {
        prev->refs++;
	tmp->hash = prev->hash * 97 + static_cast<unsigned>(output);
      } else {
        tmp->hash = static_cast<unsigned>(output);
      }
      return tmp;
    }

    inline Token* Combine(Token *tok1, Token *tok2) { // Viterbi version
      KALDI_ASSERT(tok1);
      if (!tok2) return tok1;
      if (tok1 == tok2) return tok1;
      if (tok1->c.Value() < tok2->c.Value()) {
        DeleteTok(tok2);
        return tok1;
      } else {
        DeleteTok(tok1);
        return tok2;
      }
    }
    
    inline bool CombineN(Elem *head, Token *new_tok) { // n-best version
      if (!new_tok) return false;
      Elem *e = head;
      StateId state = e->key;
      BaseFloat new_weight = static_cast<BaseFloat>(new_tok->c.Value());
      size_t count = 0;
      BaseFloat worst_weight = 0.0;  // small == low cost
      Elem *worst_elem = NULL;
      do {
        count++;
        Token *tok = e->val;
        if (tok == new_tok) { return false; }
        BaseFloat w = static_cast<BaseFloat>(tok->c.Value());
        if (w > worst_weight) {
           worst_weight = w;
           worst_elem = e;
        }
        if (tok->Equal(new_tok)) { // if they have the same output sequence
          if (w < new_weight) {
            DeleteTok(new_tok);
            return false;
          } else {
            DeleteTok(tok);
	        e->val = new_tok;
            return true;
          }
        }
        e = e->tail;
      } while ( (e != NULL) && (e->key == state) );
      // if we are here, no Token with the same output sequence was found
      if (count < n_best_) {
        toks_->InsertMore(state, new_tok);
        return true;
      } else {
        if (worst_weight < new_weight) {
          DeleteTok(new_tok);
          return false;
        } else {
          DeleteTok(worst_elem->val);
          worst_elem->val = new_tok;
          return true;
        }
      }
    }
    inline Token* Advance(Token *source, Arc &arc, int32 frame,
        BaseFloat cutoff) {
      // compute new weight	
      Weight w = Times(source->c, arc.weight);
      Weight amscore = Weight::One();
      if (arc.ilabel > 0) { // emitting arc
        amscore = Weight(- decodable_->LogLikelihood(frame, arc.ilabel));
        w = Times(w, amscore);
      }
      Weight wa = Times(source->ca, amscore);
      if (w.Value() > cutoff) {  // prune
          return NULL;
      }
      // create new token  
      Token *tok;
      if (arc.olabel > 0) { // create new token
        // find or create corresponding Token
        tok = CreateTok(arc.olabel, source); // "new" Token
        tok->I = CreateSeq(arc.ilabel, NULL); // new sequence I starts
      } else { // append sequence I
        tok = CreateTok(source->o, source->previous); // copy previous Token
        tok->I = CreateSeq(arc.ilabel, source->I);
      }
      tok->c = w;
      tok->ca = wa;
      return tok;
    }

    void Clear() {
      // check that all seq tokens are freed
      for (size_t i = 0; i < allocated_t_.size(); i++) delete[] allocated_t_[i];
      allocated_t_.clear();
      free_t_head_ = NULL;
      // check that all seq tokens are freed
      for (size_t i = 0; i < allocated_s_.size(); i++) delete[] allocated_s_[i];
      allocated_s_.clear();
      free_st_head_ = NULL;
    }
    TokenStore() {
      free_t_head_ = NULL;
      free_st_head_ = NULL;
    }
    ~TokenStore() {
      Clear();
    }
   private:
    // head of list of currently free Tokens ready for allocation
    Token *free_t_head_;
    std::vector<Token*> allocated_t_;  // list of allocated tokens
    // head of list of currently free SeqTokens ready for allocation
    SeqToken *free_st_head_;
    std::vector<SeqToken*> allocated_s_;  // list of allocated seq tokens
    DecodableInterface *decodable_;
    TokenHash *toks_;
    int32 n_best_;
    // number of tokens to allocate in one block
    static const size_t allocate_block_size_ = 8192;
  }; // class TokenStore
  typedef TokenStore::Token Token;
  typedef TokenStore::SeqToken SeqToken;
  // typedef HashList<StateId, Token*>::Elem Elem;
  typedef HashList<StateId, Token*> TokenHash;
  typedef TokenHash::Elem Elem;

  /// Gets the weight cutoff.  Also counts the active tokens.
  BaseFloat GetCutoff(Elem *list_head, size_t *tok_count,
                      BaseFloat *adaptive_beam, Elem **best_elem) {
    BaseFloat best_weight = 1.0e+10;  // positive == high cost == bad.
    size_t count = 0;
    // find best token
    tmp_array_.clear();
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = static_cast<BaseFloat>(e->val->c.Value());
      tmp_array_.push_back(w); // ??check this - not always necessary
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    // compute adaptive beam
    if ((opts_.max_active == std::numeric_limits<int32>::max()) ||
        (tmp_array_.size() <= static_cast<size_t>(opts_.max_active))) {
      if (adaptive_beam != NULL) *adaptive_beam = opts_.beam;
      return best_weight + opts_.beam;
    } else {
      // the lowest elements (lowest costs, highest likes)
      // will be put in the left part of tmp_array.
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin() + opts_.max_active,
                       tmp_array_.end());
      // return the tighter of the two beams.
      BaseFloat ans = std::min(best_weight + opts_.beam,
                               *(tmp_array_.begin() + opts_.max_active));
      if (adaptive_beam)
        *adaptive_beam = std::min(opts_.beam,
                                  ans - best_weight + opts_.beam_delta);
      return ans;
    }
  }

  void PossiblyResizeHash(size_t num_toks) {
    size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                        * opts_.hash_ratio);
    if (new_sz > toks_.Size()) {
      toks_.SetSize(new_sz);
    }
  }

  // PropagateEmitting returns the likelihood cutoff used.
  BaseFloat PropagateEmitting(int32 frame) {
    Elem *last_toks = toks_.Clear(); // P <- C , C = {}
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
        Arc arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          arc.weight = Times(arc.weight,
                             Weight(- decodable_->LogLikelihood(frame, arc.ilabel)));
          BaseFloat new_weight = arc.weight.Value() + tok->c.Value();
          if (new_weight + adaptive_beam < next_weight_cutoff)
            next_weight_cutoff = new_weight + adaptive_beam;
        }
      }
    }

    // int32 n = 0, np = 0;

    // the tokens are now owned here, in last_toks, and the hash is empty.
    // 'owned' is a complex thing here; the point is we need to call DeleteElem
    // on each elem 'e' to let toks_ know we're done with them.

    for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) {
      // for all (s,t) in P
      // n++;
      // because we delete "e" as we go.
      StateId state = e->key;
      Token *tok = e->val;
      if (tok->c.Value() < weight_cutoff) {  // not pruned.
        // np++;
        // KALDI_ASSERT(state == tok->arc_.nextstate);
        for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
            !aiter.Done(); aiter.Next()) {
        // for all a in A(state)    
          Arc arc = aiter.Value();
          if (arc.ilabel != 0) {  // propagate only emitting
            Token *new_tok = 
	      token_store_.Advance(tok, arc, frame, next_weight_cutoff);
            if (new_tok) {
              Elem *e_found = toks_.Find(arc.nextstate);
              if (e_found == NULL) {
                toks_.Insert(arc.nextstate, new_tok);
              } else {
                token_store_.CombineN(e_found, new_tok);
              }
            }
          }
        }
      }
      e_tail = e->tail; // several tokens with the same key can follow
      token_store_.DeleteTok(e->val);
      toks_.Delete(e);
    }
    //  std::cerr << n << ', ' << np << ', ' <<adaptive_beam<<' ';
    return adaptive_beam;
  }

  void PropagateEpsilon(BaseFloat adaptive_beam) {
    // Processes nonemitting arcs for one frame.  Propagates within
    // cur_toks_.
    KALDI_ASSERT(queue_.empty());
    queue_.max_load_factor(1.0);
    float best_weight = 1.0e+10;
    for (Elem *e = toks_.GetList(); e != NULL;  e = e->tail) {
      // queue_.push_back(e->key);
      queue_.insert(e->key);
      best_weight = std::min(best_weight, e->val->c.Value());
    }
    BaseFloat cutoff = best_weight + adaptive_beam;

    while (!queue_.empty()) {
      // StateId state = queue_.back();
      StateId state = *(queue_.begin());
      // queue_.pop_back();
      queue_.erase(queue_.begin());
      Elem *elem = toks_.Find(state);  // would segfault if state not
      // in toks_ but this can't happen.
      
      // we have to pop all tokens with the same state
      // this may create some unneccessary repetitions, since only the new token
      // needs to be forwarded, but I don't know yet how to solve this
      while (elem && elem->key == state) {
        Token *tok = elem->val;
        elem = elem->tail;

        if (tok->c.Value() > cutoff) {  // Don't bother processing successors.
          continue;
        }
        // KALDI_ASSERT(tok != NULL && state == tok->arc_.nextstate);
        KALDI_ASSERT(tok != NULL);
        for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_, state);
            !aiter.Done(); aiter.Next()) {
          // for all a in A(state)
          Arc arc = aiter.Value();
          if (arc.ilabel == 0) {  // propagate nonemitting only...
            Token *new_tok = token_store_.Advance(tok, arc, -1, cutoff); // -1:eps
            if (new_tok) {
              Elem *e_found = toks_.Find(arc.nextstate);
              if (e_found == NULL) {
                toks_.Insert(arc.nextstate, new_tok);
                // queue_.push_back(arc.nextstate);
                queue_.insert(arc.nextstate); // might be pushed several times
              } else {
                if (token_store_.CombineN(e_found, new_tok)) { // C was updated
                  // queue_.push_back(arc.nextstate);
                  queue_.insert(arc.nextstate);
                }
              }
            }
          } // if nonemitting
        } // for Arciterator
      } // while
    }
  }

  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.
  TokenHash toks_;
  const fst::Fst<fst::StdArc> &fst_;
  NBestDecoderOptions opts_;
  typedef std::tr1::unordered_set<StateId> StateQueue;
  StateQueue queue_; // used in PropagateEpsilon,
  // std::vector<StateId> queue_;  // temp variable used in PropagateEpsilon,
  std::vector<BaseFloat> tmp_array_;  // used in GetCutoff.
  // make it class member to avoid internal new/delete.
  TokenStore token_store_;
  DecodableInterface *decodable_;

  // It might seem unclear why we call ClearToks(toks_.Clear()).
  // There are two separate cleanup tasks we need to do at when we start a new file.
  // one is to delete the Token objects in the list; the other is to delete
  // the Elem objects.  toks_.Clear() just clears them from the hash and gives ownership
  // to the caller, who then has to call toks_.Delete(e) for each one.  It was designed
  // this way for convenience in propagating tokens from one frame to the next.
  void ClearToks(Elem *list) {
    for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
      token_store_.DeleteTok(e->val);
      e_tail = e->tail;
      toks_.Delete(e);
    }
  }

};


} // end namespace kaldi.


#endif
