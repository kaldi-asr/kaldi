// decoder/kaldi-decoder.h

// Copyright 2011  Mirko Hannemann;  Lukas Burget

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

#ifndef KALDI_DECODER_KALDI_DECODER_H_
#define KALDI_DECODER_KALDI_DECODER_H_


/**
 * @brief Main decoder class implementation
 *
 * According to legend, Kaldi was the Ethiopian goatherder who discovered the
 * coffee plant. He found his goats' temperaments to be greatly excited after
 * feasting on the ripe red cherries of a small tree that grew on the side of a
 * mountain (they had become dancing goats).
 *
 *                                          http://en.wikipedia.org/wiki/Kaldi
 */

#include <iostream>
#include <algorithm>
#include <queue>
#include <map>
#include <list>
#include <vector>
#include <set>

#include <fst/fstlib.h>
#include <fst/fst-decl.h>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "util/parse-options.h"

// macros to switch off all debugging messages without runtime cost
//#define DEBUG_CMD(x) x;
//#define DEBUG_OUT3(x) KALDI_VLOG(3) << x;
//#define DEBUG_OUT2(x) KALDI_VLOG(2) << x;
//#define DEBUG_OUT1(x) KALDI_VLOG(1) << x;
#define DEBUG_OUT1(x)
#define DEBUG_OUT2(x)
#define DEBUG_OUT3(x)
#define DEBUG_CMD(x)

namespace kaldi {

const int kDecoderBlock = 100000;
// the number of additional entries to be allocated with each vector.resize()

// state coloring: initially kWhite; kGray : touched and kBlack/kBrown : finished
const int kQueue = 1; //black >=0: realized by assigning a topological number
const int kBlack = 0; //black >=0: realized by assigning a topological number
const int kBrown = -1;
const int kGray  = -2;
const int kWhite = -3;
DEBUG_CMD(int wlnumber = 0)
DEBUG_CMD(int dwlnumber = 0)
DEBUG_CMD(int tnumber = 0)
DEBUG_CMD(int dtnumber = 0)


struct KaldiDecoderOptions {
  BaseFloat lm_scale;
  BaseFloat word_penalty;
  int32     max_active_tokens;
  BaseFloat beamwidth;
  BaseFloat beamwidth2;

  KaldiDecoderOptions(): lm_scale(1.0), word_penalty(0.0),
                         max_active_tokens(std::numeric_limits<int32>::max()),
                         beamwidth(16.0), beamwidth2(16.0) { }
  void Register(ParseOptions *po, bool full) {  // "full":use obscure options too
	// depends on program
	po->Register("beam", &beamwidth, "Decoder beam");
	po->Register("max-active", &max_active_tokens, "Decoder max active states.");
	if (full) {
          po->Register("beam2", &beamwidth2,
                       "Decoder beam (tighter after acoustic model)");
          po->Register("lm-scale", &lm_scale, "Language model scaling factor");
          po->Register("word-penalty", &word_penalty, "Word insertion penalty");
    }
  }
};

//****************************************************************************
//****************************************************************************
/**
 * @brief The main decoder class
 */
template<class Decodable, class Fst>
class KaldiDecoder {
 public:
  typedef typename Fst::Arc MyArc;
  typedef typename MyArc::StateId StateId;
  typedef typename MyArc::Weight Weight;
  typedef typename MyArc::Label Label;
  typedef typename fst::ArcIterator<Fst> ArcIterator;

  // WordLinkStore is a store of linked word links with its own allocator
  class WordLinkStore {
  // WordLink and Token are very similar, but not all fields needed in both
  // a WordLink stores the decoding history
  // a Token stores information in the active tokens
  // the pointer *previous has a double function:
  // it's also used in linked lists to store allocated tokens/wordlinks
   public:
    struct WordLink {
      int state;      // time information or state number
      Weight weight;  // accumulated path score (could be stored in arc.weight)
      Label olabel;   // (word) output label
      WordLink *previous;  // lattice backward pointer (also as linked list)
      DEBUG_CMD(int unique)
      // MyArc arc;    // arc that lead to this node (so far only used olabel)
      int refs;       // reference counter (for memory management)
    };

    class Token {
     public:
      StateId state;  // state number or time information
      Weight weight;  // accumulated path score (could be stored in arc.weight)
      Label olabel;   // (word) output label
      WordLink *previous;  // lattice backward pointer (also as linked list)
      //Token *next;
      DEBUG_CMD(int unique)
      // MyArc arc;    // arc that lead to this node (so far only used olabel)
      // int refs;     // reference counter only for WordLink, not for Token

      inline void UpdateWordLink(Token *source_token, WordLinkStore *wl_store) {
        // stores a copy of the back-track pointer from the source token
        // if our token has an old wordlink, remove it:
        if (previous != NULL) wl_store->Delete(previous);
        previous = source_token->previous;  // copy wordlink from source token
        previous->refs++;  // remember the new usage in ref counter
        DEBUG_OUT3("copy:"<< previous->unique << " (" << previous->refs << "x)")
      }
    };  // class Token

    inline void Delete(WordLink *wordlink) {
      // delete wordlink: either decrease reference count or put to linked list
      DEBUG_CMD(assert(wordlink->refs>0))
      wordlink->refs--;
      DEBUG_OUT3("dec:" << wordlink->unique << " (" << wordlink->refs << "x)")
      if (wordlink->refs > 0) return;

      // really kill wordlink
      DEBUG_OUT3( "kill" )
      // clean up unused wordlinks recursively backwards
      if (wordlink->previous != NULL) Delete(wordlink->previous);
      DEBUG_CMD(dwlnumber++)
      // save the unused wordlink in linked list (abusing *previous)
      wordlink->previous = free_wl_head_;
      free_wl_head_ = wordlink;
    }
    inline WordLink *New() {
      // new wordlink: either take from linked list or extend list
      if (free_wl_head_) {  // take from free list
        WordLink *tmp = free_wl_head_;
        free_wl_head_ = free_wl_head_->previous;
        return tmp;
      } else {  // make new entries in free list
        WordLink *tmp = new WordLink[allocate_block_size_];
        for (size_t i = 0; i + 1 < allocate_block_size_; i++) {
          tmp[i].previous = tmp + i + 1;
        }
        tmp[allocate_block_size_ - 1].previous = NULL;
        allocated_.push_back(tmp);
        free_wl_head_ = tmp->previous;
        return tmp;
      }
    }
    inline void LinkToken(Token *token, int frame) {
      // creates WordLink for word labels and fills data fields
      // create the wordlink only at beginning: previous == NULL
      // or at word links: olabel>0
      if ((token->previous != NULL) && (token->olabel <= 0)) return;
      // in future this could be implemented with an ArcFilter

      DEBUG_OUT3( "create WL:" << wlnumber )
      if (token->previous != NULL) { DEBUG_OUT3(" from:"
        << token->previous->unique << " (" << token->previous->refs << "x)") }

      WordLink *ans = New();
      // initializes all data fields
      ans->refs = 1;  // can it be zero if not yet assigned?
      DEBUG_CMD(ans->unique = wlnumber++)
      ans->state = frame;
      ans->weight = token->weight;
      ans->olabel = token->olabel;
      ans->previous = token->previous;  // lattice backward pointer
      token->previous = ans;  // store new wordlink in token
    }
    void Clear() {
      DEBUG_OUT1("wordlinks created: " << wlnumber << " deleted: " << dwlnumber)
      DEBUG_CMD(assert(wlnumber == dwlnumber))
      // check that all wordlinks are freed
      for (size_t i = 0; i < allocated_.size(); i++) delete[] allocated_[i];
      allocated_.clear();
      free_wl_head_ = NULL;
    }
    WordLinkStore() {
      free_wl_head_ = NULL;
    }
    ~WordLinkStore() {
      Clear();
    }
   private:
    // head of list of currently free wordlinks ready for allocation
    WordLink *free_wl_head_;
    std::vector<WordLink*> allocated_;  // list of allocated wordlinks
    // number of wordlinks to allocate in one block
    static const size_t allocate_block_size_ = 8096;
  };  // class WordLinkStore
  typedef typename WordLinkStore::WordLink WordLink;
  typedef typename WordLinkStore::Token Token;


  // TokenStore is a store of tokens with its own allocator
  class TokenStore {
   public:
    inline void Delete(Token *token) {
      // delete token: put to linked list
      DEBUG_OUT3( "kill token:" << token->unique << ":" << token->state)
      DEBUG_CMD(assert(token->state != fst::kNoStateId))
      DEBUG_CMD(token->state = fst::kNoStateId)
      DEBUG_CMD(dtnumber++)
      // clean up unused wordlinks recursively backwards
      if (token->previous != NULL) wl_store_->Delete(token->previous);
      // save the unused token (abusing *previous as linked list)
      token->previous = reinterpret_cast<WordLink*>(free_tok_head_);
      free_tok_head_ = token;
    }
    inline Token *New() {
      // new token: either take from linked list or extend list
      if (free_tok_head_) {  // take from free list
        Token *tmp = free_tok_head_;
        free_tok_head_ = reinterpret_cast<Token*>(free_tok_head_->previous);
        return tmp;
      } else {  // make new entries in free list
        Token *tmp = new Token[allocate_block_size_];
        for (size_t i = 0; i + 1 < allocate_block_size_; i++) {
          tmp[i].previous = reinterpret_cast<WordLink*>(tmp + i + 1);
        }
        tmp[allocate_block_size_ - 1].previous = NULL;
        allocated_.push_back(tmp);
        free_tok_head_ = reinterpret_cast<Token*>(tmp->previous);
        return tmp;
      }
    }
    inline Token *NewToken(int state) {  // initialize Token
      Token *ans = New();
      ans->state = state;
      ans->weight = Weight::Zero();
      ans->previous = NULL;
      //ans->next = NULL;
      ans->olabel = fst::kNoLabel;
      DEBUG_CMD(ans->unique = tnumber++)
  // ans->arc = MyArc(fst::kNoLabel, fst::kNoLabel, Weight::Zero(), fst::kNoStateId);
      DEBUG_OUT3("create token:" << ans->unique)
      return ans;
    }
    void Defragment() {  // write all links consecutively in order
      if (allocated_.size() < 1) return;
      free_tok_head_ = allocated_[0];
      for (size_t j = 0; j < allocated_.size(); j++) {
        Token *tmp = allocated_[j];
        for (size_t i = 0; i + 1 < allocate_block_size_; i++) {
          tmp[i].previous = reinterpret_cast<WordLink*>(tmp + i + 1);
        }
        if (j == allocated_.size() - 1) {
          tmp[allocate_block_size_ - 1].previous = NULL;
        } else {
          tmp[allocate_block_size_ - 1].previous =
            reinterpret_cast<WordLink*>(allocated_[j+1]);
        }
      }
    }
    void DefragmentBack() { // write all links consecutively in order
      if (allocated_.size() < 1) return;
      free_tok_head_ = allocated_[allocated_.size()-1] + allocate_block_size_-1;
      for(size_t j = 0; j < allocated_.size(); j++) {
        Token *tmp = allocated_[j];
        for(size_t i = 1; i < allocate_block_size_; i++) {
          tmp[i].previous = reinterpret_cast<WordLink*>(tmp + i - 1);
        }
        if (j == 0) {
          tmp[0].previous = NULL;
        } else {
          tmp[0].previous = 
            reinterpret_cast<WordLink*>(allocated_[j-1]+allocate_block_size_-1);
        }
      }
    }
    void Clear() {
      DEBUG_OUT1("tokens created: " << tnumber << " deleted: " << dtnumber)
      DEBUG_CMD(assert(tnumber == dtnumber))
      // check that all tokens are freed
      for (size_t i = 0; i < allocated_.size(); i++) delete[] allocated_[i];
      allocated_.clear();
      free_tok_head_ = NULL;
    }
    TokenStore(WordLinkStore *wl_store) {
      free_tok_head_ = NULL;
      wl_store_ = wl_store;
    }
    ~TokenStore() {
      Clear();
    }
   private:
    // head of list of currently free tokens ready for allocation
    Token *free_tok_head_;
    std::vector<Token*> allocated_;  // list of allocated tokens
    // number of tokens to allocate in one block
    static const size_t allocate_block_size_ = 8096;
    WordLinkStore *wl_store_;
  };  // class TokenStore


  // TokenSet is an unordered set of Tokens
  // with random access and consecutive member access
  // used to access the active tokens in decoding

  // StateOrder is a multi-functional data structure:
  // a) a set of states with topological state coloring for recursive visit
  //    used to determine the order of evaluating non-emitting states
  // b) a topologically sorted priority queue of states,
  //    it contains the ordered non-emitting states
  template<class StateId>
  class TokenSet { // TokenSet and StateOrder merged
   private:
    std::vector<Token*> hash_;  // big, sparse vector for random access
    // initially NULL, later stores the pointer to token (also in members_)
    // the hash is only needed for members_next_
    TokenStore *tokens_, *tokens_next_;  // to allocate new tokens
    std::vector<Token*> *members_;  // small, dense vector for consecutive access
    std::vector<Token*> *members_next_;  // the same for next frame
    typename std::vector<Token*>::iterator iter_;  // internal iterator for members
    //Token *members_, *members_next_; // pointers for consecutive access
    //DEBUG_CMD(size_t members_size_)
    //DEBUG_CMD(size_t next_size_)
    //Token *current_; // internal iterator for members

    std::vector<int> color_;  // map state to color (big, sparse vector)
// state coloring: initially kWhite; kGray: touched and kBlack/kBrown : finished
    std::vector<StateId> queue_;  // used as topologically sorted priority queue

   public:
    inline bool QueueEmpty() { return queue_.empty(); }
    inline size_t QueueSize() { return queue_.size(); }
    inline size_t ColorSize() { return color_.size(); }
    void QueueReset() { // reset states to unvisited, but keep memory allocated
      while (!queue_.empty()) {
        color_[queue_.back()] = kWhite;
        queue_.pop_back();
      }
    }
    void AssertQueueEmpty() {
      assert(queue_.size() == 0);
      for(std::vector<int>::iterator it = color_.begin(); 
          it != color_.end(); ++it) { assert(*it == kWhite); }
    }
    //**** functions for the recursive depth first visitor
    inline void Touch(StateId state) { // mark kGray: for detecting self-loops
      if (color_.size() <= state) color_.resize(state+1+kDecoderBlock, kWhite);
      color_[state] = kGray;
    }
    inline void Brown(StateId state) { // mark kBrown: explored as emitting
      if (color_.size() <= state) color_.resize(state+1+kDecoderBlock, kWhite);
      color_[state] = kBrown;
    }
    inline void Untouch(StateId state) {
      // if we assume, that a state number is unique over the whole decoding
      // we can leave this function and cache the state color
      // which saves about 10% of total time!
      //color_[state] = kWhite;
    }
    inline void Finish(StateId state) { // assign kBlack and push to queue
      if (color_.size() <= state) color_.resize(state+1+kDecoderBlock, kWhite);
      color_[state] = kBlack;
      queue_.push_back(state);         // remember inverse mapping
    }
    inline void Enqueue(StateId state) { // put state on priority queue
      DEBUG_CMD(assert(state < color_.size()))
      DEBUG_CMD(assert(color_[state] >= kBlack))
      color_[state] = kQueue; // the state must be in queue_
    }
    inline int GetKey(StateId state) { // retrieve state color
      if (state < color_.size())
        return color_[state];
      else
        return kWhite;
    }

    // functions for implementing the priority queue
    inline Token *PopQueue() { // get next state from queue_ and remove hash
      // we start popping from the back (the last finished states)
      do {
        StateId state = queue_.back();
        queue_.pop_back();
        int color = color_[state];
        color_[state] = kWhite;  // reset all states
        if (color == kQueue) {
          Token *ans = hash_[state];
          DEBUG_CMD(assert(ans->state == state))
          hash_[state] = NULL;
          return ans;
        }
      } while(!queue_.empty());
      return NULL; // end of queue was reached
    }

    // functions for random access of members_next_ via hash
    inline size_t HashSize() { return hash_.size(); }
    inline Token *HashLookup(StateId state) {
      // retrieves token for state or creates a new one if necessary
      DEBUG_CMD(assert(hash_.size() > state)) // checked by Resize(AllocateLists)
      if (hash_[state]) {
        return hash_[state];  // return corresponding token
      } else {  // create new token for state
        Token *ans = tokens_next_->NewToken(state);
        hash_[state] = ans;
        return ans;  // ans->weight == Weight::Zero() means new token
      }
    }
    inline void HashRemove(Token *token) {
      hash_[token->state] = NULL;
    }
    inline void ResizeHash(size_t newsize) {  // memory allocation
      if (hash_.size() <= newsize) {
        DEBUG_OUT1("resize hash:" << newsize + 1 + kDecoderBlock)
        // resize to new size: newsize+1 and a bit ahead
        hash_.resize(newsize + 1 + kDecoderBlock, NULL);  // NULL: not in list
      }
    }
    inline void Defragment() {  // write pointers to store tokens consecutively
      tokens_next_->Defragment();
    }
    inline void DefragmentBack() { // write pointers to store tokens consecutively
      tokens_next_->DefragmentBack();
    }

    //inline size_t NextSize() { DEBUG_CMD(return next_size_) }
    inline size_t NextSize() { return members_next_->size(); }
    inline void NextReserve(int n_tokens) { members_next_->reserve(n_tokens); }
    inline void NextPush(Token *token) {  // push already existing state
      //token->next = members_next_;
      //members_next_ = token;
      //DEBUG_CMD(next_size_++)
      members_next_->push_back(token);  // for consecutive access
    }
    inline void NextDelete(Token *token) {
      tokens_next_->Delete(token);
    }
    
    // functions for consecutive access of members_ via internal iterator
    inline size_t Size() { return members_->size(); }
    //inline size_t Size() { DEBUG_CMD(return members_size_) }
    inline bool Empty() { return members_->size() == 0; }
    //inline bool Empty() { return members_ == NULL; }
    inline void Start() { iter_ = members_->begin(); }
    //inline void Start() { current_ = members_; }    
    inline bool Ended() { return iter_ == members_->end(); }
    //inline bool Ended() { return current_ == NULL; }    
    inline Token *GetNext() {  // consecutive access
      if (iter_ == members_->end()) { return NULL; } else { return *iter_++; }
      //Token *ans = current_;
      //current_ = current_->next;
      //return ans;
    }
    inline Token *PopNext() {  // destructive, consecutive access to members_ and remove from hash
      if (members_->empty()) return NULL;
      Token *ans = members_->back();
      members_->pop_back();
      //Token *ans = members_;
      //members_ = members_->next;
      //DEBUG_CMD(members_size_--)
      return ans;
    }
/*    inline void KillThis() {  // delete current token from members_ and tokens_
      if (iter_ != members_->begin()) iter_--;  // adjust iterator for GetNext()
      tokens_->Delete(*iter_);                 // delete old token
      *iter_ = members_->back();               // fill hole with new token
      members_->pop_back();
    }*/
    inline void ThisDelete(Token *token) {
      tokens_->Delete(token);
    }

    // general functions
    inline void Swap() {  // flip dual token lists
     std::swap(members_, members_next_);
     //DEBUG_CMD(std::swap(members_size_, next_size_))     
     std::swap(tokens_, tokens_next_);
    }
    inline void SwapMembers() {  // do not flip token stores
     std::swap(members_, members_next_);
     //DEBUG_CMD(std::swap(members_size_, next_size_))     
    }
    void Clear() {
      hash_.clear();
      //DEBUG_CMD(members_size_ = 0)
      //DEBUG_CMD(next_size_ = 0)
      members_->clear();
      members_next_->clear();
      color_.clear();
      queue_.clear();
    }
    void AssertNextEmpty() {
      //DEBUG_CMD(assert(next_size_ == 0))      
      assert(members_next_->size() == 0);
      int i = 0;
      for (typename std::vector<Token*>::iterator it = hash_.begin();
          it != hash_.end(); ++it) {
        if (*it) { DEBUG_OUT1("failed:" << i << ":" << (*it)->state)
          //*it = NULL;
        }
        assert(*it == NULL);
        i++;
      }
    }
    void Init(TokenStore *tokens, TokenStore *tokens_next) {
      tokens_ = tokens;
      tokens_next_ = tokens_next;
      members_ = new std::vector<Token*>;
      members_next_ = new std::vector<Token*>;
      //members_ = NULL;
      //members_next_ = NULL;
      //DEBUG_CMD(members_size_ = 0)
      //DEBUG_CMD(next_size_ = 0)
    }
    TokenSet() {} // it's not really nice to leave tokens_[next_] uninitialized
    ~TokenSet() {
      Clear();
      delete members_;
      delete members_next_;
    }
  };  // class TokenSet


 public:
  KaldiDecoder(KaldiDecoderOptions opts);
  ~KaldiDecoder();

  // functions to set decoder options
  void SetMaxActiveTokens(int32 max_active_tokens) {
    options_.max_active_tokens =
      max_active_tokens ? max_active_tokens : std::numeric_limits<int32>::max();
  }
  void SetBeamPruning(BaseFloat beam_width) {
    options_.beamwidth = beam_width;
    options_.beamwidth2 = beam_width;
  }
  void SetLmScale(BaseFloat lm_scale) {
    options_.lm_scale = lm_scale;
  }
  void SetWordPenalty(BaseFloat word_penalty) {
    options_.word_penalty = word_penalty;
  }

  /// return best path weight
  inline BaseFloat Likelihood() {
    if (final_token_) return final_token_->weight.Value();
    else return Weight::Zero().Value();
  }
  /// comparison of weights
  inline bool is_less(const Weight &w1, const Weight &w2) const {
    return (Plus(w1, w2) == w1) && w1 != w2;
  }

  // functions in main decoding loop
  /// performs the decoding
  fst::VectorFst<fst::StdArc>* Decode(const Fst &fst, Decodable *decodable);
  // fst: recognition network
  // decodable: acoustic model/features
  // output: linear FST of wordlinks in best path
  // must not necessary have the same arc type as the recognition network!

  /// prepares data structures for new decoding run
  void InitDecoding(const Fst &fst, Decodable *decodable);

  /// follow non-emitting arcs for states on priority queue
  void ProcessNonEmitting();

  /// evaluate acoustic model for current frame for all states on active_tokens_
  void EvaluateAndPrune();

  /// follows emitting arcs for states on active_tokens_(this), applies pruning
  void ProcessEmitting();

  /// forward to final states, backtracking, build output FST, memory clean-up
  void FinalizeDecoding();

  /**
   * @brief recursive depth first search visitor
   * @param s the FST state
   * @return outputs true, if the state has emitting links
   */
  int VisitNode(StateId s);

  /**
   * @brief recursive depth first search visitor,
   *   checks, that FST has all required properties
   * @param s the FST state
   * @return outputs true, if the state has emitting links
   */
  int VerifyNode(StateId s);

  /**
   * @brief creates new token, computes score/penalties,
   * @param token, arc
   * @return new token, if arc should be followed, otherwise NULL
   */
  inline bool PassTokenThroughArc(Token *token, const MyArc &arc);

  /// processes final states
  inline bool ReachedFinalState(Token *token);


 private:
  KaldiDecoderOptions options_;         // stores pruning beams, etc.
  Decodable *p_decodable_;         // accessor for the GMMs
  const Fst* reconet_;              // recognition network as FST
  // const fst::SymbolTable *model_list;     // input symbol table
  // const fst::SymbolTable *word_list;     // output symbol table
  fst::VectorFst<fst::StdArc>* output_arcs_;  // recogn. output as word link FST

  // arrays for token passing
  WordLinkStore wl_store_;           // data structure for allocating word links
  TokenStore *tokens_;                   // for allocating tokens
  TokenStore *tokens_next_;
  // list of active tokens for this and next frame
  TokenSet<StateId> active_tokens_;

  int frame_index_;                      // index of currently processed frame
  Token final_token_;                    // best token reaching final state
  Weight beam_threshold_;                // cut-off after evaluating PDFs
  std::vector<BaseFloat> scores_;        // used in pruning
  // it's a class member to avoid internal new/delete
};  // class KaldiDecoder

};  // namespace kaldi

#include "decoder/kaldi-decoder-inl.h"

#endif
