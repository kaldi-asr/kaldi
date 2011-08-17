// decoder/kaldi-decoder-left.h

// Copyright 2009-2011  Mirko Hannemann;  Lukas Burget

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
#include <vector>
#include <set>

#include <fst/fstlib.h>
#include <fst/fst-decl.h>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "util/parse-options.h"
#include "hmm/transition-model.h"

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

DEBUG_CMD(int lnumber = 0)
DEBUG_CMD(int dlnumber = 0)
DEBUG_CMD(int tnumber = 0)
DEBUG_CMD(int dtnumber = 0)

struct KaldiDecoderOptions {
  BaseFloat lm_scale;
  BaseFloat word_penalty;
  int32     max_active_tokens;
  BaseFloat beamwidth;
  BaseFloat beamwidth2;

  KaldiDecoderOptions(): lm_scale(1.0), word_penalty(0.0),
                         max_active_tokens(10000),
                         beamwidth(16.0), beamwidth2(16.0) { }
  void Register(ParseOptions *po, bool full) {  // "full":use obscure options too
	// depends on program
	po->Register("beam", &beamwidth, "Decoder beam");
	po->Register("max-active", &max_active_tokens, "Decoder max active tokens.");
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

  // LinkStore is a store of input arcs (lattice links) with its own allocator
  class LinkStore {
  // Link and Token are very similar - just some fields interpreted differently
  // a Link stores the decoding history (incoming arcs)
  // a Token stores information for destination nodes
  // the pointer *next has several functions:
  // a) in linked lists to store free allocated tokens/links
  // b) as linked list of incoming arcs
  // c) as lattice backward pointer

   public:
    class Token;
    
    struct Link {
      //int state;       // source state number or time information
      Token *source;   // token in source state
      Label olabel;    // (word) output label
      Weight weight;   // arc weight or accumulated path score
      Link *next;      // linked list of arcs or lattice backward pointer
      int refs;        // reference counter (for memory management)
      DEBUG_CMD(int unique) // identifier for debugging
    };

    class Token {
     public:
      StateId state;  // state number or time information
      Label ilabel;   // input label (model) or node color
      Weight weight;  // accumulated path score
      Link *arcs;     // multifunctional: linked list of incoming arcs, 
        // lattice backward pointer and linked list of free arcs
      //int refs;     // reference counter only for Link, not for Token
      DEBUG_CMD(int unique)

      inline void Init(StateId state_) {
        state = state_;
        ilabel = fst::kNoLabel;
        weight = Weight::Zero();
        arcs = NULL;
        DEBUG_CMD(unique = tnumber++)
        DEBUG_OUT3("create token:" << unique)
      }
      inline void AddInputArc(Token *source, const MyArc &arc,
          LinkStore *link_store) { // link a new arc to the incoming arc list
        DEBUG_OUT3("add arc:" << source->state << "->" << arc.nextstate)
        Link *tmp = link_store->FastNewLink();
        tmp->source = source;
        tmp->olabel = arc.olabel;
        tmp->weight = arc.weight;
        tmp->next = arcs;
        arcs = tmp;
      }
      inline void UpdateLink(Link *source, LinkStore *link_store) {
        // stores a copy of the back-track pointer from the source token
        // if our token has an old lattice link, remove it:
        if (arcs != NULL) link_store->SlowDelete(arcs);
        arcs = source;  // copy link from source (must be "slow" link!!)
        arcs->refs++;   // remember the new usage in ref counter
        DEBUG_OUT3("copy:"<< arcs->unique << " (" << arcs->refs << "x)")
      }
    };  // class Token

    // functions for fast creation/deletion of temporary links
    inline void FastDelete(Link *link) {  // delete arc: put to linked list
      DEBUG_CMD(assert(link->state != fst::kNoStateId))
      DEBUG_CMD(link->state = fst::kNoStateId)
      // save the unused arc using *next as linked list
      link->next = fast_link_head_;
      fast_link_head_ = link;
    }
    inline Link *FastNewLink() {  // create new link
      // allocate: either take from linked list or extend list
      if (fast_link_head_) {  // take from free list
        Link *tmp = fast_link_head_;
        fast_link_head_ = fast_link_head_->next;
        return tmp;
      } else {  // make new entries in free list
        Link *tmp = new Link[allocate_block_size_];
        for (size_t i = 0; i + 1 < allocate_block_size_; i++) {
          tmp[i].next = tmp + i + 1;
        }
        tmp[allocate_block_size_ - 1].next = NULL;
        fast_alloc_.push_back(tmp);
        fast_link_head_ = tmp->next;
        return tmp;
      }
    }
    void FastDefragment() {  // write all links consecutively in order
      if (fast_alloc_.size() < 1) return;
      fast_link_head_ = fast_alloc_[0];
      for (size_t j = 0; j < fast_alloc_.size(); j++) {
        Link *tmp = fast_alloc_[j];
        for (size_t i = 0; i + 1 < allocate_block_size_; i++) {
          tmp[i].next = tmp + i + 1;
        }
        if (j == fast_alloc_.size() - 1) {
          tmp[allocate_block_size_ - 1].next = NULL;
        } else {
          tmp[allocate_block_size_ - 1].next = fast_alloc_[j+1];
        }
      }
    }
    // functions for creation/deletion of permanent links
    inline void SlowDelete(Link *link) {
      // delete link: either decrease reference count or put to linked list
      DEBUG_CMD(assert(link->refs > 0))
      link->refs--;
      DEBUG_OUT3("dec:" << link->unique << " (" << link->refs << "x)")
      if (link->refs > 0) return;
      // really kill link
      DEBUG_OUT3( "kill" )
      // clean up unused links recursively backwards
      if (link->next != NULL) SlowDelete(link->next);
      DEBUG_CMD(dlnumber++)
      // save the unused link in linked list (abusing *next)
      link->next = slow_link_head_;
      slow_link_head_ = link;
    }
    inline Link *SlowNewLink() {
      // new link: either take from linked list or extend list
      DEBUG_OUT3( "create link:" << lnumber )
      if (slow_link_head_) {  // take from free list
        Link *tmp = slow_link_head_;
        slow_link_head_ = slow_link_head_->next;
        return tmp;
      } else {  // make new entries in free list
        Link *tmp = new Link[allocate_block_size_];
        for (size_t i = 0; i + 1 < allocate_block_size_; i++) {
          tmp[i].next = tmp + i + 1;
        }
        tmp[allocate_block_size_ - 1].next = NULL;
        slow_alloc_.push_back(tmp);
        slow_link_head_ = tmp->next;
        return tmp;
      }
    }
    inline void LinkToken(Token *token, int frame, Label label) {
      // creates Link for word labels and fills data fields
      // create link only at beginning (next==NULL) or at word arcs (olabel>0)
      if ((token->arcs != NULL) && (token->arcs->olabel <= 0)) return;
      if (token->arcs != NULL) { DEBUG_OUT3(" from:"
        << token->arcs->unique << " (" << token->arcs->refs << "x)") }
      Link *ans = SlowNewLink();
      // initialize all data fields
      //?? ans->state = frame;
      ans->source = NULL;
      ans->olabel = label;
      ans->weight = token->weight;
      ans->next = token->arcs;  // lattice backward pointer
      ans->refs = 1;  // can it be zero if not yet assigned?
      DEBUG_CMD(ans->unique = lnumber++)
      token->arcs = ans;  // store new link in token
    }

    void Clear() {
      for (size_t i = 0; i < fast_alloc_.size(); i++) delete[] fast_alloc_[i];
      fast_alloc_.clear();
      fast_link_head_ = NULL;
      DEBUG_OUT1("links created: " << lnumber << " deleted: " << dlnumber)
      DEBUG_CMD(assert(lnumber == dlnumber)) // check that all links are freed
      for (size_t i = 0; i < slow_alloc_.size(); i++) delete[] slow_alloc_[i];
      slow_alloc_.clear();
      slow_link_head_ = NULL;
    }
    LinkStore() {
      fast_link_head_ = NULL;
      slow_link_head_ = NULL;
    }
    ~LinkStore() {
      Clear();
    }
   private:
    // head of list of currently free tokens ready for allocation
    Link *fast_link_head_;
    std::vector<Link*> fast_alloc_;  // list of allocated links (temporary)
    Link *slow_link_head_;
    std::vector<Link*> slow_alloc_;  // list of allocated links (permanent)
    // number of arcs to allocate in one block
    static const size_t allocate_block_size_ = 16384;
  };  // class LinkStore
  typedef typename LinkStore::Link Link;
  typedef typename LinkStore::Token Token;


  // TokenStore is a store of tokens with its own allocator
  class TokenStore {
   public:
    inline void Delete(Token *token) {
      // delete token: put to linked list
      DEBUG_OUT3( "kill token:" << token->unique << ":" << token->state)
      DEBUG_CMD(assert(token->state != fst::kNoStateId))
      DEBUG_CMD(token->state = fst::kNoStateId)
      DEBUG_CMD(dtnumber++)
      token->state = fst::kNoStateId; // to indicate deleted tokens
      // clean up unused links recursively backwards
      if (token->arcs != NULL) link_store_->SlowDelete(token->arcs);
      // save the unused token (abusing *arcs as linked list)
      token->arcs = reinterpret_cast<Link*>(free_tok_head_);
      free_tok_head_ = token;
    }
    inline Token *NewToken(int state) { // create and initialize new token
      // allocate: either take from linked list or extend list
      Token *tmp;
      if (free_tok_head_) {  // take from free list
        tmp = free_tok_head_;
        free_tok_head_ = reinterpret_cast<Token*>(free_tok_head_->arcs);
      } else {  // make new entries in free list
        tmp = new Token[allocate_block_size_];
        for (size_t i = 0; i + 1 < allocate_block_size_; i++) {
          tmp[i].arcs = reinterpret_cast<Link*>(tmp + i + 1);
        }
        tmp[allocate_block_size_ - 1].arcs = NULL;
        allocated_.push_back(tmp);
        free_tok_head_ = reinterpret_cast<Token*>(tmp->arcs);
      }
      tmp->Init(state);
      return tmp;
    }
    void Defragment() {  // write all links consecutively in order
      if (allocated_.size() < 1) return;
      free_tok_head_ = allocated_[0];
      for (size_t j = 0; j < allocated_.size(); j++) {
        Token *tmp = allocated_[j];
        for (size_t i = 0; i + 1 < allocate_block_size_; i++) {
          tmp[i].arcs = reinterpret_cast<Link*>(tmp + i + 1);
        }
        if (j == allocated_.size() - 1) {
          tmp[allocate_block_size_ - 1].arcs = NULL;
        } else {
          tmp[allocate_block_size_ - 1].arcs =
            reinterpret_cast<Link*>(allocated_[j+1]);
        }
      }
    }
    void DefragmentBack() { // write all links consecutively in order
      if (allocated_.size() < 1) return;
      free_tok_head_ = allocated_[allocated_.size()-1] + allocate_block_size_-1;
      for(size_t j = 0; j < allocated_.size(); j++) {
        Token *tmp = allocated_[j];
        for(size_t i = 1; i < allocate_block_size_; i++) {
          tmp[i].arcs = reinterpret_cast<Link*>(tmp + i - 1);
        }
        if (j == 0) {
          tmp[0].arcs = NULL;
        } else {
          tmp[0].arcs = 
            reinterpret_cast<Link*>(allocated_[j-1]+allocate_block_size_-1);
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
    TokenStore(LinkStore *link_store) {
      free_tok_head_ = NULL;
      link_store_ = link_store;
    }
    ~TokenStore() {
      Clear();
    }
   private:
    // head of list of currently free tokens ready for allocation
    Token *free_tok_head_;
    std::vector<Token*> allocated_;  // list of allocated tokens
    // number of tokens to allocate in one block
    static const size_t allocate_block_size_ = 8192;
    LinkStore *link_store_;
  };  // class TokenStore

  // TokenSet is an unordered set of Tokens
  // with random access and consecutive member access
  // used to access the active tokens in decoding

  // StateOrder contains topological state coloring for recursive visit
  // used to determine the order of evaluating (non-emitting) arcs
  // and the topologically sorted priority queue of (non-emitting) arcs
  template<class StateId>
  class TokenSet { //TokenSet and StateOrder
   private:
    std::vector<Token*> hash_;  // big, sparse vector for random access
    // initially NULL, later stores the pointer to token (also in members_)
    // the hash is only needed for members_next_
    std::vector<StateId> queue_;  // used as topologically sorted priority queue
    // always two tokens form one entry: source and destination
    TokenStore *tokens_, *tokens_next_;  // to allocate new tokens
    std::vector<Token*> *members_;  // small, dense vector for consecutive access
    std::vector<Token*> *members_next_;  // the same for next frame
    typename std::vector<Token*>::iterator iter_;  // internal iterator for members
    //Token *members_, *members_next_; // pointers for consecutive access
    //DEBUG_CMD(size_t members_size_)
    //DEBUG_CMD(size_t next_size_)
    //Token *current_; // internal iterator for members

   public:
    // functions for implementing the priority queue
    inline bool QueueEmpty() { return queue_.empty(); }
    inline size_t QueueSize() { return queue_.size(); }
    void QueueReset() { // reset states to unvisited, but keep memory allocated
      while (!queue_.empty()) {
        DEBUG_CMD(assert(queue_.back() != fst::kNoStateId))
        //?? queue_.back()->ilabel = fst::kNoStateId;
        queue_.pop_back();
      }
    }
    void AssertQueueEmpty() {
      assert(queue_.size() == 0);
      for(typename std::vector<Token*>::iterator it = hash_.begin(); 
          it != hash_.end(); ++it) {
        if (*it) assert((*it)->ilabel == fst::kNoStateId);
      }
    }
    inline void Untouch(StateId state) {
      // if we assume, that a state number is unique over the whole decoding
      // we can leave this function and cache the state color
      // which saves about 10% of total time!
      hash_[state]->ilabel = fst::kNoStateId;
    }
    inline void PushQueue(Token *token) { // mark as explored
      DEBUG_OUT2("push queue:" << token->state << ":" << token->ilabel)
      DEBUG_CMD(assert(token->ilabel < 0)) //emitting (>0) not allowed
      token->ilabel = 0; // non-emitting and finished
      queue_.push_back(token->state);
    }
    inline StateId PopQueue() { // get next destination node from queue_
      StateId state = queue_.back();
      queue_.pop_back(); // start popping from last finished states
      return state;
    }

    // functions for random access of members_next_ via hash
    inline size_t HashSize() { return hash_.size(); }
    // touched non-emitting: ilabel == fst::kNoLabel (-1)
    // finished non-emitting: ilabel == 0
    // finished emitting: ilabel > 0
    // queued: ilabel < 0
    inline Token *HashCheck(StateId state, Label ilabel, Decodable *decodable) {
      // retrieves token for state or creates a new one if necessary
      if (hash_.size() <= state) {
        DEBUG_OUT1("resize hash:" << state + 1 + kDecoderBlock)
        hash_.resize(state + 1 + kDecoderBlock, NULL);  // NULL: not in list
      }
      //DEBUG_CMD(assert(hash_.size() > state)) // checked by ResizeHash
      if (hash_[state] != NULL) {
        DEBUG_OUT2("hashed:"<<hash_[state]->state<<":"<<hash_[state]->ilabel)
        DEBUG_CMD(assert(hash_[state]->state == state))
        // check that all incoming arcs have the same model!
        DEBUG_CMD(if (ilabel > 0)
          assert(decodable->TransModel()->TransitionIdToPdf(hash_[state]->ilabel)
              == decodable->TransModel()->TransitionIdToPdf(ilabel)))
        //DEBUG_CMD(if (ilabel > 0) assert(hash_[state]->ilabel == ilabel))
        DEBUG_CMD(if (ilabel <= 0) assert(hash_[state]->ilabel == 0))
        // this also checks that transducer doesn't contain epsilon loop!
        // it implies that: assert(hash_[state]->ilabel != fst::kNoLabel)
        return hash_[state];  // return corresponding token
      } else {  // create new token for state
        Token *ans;
        if (ilabel > 0) {  // next state emitting
          ans = tokens_next_->NewToken(state);
          ans->ilabel = ilabel; // code: finished emitting
          DEBUG_OUT2("state " << state << " to hash (next)")
          //PushFront(ans); // put to queue, to handle all states on one queue
          NextPush(ans); // to be fetched in ProcessTokens
        } else {  // non-emitting state
          ans = tokens_->NewToken(state); //?? tokens_ or tokens_next_
          DEBUG_OUT2("state " << state << " to hash (this)")
        }
        hash_[state] = ans;
        return ans;  // ans->weight == Weight::Zero() means new token
      }
    }
    inline Token *HashLookup(StateId state) {  // retrieves token for state
      DEBUG_CMD(assert(hash_.size() > state)) // checked by ResizeHash
      DEBUG_CMD(assert(hash_[state] != NULL))
      return hash_[state];  // return corresponding token
    }
    inline void HashRemove(Token *token) {
      DEBUG_CMD(assert(hash_.size() > token->state))
      DEBUG_CMD(assert(hash_[token->state] != NULL))
      DEBUG_CMD(assert(hash_[token->state]->state == token->state))
      hash_[token->state] = NULL;
    }
    inline void HashRemove(StateId state) {
      DEBUG_CMD(assert(hash_.size() > state))
      DEBUG_CMD(assert(hash_[state] != NULL))
      DEBUG_CMD(assert(hash_[state]->state == state))
      hash_[state] = NULL;
    }
    inline void ResizeHash(size_t newsize) {  // memory allocation
      if (hash_.size() <= newsize) {
        DEBUG_OUT1("resize hash:" << newsize + 1 + kDecoderBlock)
        // resize to new size: newsize+1 and a bit ahead
        hash_.resize(newsize + 1 + kDecoderBlock, NULL);  // NULL: not in list
      }
    }
    void AssertHashEmpty() {
      DEBUG_OUT2("hash empty?")
      int i = 0;
      for (typename std::vector<Token*>::iterator it = hash_.begin();
          it != hash_.end(); ++it) {
        if (*it) { DEBUG_OUT1("failed:" << i << ":" << (*it)->state)
          //*it = NULL;
        }
        //assert(*it == NULL);
        i++;
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
    inline void NextDelete(Token *token) {
      tokens_next_->Delete(token);
    }
    inline void NextPush(Token *token) {  // push already existing state
      members_next_->push_back(token);  // for consecutive access
    }
    inline void ThisPush(Token *token) {  // push already existing state
      //token->next = members_;
      //members_ = token;
      //DEBUG_CMD(members_size_++)
      members_->push_back(token);  // for consecutive access
    }
    
    // functions for consecutive access of members_ via internal iterator
    inline size_t Size() { return members_->size(); }
    //inline size_t Size() { DEBUG_CMD(return members_size_) }
    inline bool NextEmpty() { return members_next_->size() == 0; }
    inline bool Empty() { return members_->size() == 0; }
    //inline bool Empty() { return members_ == NULL; }
    inline void Start() { iter_ = members_->begin(); }
    //inline void Start() { current_ = members_; }    
    inline void NextStart() { iter_ = members_next_->begin(); }
    inline bool Ended() { return iter_ == members_->end(); }
    //inline bool Ended() { return current_ == NULL; }    
    inline bool NextEnded() { return iter_ == members_next_->end(); }
    inline Token *GetNext() {  // consecutive access
      if (iter_ == members_->end()) { return NULL; } else { return *iter_++; }
      //Token *ans = current_;
      //current_ = current_->next;
      //return ans;
    }
    inline Token *NextGetNext() {  // consecutive access
      if (iter_ == members_next_->end()) {
        return NULL;
      } else {
        return *iter_++;
      }
    }
    inline Token *Get(size_t s) { return (*members_)[s]; }
    inline Token *PopNext() {  // destructive, consecutive access to members_ and remove from hash
      if (members_->empty()) return NULL;
      Token *ans = members_->back();
      members_->pop_back();
      //Token *ans = members_;
      //members_ = members_->next;
      //DEBUG_CMD(members_size_--)
      return ans;
    }
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
      queue_.clear();
      //DEBUG_CMD(members_size_ = 0)
      //DEBUG_CMD(next_size_ = 0)
      members_->clear();
      members_next_->clear();
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
  // output: linear FST of lattice links in best path
  // must not necessary have the same arc type as the recognition network!

  /// prepares data structures for new decoding run
  void InitDecoding(const Fst &fst, Decodable *decodable);

  /// process active states list and find topological arc order
  void ProcessToDoList();

  /// processes incoming arcs to a state in viterbi style
  inline void ProcessState(Token *dest, bool final_frame);

  /// follow non-emitting and emitting arcs for tokens on queue and members
  void ProcessTokens(bool final_frame);

  /// evaluate acoustic model for current frame for all states on active_tokens_
  void EvaluateAndPrune();

  /// forward to final states, backtracking, build output FST, memory clean-up
  void FinalizeDecoding();

  /**
   * @brief recursive depth first search visitor
   *   checks, that FST has all required properties
   * @param s the FST state
   */
  void VisitNode(Token *token);
  void VisitNode2(Token *token);

  /**
   * @brief creates new token, computes score/penalties, update word links
   * @param token, arc
   * @return new token, if arc should be followed, otherwise NULL
   */
  inline bool PassTokenThroughArc(Token *source, Token *dest);

  /// processes final states
  inline bool ReachedFinalState(Token *token);


 private:
  KaldiDecoderOptions options_;         // stores pruning beams, etc.
  Decodable *p_decodable_;              // accessor for the GMMs
  const Fst* reconet_;                  // recognition network as FST
  // const fst::SymbolTable *model_list;     // input symbol table
  // const fst::SymbolTable *word_list;     // output symbol table
  fst::VectorFst<fst::StdArc>* output_arcs_;  // recogn. output as word link FST

  // arrays for token passing
  LinkStore link_store_;          // data structure for allocating lattice links
  TokenStore *tokens_;            // for allocating tokens
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

#include "decoder/kaldi-decoder-left-inl.h"

#endif
