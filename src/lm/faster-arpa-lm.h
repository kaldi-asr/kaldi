// lm/const-arpa-lm.h

// Copyright 2018  Zhehuai Chen

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

#ifndef KALDI_LM_FASTER_ARPA_LM_H_
#define KALDI_LM_FASTER_ARPA_LM_H_

#include <string>
#include <vector>
#include <math.h>

#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "lm/arpa-file-parser.h"
#include "util/common-utils.h"

namespace kaldi {

#define MAX_NGRAM 5+1
#define RAND_TYPE int32
class FasterArpaLm {
 public:

  // LmState in FasterArpaLm: the basic storage unit
  class LmState {
   public:
    LmState(): logprob_(0), h_value(0), next(NULL) { }
    LmState(float logprob, float backoff_logprob): 
      logprob_(logprob), backoff_logprob_(backoff_logprob), h_value(0),
    next(NULL) { }
    void Allocate(const NGram* ngram, float lm_scale=1) {
      logprob_ = ngram->logprob*lm_scale;
      backoff_logprob_ = ngram->backoff*lm_scale;
      /*
      std::vector<int32> &word_ids = ngram->words;
      int32 ngram_order = word_ids.size();
      int32 sz= sizeof(int32)*(ngram_order);
      */
    }
    bool IsExist() const { return logprob_!=0; }
    ~LmState() { }

    // for current query
    float logprob_;
    // for next query; can be optional
    float backoff_logprob_;
    int32 h_value;
    LmState* next; // for colid
  };

  // Class to build FasterArpaLm from Arpa format language model. It relies on the
  // auxiliary class LmState above.
  class FasterArpaLmBuilder : public ArpaFileParser {
   public:
    FasterArpaLmBuilder(ArpaParseOptions &options, FasterArpaLm *lm, 
      float lm_scale = 1): ArpaFileParser(options, NULL),
    lm_(lm), lm_scale_(lm_scale) { }
    ~FasterArpaLmBuilder() { }

   protected:
    // ArpaFileParser overrides.
    virtual void HeaderAvailable() {
      lm_->Allocate(NgramCounts(), 
          Options().bos_symbol, Options().eos_symbol, Options().unk_symbol);
    }
    virtual void ConsumeNGram(const NGram& ngram) {
      LmState lm_state(ngram.logprob * lm_scale_, ngram.backoff * lm_scale_);
      lm_->SaveHashedState(ngram.words, lm_state, true);
    }

    virtual void ReadComplete()  { }

   private:
    FasterArpaLm *lm_;
    float lm_scale_;
  };

  FasterArpaLm(ArpaParseOptions &options, const std::string& arpa_rxfilename,
    int32 symbol_size, float lm_scale=1): symbol_size_(symbol_size), options_(options) {
    assert(symbol_size_);
    is_built_ = false;
    ngram_order_ = 0;
    num_words_ = 0;
    lm_states_size_ = 0;
    randint_per_word_gram_ = NULL;
    max_collision_ = 0;

    BuildFasterArpaLm(arpa_rxfilename, lm_scale);
    KALDI_VLOG(2) << max_collision_;
  }

  ~FasterArpaLm() {
    if (is_built_) Free();
  }

  int32 BosSymbol() const { return bos_symbol_; }
  int32 EosSymbol() const { return eos_symbol_; }
  int32 UnkSymbol() const { return unk_symbol_; }
  int32 NgramOrder() const { return ngram_order_; }

  inline int32 GetHashedIdx(const int32* word_ids, 
      int query_ngram_order, int32 *h_value=NULL) const {
    assert(query_ngram_order > 0 && query_ngram_order <= ngram_order_);
    int32 ngram_order = query_ngram_order;
    int32 hashed_idx;
    if (ngram_order == 1) {
      hashed_idx = word_ids[ngram_order-1];
    } else {
      hashed_idx=randint_per_word_gram_[0][word_ids[0]];
      for (int i=1; i<ngram_order_; i++) {
        int word_id=i<ngram_order?word_ids[i]:
          (word_ids[i-ngram_order] + i + ngram_order); // this is totally a hack
        hashed_idx ^= randint_per_word_gram_[i][word_id];
      }
      if (h_value) *h_value = hashed_idx; // to check colid
      hashed_idx &= 
          (hash_size_except_uni_ - 1);
    }
    return hashed_idx;
  }
  inline void InsertHash(int32 hashed_idx, int32 ngrams_saved_num_) {
    if (ngrams_map_.at(hashed_idx)) {
      LmState *lm_state = ngrams_map_[hashed_idx];
      int32 cnt=0;
      while (lm_state->next) {
        lm_state = lm_state->next;
        cnt++;
      }
      lm_state->next = &ngrams_[ngrams_saved_num_];
      max_collision_=std::max(cnt,max_collision_);
    } else {
      ngrams_map_[hashed_idx] = &ngrams_[ngrams_saved_num_];
    }
  }
  inline void SaveHashedState(const int32* word_ids, 
      int query_ngram_order, LmState &lm_state_pattern) {
    int32 h_value=0;
    int32 hashed_idx = GetHashedIdx(word_ids, query_ngram_order, &h_value);
    lm_state_pattern.h_value = h_value;
    int32 ngram_order = query_ngram_order;
    if (ngram_order == 1) {
      ngrams_[hashed_idx] = lm_state_pattern;
    } else {
      ngrams_[ngrams_saved_num_] = lm_state_pattern;
      InsertHash(hashed_idx, ngrams_saved_num_++);
    }
  }
  inline void SaveHashedState(const std::vector<int32> &word_ids, LmState &lm_state_pattern,
       bool reverse = false, int query_ngram_order = 0)  {
    int32 ngram_order = query_ngram_order==0? word_ids.size(): query_ngram_order;
    int32 word_ids_arr[MAX_NGRAM];
    if (reverse)
      for (int i=0; i<ngram_order;i++) word_ids_arr[ngram_order - i - 1]=word_ids[i];
    else
      for (int i=0; i<ngram_order;i++) word_ids_arr[i]=word_ids[i];

    return SaveHashedState(word_ids_arr, ngram_order, lm_state_pattern);
  }


  inline const LmState* GetHashedState(const int32* word_ids, 
      int query_ngram_order) const {
    int32 h_value;
    int32 hashed_idx = GetHashedIdx(word_ids, query_ngram_order, &h_value);
    int32 ngram_order = query_ngram_order;
    if (ngram_order == 1) {
      return &ngrams_[hashed_idx];
    } else {
      LmState *lm_state = ngrams_map_[hashed_idx];
      while (lm_state) {
        if (lm_state->h_value == h_value) {
          return lm_state;
        }
        lm_state = lm_state->next;
      }
    }
   
    // not found, can be bug or really not found the corresponding ngram 
    return NULL;
  }
  inline const LmState* GetHashedState(const std::vector<int32> &word_ids, 
       bool reverse = false, int query_ngram_order = 0) const {
    int32 ngram_order = query_ngram_order==0? word_ids.size(): query_ngram_order;
    int32 word_ids_arr[MAX_NGRAM];
    if (reverse)
      for (int i=0; i<ngram_order;i++) word_ids_arr[ngram_order - i - 1]=word_ids[i];
    else
      for (int i=0; i<ngram_order;i++) word_ids_arr[i]=word_ids[i];
    return GetHashedState(word_ids_arr, ngram_order);
  }

  // if exist, get logprob_, else get backoff_logprob_
  // memcpy(n_wids+1, wids, len(wids)); n_wids[0] = cur_wrd;
  inline float GetNgramLogprob(const int32 *word_ids, 
      const int32 word_ngram_order, 
      std::vector<int32>& o_word_ids) const {
    float prob;
    int32 ngram_order = word_ngram_order;
    assert(ngram_order > 0);
    if (ngram_order > ngram_order_) {
      //while (wseq.size() >= lm_.NgramOrder()) {
      // History state has at most lm_.NgramOrder() -1 words in the state.
      // wseq.erase(wseq.begin(), wseq.begin() + 1);
      //}
      // we don't need to do above things as we do in reverse fashion:
      //  memcpy(n_wids+1, wids, len(wids)); n_wids[0] = cur_wrd;
      ngram_order = ngram_order_;
    }

    const LmState *lm_state = GetHashedState(word_ids, ngram_order);
    if (lm_state) { //found out
      assert(lm_state->IsExist());
      //assert(ngram_order==1 || GetHashedState(word_ids, ngram_order-1)->IsExist());
      prob = lm_state->logprob_;
      /*
      for (int i=0; i<ngram_order; i++) {
        std::cout<<word_ids[i]<<" ";
      }
      std::cout<<ngram_order<<" "<<prob<<"\n";
      */
      // below code is to make sure the LmState exist, so un-exist states can be recombined to a same state
      ngram_order = std::min(ngram_order,ngram_order_-1);
      while(!GetHashedState(word_ids, ngram_order)) ngram_order--;
      assert(ngram_order>0);

      o_word_ids.resize(ngram_order);
      for (int i=0; i<ngram_order; i++) {
        o_word_ids[i] = word_ids[i]; 
      }

    } else {
      assert(ngram_order > 1); // thus we can do backoff
      const LmState *lm_state_bo = GetHashedState(word_ids + 1, ngram_order-1); 

      //assert(lm_state_bo && lm_state_bo->IsExist()); // TODO: assert will fail because some place has false-exist? 84746 4447 8537 without 4447 8537 in LM

      prob = lm_state_bo? lm_state_bo->backoff_logprob_:0;
      prob += GetNgramLogprob(word_ids, ngram_order - 1, o_word_ids);
    }
    return prob;
  }

  bool BuildFasterArpaLm(const std::string& arpa_rxfilename, float lm_scale) {
    FasterArpaLmBuilder lm_builder(options_, this, lm_scale);
    KALDI_VLOG(1) << "Reading " << arpa_rxfilename;
    Input ki(arpa_rxfilename);
    lm_builder.Read(ki.Stream());
    return true;
  }

 private:
  void Allocate(const std::vector<int32>& ngram_count, 
                int32 bos_symbol, int32 eos_symbol, 
                int32 unk_symbol) {
    bos_symbol_ = bos_symbol;
    eos_symbol_ = eos_symbol;
    unk_symbol_ = unk_symbol;
    ngram_order_ = ngram_count.size();
    RAND_TYPE max_rand = RAND_MAX;
    kaldi::RandomState rstate;
    rstate.seed = 27437;
    randint_per_word_gram_ = (RAND_TYPE **)malloc(ngram_order_ * sizeof(void*));
    ngrams_hashed_size_ = (int32*)malloc(ngram_order_ * sizeof(int32));
    int32 acc=0;
    for (int i=0; i< ngram_order_; i++) {
      if (i == 0) ngrams_hashed_size_[i] = symbol_size_; // uni-gram
      else {
        ngrams_hashed_size_[i] = ngram_count[i];
      }
      randint_per_word_gram_[i] = (RAND_TYPE* )malloc(symbol_size_ * sizeof(RAND_TYPE)) ;
      for (int j=0; j<symbol_size_; j++) {
        randint_per_word_gram_[i][j] = kaldi::RandInt(0, max_rand, &rstate);
      }
      acc+= ngrams_hashed_size_[i];
    }
    hash_size_except_uni_ = acc - symbol_size_;
    hash_size_except_uni_  = (1<<(int)ceil(log(hash_size_except_uni_) / 
                                 M_LN2 + 0.5));
    KALDI_VLOG(2) << " hashed_size/size = "<< 
        1.0 * (hash_size_except_uni_+symbol_size_) / acc <<" "<<acc;
    
    ngrams_ = (LmState* )calloc(sizeof(LmState), acc); //use default constructor
    ngrams_saved_num_ = symbol_size_; // assume uni-gram is allocated
    ngrams_map_.resize(hash_size_except_uni_, NULL);
    is_built_ = true;
  }
  void Free() {
    for (int i=0; i< ngram_order_; i++) {
      free(randint_per_word_gram_[i]);
    }
    free(randint_per_word_gram_);
    free(ngrams_);
  }

 private:
  // configurations

  // Indicating if FasterArpaLm has been built or not.
  bool is_built_;
  // Integer corresponds to <s>.
  int32 bos_symbol_;
  // Integer corresponds to </s>.
  int32 eos_symbol_;
  // Integer corresponds to unknown-word. -1 if no unknown-word symbol is
  // provided.
  int32 unk_symbol_;  
  // N-gram order of language model. This can be figured out from "/data/"
  // section in Arpa format language model.
  int32 ngram_order_;
  int32 symbol_size_;
  // Index of largest word-id plus one. It defines the end of <unigram_states_>
  // array.
  int32 num_words_;
  // Size of the <lm_states_> array, which will be needed by I/O.
  int64 lm_states_size_;
  // Hash table from word sequences to LmStates.
  unordered_map<std::vector<int32>,
                LmState*, VectorHasher<int32> > seq_to_state_;
  ArpaParseOptions &options_;

  // data

  // Memory blcok for storing N-gram; ngrams_[ngram_order][hashed_idx]
  LmState* ngrams_;
  int32 ngrams_saved_num_;

  std::vector<LmState *> ngrams_map_; // hash to ngrams_ index
  // used to obtain hash value; randint_per_word_gram_[ngram_order][word_id]
  RAND_TYPE** randint_per_word_gram_;
  int32* ngrams_hashed_size_;
  int32 hash_size_except_uni_;
  int32 max_collision_;
};


/**
 This class wraps a FasterArpaLm format language model with the interface defined
 in DeterministicOnDemandFst.
 */
class FasterArpaLmDeterministicFst
  : public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;
  typedef FasterArpaLm::LmState LmState;

  explicit FasterArpaLmDeterministicFst(const FasterArpaLm& lm): 
    start_state_(0), lm_(lm) { 
    // Creates a history state for <s>.
    std::vector<Label> bos_state(1, lm_.BosSymbol());
    state_to_wseq_.push_back(bos_state);
    wseq_to_state_[bos_state] = 0;
  }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return start_state_; }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual Weight Final(StateId s) {
    // At this point, we should have created the state.
    KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());
    const std::vector<Label>& wseq = state_to_wseq_[s];
    std::vector<Label> owseq;
    float logprob = GetNgramLogprob(wseq, lm_.EosSymbol(), owseq);
    return Weight(-logprob);
  }

  float GetNgramLogprob(const std::vector<int32> &wseq, int32 ilabel,
    std::vector<int32> &owseq) {
    int32 n = wseq.size();
    int32 word_ids[MAX_NGRAM];
    assert(n+1 <= MAX_NGRAM);

    word_ids[0] = ilabel;
    for (int i=0; i<n; i++ ) {
      word_ids[i+1] = wseq[i];
    }

    return lm_.GetNgramLogprob(word_ids, n+1, owseq);
  }
  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc) {
    // At this point, we should have created the state.
    KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

    std::vector<Label> wseq = state_to_wseq_[s];
    std::vector<Label> owseq;
    float logprob = GetNgramLogprob(wseq, ilabel, owseq);
    if (logprob == std::numeric_limits<float>::min()) {
      return false;
    }

    std::pair<const std::vector<Label>, StateId> wseq_state_pair(
        owseq, static_cast<Label>(state_to_wseq_.size()));

    // Attemps to insert the current <wseq_state_pair>. If the pair already exists
    // then it returns false.
    typedef MapType::iterator IterType;
    std::pair<IterType, bool> result = wseq_to_state_.insert(wseq_state_pair);

    // If the pair was just inserted, then also add it to <state_to_wseq_>.
    if (result.second == true)
      state_to_wseq_.push_back(owseq);

    // Creates the arc.
    oarc->ilabel = ilabel;
    oarc->olabel = ilabel;
    oarc->nextstate = result.first->second;
    oarc->weight = Weight(-logprob);

    return true;
  }

 private:
  typedef unordered_map<std::vector<Label>,
                        StateId, VectorHasher<Label> > MapType;
  StateId start_state_;
  MapType wseq_to_state_;
  std::vector<std::vector<Label> > state_to_wseq_;

  const FasterArpaLm& lm_;
};


}  // namespace kaldi

#endif  // KALDI_LM_CONST_ARPA_LM_H_
