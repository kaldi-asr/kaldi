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

#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "lm/arpa-file-parser.h"
#include "util/common-utils.h"

namespace kaldi {

#define MAX_NGRAM 5+1

class FasterArpaLm {
 public:

  // LmState in FasterArpaLm: the basic storage unit
  class LmState {
   public:
    LmState() logprob_(0) { }
    Allocate(NGram* ngram, float lm_scale=1): 
    logprob_(ngram->logprob_*lm_scale), 
    backoff_logprob_(ngram->backoff_logprob_*lm_scale) {
      /*
      std::vector<int32> &word_ids = ngram->words;
      int32 ngram_order = word_ids.size();
      int32 sz= sizeof(int32)*(ngram_order);
      */
    }
    bool IsExist() { return logprob_!=0; };
    ~LmState() { }

    // for current query
    float logprob_;
    // for next query; can be optional
    float backoff_logprob_;
  };

  // Class to build FasterArpaLm from Arpa format language model. It relies on the
  // auxiliary class LmState above.
  class FasterArpaLmBuilder : public ArpaFileParser {
   public:
    FasterArpaLmBuilder(ArpaParseOptions &options, FasterArpaLm *lm, 
      float lm_scale = 1): 
    lm_(lm), lm_scale_(lm_scale) { ArpaFileParser(options, NULL); }
    ~FasterArpaLmBuilder() { }

   protected:
    // ArpaFileParser overrides.
    virtual void HeaderAvailable() {
      lm_->Allocate(NgramCounts(), Symbols());
    }
    virtual void ConsumeNGram(const NGram& ngram) {
      LmState *lmstate = lm_->GetHashedState(ngram.words);
      lmstate->Allocate(&ngram, lm_scale_);
    }

    virtual void ReadComplete()  { }

   private:
    FasterArpaLm *lm_;
    float lm_scale_;
  };

  FasterArpaLm(ArpaParseOptions &options, const std::string& arpa_rxfilename,
    float lm_scale=0) {
    is_built_ = false;
    ngram_order_ = 0;
    num_words_ = 0;
    lm_states_size_ = 0;
    ngrams_ = NULL;
    randint_per_word_gram_ = NULL;
    options_ = options;

    BuildFasterArpaLm(arpa_rxfilename, lm_scale);
  }

  ~FasterArpaLm() {
    if (is_built_) free();
  }

  inline LmState* GetHashedState(int32* word_ids, 
      int query_ngram_order) {
    assert(query_ngram_order > 0 && query_ngram_order <= ngram_order_);
    int32 ngram_order = query_ngram_order;
    if (ngram_order == 1) {
      return &ngrams_[ngram_order-1][word_ids[ngram_order-1]];
    } else {
      int32 hashed_idx=randint_per_word_gram_[0][word_ids[0]];
      for (int i=1; i<ngram_order; i++) {
        hashed_idx ^= randint_per_word_gram_[i][word_ids[i]];
      }
      return &ngrams_[ngram_order-1][hashed_idx & 
          (ngrams_hashed_size_[ngram_order-1] - 1)];
    }
  }
  inline LmState* GetHashedState(std::vector<int32> &word_ids, 
      int query_ngram_order = 0) {
    int32 ngram_order = query_ngram_order==0? word_ids.size(): query_ngram_order;
    int32 word_ids_arr[MAX_NGRAM];
    for (int i=0; i<query_ngram_order;i++) word_ids_arr[i]=word_ids[i];
    return GetHashedState(word_ids_arr, ngram_order)
  }

  // if exist, get logprob_, else get backoff_logprob_
  // memcpy(n_wids+1, wids, len(wids)); n_wids[0] = cur_wrd;
  inline float GetNgramLogprob(const int32 *word_ids, 
      const int32 ngram_order, 
      std::std::vector<int32>& o_word_ids) {
    float prob;
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

    LmState *lm_state = GetHashedState(word_ids, ngram_order);
    assert(lm_state);
    if (lm_state->IsExist()) {
      prob = lm_state->logprob_;
      o_word_ids.resize(ngram_order);
      for (int i=0; i<ngram_order; i++) {
        o_word_ids[i] = word_ids[i];
      }
    } else {
      LmState *lm_state_bo = GetHashedState(word_ids + 1, ngram_order-1); 
      prob = lm_state_bo->backoff_logprob_ + 
        GetNgramLogprob(word_ids, ngram_order - 1, o_word_ids);
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
                const fst::SymbolTable* symbols) {
    ngram_order_ = ngram_count.size();
    uint64 max_rand = -1;
    kaldi::RandomState rstate;
    rstate.seed = 27437;
    ngrams_ = malloc(ngram_order_ * sizeof(void*));
    randint_per_word_gram_ = malloc(ngram_order_ * sizeof(void*));
    ngrams_hashed_size_ = malloc(ngram_order_ * sizeof(int32));
    for (int i=0; i< ngram_order_; i++) {
      if (i == 0) ngrams_hashed_size_[i] = ngram_count[i]; // uni-gram
      else {
        ngrams_hashed_size_[i] = (1<<(int)ceil(log(ngram_count[i]) / 
                                 M_LN2 + 0.3));
      }
      KALDI_VLOG(2) << "ngram: "<< i <<" hashed_size/size = "<< 
        ngrams_hashed_size_[i] / ngram_count[i];
      ngrams_[i] = new LmState[ngrams_hashed_size_[i]];
      randint_per_word_gram_[i] = new int32[symbols->NumSymbols()];
      for (int j=0; j<symbols->NumSymbols(); j++) {
        randint_per_word_gram_[i][j] = kaldi::RandInt(0, max_rand, &rstate);
      }
    }
    is_built_ = true;
  }
  void free() {
    for (int i=0; i< ngram_order_; i++) {
      delete ngrams_[i];
      delete randint_per_word_gram_[i];
    }
    delete ngrams_;
    delete randint_per_word_gram_;
  }

 private:
  // configurations

  // Indicating if FasterArpaLm has been built or not.
  bool is_built_;
  // N-gram order of language model. This can be figured out from "/data/"
  // section in Arpa format language model.
  int32 ngram_order_;
  // Index of largest word-id plus one. It defines the end of <unigram_states_>
  // array.
  int32 num_words_;
  // Size of the <lm_states_> array, which will be needed by I/O.
  int64 lm_states_size_;
  // Hash table from word sequences to LmStates.
  unordered_map<std::vector<int32>,
                LmState*, VectorHasher<int32> > seq_to_state_;
  ArpaParseOptions &options;

  // data

  // Memory blcok for storing N-gram; ngrams_[ngram_order][hashed_idx]
  LmState** ngrams_;
  // used to obtain hash value; randint_per_word_gram_[ngram_order][word_id]
  uint64** randint_per_word_gram_;
  int32* ngrams_hashed_size_;
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
    lm_(lm), start_state_(0) { 
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
    std::vector<Label> wseq = state_to_wseq_[s];
    std::vector<Label> owseq;
    float logprob = GetNgramLogprob(wseq, ilabel, owseq);
    return Weight(-logprob);
  }

  float GetNgramLogprob(std::std::vector<int32> &wseq, int32 ilabel,
    std::std::vector<int32> &owseq) {
    int32 n = wseq.size();
    int32 word_ids[MAX_NGRAM];
    assert(n+1 <= MAX_NGRAM);

    word_ids[0] = ilabel;
    for (int i=n-1; i>=0; i-- ) {
      word_ids[n-i] = wseq[i];
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
