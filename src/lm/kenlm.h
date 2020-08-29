#ifndef KALDI_LM_KENLM_H
#define KALDI_LM_KENLM_H

#include "lm/model.hh"
#include "util/murmur_hash.hh"

#include <base/kaldi-common.h>
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <fstext/deterministic-fst.h>

namespace kaldi {

// This class is a thin wrapper of KenLM model,
// the model itself can be either "trie" or "probing" mode.
// Kaldi algorithms shouldn't explicitly interact with KenLm class,
// instead, use provided methods from the fst wrapper (class KenLmFst)
class KenLm {
 public:
  typedef lm::WordIndex WordIndex;
  typedef lm::ngram::State State;

 public:
  KenLm() : model_(NULL), vocab_(NULL) { }

  ~KenLm() {
    delete model_;
    model_ = NULL;
    vocab_ = NULL;
    reindex_.clear();
  }

  int Load(std::string kenlm_filename, std::string symbol_table_filename);

  inline int32 GetSymbolIndex(std::string symbol) { return symbol_to_symbol_id_[symbol]; }
  inline WordIndex GetWordIndex(std::string word) { return vocab_->Index(word.c_str()); }
  inline WordIndex GetWordIndex(int32 symbol_id) { return reindex_[symbol_id]; }

  void SetStateToBos(State *s) { model_->BeginSentenceWrite(s); }
  void SetStateToNull(State *s) { model_->NullContextWrite(s); }

  inline BaseFloat Score(const State *istate, WordIndex word, State *ostate) {
    return model_->BaseScore(istate, word, ostate);
  }

  struct StateHasher {
    inline size_t operator()(const State &s) const noexcept {
      return util::MurmurHashNative(s.words, sizeof(WordIndex) * s.Length());
    }
  };
  
 private:
  lm::base::Model *model_;

  // vocab_ points to models_'s internal vocabulary
  // no ownership, just for quick reference.
  const lm::base::Vocabulary* vocab_;

  // Kaldi's output symbol table
  std::unordered_map<std::string, int32> symbol_to_symbol_id_;

  // There are really two indexing systems here:
  // Kaldi's fst output symbol id(defined in words.txt),
  // and KenLM's word index(obtained by hashing the word string).
  // in order to incorperate KenLM into Kaldi,
  // we need to know the mapping between (Kaldi's symbol id -> KenLM's word id)
  // notes:
  // <eps> and #0 in symbol table are special,
  // they do not correspond to any word in KenLM,
  // Normally, these two symbols shouldn't consume any KenLM word,
  // so the mapping of these two symbols are logically undefined,
  // we just map them to KenLM's <unk>(which is always indexed as 0),
  // to avoid random invalid mapping.
  std::vector<WordIndex> reindex_; // reindex_[kaldi_symbol_index] -> kenlm word index
}; // class KenLm


//This class wraps KenLM into Kaldi's DeterministicOnDemandFst class,
//so that Kaldi's fst framework can utilize KenLM as language model
template<class Arc>
class KenLmFst : public fst::DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename KenLm::State State;
  typedef typename KenLm::WordIndex WordIndex;
 
  explicit KenLmFst(KenLm *lm)
   : lm_(lm), num_states_(0), bos_state_id_(0)
  {
    MapElem elem;
    lm->SetStateToBos(&elem.first);
    elem.second = num_states_;

    std::pair<IterType, bool> ins_result = state_map_.insert(elem);
    KALDI_ASSERT(ins_result.second == true);
    state_vec_.push_back(&ins_result.first->first);
    num_states_++;

    eos_symbol_id_ = lm_->GetSymbolIndex("</s>");
  }
  virtual ~KenLmFst() { }

  virtual StateId Start() { 
    return bos_state_id_;
  }

  virtual bool GetArc(StateId s, Label label, Arc *oarc) {
    KALDI_ASSERT(s < static_cast<StateId>(state_vec_.size()));
    const State* istate = state_vec_[s];
    MapElem e;
    WordIndex word = lm_->GetWordIndex(label);
    BaseFloat log_10_prob = lm_->Score(istate, word, &e.first);
    e.second = num_states_;
    std::pair<IterType, bool> result = state_map_.insert(e);
    if (result.second == true) {
      state_vec_.push_back(&(result.first->first));
      num_states_++;
    }

    oarc->ilabel = label;
    oarc->olabel = oarc->ilabel;
    oarc->nextstate = result.first->second;
    oarc->weight = Weight(-log_10_prob * M_LN10); // KenLM's log10() -> Kaldi's ln()

    return true;
  }

  virtual Weight Final(StateId s) {
    Arc oarc;
    GetArc(s, eos_symbol_id_, &oarc);
    return oarc.weight;
  }

 private:
  KenLm *lm_; // no ownership

 private:
  typedef std::pair<State, StateId> MapElem;
  typedef unordered_map<State, StateId, KenLm::StateHasher> MapType;
  typedef typename MapType::iterator IterType;

  MapType state_map_;
  std::vector<const State*> state_vec_;
  StateId num_states_; // state vector index range, [0, num_states_)

  StateId bos_state_id_;  // fst start state id
  WordIndex eos_symbol_id_;

}; // class KenLmFst

} // namespace kaldi

#endif
