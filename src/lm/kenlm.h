#ifndef KALDI_LM_KENLM_H
#define KALDI_LM_KENLM_H

#include <base/kaldi-common.h>
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <fstext/deterministic-fst.h>

#include "lm/model.hh"
#include "util/murmur_hash.hh"

namespace kaldi {

// This class is a thin wrapper of KenLM model,
// the underlying model structure can be either "trie" or "probing".
// Its primary jobs are:
//   1. loads & holds kenlm model resources with ownership
//   2. handles the index mapping between kaldi fst symbol and kenlm word
//   3. provides ngram query method to upper level fst wrapper(i.e. KenLmDeterministicOnDemandFst)
// Kaldi algorithms should interact with the fst wrapper instead of KenLm class.
class KenLm {
 public:
  typedef lm::WordIndex WordIndex;
  typedef lm::ngram::State State;

 public:
  KenLm() : 
    model_(NULL), vocab_(NULL),
    bos_symid_(0), eos_symid_(0), unk_symid_(0)
  { }

  ~KenLm() {
    delete model_;
    model_ = NULL;
    vocab_ = NULL;
    symid_to_wid_.clear();
  }

  int Load(std::string kenlm_filename, std::string symbol_table_filename);

  inline WordIndex GetWordIndex(std::string word) {
    return vocab_->Index(word.c_str());
  }

  inline WordIndex GetWordIndex(int32 symbol_id) {
    return symid_to_wid_[symbol_id];
  }

  void SetStateToBeginOfSentence(State *s) { model_->BeginSentenceWrite(s); }
  void SetStateToNull(State *s) { model_->NullContextWrite(s); }

  int32 BosSymbolIndex() { return bos_symid_; }
  int32 EosSymbolIndex() { return eos_symid_; }
  int32 UnkSymbolIndex() { return unk_symid_; }

  inline BaseFloat Score(const State *istate, WordIndex word, State *ostate) {
    return model_->BaseScore(istate, word, ostate);
  }

  // this provides a fast hash function to upper level fst wrapper class,
  // to maintain the mapping between underlying lm states and fst state indexes
  struct StateHasher {
    inline size_t operator()(const State &s) const noexcept {
      return util::MurmurHashNative(s.words, sizeof(WordIndex) * s.Length());
    }
  };
  
 private:
  lm::base::Model *model_; // has ownership

  // no ownership, points to models_'s internal vocabulary
  const lm::base::Vocabulary* vocab_;

  // There are really two indexing systems here:
  // Kaldi's fst output symbol id(defined in words.txt),
  // and KenLM's word index(obtained by hashing the word string).
  // in order to incorperate KenLM into Kaldi during runtime, 
  // we need to know the mapping between the two indexing systems.
  // by keeping this mapping explicity in this class,
  // we avoid modifing any kaldi & kenlm runtime resources,
  // (e.g. HCLG.fst/lattices & kenlm model file)
  // notes:
  // <eps> and #0 symbols are special,
  // they do not correspond to any word in KenLM,
  // Normally, these two symbols shouldn't consume any KenLM word,
  // so the mapping of these two symbols are logically undefined,
  // we just map them to KenLM's <unk>(which is always indexed as 0),
  // to avoid random invalid mapping.
  // usage: symid_to_wid_[kaldi_symbol_index] -> kenlm word index
  std::vector<WordIndex> symid_to_wid_;

  // special lm symbols
  int32 bos_symid_;
  int32 eos_symid_;
  int32 unk_symid_;
}; // class KenLm


// This class wraps KenLm into Kaldi's DeterministicOnDemandFst class,
// so that Kaldi's fst framework can utilize KenLM as a simple FST.
// objects of this class have internal states(so not thread-safe),
// different threads should create their own objects, they are lightweight.
// Globally, all KenLmDeterministicOnDemandFst objects should share 
// a single KenLm object (which is stateless and heavy)
template<class Arc>
class KenLmDeterministicOnDemandFst : public fst::DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename KenLm::State State;
  typedef typename KenLm::WordIndex WordIndex;
 
  explicit KenLmDeterministicOnDemandFst(KenLm *lm)
   : lm_(lm), num_states_(0), bos_state_id_(0)
  {
    MapElem elem;
    lm->SetStateToBeginOfSentence(&elem.first);
    elem.second = num_states_;

    std::pair<IterType, bool> ins_result = state_map_.insert(elem);
    KALDI_ASSERT(ins_result.second == true);
    state_vec_.push_back(&ins_result.first->first);
    num_states_++;

    eos_symbol_id_ = lm_->EosSymbolIndex();
  }
  virtual ~KenLmDeterministicOnDemandFst() { }

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

}; // class KenLmDeterministicOnDemandFst

} // namespace kaldi

#endif
