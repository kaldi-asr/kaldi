#ifndef KALDI_LM_KENLM_H
#define KALDI_LM_KENLM_H

#include <base/kaldi-common.h>
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <fstext/deterministic-fst.h>

#include "lm/model.hh"
#include "util/murmur_hash.hh"

namespace kaldi {

// Kaldi wrapper class for KenLm model, 
// the underlying model structure can be either "trie" or "probing".
// Its main purposes:
//  1. loads & holds kenlm model resources(with ownership)
//  2. handles the index mapping between kaldi symbol index & kenlm word index
//  3. provides ngram query method to upper level fst wrapper
// This class is stateless and thread-safe, so it can be shared by its FST wrapper class
// (i.e. KenLmDeterministicOnDemandFst class)
class KenLm {
 public:
  typedef lm::WordIndex WordIndex;
  typedef lm::ngram::State State;

 public:
  KenLm() : 
    model_(nullptr), vocab_(nullptr),
    bos_sym_(), eos_sym_(), unk_sym_(),
    bos_symid_(0), eos_symid_(0), unk_symid_(0)
  { }

  ~KenLm() {
    if (model_ != nullptr) {
      delete model_;
    }
    model_ = nullptr;
    vocab_ = nullptr;
    symid_to_wid_.clear();
  }

  // with giant LM on SSD hard drive,
  // you can set load method to util::LoadMethod::LAZY,
  // which enables on-demand model loading via POSIX mmap(),
  // refer to kenlm/util/mmap.hh for more load methods.
  int Load(std::string kenlm_filename, 
           std::string kaldi_symbol_table_filename,
           util::LoadMethod load_method = util::LoadMethod::POPULATE_OR_READ);

  inline WordIndex GetWordIndex(std::string word) const {
    return vocab_->Index(word.c_str());
  }

  inline WordIndex GetWordIndex(int32 symbol_id) const {
    return symid_to_wid_[symbol_id];
  }

  void SetStateToBeginOfSentence(State *s) const { model_->BeginSentenceWrite(s); }
  void SetStateToNull(State *s) const { model_->NullContextWrite(s); }

  int32 BosSymbolIndex() const { return bos_symid_; }
  int32 EosSymbolIndex() const { return eos_symid_; }
  int32 UnkSymbolIndex() const { return unk_symid_; }

  inline BaseFloat Score(const State *in_state,
                         WordIndex word,
                         State *out_state) const {
    return model_->BaseScore(in_state, word, out_state);
  }

  // this provides a fast hash function to upper level fst wrapper class,
  // to maintain the mapping between underlying lm states and fst state indexes
  struct StateHasher {
    inline size_t operator()(const State &s) const noexcept {
      return util::MurmurHashNative(s.words, sizeof(WordIndex) * s.Length());
    }
  };

 private:
  void ComputeSymbolToWordIndexMapping(std::string symbol_table);
  
 private:
  lm::base::Model *model_; // has ownership

  // no ownership, points to models_'s internal vocabulary
  const lm::base::Vocabulary* vocab_;

  // There are really two indexing systems here:
  // Kaldi's fst output symbol id(defined in words.txt),
  // and KenLm's word index(obtained by hashing the word string).
  // in order to incorperate KenLm into Kaldi during runtime, 
  // we need to know the mapping between the two indexing systems.
  // by keeping this mapping explicity in this class,
  // we avoid modifing any Kaldi & KenLm runtime resources,
  // (e.g. HCLG.fst/lattices & KenLm model file)
  // notes:
  // <eps> and #0 symbols are special,
  // they do not correspond to any word in KenLm,
  // Normally, these two symbols shouldn't consume any KenLm word,
  // so the mapping of these two symbols are logically undefined,
  // we just map them to KenLm's <unk>(which is always indexed as 0),
  // to avoid random invalid mapping.
  // usage: symid_to_wid_[kaldi_symbol_index] -> kenlm word index
  std::vector<WordIndex> symid_to_wid_;

  // special lm symbols
  std::string bos_sym_;
  std::string eos_sym_;
  std::string unk_sym_;

  int32 bos_symid_;
  int32 eos_symid_;
  int32 unk_symid_;
}; // class KenLm


// This class wraps KenLm into DeterministicOnDemandFst interface,
// so that Kaldi's fst framework can utilize KenLm as a deterministic FST.
// objects of this class have states(so it's not thread-safe),
// different threads should create their own instances, they are lightweight.
// Globally, KenLmDeterministicOnDemandFst objects should share 
// the same KenLm object (which is heavy, stateless and thread-safe)
template<class Arc>
class KenLmDeterministicOnDemandFst : public fst::DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename KenLm::State State;
  typedef typename KenLm::WordIndex WordIndex;
 
  explicit KenLmDeterministicOnDemandFst(const KenLm *lm)
   : lm_(lm), num_states_(0), bos_state_id_(0)
  {
    // create bos to be FST start state
    MapElem e;
    lm->SetStateToBeginOfSentence(&e.first);
    e.second = bos_state_id_;
    std::pair<IterType, bool> r = state_map_.insert(e);
    KALDI_ASSERT(r.second == true); // bos successfully inserted into state map
    state_vec_.push_back(&r.first->first);
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
    std::pair<IterType, bool> r = state_map_.insert(e);
    if (r.second == true) { // new state
      state_vec_.push_back(&(r.first->first));
      num_states_++;
    }

    oarc->ilabel = label;
    oarc->olabel = oarc->ilabel;
    oarc->nextstate = r.first->second;
    oarc->weight = Weight(-log_10_prob * M_LN10); // KenLm log10() -> Kaldi ln()

    return true;
  }

  virtual Weight Final(StateId s) {
    Arc oarc;
    GetArc(s, eos_symbol_id_, &oarc);
    return oarc.weight;
  }

 private:
  typedef std::pair<State, StateId> MapElem;
  typedef unordered_map<State, StateId, KenLm::StateHasher> MapType;
  typedef typename MapType::iterator IterType;

  const KenLm *lm_; // no ownership
  MapType state_map_;
  std::vector<const State*> state_vec_;
  StateId num_states_; // state vector index range, [0, num_states_)
  StateId bos_state_id_;  // fst start state id
  Label eos_symbol_id_;
}; // class KenLmDeterministicOnDemandFst
} // namespace kaldi
#endif
