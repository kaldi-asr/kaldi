#ifndef KALDI_LM_KENLM_H
#define KALDI_LM_KENLM_H

#include "lm/model.hh"
#include "util/murmur_hash.hh"

namespace kaldi {

class KenLm {
 public:
  typedef lm::WordIndex WordIndex;
  typedef lm::ngram::State State;

 public:
  KenLm() : model_(NULL), vocab_(NULL) { }

  ~KenLm() {
    DELETE(model_);
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
  const lm::base::Vocabulary* vocab_; // no ownership, it points to models_'s internal vocabuary
  std::unordered_map<std::string, int32> symbol_to_symbol_id_;

  // There are really two indexing systems here:
  // Kaldi's fst output symbol id(defined in words.txt),
  // and KenLM's word index(obtained by hashing the word string).
  // in order to incorperate KenLM into Kaldi,
  // we need to know the mapping between (Kaldi's symbol id -> KenLM's word id)
  // special symbols:
  // Kaldi's output symbol table contains two special symbols(<eps> and #0) that do not exist in KenLM,
  // In any circumstance, these two symbols shouldn't consume any KenLM word,
  // so the mapping of these two symbols are logically undefined,
  // we just map them to KenLM's <unk>(which is always indexed as 0) to avoid random invalid mapping.
  std::vector<WordIndex> reindex_; // reindex_[kaldi_symbol_index] -> kenlm word index
}; // class KenLm

} // namespace kaldi

#endif
