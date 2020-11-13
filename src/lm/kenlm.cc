#include "lm/kenlm.h"

namespace kaldi {

void KenLm::ComputeSymbolToWordIndexMapping(std::string symbol_table_filename) {
  // count symbol table size
  int num_syms = 0;
  std::string line;
  {
    std::ifstream is(symbol_table_filename);
    while(std::getline(is, line)) { if (!line.empty()) num_syms++; }
  }

  symid_to_wid_.clear();
  symid_to_wid_.resize(num_syms, 0);

  std::ifstream is(symbol_table_filename);
  while (std::getline(is, line)) {
    std::vector<std::string> fields;
    SplitStringToVector(line, " ", true, &fields);
    if (fields.size() == 2) {
      std::string sym = fields[0];
      int32 symid = 0;  ConvertStringToInteger(fields[1], &symid);
      // update class info if this is a special LM word
      if (sym == "<s>") {
        bos_sym_ = sym;
        bos_symid_ = symid;
      } else if (sym == "</s>") {
        eos_sym_ = sym;
        eos_symid_ = symid;
      } else if (sym == "<unk>" || sym == "<UNK>") {
        unk_sym_ = sym;
        unk_symid_ = symid;
      }
      // get & check word id for this symbol
      WordIndex wid = vocab_->Index(sym.c_str());
      if ((wid == vocab_->Index("<unk>") || wid == vocab_->Index("<UNK>")) 
          && sym != "<unk>" && sym != "<UNK>"
          && sym != "<eps>"
          && sym != "#0") {
        KALDI_ERR << "found mismatched symbol: " << sym
                  << ", this symbol is in Kaldi, but is unseen in KenLM"
                  << ", they should have strictly consistent vocabulary.";
      } else {
        symid_to_wid_[symid] = wid;
      }
    }
  }
  KALDI_LOG << "Successfully mapped " << symid_to_wid_.size() 
            << " Kaldi symbols to KenLM words";
}

int KenLm::Load(std::string kenlm_filename, std::string symbol_table_filename) {
  if (model_ != NULL) {
    delete model_;
  }
  model_ = NULL;
  vocab_ = NULL;

  // load KenLm model
  // LAZY mode allows on-demands read instead of entired loading into memory
  // this is especially useful when we are dealing with very large LM
  lm::ngram::Config config;
  config.load_method = util::LoadMethod::LAZY;
  model_ = lm::ngram::LoadVirtual(kenlm_filename.c_str(), config);
  if (!model_) {
    KALDI_ERR << "Failed loading KenLm model";
  }

  // load vocabulary pointer from kenlm model internal
  vocab_ = &model_->BaseVocabulary();
  if (!vocab_) {
    KALDI_ERR << "Failed loading KenLm vocabulary";
  }

  // compute the index mapping from Kaldi symbol to KenLm word
  ComputeSymbolToWordIndexMapping(symbol_table_filename);
  return 0;
}

} // namespace kaldi
