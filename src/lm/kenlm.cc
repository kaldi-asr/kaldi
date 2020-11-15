#include "lm/kenlm.h"

namespace kaldi {

void KenLm::ComputeSymbolToWordIndexMapping(std::string symbol_table_filename) {
  // count symbol table size
  int num_syms = 0;
  std::string line;
  std::ifstream is(symbol_table_filename);
  while(std::getline(is, line)) {
    if (!line.empty()) num_syms++;
  }

  symid_to_wid_.clear();
  symid_to_wid_.resize(num_syms, 0);

  is.clear();
  is.seekg(0);
  while (std::getline(is, line)) {
    std::vector<std::string> fields;
    SplitStringToVector(line, " ", true, &fields);
    if (fields.size() == 2) {
      std::string sym = fields[0];
      int32 symid = 0;  ConvertStringToInteger(fields[1], &symid);
      // mark special LM word
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
      // check vocabulary consistency between kaldi and KenLm.
      // note here we add seemingly verbose check about unknown word
      // so we don't need to worry about mismatch error between <unk> & <UNK>
      WordIndex wid = vocab_->Index(sym.c_str());
      if ((wid == vocab_->Index("<unk>") || wid == vocab_->Index("<UNK>")) 
          && sym != "<unk>" && sym != "<UNK>"
          && sym != "<eps>"
          && sym != "#0") {
        KALDI_ERR << "found mismatched symbol: " << sym
                  << ", this symbol is in Kaldi, but is unseen in KenLm"
                  << ", they should have strictly consistent vocabulary.";
      } else {
        symid_to_wid_[symid] = wid;
      }
    }
  }
  KALDI_LOG << "Successfully mapped " << symid_to_wid_.size() 
            << " Kaldi symbols to KenLm words";
}

int KenLm::Load(std::string kenlm_filename,
                std::string symbol_table_filename,
                util::LoadMethod load_method) {
  if (model_ != nullptr) { delete model_; }
  model_ = nullptr;
  vocab_ = nullptr;

  // load KenLm model
  lm::ngram::Config config;
  config.load_method = load_method;
  model_ = lm::ngram::LoadVirtual(kenlm_filename.c_str(), config);
  if (model_ == nullptr) { KALDI_ERR << "Failed to load KenLm model"; }

  // KenLm holds vocabulary internally with ownership,
  // vocab_ here is just for convenient reference
  vocab_ = &model_->BaseVocabulary();
  if (vocab_ == nullptr) { KALDI_ERR << "Failed to get vocabulary from KenLm model"; }

  // compute the index mapping from Kaldi symbol to KenLm word
  ComputeSymbolToWordIndexMapping(symbol_table_filename);
  return 0;
}

} // namespace kaldi
