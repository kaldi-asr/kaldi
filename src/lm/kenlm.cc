#include "lm/kenlm.h"

namespace kaldi {

int KenLm::Load(std::string kenlm_filename, std::string symbol_table_filename) {
  if (model_ != NULL) {
    delete model_;
  }
  model_ = NULL;
  vocab_ = NULL;

  // load KenLm model
  // LAZY mode allows on-demands read instead of load the entired model into memory
  // this is especially useful when we are dealing with very large LM
  lm::ngram::Config config;
  config.load_method = util::LoadMethod::LAZY;
  model_ = lm::ngram::LoadVirtual(kenlm_filename.c_str(), config);
  if (!model_) {
    KALDI_ERR << "Failed loading KenLm model";
  }

  // load vocabulary, this member ponter has no ownership of the actual vocabulary
  vocab_ = &model_->BaseVocabulary();
  if (!vocab_) {
    KALDI_ERR << "Failed loading KenLm vocabulary";
  }

  // count symbol table size
  int num_syms = 0;
  std::string line;
  {
    std::ifstream is(symbol_table_filename);
    while(std::getline(is, line)) { if (!line.empty()) num_syms++; }
  }

  // compute the word reindex from Kaldi symbol id to KenLm word id
  symid_to_wid_.clear();
  symid_to_wid_.resize(num_syms, 0); // 0 always means <unk> in KenLm

  std::ifstream is(symbol_table_filename);
  while (std::getline(is, line)) {
    std::vector<std::string> fields;
    SplitStringToVector(line, " ", true, &fields);
    if (fields.size() == 2) {
      std::string sym = fields[0];
      int32 symid = 0;  ConvertStringToInteger(fields[1], &symid);
      // TO DISCUSS: should we force the kenlm have exactly the same vocab with kaldi?
      // it is ill-defined problem if we allow mismatch between kaldi and kenlm vocab.
      symid_to_wid_[symid] = vocab_->Index(sym.c_str());

      // register if it is special lm symbols
      if (sym == "<s>") {
        bos_symid_ = symid;
      } else if (sym == "</s>") {
        eos_symid_ = symid;
      } else if (sym == "<unk>" || sym == "<UNK>") {
        unk_symid_ = symid;
      }
    }
  }

  KALDI_LOG << "Kaldi word symble table size: " << symid_to_wid_.size();

  return 0;
}

} // namespace kaldi
