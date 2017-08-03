// rnnlm/rnnlm-example-test.cc

// Copyright 2017  Johns Hopkins University (author: Daniel Povey)

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


#include "rnnlm/rnnlm-example.h"
#include "rnnlm/rnnlm-test-utils.h"
#include "rnnlm/rnnlm-example-utils.h"

namespace kaldi {
namespace rnnlm {


void TestRnnlmExample() {

  std::set<std::string> forbidden_symbols;
  GetForbiddenSymbols(&forbidden_symbols);

  std::vector<std::vector<std::string> > sentences;
  GetTestSentences(forbidden_symbols, &sentences);

  fst::SymbolTable *symbol_table = GetSymbolTable(sentences);

  std::vector<std::vector<int32> > int_sentences;
  ConvertToInteger(sentences, *symbol_table, &int_sentences);

  std::vector<std::vector<int32> > int_sentences_train,
      int_sentences_test;
  for (size_t i = 0; i < int_sentences.size(); i++) {
    if (i % 10 == 0) int_sentences_test.push_back(int_sentences[i]);
    else int_sentences_train.push_back(int_sentences[i]);
  }


  std::ostringstream os;
  int32 ngram_order = 3;
  int32 bos = 1, eos = 2, brk = 3;
  EstimateAndWriteLanguageModel(ngram_order, *symbol_table,
                                int_sentences_train, bos, eos, os);

  ArpaParseOptions arpa_options;
  arpa_options.bos_symbol = bos;
  arpa_options.eos_symbol = eos;
  ArpaSampling arpa(arpa_options, symbol_table);

  // TODO: we'll add more tests from this point.
  RnnlmEgsConfig egs_config;
  egs_config.vocab_size = symbol_table->AvailableKey();
  egs_config.bos_symbol = bos;
  egs_config.eos_symbol = eos;
  egs_config.brk_symbol = brk;

  RnnlmExampleSampler sampler(egs_config, arpa);

  RnnlmExampleWriter writer("ark:tmp.ark");

  {
    RnnlmExampleCreator *creator = NULL;
    if (RandInt(0, 1) == 0) {
      // use sampling to create the egs.
      creator = new RnnlmExampleCreator(egs_config, sampler, &writer);
    } else {
      creator = new RnnlmExampleCreator(egs_config, &writer);
    }
    for (size_t i = 0; i < int_sentences_test.size(); i++) {
      BaseFloat weight = 0.5 * RandInt(1, 2);
      creator->AcceptSequence(weight, int_sentences_test[i]);
    }
    delete creator;
  }

  // TODO: read the archive and actually train.


}



}
}

int main() {
  int32 loop = 0;

  // SetVerboseLevel(2);
#if HAVE_CUDA == 1
  for (loop = 0; loop < 2; loop++) {
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif

    kaldi::rnnlm::TestRnnlmExample();


    if (loop == 0)
      KALDI_LOG << "Tests without GPU use succeeded.";
    else
      KALDI_LOG << "Tests with GPU use (if available) succeeded.";
#if HAVE_CUDA == 1
  } // No for loop if 'HAVE_CUDA != 1',
  SetVerboseLevel(4);
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}



