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
#include "rnnlm/rnnlm-training.h"

namespace kaldi {
namespace rnnlm {


// Gets a neural net that has no dependency on t greater than the current t.
nnet3::Nnet *GetTestingNnet(int32 embedding_dim) {
  std::ostringstream config_os;
  config_os << "input-node name=input dim=" << embedding_dim << std::endl;
  config_os << "component name=affine1 type=NaturalGradientAffineComponent input-dim="
            << embedding_dim << " output-dim=" << embedding_dim << std::endl;
  config_os << "component-node input=input name=affine1 component=affine1\n";
  config_os << "output-node input=affine1 name=output\n";
  std::istringstream config_is(config_os.str());
  nnet3::Nnet *ans = new nnet3::Nnet();
  ans->ReadConfig(config_is);
  return ans;
}


// test training (no sparse embedding).
void TestRnnlmTraining(const std::string &archive_rxfilename,
                       int32 vocab_size) {
  SequentialRnnlmExampleReader reader(archive_rxfilename);
  int32 embedding_dim = RandInt(10, 30);

  nnet3::Nnet *rnnlm = GetTestingNnet(embedding_dim);
  CuMatrix<BaseFloat> embedding_mat(vocab_size, embedding_dim);
  embedding_mat.SetRandn();


  RnnlmCoreTrainerOptions core_config;
  RnnlmEmbeddingTrainerOptions embedding_config;

  bool train_embedding = (RandInt(0, 1) == 0);

  {
    RnnlmTrainer trainer(train_embedding, core_config, embedding_config,
                         NULL, &embedding_mat, rnnlm);
    for (; !reader.Done(); reader.Next()) {
      trainer.Train(&reader.Value());
    }
  }

  delete rnnlm;
}



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

  {
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
        BaseFloat weight = 0.5 * RandInt(1, 3);
        creator->AcceptSequence(weight, int_sentences_test[i]);
      }
      delete creator;
    }
  }

  TestRnnlmTraining("ark:tmp.ark", egs_config.vocab_size);


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



