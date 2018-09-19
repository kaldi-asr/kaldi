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
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-device.h"

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
  RnnlmObjectiveOptions objective_config;

  bool train_embedding = (RandInt(0, 1) == 0);

  {
    RnnlmTrainer trainer(train_embedding, core_config, embedding_config,
                         objective_config, NULL, &embedding_mat, rnnlm);
    for (; !reader.Done(); reader.Next()) {
      trainer.Train(&reader.Value());
    }
  }

  delete rnnlm;
}

void TestRnnlmOutput(const std::string &archive_rxfilename) {
  SequentialRnnlmExampleReader reader(archive_rxfilename);
  int32 num_test = 10;
  for (int32 n = 0; !reader.Done() && n < num_test; reader.Next(), n++) {
    RnnlmExample &example(reader.Value());
    bool train_embedding = (RandInt(0, 1) == 0),
        train_nnet = (RandInt(0, 1) == 0);

    RnnlmExampleDerived derived;
    GetRnnlmExampleDerived(example, train_embedding, &derived);

    int32 embedding_dim = RandInt(10, 40),
        vocab_size = example.vocab_size,
        num_output_rows = example.chunk_length * example.num_chunks;



    CuMatrix<BaseFloat> embedding(vocab_size, embedding_dim),
        embedding_deriv(vocab_size, embedding_dim),
        nnet_output(num_output_rows, embedding_dim),
        nnet_output_deriv(num_output_rows, embedding_dim);

    embedding.SetRandn();
    embedding.Scale(0.05);
    nnet_output.SetRandn();
    nnet_output.Scale(0.05);

    // Make sure the embedding and nnet output are opposite
    // directions so the normalizer is reasonable.
    for (int32 i = 0; i < embedding.NumRows(); i++)
      embedding(i, 0) += 1.0;
    for (int32 i = 0; i < nnet_output.NumRows(); i++)
      nnet_output(i, 0) += -log(embedding.NumRows());

    BaseFloat weight, objf_num, objf_den, objf_den_exact;

    RnnlmObjectiveOptions objective_config;
    ProcessRnnlmOutput(objective_config,
                       example, derived, embedding, nnet_output,
                       train_embedding ? &embedding_deriv : NULL,
                       train_nnet ? &nnet_output_deriv : NULL,
                       &weight, &objf_num, &objf_den, &objf_den_exact);

    KALDI_LOG << "Weight=" << weight
              << ", objf-num=" << objf_num
              << ", objf-den=" << objf_den
              << ", objf=" << (objf_num + objf_den)
              << ", objf-den-exact is " << objf_den_exact;

    if (train_embedding) {
      BaseFloat delta = 0.0004;
      // test the embedding derivatives
      KALDI_LOG << "Testing the derivatives w.r.t. "
          "the embedding [testing ProcessOutput()].";
      // num_tries is the number of times we perturb the embedding matrix;
      // making this >1 makes the test more robust.
      int32 num_tries = 3;
      Vector<BaseFloat> objf_change_predicted(num_tries),
          objf_change_observed(num_tries);
      for (int32 i = 0; i < num_tries; i++) {
        CuMatrix<BaseFloat> embedding2(vocab_size, embedding_dim);
        embedding2.SetRandn();
        embedding2.Scale(delta);
        objf_change_predicted(i) = TraceMatMat(embedding2, embedding_deriv, kTrans);
        embedding2.AddMat(1.0, embedding);

        KALDI_LOG << "Embedding sum is " << embedding.Sum()
                  << ", nnet-output sum is " << nnet_output.Sum()
                  << ", smat sum is " << derived.output_words_smat.Sum();

        BaseFloat weight2, objf_num2, objf_den2;
        RnnlmObjectiveOptions objective_config;
        ProcessRnnlmOutput(objective_config,
                           example, derived, embedding2, nnet_output,
                           NULL, NULL,
                           &weight2, &objf_num2, &objf_den2, NULL);
        objf_change_observed(i) = (objf_num2 + objf_den2) -
            (objf_num + objf_den);
      }
      KALDI_LOG << "Objf change is " << objf_change_predicted
                << " (predicted) vs. " << objf_change_observed << " (observed), "
                << "when changing embedding.";
      if (!objf_change_predicted.ApproxEqual(objf_change_observed, 0.1)) {
        KALDI_WARN << "Embedding-deriv test failed.";
      }

    }
    if (train_nnet) {
      BaseFloat delta = 0.001;
      // test the nnet-output derivatives
      KALDI_LOG << "Testing the derivatives w.r.t. "
          "the nnet output [testing ProcessOutput()].";
      // num_tries is the number of times we perturb the embedding matrix;
      // making this >1 makes the test more robust.
      int32 num_tries = 3;
      Vector<BaseFloat> objf_change_predicted(num_tries),
          objf_change_observed(num_tries);
      for (int32 i = 0; i < num_tries; i++) {
        CuMatrix<BaseFloat> nnet_output2(num_output_rows, embedding_dim);
        nnet_output2.SetRandn();
        nnet_output2.Scale(delta);
        objf_change_predicted(i) = TraceMatMat(nnet_output2, nnet_output_deriv,
                                               kTrans);
        nnet_output2.AddMat(1.0, nnet_output);

        KALDI_LOG << "Embedding sum is " << embedding.Sum()
                  << ", nnet-output sum is " << nnet_output.Sum()
                  << ", smat sum is " << derived.output_words_smat.Sum();

        BaseFloat weight2, objf_num2, objf_den2;
        RnnlmObjectiveOptions objective_config;
        ProcessRnnlmOutput(objective_config,
                           example, derived, embedding, nnet_output2,
                           NULL, NULL,
                           &weight2, &objf_num2, &objf_den2, NULL);
        objf_change_observed(i) = (objf_num2 + objf_den2) -
            (objf_num + objf_den);
      }
      KALDI_LOG << "Objf change is " << objf_change_predicted
                << " (predicted) vs. " << objf_change_observed << " (observed), "
                << "when changing nnet output.";
      if (!objf_change_predicted.ApproxEqual(objf_change_observed, 0.1)) {
        KALDI_WARN << "Nnet-output-deriv test failed.";
      }
    }
  }
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


  std::stringstream os;
  int32 ngram_order = 3;
  int32 bos = 1, eos = 2, brk = 3;
  EstimateAndWriteLanguageModel(ngram_order, *symbol_table,
                                int_sentences_train, bos, eos, os);

  ArpaParseOptions arpa_options;
  arpa_options.bos_symbol = bos;
  arpa_options.eos_symbol = eos;
  SamplingLm arpa(arpa_options, symbol_table);
  os.seekg(0, std::ios::beg);
  arpa.Read(os);

  // TODO: we'll add more tests from this point.
  RnnlmEgsConfig egs_config;
  egs_config.vocab_size = symbol_table->AvailableKey();
  egs_config.bos_symbol = bos;
  egs_config.eos_symbol = eos;
  egs_config.brk_symbol = brk;

  RnnlmExampleSampler sampler(egs_config, arpa);

  {
    RnnlmExampleWriter writer("ark:tmp.ark");
    TaskSequencerConfig sequencer_config;

    {
      RnnlmExampleCreator *creator = NULL;
      if (RandInt(0, 1) == 0) {
        // use sampling to create the egs.

        creator = new RnnlmExampleCreator(egs_config, sequencer_config, sampler, &writer);
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

  TestRnnlmOutput("ark:tmp.ark");
  TestRnnlmTraining("ark:tmp.ark", egs_config.vocab_size);

}



}
}

int main() {
  srand(100);
  using namespace kaldi;
  using namespace kaldi::nnet3;
  int32 loop = 0;
  using namespace kaldi;

  // SetVerboseLevel(2);
#if HAVE_CUDA == 1
  for (loop = 0; loop < 2; loop++) {
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif

    for (int32 i = 0; i < 8; i++)
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

  unlink("tmp.ark");
  return 0;
}

// TODO (important): add offset to the embedding.
