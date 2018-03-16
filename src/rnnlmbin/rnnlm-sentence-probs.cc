// rnnlmbin/rnnlm-sentence-probs.cc

// Copyright 2015-2017  Johns Hopkins University (author: Daniel Povey)
//           2017 Hainan Xu

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "rnnlm/rnnlm-training.h"
#include "rnnlm/rnnlm-example-utils.h"
#include "rnnlm/rnnlm-core-compute.h"
#include "rnnlm/rnnlm-compute-state.h"
#include "nnet3/nnet-utils.h"
#include <fstream>
#include <sstream>

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::rnnlm;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program takes input of a text corpus (with words represented by\n"
        "symbol-id's), and an already trained RNNLM model, and prints the log\n"
        "-probabilities of each word in the corpus. The RNNLM resets its hidden\n"
        "state for each new line. This is used in n-best rescoring with RNNLMs\n"
        "An example the n-best rescoring usage is at "
        "egs/swbd/s5c$ vi local/rnnlm/run_tdnn_lstm.sh"
        "\n"
        "Usage:\n"
        " rnnlm-sentence-probs [options] <rnnlm> <word-embedding-matrix> "
        "<input-text-file> \n"
        "e.g.:\n"
        " rnnlm-sentence-probs rnnlm/final.raw rnnlm/final.word_embedding "
        "dev_corpus.txt > output_logprobs.txt\n";

    std::string use_gpu = "no";
    bool batchnorm_test_mode = true, dropout_test_mode = true;

    ParseOptions po(usage);
    rnnlm::RnnlmComputeStateComputationOptions opts;
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (opts.bos_index == -1 || opts.eos_index == -1) {
      KALDI_ERR << "You must set --bos-symbol and --eos-symbol options";
    }

    std::string rnnlm_rxfilename = po.GetArg(1),
                word_embedding_rxfilename = po.GetArg(2),
                text_filename = po.GetArg(3);

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().AllowMultithreading();
#endif

    kaldi::nnet3::Nnet rnnlm;
    ReadKaldiObject(rnnlm_rxfilename, &rnnlm);

    KALDI_ASSERT(IsSimpleNnet(rnnlm));
    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, &rnnlm);
    if (dropout_test_mode)
      SetDropoutTestMode(true, &rnnlm);

    CuMatrix<BaseFloat> word_embedding_mat;
    ReadKaldiObject(word_embedding_rxfilename, &word_embedding_mat);

    const rnnlm::RnnlmComputeStateInfo info(opts, rnnlm, word_embedding_mat);

    std::ifstream ifile(text_filename.c_str());

    std::string key, line;
    while (ifile >> key) {
      getline(ifile, line);
      std::vector<int32> v;
      KALDI_ASSERT(SplitStringToIntegers(line, " ", true, &v));
      RnnlmComputeState rnnlm_compute_state(info, opts.bos_index);

      std::cout << key << " ";
      for (int32 i = 0; i < v.size(); i++) {
        int32 word_id = v[i];
        std::cout << rnnlm_compute_state.LogProbOfWord(word_id) << " ";

        rnnlm_compute_state.AddWord(word_id);
      }
      // add the </s> symbol
      int32 word_id = opts.eos_index;
      std::cout << rnnlm_compute_state.LogProbOfWord(word_id) << std::endl;
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
