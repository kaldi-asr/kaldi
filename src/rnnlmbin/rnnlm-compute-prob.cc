// rnnlmbin/rnnlm-compute-prob.cc

// Copyright 2015-2017  Johns Hopkins University (author: Daniel Povey)

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
#include "nnet3/nnet-utils.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::rnnlm;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program computes the probability per word of the provided training\n"
        "data in 'egs' format as prepared by rnnlm-get-egs.  The interface is similar\n"
        "to rnnlm-train, except that it doesn't train, and doesn't write the model;\n"
        "it just prints the average probability to the standard output (in addition\n"
        "to printing various diagnostics to the standard error).\n"
        "\n"
        "Usage:\n"
        " rnnlm-compute-prob [options] <rnnlm> <word-embedding-matrix> <egs-rspecifier>\n"
        "e.g.:\n"
        " rnnlm-get-egs ... ark:- | \\\n"
        " rnnlm-compute-prob 0.raw 0.word_embedding ark:-\n"
        "(note: use rnnlm-get-word-embedding to get the word embedding matrix if\n"
        "you are using sparse word features.)\n";

    std::string use_gpu = "no";
    bool batchnorm_test_mode = true, dropout_test_mode = true;

    ParseOptions po(usage);
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("batchnorm-test-mode", &batchnorm_test_mode,
                "If true, set test-mode to true on any BatchNormComponents.");
    po.Register("dropout-test-mode", &dropout_test_mode,
                "If true, set test-mode to true on any DropoutComponents and "
                "DropoutMaskComponents.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string rnnlm_rxfilename = po.GetArg(1),
        word_embedding_rxfilename = po.GetArg(2),
        egs_rspecifier = po.GetArg(3);

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().AllowMultithreading();
#endif

    kaldi::nnet3::Nnet rnnlm;
    ReadKaldiObject(rnnlm_rxfilename, &rnnlm);

    if (!IsSimpleNnet(rnnlm))
      KALDI_ERR << "Input RNNLM in " << rnnlm_rxfilename
                << " is not the type of neural net we were looking for; "
          "failed IsSimpleNnet().";
    if (batchnorm_test_mode)
      SetBatchnormTestMode(true, &rnnlm);
    if (dropout_test_mode)
      SetDropoutTestMode(true, &rnnlm);

    CuMatrix<BaseFloat> word_embedding_mat;
    ReadKaldiObject(word_embedding_rxfilename, &word_embedding_mat);


    SequentialRnnlmExampleReader example_reader(egs_rspecifier);

    double tot_weight = 0.0, tot_objf = 0.0;
    {
      RnnlmCoreComputer computer(rnnlm);

      int32 num_minibatches = 0;

      for (; !example_reader.Done(); example_reader.Next()) {
        const RnnlmExample &minibatch = example_reader.Value();
        RnnlmExampleDerived derived;
        bool need_embedding_deriv = false;
        GetRnnlmExampleDerived(minibatch, need_embedding_deriv, &derived);
        // compute for this minibatch.
        BaseFloat weight,
            objf = computer.Compute(minibatch, derived, word_embedding_mat,
                                    &weight, NULL);
        tot_weight += weight;
        tot_objf += objf;
        num_minibatches++;
      }
      if (num_minibatches == 0)
        KALDI_ERR << "Processed no data.";
      // The destructor of 'computer' prints diagnostics so we don't need to
      // print any diagnostic messages.
    }


    std::cout << (tot_objf / tot_weight) << std::endl;

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
