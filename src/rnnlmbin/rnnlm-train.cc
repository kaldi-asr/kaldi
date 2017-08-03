// rnnlmbin/rnnlm-train.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::rnnlm;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3-based RNNLM language model (reads minibatches prepared\n"
        "by rnnlm-get-egs).  Supports various modes depending which parameters\n"
        "we are training.\n"
        "Usage:\n"
        " rnnlm-train [options] <egs-rspecifier>\n"
        "e.g.:\n"
        " rnnlm-get-egs ... ark:- | \\\n"
        " rnnlm-train --read-rnnlm=foo/0.raw --write-rnnlm=foo/1.raw --read-embedding=foo/0.embedding\\\n"
        "       --write-embedding=foo/1.embedding --read-sparse-word-features=foo/word_feats.txt ark:-\n"
        "See also: rnnlm-get-egs\n";


    std::string use_gpu = "yes";
    RnnlmTrainerOptions train_config;

    ParseOptions po(usage);
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    train_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 1 || !train_config.HasRequiredOptions()) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif


    {
      RnnlmTrainer trainer(train_config);


      SequentialRnnlmExampleReader example_reader(examples_rspecifier);

      for (; !example_reader.Done(); example_reader.Next())
        trainer.Train(&(example_reader.Value()));

      if (trainer.NumMinibatchesProcessed() == 0)
        KALDI_ERR << "There was no data to train on.";
      // The destructor of 'trainer' trains on the last minibatch
      // and writes out anything we need to write out.
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


