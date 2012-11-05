// nnet-cpubin/nnet-copy-egs.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet-cpu/nnet-randomize.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples (typically single frames) for neural network training,\n"
        "possibly changing the binary mode.\n"
        "\n"
        "Usage:  nnet-copy-egs [options] <egs-rspecifier> <egs-wspecifier>\n"
        "\n"
        "e.g.\n"
        "nnet-copy-egs ark:train.egs ark,t:text.egs\n";
        

    ParseOptions po(usage);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    SequentialNnetTrainingExampleReader example_reader(examples_rspecifier);
    NnetTrainingExampleWriter example_writer(examples_wspecifier);
    
    int64 num_done = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_done++)
      example_writer.Write(example_reader.Key(), example_reader.Value());

    KALDI_LOG << "Copied " << num_done << " neural-network training examples "
              << " to " << examples_wspecifier;
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


