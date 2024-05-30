// nnet3bin/nnet3-merge-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar

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
#include "hmm/transition-model.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {
// returns the number of indexes/frames in the NnetIo with output name
// including string "output" as part of its name in the eg.
// e.g. output-0, output-xent
int32 NumOutputIndexes(const NnetExample &eg) {
  for (size_t i = 0; i < eg.io.size(); i++)
    if (eg.io[i].name.find("output") != std::string::npos)
      return eg.io[i].indexes.size();
  return 1;  // Suppress compiler warning.
}

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This copies nnet training examples from input to output, but while doing so it\n"
        "merges many NnetExample objects into one, forming a minibatch consisting of a\n"
        "single NnetExample.\n"
        "\n"
        "Usage:  nnet3-merge-egs [options] <egs-rspecifier> <egs-wspecifier>\n"
        "e.g.\n"
        "nnet3-merge-egs --minibatch-size=512 ark:1.egs ark:- | nnet3-train-simple ... \n"
        "See also nnet3-copy-egs\n";

    ParseOptions po(usage);

    ExampleMergingConfig merging_config;
    merging_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    merging_config.ComputeDerived();

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);

    ExampleMerger merger(merging_config, &example_writer);

    for (; !example_reader.Done(); example_reader.Next()) {
      const NnetExample &cur_eg = example_reader.Value();
      merger.AcceptExample(new NnetExample(cur_eg));
    }
    // the merger itself prints the necessary diagnostics.
    merger.Finish();
    return merger.ExitStatus();
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
