// nnet3bin/nnet3-discriminative-merge-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//           2014-2015  Vimal Manohar

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
#include "nnet3/nnet-discriminative-example.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This copies nnet3 discriminative training examples from input to output, merging them\n"
        "into composite examples.  The --minibatch-size option controls how many egs\n"
        "are merged into a single output eg.\n"
        "\n"
        "Usage:  nnet3-discriminative-egs [options] <egs-rspecifier> <egs-wspecifier>\n"
        "e.g.\n"
        "nnet3-discriminative-merge-egs --minibatch-size=128 ark:1.degs ark:- | nnet3-discriminative-train ... \n"
        "See also nnet3-discriminative-copy-egs\n";

    ExampleMergingConfig merging_config("64");  // 64 is default minibatch size.

    ParseOptions po(usage);
    merging_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    SequentialNnetDiscriminativeExampleReader example_reader(examples_rspecifier);
    NnetDiscriminativeExampleWriter example_writer(examples_wspecifier);

    merging_config.ComputeDerived();
    DiscriminativeExampleMerger merger(merging_config, &example_writer);
    for (; !example_reader.Done(); example_reader.Next()) {
      const NnetDiscriminativeExample &cur_eg = example_reader.Value();
      merger.AcceptExample(new NnetDiscriminativeExample(cur_eg));
    }
    // the merger itself prints the necessary diagnostics.
    merger.Finish();
    return merger.ExitStatus();
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
