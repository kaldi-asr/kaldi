// nnet2bin/nnet-select-egs.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet2/nnet-randomize.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Select a subset of the input examples, every k'th example of n.\n"
        "More precisely, numbering examples from 0, selects every example such\n"
        "that the number m of the example is equivalent to k modulo n (so k\n"
        "does not have to be < n)."
        "\n"
        "Usage:  nnet-select-egs [options] <egs-rspecifier> <egs-wspecifier1>\n"
        "\n"
        "e.g.\n"
        "nnet-select-egs --n=3 --k=1 ark:train.egs ark:-\n";
    
    int32 n = 1;
    int32 k = 0;
    
    ParseOptions po(usage);
    po.Register("k", &k, "Which number modulo n to take, with 0 <= k < n.");
    po.Register("n", &n, "Modulus (we'll take one in every n examples).");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(n > 0);
    
    std::string examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);

    
    int64 num_read = 0, num_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      if (num_read % n == k % n) {
        example_writer.Write(example_reader.Key(),
                             example_reader.Value());
        num_written++;
      }
    }

    KALDI_LOG << "Copied " << num_written << " of "
              << num_read << " neural-network training examples ";
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


