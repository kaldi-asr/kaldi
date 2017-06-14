// chainbin/nnet3-chain-shuffle-egs.cc

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
#include "nnet3/nnet-chain-example.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy nnet3+chain examples for neural network training, from the input to output,\n"
        "while randomly shuffling the order.  This program will keep all of the examples\n"
        "in memory at once, unless you use the --buffer-size option\n"
        "\n"
        "Usage:  nnet3-chain-shuffle-egs [options] <egs-rspecifier> <egs-wspecifier>\n"
        "\n"
        "nnet3-chain-shuffle-egs --srand=1 ark:train.egs ark:shuffled.egs\n";

    int32 srand_seed = 0;
    int32 buffer_size = 0;
    ParseOptions po(usage);
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("buffer-size", &buffer_size, "If >0, size of a buffer we use "
                "to do limited-memory partial randomization.  Otherwise, do "
                "full randomization.");

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    int64 num_done = 0;

    std::vector<std::pair<std::string, NnetChainExample*> > egs;

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);
    NnetChainExampleWriter example_writer(examples_wspecifier);
    if (buffer_size == 0) { // Do full randomization
      // Putting in an extra level of indirection here to avoid excessive
      // computation and memory demands when we have to resize the vector.

      for (; !example_reader.Done(); example_reader.Next())
        egs.push_back(std::pair<std::string, NnetChainExample*>(
            example_reader.Key(),
            new NnetChainExample(example_reader.Value())));

      std::random_shuffle(egs.begin(), egs.end());
    } else {
      KALDI_ASSERT(buffer_size > 0);
      egs.resize(buffer_size,
                 std::pair<std::string, NnetChainExample*>("", NULL));
      for (; !example_reader.Done(); example_reader.Next()) {
        int32 index = RandInt(0, buffer_size - 1);
        if (egs[index].second == NULL) {
          egs[index] = std::pair<std::string, NnetChainExample*>(
              example_reader.Key(),
              new NnetChainExample(example_reader.Value()));
        } else {
          example_writer.Write(egs[index].first, *(egs[index].second));
          egs[index].first = example_reader.Key();
          *(egs[index].second) = example_reader.Value();
          num_done++;
        }
      }
    }
    for (size_t i = 0; i < egs.size(); i++) {
      if (egs[i].second != NULL) {
        example_writer.Write(egs[i].first, *(egs[i].second));
        delete egs[i].second;
        num_done++;
      }
    }

    KALDI_LOG << "Shuffled order of " << num_done
              << " neural-network training examples "
              << (buffer_size ? "using a buffer (partial randomization)" : "");

    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


