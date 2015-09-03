// nnet2bin/nnet-shuffle-egs-discriminative.cc

// Copyright 2012-2013  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet2/nnet-example-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples (typically single frames) for neural network training,\n"
        "from the input to output, but randomly shuffle the order.  This program will keep\n"
        "all of the examples in memory at once, so don't give it too many.\n"
        "\n"
        "Usage:  nnet-shuffle-egs-discriminative [options] <egs-rspecifier> <egs-wspecifier>\n"
        "\n"
        "nnet-shuffle-egs-discriminative --srand=1 ark:train.degs ark:shuffled.degs\n";
    
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

    std::vector<DiscriminativeNnetExample*> egs;
    SequentialDiscriminativeNnetExampleReader example_reader(
        examples_rspecifier);
    DiscriminativeNnetExampleWriter example_writer(
        examples_wspecifier);
    if (buffer_size == 0) { // Do full randomization
      // Putting in an extra level of indirection here to avoid excessive
      // computation and memory demands when we have to resize the vector.
    
      for (; !example_reader.Done(); example_reader.Next())
        egs.push_back(new DiscriminativeNnetExample(
            example_reader.Value()));
      
      std::random_shuffle(egs.begin(), egs.end());
    } else {
      KALDI_ASSERT(buffer_size > 0);
      egs.resize(buffer_size, NULL);
      for (; !example_reader.Done(); example_reader.Next()) {
        int32 index = RandInt(0, buffer_size - 1);
        if (egs[index] == NULL) {
          egs[index] = new DiscriminativeNnetExample(example_reader.Value());
        } else {
          std::ostringstream ostr;
          ostr << num_done;
          example_writer.Write(ostr.str(), *(egs[index]));
          *(egs[index]) = example_reader.Value();
          num_done++;
        }
      }      
    }
    for (size_t i = 0; i < egs.size(); i++) {
      std::ostringstream ostr;
      ostr << num_done;
      if (egs[i] != NULL) {
        example_writer.Write(ostr.str(), *(egs[i]));
        delete egs[i];
      }
      num_done++;
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


