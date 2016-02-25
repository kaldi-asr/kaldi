// chainbin/nnet3-chain-copy-egs.cc

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

namespace kaldi {
// returns an integer randomly drawn with expected value "expected_count"
// (will be either floor(expected_count) or ceil(expected_count)).
int32 GetCount(double expected_count) {
  KALDI_ASSERT(expected_count >= 0.0);
  int32 ans = floor(expected_count);
  expected_count -= ans;
  if (WithProb(expected_count))
    ans++;
  return ans;
}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples for nnet3+chain network training, possibly changing the binary mode.\n"
        "Supports multiple wspecifiers, in which case it will write the examples\n"
        "round-robin to the outputs.\n"
        "\n"
        "Usage:  nnet3-chain-copy-egs [options] <egs-rspecifier> <egs-wspecifier1> [<egs-wspecifier2> ...]\n"
        "\n"
        "e.g.\n"
        "nnet3-chain-copy-egs ark:train.cegs ark,t:text.cegs\n"
        "or:\n"
        "nnet3-chain-copy-egs ark:train.cegs ark:1.cegs ark:2.cegs\n";

    bool random = false;
    int32 srand_seed = 0;
    int32 frame_shift = 0;
    int32 truncate_deriv_weights = 0;
    BaseFloat keep_proportion = 1.0;

    ParseOptions po(usage);
    po.Register("random", &random, "If true, will write frames to output "
                "archives randomly, not round-robin.");
    po.Register("keep-proportion", &keep_proportion, "If <1.0, this program will "
                "randomly keep this proportion of the input samples.  If >1.0, it will "
                "in expectation copy a sample this many times.  It will copy it a number "
                "of times equal to floor(keep-proportion) or ceil(keep-proportion).");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(only relevant if --random=true or --keep-proportion != 1.0)");
    po.Register("frame-shift", &frame_shift, "Allows you to shift time values "
                "in the supervision data (excluding iVector data) - useful in "
                "augmenting data.  Note, the outputs will remain at the closest "
                "exact multiples of the frame subsampling factor");
    po.Register("truncate-deriv-weights", &truncate_deriv_weights,
                "If nonzero, the number of initial/final subsample frames that "
                "will have their derivatives' weights set to zero.");

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);

    int32 num_outputs = po.NumArgs() - 1;
    std::vector<NnetChainExampleWriter*> example_writers(num_outputs);
    for (int32 i = 0; i < num_outputs; i++)
      example_writers[i] = new NnetChainExampleWriter(po.GetArg(i+2));

    std::vector<std::string> exclude_names; // names we never shift times of;
                                            // not configurable for now.
    exclude_names.push_back(std::string("ivector"));


    int64 num_read = 0, num_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      // count is normally 1; could be 0, or possibly >1.
      int32 count = GetCount(keep_proportion);
      std::string key = example_reader.Key();
      if (frame_shift == 0 && truncate_deriv_weights == 0) {
        const NnetChainExample &eg = example_reader.Value();
        for (int32 c = 0; c < count; c++) {
          int32 index = (random ? Rand() : num_written) % num_outputs;
          example_writers[index]->Write(key, eg);
          num_written++;
        }
      } else if (count > 0) {
        NnetChainExample eg = example_reader.Value();
        if (frame_shift != 0)
          ShiftChainExampleTimes(frame_shift, exclude_names, &eg);
        if (truncate_deriv_weights != 0)
          TruncateDerivWeights(truncate_deriv_weights, &eg);
        for (int32 c = 0; c < count; c++) {
          int32 index = (random ? Rand() : num_written) % num_outputs;
          example_writers[index]->Write(key, eg);
          num_written++;
        }
      }
    }
    for (int32 i = 0; i < num_outputs; i++)
      delete example_writers[i];
    KALDI_LOG << "Read " << num_read
              << " neural-network training examples, wrote " << num_written;
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


