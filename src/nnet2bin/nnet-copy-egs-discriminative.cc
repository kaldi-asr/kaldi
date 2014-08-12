// nnet2bin/nnet-copy-egs-discriminative.cc

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
#include "nnet2/nnet-randomize.h"

namespace kaldi {
namespace nnet2 {
// returns an integer randomly drawn with expected value "expected_count"
// (will be either floor(expected_count) or ceil(expected_count)).
// this will go into an infinite loop if expected_count is very huge, but
// it should never be that huge.
int32 GetCount(double expected_count) {
  KALDI_ASSERT(expected_count >= 0.0);
  int32 ans = 0;
  while (expected_count > 1.0) {
    ans++;
    expected_count--;
  }
  if (WithProb(expected_count))
    ans++;
  return ans;
}

} // namespace nnet2
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples for discriminative neural\n"
        "network training.  Supports multiple wspecifiers, in\n"
        "which case it will write the examples round-robin to the outputs.\n"
        "\n"
        "Usage:  nnet-copy-egs-discriminative [options] <egs-rspecifier> <egs-wspecifier1> [<egs-wspecifier2> ...]\n"
        "\n"
        "e.g.\n"
        "nnet-copy-egs-discriminative ark:train.degs ark,t:text.degs\n"
        "or:\n"
        "nnet-copy-egs-discriminative ark:train.degs ark:1.degs ark:2.degs\n";
        
    bool random = false;
    int32 srand_seed = 0;
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
    
    po.Read(argc, argv);

    srand(srand_seed);
    
    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);

    SequentialDiscriminativeNnetExampleReader example_reader(
        examples_rspecifier);

    int32 num_outputs = po.NumArgs() - 1;
    std::vector<DiscriminativeNnetExampleWriter*> example_writers(num_outputs);
    for (int32 i = 0; i < num_outputs; i++)
      example_writers[i] = new DiscriminativeNnetExampleWriter(
          po.GetArg(i+2));

    
    int64 num_read = 0, num_written = 0, num_frames_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      int32 count = GetCount(keep_proportion);
      for (int32 c = 0; c < count; c++) {
        int32 index = (random ? Rand() : num_written) % num_outputs;
        std::ostringstream ostr;
        ostr << num_written;
        example_writers[index]->Write(ostr.str(),
                                      example_reader.Value());
        num_written++;
        num_frames_written +=
            static_cast<int64>(example_reader.Value().num_ali.size());
      }
    }
    
    for (int32 i = 0; i < num_outputs; i++)
      delete example_writers[i];
    KALDI_LOG << "Read " << num_read << " discriminative neural-network training"
              << " examples, wrote " << num_written << ", consisting of "
              << num_frames_written << " frames.";
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


