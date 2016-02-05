// nnet3/nnet-example-test.cc

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

#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-compile.h"
#include "nnet3/nnet-analyze.h"
#include "nnet3/nnet-test-utils.h"
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-compute.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"
#include "base/kaldi-math.h"

namespace kaldi {
namespace nnet3 {



void UnitTestNnetExample() {
  for (int32 n = 0; n < 50; n++) {

    NnetExample eg;
    int32 num_supervised_frames = RandInt(1, 10),
                   left_context = RandInt(0, 5),
                  right_context = RandInt(0, 5),
                      input_dim = RandInt(1, 10),
                     output_dim = RandInt(5, 10),
                    ivector_dim = RandInt(-1, 2);
    GenerateSimpleNnetTrainingExample(num_supervised_frames, left_context,
                                      right_context, input_dim, output_dim,
                                      ivector_dim, &eg);
    bool binary = (RandInt(0, 1) == 0);
    std::ostringstream os;
    eg.Write(os, binary);
    NnetExample eg_copy;
    if (RandInt(0, 1) == 0)
      eg_copy = eg;
    std::istringstream is(os.str());
    eg_copy.Read(is, binary);
    std::ostringstream os2;
    eg_copy.Write(os2, binary);
    if (binary) {
      KALDI_ASSERT(os.str() == os2.str());
      KALDI_ASSERT(eg_copy == eg);
    }
    KALDI_ASSERT(ExampleApproxEqual(eg, eg_copy, 0.1));
  }
}


void UnitTestNnetMergeExamples() {
  for (int32 n = 0; n < 50; n++) {
    int32 num_supervised_frames = RandInt(1, 10),
                   left_context = RandInt(0, 5),
                  right_context = RandInt(0, 5),
                      input_dim = RandInt(1, 10),
                     output_dim = RandInt(5, 10),
                    ivector_dim = RandInt(-1, 2);

    int32 num_egs = RandInt(1, 4);
    std::vector<NnetExample> egs_to_be_merged(num_egs);
    for (int32 i = 0; i < num_egs; i++) {
      NnetExample eg;
      // sometimes omit the ivector.  just tests things a bit more
      // thoroughly.
      GenerateSimpleNnetTrainingExample(num_supervised_frames, left_context,
                                        right_context, input_dim, output_dim,
                                        RandInt(0, 1) == 0 ? 0 : ivector_dim,
                                        &eg);
      KALDI_LOG << i << "'th example to be merged is: ";
      eg.Write(std::cerr, false);
      egs_to_be_merged[i].Swap(&eg);
    }
    NnetExample eg_merged;
    bool compress = (RandInt(0, 1) == 0);
    MergeExamples(egs_to_be_merged, compress, &eg_merged);
    KALDI_LOG << "Merged example is: ";
    eg_merged.Write(std::cerr, false);
  }
}



} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;

  UnitTestNnetExample();
  UnitTestNnetMergeExamples();

  KALDI_LOG << "Nnet-example tests succeeded.";

  return 0;
}

