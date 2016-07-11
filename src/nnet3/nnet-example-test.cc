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


void UnitTestNnetSplitExampleUsingSplitChunk() {
  struct NnetGenerationOptions gen_config;
  std::vector<std::string> configs;
  GenerateConfigSequenceStatePreservingLstm(gen_config, &configs);
  Nnet nnet;
  std::istringstream is(configs[0]);
  nnet.ReadConfig(is);

  for (int32 k = 0; k < 50; k++) {
    int32 num_supervised_frames_after_split = RandInt(1, 10),
                   num_examples_after_split = RandInt(1, 10),
         num_supervised_frames_before_split = 
	 num_supervised_frames_after_split * num_examples_after_split,
                               left_context = RandInt(0, 5),
                              right_context = RandInt(0, 5),
                         chunk_left_context = RandInt(0, 20),
                        chunk_right_context = RandInt(0, 20),
                                  input_dim = RandInt(1, 10),
                                 output_dim = RandInt(5, 10),
                                ivector_dim = RandInt(-1, 2);

    int32 num_egs = RandInt(1, 5);
    // generate examples and then merge these examples to create a minibatch
    std::vector<NnetExample> egs_to_be_merged(num_egs);
    for (int32 n = 0; n < num_egs; n++) {
      NnetExample eg;
      GenerateSimpleNnetTrainingExample(num_supervised_frames_before_split,
                                        left_context + chunk_left_context,
                                        right_context + chunk_right_context,
                                        input_dim, output_dim, ivector_dim,
                                        &eg);
      egs_to_be_merged[n].Swap(&eg);
    }
    NnetExample eg_merged;
    bool compress = (RandInt(0, 1) == 0);
    MergeExamples(egs_to_be_merged, compress, &eg_merged);

    KALDI_LOG << k << "'th merged example (num_egs=" << num_egs
              << ", left_context=" << left_context
              << ", right_context=" << right_context
              << ", chunk_left_context=" << chunk_left_context
              << ", chunk_right_context=" << chunk_right_context
              << ", chunk_size=" << num_supervised_frames_before_split
              << ") to be splitted is: ";
    eg_merged.Write(std::cerr, false);
    std::vector<NnetExample> egs_splitted;
    SplitChunk(num_supervised_frames_after_split, left_context, right_context,
               eg_merged, nnet, &egs_splitted);
    // test if num of splitted examples agree
    KALDI_ASSERT(egs_splitted.size() == num_examples_after_split);

    const int32 num_input_frames_before_split = left_context + right_context +
                    num_supervised_frames_before_split +
                    chunk_left_context + chunk_right_context,
                num_supervised_frames_with_model_contexts_after_split =
                    left_context + right_context +
                    num_supervised_frames_after_split;
    KALDI_LOG << num_examples_after_split << " splitted examples (chunk_size="
              << num_supervised_frames_after_split << ") are: ";
    for (int32 f = 0; f < eg_merged.io.size(); f++) {
      Matrix<BaseFloat> feat;
      eg_merged.io[f].features.GetMatrix(&feat);
      for (int32 i = 0; i < egs_splitted.size(); i++) {
        if (f == 0) {
          KALDI_LOG << i << "'th:";
          egs_splitted[i].Write(std::cerr, false);
        }
        const std::vector<NnetIo> &io = egs_splitted[i].io;
        // test if the number of data and indexes in a splitted example agree
        KALDI_ASSERT(io[f].features.NumRows() == io[f].indexes.size());
        // test if io names are unchanged after split
        KALDI_ASSERT(io[f].name == eg_merged.io[f].name);

        // get data matrix for the f-th io of the i-th splitted example
        Matrix<BaseFloat> feat_splitted;
        io[f].features.GetMatrix(&feat_splitted);

        int32 row_offset = 0, num_rows = 0;
        for (int32 n = 0; n < num_egs; n++) {
          if (io[f].name == "input") {
            row_offset = n * num_input_frames_before_split +
                         i * num_supervised_frames_after_split +
                         (i == 0 ? 0 : chunk_left_context);
            num_rows = num_supervised_frames_with_model_contexts_after_split +
                (i == 0 ? chunk_left_context : 0) +
                (i == num_examples_after_split - 1 ? chunk_right_context : 0);
          } else if (io[f].name == "output") {
            row_offset = n * num_supervised_frames_before_split +
                         i * num_supervised_frames_after_split;
            num_rows = num_supervised_frames_after_split;
          } else if (io[f].name == "ivector") {
            row_offset = n;
            num_rows = 1;
          }
          // test if indexes are as expected after split
          for (int32 row = 0; row < num_rows; row++) {
            const std::vector<Index>::const_iterator iter_src =
                eg_merged.io[f].indexes.begin() + row_offset + row;
            const std::vector<Index>::const_iterator iter_dst =
                io[f].indexes.begin() + n * num_rows + row; 
            KALDI_ASSERT(iter_src->n == iter_dst->n);
            if (io[f].name == "ivector")
              KALDI_ASSERT(iter_src->t == iter_dst->t);
            else
              KALDI_ASSERT(iter_src->t == iter_dst->t +
                           i * num_supervised_frames_after_split);
          }
          SubMatrix<BaseFloat> feat_sub = feat.RowRange(row_offset, num_rows);
          SubMatrix<BaseFloat> feat_splitted_sub =
                               feat_splitted.RowRange(n * num_rows, num_rows);
          // test if the data matrices are correctly splitted
          KALDI_ASSERT(ApproxEqual(feat_sub, feat_splitted_sub,
                                   static_cast<BaseFloat>(0.0001)));
        }
      }
    } 
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;

  UnitTestNnetExample();
  UnitTestNnetMergeExamples();
  UnitTestNnetSplitExampleUsingSplitChunk();

  KALDI_LOG << "Nnet-example tests succeeded.";

  return 0;
}

