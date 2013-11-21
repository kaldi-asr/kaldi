// nnet2bin/nnet-combine-egs-discriminative.cc

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
        "Copy examples for discriminative neural network training,\n"
        "and combine successive examples if their combined length will\n"
        "be less than --max-length.  This can help to improve efficiency\n"
        "(--max-length corresponds to minibatch size)\n"
        "\n"
        "Usage:  nnet-combine-egs-discriminative [options] <egs-rspecifier> <egs-wspecifier>\n"
        "\n"
        "e.g.\n"
        "nnet-combine-egs-discriminative --max-length=512 ark:temp.1.degs ark:1.degs\n";
        
    int32 max_length = 512;
    int32 hard_max_length = 2048;
    int32 batch_size = 250;
    ParseOptions po(usage);
    po.Register("max-length", &max_length, "Maximum length of example that we "
                "will create when combining");
    po.Register("batch-size", &batch_size, "Size of batch used when combinging "
                "examples");
    po.Register("hard-max-length", &hard_max_length, "Length of example beyond "
                "which we will discard (very long examples may cause out of "
                "memory errors)");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(hard_max_length >= max_length);
    KALDI_ASSERT(batch_size >= 1);
    
    std::string examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    SequentialDiscriminativeNnetExampleReader example_reader(
        examples_rspecifier);
    DiscriminativeNnetExampleWriter example_writer(
        examples_wspecifier);

    int64 num_read = 0, num_written = 0, num_discarded = 0;

    while (!example_reader.Done()) {
      std::vector<DiscriminativeNnetExample> buffer;
      size_t size = batch_size;
      buffer.reserve(size);

      for (; !example_reader.Done() && buffer.size() < size;
           example_reader.Next()) {
        buffer.push_back(example_reader.Value());
        num_read++;
      }

      std::vector<DiscriminativeNnetExample> combined;
      CombineDiscriminativeExamples(max_length, buffer, &combined);
      buffer.clear();
      for (size_t i = 0; i < combined.size(); i++) {
        const DiscriminativeNnetExample &eg = combined[i];
        int32 num_frames = eg.input_frames.NumRows();
        if (num_frames > hard_max_length) {
          KALDI_WARN << "Discarding segment of length " << num_frames
                     << " because it exceeds --hard-max-length="
                     << hard_max_length;
          num_discarded++;
        } else {
          std::ostringstream ostr;
          ostr << (num_written++);
          example_writer.Write(ostr.str(), eg);
        }
      }
    }
    
    KALDI_LOG << "Read " << num_read << " discriminative neural-network training"
              << " examples, wrote " << num_written << ", discarded "
              << num_discarded;
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


