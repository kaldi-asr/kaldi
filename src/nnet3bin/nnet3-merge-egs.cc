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
// returns the number of indexes/frames in the NnetIo named "output" in the eg,
// or crashes if it is not there.
int32 NumOutputIndexes(const NnetExample &eg) {
  for (size_t i = 0; i < eg.io.size(); i++)
    if (eg.io[i].name == "output")
      return eg.io[i].indexes.size();
  KALDI_ERR << "No output named 'output' in the eg.";
  return 0;  // Suppress compiler warning.
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
        "single NnetExample.  Note: if --measure-output-frames=true, which it is by default,\n"
        "the --minibatch-size option will be interpreted as a target number of output frames;\n"
        "otherwise as a number of input examples to combine.  This makes a difference\n"
        "if the input examples have multiple supervised frames in them.\n"
        "\n"
        "Usage:  nnet3-merge-egs [options] <egs-rspecifier> <egs-wspecifier>\n"
        "e.g.\n"
        "nnet3-merge-egs --minibatch-size=512 ark:1.egs ark:- | nnet3-train-simple ... \n"
        "See also nnet3-copy-egs\n";

    bool compress = false;
    int32 minibatch_size = 512;
    bool measure_output_frames = true;
    bool discard_partial_minibatches = false;

    ParseOptions po(usage);
    po.Register("minibatch-size", &minibatch_size, "Target size of minibatches "
                "when merging (see also --measure-output-frames)");
    po.Register("measure-output-frames", &measure_output_frames, "If true, "
                "--minibatch-size is a target number of total output frames; if "
                "false, --minibatch-size is the number of input examples to "
                "merge.");
    po.Register("compress", &compress, "If true, compress the output examples "
                "(not recommended unless you are writing to disk)");
    po.Register("discard-partial-minibatches", &discard_partial_minibatches,
                "discard any partial minibatches of 'uneven' size that may be "
                "encountered at the end; 'true' is recommended, to avoid "
                "incurring compilation costs.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);

    std::vector<NnetExample> examples;
    examples.reserve(minibatch_size);

    int32 cur_num_output_frames = 0;

    int64 num_read = 0, num_written = 0;
    while (!example_reader.Done()) {
      const NnetExample &cur_eg = example_reader.Value();
      examples.resize(examples.size() + 1);
      examples.back() = cur_eg;
      cur_num_output_frames += NumOutputIndexes(cur_eg);
      bool minibatch_ready =
          (measure_output_frames ?
           cur_num_output_frames >= minibatch_size :
           static_cast<int32>(examples.size()) >= minibatch_size);

      // Do Next() now, so we can test example_reader.Done() below .
      example_reader.Next();
      num_read++;

      if (minibatch_ready || (!discard_partial_minibatches &&
                              (example_reader.Done() && !examples.empty()))) {
        NnetExample merged_eg;
        MergeExamples(examples, compress, &merged_eg);
        std::ostringstream ostr;
        ostr << "merged-" << num_written;
        num_written++;
        std::string output_key = ostr.str();
        example_writer.Write(output_key, merged_eg);
        examples.clear();
        cur_num_output_frames = 0;
      }
    }
    KALDI_LOG << "Merged " << num_read << " egs to " << num_written << '.';
    return (num_written != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
