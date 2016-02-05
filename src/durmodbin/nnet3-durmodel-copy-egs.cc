// durmodbin/nnet3-durmodel-copy-egs.cc

// Copyright 2015 Hossein Hadian

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
#include "nnet3/nnet-example.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples for nnet3 duration model\n"
        "training, adding small random noise to duration values.\n"
        "\n"
        "Usage:  nnet3-durmodel-copy-egs [options] <egs-rspecifier>"
        " <egs-wspecifier1>\n"
        "e.g.:\n"
        "nnet3-durmodel-copy-egs --noise-magnitude 0.1 ark:train.egs "
        "ark,t:text.egs\n";

    int32 srand_seed = 0;
    BaseFloat noise_magnitude = 0.1;

    ParseOptions po(usage);
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("noise-magnitude", &noise_magnitude, "Magnitude of noise"
                " (in percentage) to be added to duration values.");


    po.Read(argc, argv);
    srand(srand_seed);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1),
                examples_wspecifier = po.GetArg(2);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);

    int64 num_read = 0, num_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      std::string key = example_reader.Key();
      const NnetExample &eg = example_reader.Value();
      NnetExample eg_out(eg);

      if (noise_magnitude != 0.0) {
        SparseMatrix<BaseFloat> output_smat(
                                       eg_out.io[1].features.GetSparseMatrix());
        int32 num_rows = output_smat.NumRows();
        for (int32 row = 0; row < num_rows; row++) {
          int32 duration = output_smat.Row(row).GetElement(0).first + 1;
          int32 num_cols = eg_out.io[1].features.NumCols();

          int32 duration_lower = 0.5 + duration * (1.0 - noise_magnitude),
                duration_upper = 0.5 + duration * (1.0 + noise_magnitude),
                new_duration = RandInt(duration_lower, duration_upper);
          if (new_duration < 1)
            new_duration = 1;
          if (new_duration > num_cols)
            new_duration = num_cols;

          std::vector<std::pair<MatrixIndexT, BaseFloat> > output_elements;
          output_elements.push_back(std::make_pair(new_duration - 1, 1.0f));
          SparseVector<BaseFloat> output(num_cols, output_elements);
          output_smat.SetRow(row, output);
        }
        eg_out.io[1].features = output_smat;
      }
      example_writer.Write(key, eg_out);
      num_written++;
    }

    KALDI_LOG << "Read " << num_read << " nnet examples, wrote "
              << num_written;
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


