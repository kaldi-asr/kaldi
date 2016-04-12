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
        "--\n";
    ParseOptions po(usage);


    po.Read(argc, argv);


    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    int64 num_read = 0;

    float factor[10] = {0.0};
    BaseFloat alpha[10];
    BaseFloat probability_normalization_sum[10];
    for (int m = 1; m <= 10; m++) {
      alpha[m - 1] = Exp(-1.0f / (m * 10));
      probability_normalization_sum[m - 1] = alpha[m - 1] / (1 - alpha[m - 1]);////
    }

    for (; !example_reader.Done(); example_reader.Next()) {
      std::string key = example_reader.Key();
      const NnetExample &eg = example_reader.Value();
      Matrix<BaseFloat> output(eg.io[1].features.NumRows(),
                                 eg.io[1].features.NumCols(),
                                 kUndefined);
        eg.io[1].features.CopyToMat(&output);
        if (output.NumCols() == 1) {  // the objective type is log-normal
          int32 duration = static_cast<int32>(output(0, 0));
          for (int m = 1; m <= 10; m++)
            if (duration >= m*10) {
              factor[m - 1] += (duration - (m * 10) + 1) * Log(alpha[m - 1]) -
                                           Log(probability_normalization_sum[m - 1]);
            }
          num_read++;
        } else {
          /*SparseMatrix<BaseFloat> output_smat(
                                       eg.io[1].features.GetSparseMatrix());
          int32 num_rows = output_smat.NumRows();
          for (int32 row = 0; row < num_rows; row++) {
            int32 duration = output_smat.Row(row).GetElement(0).first + 1;
            num_read += num_rows;
            for (int m = 1; m <= 10; m++)
              if (duration >= m*10) {
                factor[m - 1] += (duration - (m * 10) + 1) * Log(alpha[m - 1]) -
                                             Log(probability_normalization_sum[m - 1]);
              }
          }*/
        }
      }

    KALDI_LOG << "Read " << num_read;
    for (int m=1; m <= 10; m++)
      KALDI_LOG << "Avg factor for max-dur=" << m * 10 << " is "<< factor[m - 1]/num_read;

    return (0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


