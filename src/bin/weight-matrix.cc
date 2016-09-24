// bin/weight-matrix.cc

// Copyright 2016   Vimal Manohar

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Takes archives (typically per-utterance) of features and "
        "per-frame weights,\n"
        "and weights the features by the per-frame weights\n"
        "\n"
        "Usage: weight-matrix <matrix-rspecifier> <weights-rspecifier> "
        "<matrix-wspecifier>\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string matrix_rspecifier = po.GetArg(1),
               weights_rspecifier = po.GetArg(2),
                matrix_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader matrix_reader(matrix_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);
    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);

    int32 num_done = 0, num_err = 0;

    for (; !matrix_reader.Done(); matrix_reader.Next()) {
      std::string key = matrix_reader.Key();
      Matrix<BaseFloat> mat = matrix_reader.Value();
      if (!weights_reader.HasKey(key)) {
        KALDI_WARN << "No weight vectors for utterance " << key;
        num_err++;
        continue;
      }
      const Vector<BaseFloat> &weights = weights_reader.Value(key);
      if (weights.Dim() != mat.NumRows()) {
        KALDI_WARN << "Weights for utterance " << key
                   << " have wrong size, " << weights.Dim()
                   << " vs. " << mat.NumRows();
        num_err++;
        continue;
      }
      mat.MulRowsVec(weights);
      matrix_writer.Write(key, mat);
      num_done++;
    }
    KALDI_LOG << "Applied per-frame weights for " << num_done
              << " matrices; errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


