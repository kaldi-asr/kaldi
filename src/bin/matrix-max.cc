// featbin/matrix-mean.cc

// Copyright 2021  Desh Raj

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
#include "matrix/kaldi-matrix.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Compute element-wise max of given matrices (useful for some posterior\n"
        "combination methods). If number of rows differ in matrices, crop to\n"
        "smallest one. A spk2utt-like file must be provided to specify which\n"
        "sets of matrices to combine.\n"
        "Usage: matrix-max <spk2utt-rspecifier> <matrix-rspecifier> "
        "<matrix-wspecifier>\n"
        "e.g.: matrix-max ark:data/spk2utt scp:exp/output.scp ark:exp/outputs_max.ark\n";

    ParseOptions po(usage);
    bool binary_write = false;
    po.Register("binary", &binary_write, "If true, write output in binary "
                "(only applicable when writing files, not archives/tables).");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string spk2utt_rspecifier = po.GetArg(1),
        matrix_rspecifier = po.GetArg(2),
        matrix_wspecifier = po.GetArg(3);

    int64 num_spk_done = 0, num_spk_err = 0,
        num_utt_done = 0, num_utt_err = 0;

    RandomAccessBaseFloatMatrixReader matrix_reader(matrix_rspecifier);
    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    BaseFloatMatrixWriter matrix_writer(matrix_wspecifier);

    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      if (uttlist.empty()) {
        KALDI_WARN << "Speaker " << spk << " has no utterances. Not processing further.";
        continue;
      }
      Matrix<BaseFloat> spk_max;
      int32 utt_count = 0;

      // All matrices may be unequal size. First we drop last
      // rows of matrices and make them the same size.
      int32 min_length = std::numeric_limits<int32>::max();
      for (const std::string& utt : uttlist) {
        if (matrix_reader.HasKey(utt)) {
          min_length = std::min(min_length, matrix_reader.Value(utt).NumRows());
        }
      }
      for (const std::string& utt : uttlist) {
        if (!matrix_reader.HasKey(utt)) {
          KALDI_WARN << "No matrix present in input for utterance " << utt;
          num_utt_err++;
          continue;
        }
        if (utt_count == 0) {
          spk_max.Resize(min_length, matrix_reader.Value(utt).NumCols());
        }
        if (matrix_reader.Value(utt).NumRows() > min_length) {
          KALDI_WARN << "Cropping rows of matrix " << utt << " from "
                      << matrix_reader.Value(utt).NumRows() << " to "
                      << min_length;
        }
        SubMatrix<BaseFloat> temp(matrix_reader.Value(utt), 0, min_length,
            0, matrix_reader.Value(utt).NumCols());
        spk_max.Max(temp);
        num_utt_done++;
        utt_count++;
      }
      if (utt_count == 0) {
        KALDI_WARN << "Not producing output for speaker " << spk
                   << " since no utterances had matrices";
        num_spk_err++;
      } else {
        matrix_writer.Write(spk, spk_max);
        num_spk_done++;
      }
    }

    KALDI_LOG << "Computed max for " << num_spk_done << " speakers ("
              << num_spk_err << " with no utterances), consisting of "
              << num_utt_done << " utterances (" << num_utt_err
              << " absent from input).";

    return num_spk_done != 0 ? 0 : 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
