// bin/matrix-dot-product.cc

// Copyright  2016  Vimal Manohar

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
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Get element-wise dot product of matrices. Always returns a matrix "
        "that is the same size as the first matrix.\n"
        "If there is a mismatch in number of rows, the utterance is skipped, "
        "unless the mismatch is within a tolerance. If the second matrix has "
        "number of rows that is larger than the first matrix by less than the "
        "specified tolerance, then a submatrix of the second matrix is "
        "multiplied element-wise with the first matrix.\n"
        "\n"
        "Usage: matrix-dot-product [options] <matrix-in-rspecifier1> "
        "[<matrix-in-rspecifier2> ...<matrix-in-rspecifierN>] "
        "<matrix-out-wspecifier>\n"
        " e.g.: matrix-dot-product ark:1.weights ark:2.weights "
        "ark:combine.weights\n"
        "or \n"
        "Usage: matrix-dot-product [options] <matrix-in-rxfilename1> "
        "[<matrix-in-rxfilename2> ...<matrix-in-rxfilenameN>] "
        "<matrix-out-wxfilename>\n"
        " e.g.: matrix-sum --binary=false 1.mat 2.mat product.mat\n"
        "See also: matrix-sum, matrix-sum-rows\n";

    bool binary = true;
    int32 length_tolerance = 0;

    ParseOptions po(usage);

    po.Register("binary", &binary, "If true, write output as binary (only "
                "relevant for usage types two or three");
    po.Register("length-tolerance", &length_tolerance,
                "Tolerance length mismatch of this many frames");

    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 N = po.NumArgs();
    std::string matrix_in_fn1 = po.GetArg(1),
                matrix_out_fn = po.GetArg(N);

    if (ClassifyWspecifier(matrix_out_fn, NULL, NULL, NULL) != kNoWspecifier) {
      // output to table.

      // Output matrix
      BaseFloatMatrixWriter matrix_writer(matrix_out_fn);

      // Input matrices
      SequentialBaseFloatMatrixReader matrix_reader1(matrix_in_fn1);
      std::vector<RandomAccessBaseFloatMatrixReader*>
        matrix_readers(N-2,
            static_cast<RandomAccessBaseFloatMatrixReader*>(NULL));
      std::vector<std::string> matrix_in_fns(N-2);
      for (int32 i = 2; i < N; ++i) {
        matrix_readers[i-2] = new RandomAccessBaseFloatMatrixReader(
            po.GetArg(i));
        matrix_in_fns[i-2] = po.GetArg(i);
      }
      int32 n_utts = 0, n_total_matrices = 0,
          n_success = 0, n_missing = 0, n_other_errors = 0;

      for (; !matrix_reader1.Done(); matrix_reader1.Next()) {
        std::string key = matrix_reader1.Key();
        Matrix<BaseFloat> matrix1 = matrix_reader1.Value();
        matrix_reader1.FreeCurrent();
        n_utts++;
        n_total_matrices++;

        Matrix<BaseFloat> matrix_out(matrix1);

        int32 i = 0;
        for (i = 0; i < N-2; ++i) {
          bool failed = false;  // Indicates failure for this key.
          if (matrix_readers[i]->HasKey(key)) {
            const Matrix<BaseFloat> &matrix2 = matrix_readers[i]->Value(key);
            n_total_matrices++;
            if (SameDim(matrix2, matrix_out)) {
              matrix_out.MulElements(matrix2);
            } else {
              KALDI_WARN << "Dimension mismatch for utterance " << key
                         << " : " << matrix2.NumRows() << " by "
                         << matrix2.NumCols() << " for "
                         << "system " << (i + 2) << ", rspecifier: "
                         << matrix_in_fns[i] << " vs " << matrix_out.NumRows()
                         << " by " << matrix_out.NumCols()
                         << " primary matrix, rspecifier:" << matrix_in_fn1;
              if (matrix2.NumRows() - matrix_out.NumRows() <=
                  length_tolerance) {
                KALDI_WARN << "Tolerated length mismatch for key " << key;
                matrix_out.MulElements(matrix2.Range(0, matrix_out.NumRows(),
                                                     0, matrix2.NumCols()));
              } else {
                KALDI_WARN << "Skipping key " << key;
                failed = true;
                n_other_errors++;
              }
            }
          } else {
            KALDI_WARN << "No matrix found for utterance " << key << " for "
                       << "system " << (i + 2) << ", rspecifier: "
                       << matrix_in_fns[i];
            failed = true;
            n_missing++;
          }

          if (failed) break;
        }

        if (i != N-2)   // Skipping utterance
          continue;

        matrix_writer.Write(key, matrix_out);
        n_success++;
      }

      KALDI_LOG << "Processed " << n_utts << " utterances: with a total of "
                << n_total_matrices << " matrices across " << (N-1)
                << " different systems.";
      KALDI_LOG << "Produced output for " << n_success << " utterances; "
                << n_missing << " total missing matrices and skipped "
                << n_other_errors << "matrices.";

      DeletePointers(&matrix_readers);

      return (n_success != 0 && n_missing < (n_success - n_missing)) ? 0 : 1;
    } else {
      for (int32 i = 1; i < N; i++) {
        if (ClassifyRspecifier(po.GetArg(i), NULL, NULL) != kNoRspecifier) {
          KALDI_ERR << "Wrong usage: if last argument is not "
                    << "table, the other arguments must not be tables.";
        }
      }

      Matrix<BaseFloat> mat1;
      ReadKaldiObject(po.GetArg(1), &mat1);

      for (int32 i = 2; i < N; i++) {
        Matrix<BaseFloat> mat;
        ReadKaldiObject(po.GetArg(i), &mat);

        mat1.MulElements(mat);
      }

      WriteKaldiObject(mat1, po.GetArg(N), binary);
      KALDI_LOG << "Multiplied " << (po.NumArgs() - 1) << " matrices; "
                << "wrote product to " << PrintableWxfilename(po.GetArg(N));

      return 0;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

