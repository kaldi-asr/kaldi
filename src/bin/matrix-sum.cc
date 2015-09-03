// bin/matrix-sum.cc

// Copyright  2012-2014  Johns Hopkins University (author: Daniel Povey)
//                 2014  Vimal Manohar

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

namespace kaldi {

// sums a bunch of archives to produce one archive
// for back-compatibility with an older form, we support scaling
// of the first two input archives.
int32 TypeOneUsage(const ParseOptions &po,
                   BaseFloat scale1,
                   BaseFloat scale2) {
  int32 num_args = po.NumArgs();
  std::string matrix_in_fn1 = po.GetArg(1),
      matrix_out_fn = po.GetArg(num_args);

  // Output matrix
  BaseFloatMatrixWriter matrix_writer(matrix_out_fn);

  // Input matrices
  SequentialBaseFloatMatrixReader matrix_reader1(matrix_in_fn1);
  std::vector<RandomAccessBaseFloatMatrixReader*>
      matrix_readers(num_args-2, 
                     static_cast<RandomAccessBaseFloatMatrixReader*>(NULL));
  std::vector<std::string> matrix_in_fns(num_args-2);
  for (int32 i = 2; i < num_args; ++i) {
    matrix_readers[i-2] = new RandomAccessBaseFloatMatrixReader(po.GetArg(i));
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

    matrix1.Scale(scale1);
    
    Matrix<BaseFloat> matrix_out(matrix1);

    for (int32 i = 0; i < num_args-2; ++i) {
      if (matrix_readers[i]->HasKey(key)) {
        Matrix<BaseFloat> matrix2 = matrix_readers[i]->Value(key);
        n_total_matrices++;
        if (SameDim(matrix2, matrix_out)) {
          BaseFloat scale = (i == 0 ? scale2 : 1.0);
          // note: i == 0 corresponds to the 2nd input archive.
          matrix_out.AddMat(scale, matrix2, kNoTrans);
        } else {
          KALDI_WARN << "Dimension mismatch for utterance " << key 
                     << " : " << matrix2.NumRows() << " by "
                     << matrix2.NumCols() << " for "
                     << "system " << (i + 2) << ", rspecifier: "
                     << matrix_in_fns[i] << " vs " << matrix_out.NumRows()
                     << " by " << matrix_out.NumCols()
                     << " primary matrix, rspecifier:" << matrix_in_fn1;
          n_other_errors++;
        }
      } else {
        KALDI_WARN << "No matrix found for utterance " << key << " for "
                   << "system " << (i + 2) << ", rspecifier: "
                   << matrix_in_fns[i];
        n_missing++;
      }
    }

    matrix_writer.Write(key, matrix_out);
    n_success++;
  }

  KALDI_LOG << "Processed " << n_utts << " utterances: with a total of "
            << n_total_matrices << " matrices across " << (num_args-1)
            << " different systems";
  KALDI_LOG << "Produced output for " << n_success << " utterances; "
            << n_missing << " total missing matrices";
  
  DeletePointers(&matrix_readers);
  
  return (n_success != 0 && n_missing < (n_success - n_missing)) ? 0 : 1;
}

int32 TypeTwoUsage(const ParseOptions &po,
                   bool binary) {
  KALDI_ASSERT(po.NumArgs() == 2);
  KALDI_ASSERT(ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier &&
               "matrix-sum: first argument must be an rspecifier");
  // if next assert fails it would be bug in the code as otherwise we shouldn't
  // be called.
  KALDI_ASSERT(ClassifyRspecifier(po.GetArg(2), NULL, NULL) == kNoRspecifier);

  SequentialBaseFloatMatrixReader mat_reader(po.GetArg(1));

  Matrix<double> sum;
  
  int32 num_done = 0, num_err = 0;

  for (; !mat_reader.Done(); mat_reader.Next()) {
    const Matrix<BaseFloat> &mat = mat_reader.Value();
    if (mat.NumRows() == 0) {
      KALDI_WARN << "Zero matrix input for key " << mat_reader.Key();
      num_err++;
    } else {
      if (sum.NumRows() == 0) sum.Resize(mat.NumRows(), mat.NumCols());
      if (sum.NumRows() != mat.NumRows() || sum.NumCols() != mat.NumCols()) {
        KALDI_WARN << "Dimension mismatch for key " << mat_reader.Key()
                   << ": " << mat.NumRows() << " by " << mat.NumCols() << " vs. "
                   << sum.NumRows() << " by " << sum.NumCols();
        num_err++;
      } else {
        Matrix<double> dmat(mat);
        sum.AddMat(1.0, dmat, kNoTrans);
        num_done++;
      }
    }
  }

  Matrix<BaseFloat> sum_float(sum);
  WriteKaldiObject(sum_float, po.GetArg(2), binary);

  KALDI_LOG << "Summed " << num_done << " matrices, "
            << num_err << " with errors; wrote sum to "
            << PrintableWxfilename(po.GetArg(2));
  return (num_done > 0 && num_err < num_done) ? 0 : 1;
}

// sum a bunch of single files to produce a single file [including
// extended filenames, of course]
int32 TypeThreeUsage(const ParseOptions &po,
                     bool binary) {
  KALDI_ASSERT(po.NumArgs() >= 2);
  for (int32 i = 1; i <= po.NumArgs(); i++) {
    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier) {
      KALDI_ERR << "Wrong usage (type 3): if first and last arguments are not "
                << "tables, the intermediate arguments must not be tables.";
    }
  }

  bool add = true;
  Matrix<BaseFloat> mat;
  for (int32 i = 1; i < po.NumArgs(); i++) {
    bool binary_in;
    Input ki(po.GetArg(i), &binary_in);
    // this Read function will throw if there is a size mismatch.
    mat.Read(ki.Stream(), binary_in, add);
  }
  WriteKaldiObject(mat, po.GetArg(po.NumArgs()), binary);
  KALDI_LOG << "Summed " << (po.NumArgs() - 1) << " matrices; "
            << "wrote sum to " << PrintableWxfilename(po.GetArg(po.NumArgs()));
  return 0;
}


} // namespace kaldi


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;


    const char *usage =
        "Add matrices (supports various forms)\n"
        "\n"
        "Type one usage:\n"
        " matrix-sum [options] <matrix-in-rspecifier1> [<matrix-in-rspecifier2>"
        " <matrix-in-rspecifier3> ...] <matrix-out-wspecifier>\n"
        "  e.g.: matrix-sum ark:1.weights ark:2.weights ark:combine.weights\n"
        "  This usage supports the --scale1 and --scale2 options to scale the\n"
        "  first two input tables.\n"
        "Type two usage (sums a single table input to produce a single output):\n"
        " matrix-sum [options] <matrix-in-rspecifier> <matrix-out-wxfilename>\n"
        " e.g.: matrix-sum --binary=false mats.ark sum.mat\n"
        "Type three usage (sums single-file inputs to produce a single output):\n"
        " matrix-sum [options] <matrix-in-rxfilename1> <matrix-in-rxfilename2> ..."
        " <matrix-out-wxfilename>\n"
        " e.g.: matrix-sum --binary=false 1.mat 2.mat 3.mat sum.mat\n"
        "See also: matrix-sum-rows\n";


    BaseFloat scale1 = 1.0, scale2 = 1.0;
    bool binary = true;

    ParseOptions po(usage);

    po.Register("scale1", &scale1, "Scale applied to first matrix "
                "(only for type one usage)");
    po.Register("scale2", &scale2, "Scale applied to second matrix "
                "(only for type one usage)");
    po.Register("binary", &binary, "If true, write output as binary (only "
                "relevant for usage types two or three");
    
    po.Read(argc, argv);
    
    int32 N = po.NumArgs(), exit_status;
    
    if (po.NumArgs() >= 2 &&
        ClassifyRspecifier(po.GetArg(N), NULL, NULL) != kNoRspecifier) {
      // output to table.
      exit_status = TypeOneUsage(po, scale1, scale2);
    } else if (po.NumArgs() == 2 &&
               ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier &&
               ClassifyRspecifier(po.GetArg(N), NULL, NULL) ==
               kNoRspecifier) {
      KALDI_ASSERT(scale1 == 1.0 && scale2 == 1.0);
      // input from a single table, output not to table.
      exit_status = TypeTwoUsage(po, binary);
    } else if (po.NumArgs() >= 2 &&
               ClassifyRspecifier(po.GetArg(1), NULL, NULL) == kNoRspecifier &&
               ClassifyRspecifier(po.GetArg(N), NULL, NULL) == kNoRspecifier) {
      KALDI_ASSERT(scale1 == 1.0 && scale2 == 1.0);
      // summing flat files.
      exit_status = TypeThreeUsage(po, binary);
    } else {      
      po.PrintUsage();
      exit(1);
    }
    return exit_status;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


