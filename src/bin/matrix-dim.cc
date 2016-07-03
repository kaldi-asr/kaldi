// bin/matrix-dim.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Print dimension info on an input matrix (rows then cols, separated by tab), to\n"
        "standard output.  Output for single filename: rows[tab]cols.  Output per line for\n"
        "archive of files: key[tab]rows[tab]cols\n"
        "Usage: matrix-dim [options] <matrix-in>|<in-rspecifier>\n"
        "e.g.: matrix-dim final.mat | cut -f 2\n"
        "See also: feat-to-len, feat-to-dim\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier) {
      std::string matrix_rspecifier = po.GetArg(1);
      SequentialBaseFloatMatrixReader matrix_reader(matrix_rspecifier);
      int32 num_read = 0;
      for (; !matrix_reader.Done(); matrix_reader.Next(), num_read++) {
        const Matrix<BaseFloat> &mat = matrix_reader.Value();
        std::cout << matrix_reader.Key() << '\t'
                  << mat.NumRows() << '\t' << mat.NumCols() << '\n';
      }
      if (num_read == 0)
        KALDI_WARN << "No features read from rspecifier '"
                   << matrix_rspecifier << "'";
      return (num_read == 0 ? 1 : 0);
    } else {
      std::string matrix_rxfilename = po.GetArg(1);
      Matrix<BaseFloat> mat;
      ReadKaldiObject(matrix_rxfilename, &mat);
      std::cout << mat.NumRows() << '\t' << mat.NumCols() << '\n';
      return 0;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


