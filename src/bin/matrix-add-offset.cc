// bin/matrix-add-offset.cc

// Copyright 2015  Vimal Manohar

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
        "Add an offset vector to the rows of matrices in a table.\n"
        "\n"
        "Usage: matrix-add-offset [options] <matrix-rspecifier> "
        "<vector-wxfilename> <matrix-wspecifier>\n"
        "e.g.: matrix-add-offset log_post.mat neg_priors.vec log_like.mat\n"
        "See also: matrix-sum-rows, matrix-sum, vector-sum\n";


    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string rspecifier = po.GetArg(1);
    std::string vector_rxfilename = po.GetArg(2);
    std::string wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader mat_reader(rspecifier);
    BaseFloatMatrixWriter mat_writer(wspecifier);

    int32 num_done = 0;

    Vector<BaseFloat> vec;
    {
      bool binary_in;
      Input ki(vector_rxfilename, &binary_in);
      vec.Read(ki.Stream(), binary_in);
    }

    for (; !mat_reader.Done(); mat_reader.Next()) {
      std::string key = mat_reader.Key();
      Matrix<BaseFloat> mat(mat_reader.Value());
      if (vec.Dim() != mat.NumCols()) {
        KALDI_ERR << "Mismatch in vector dimension and "
                  << "number of columns in matrix; "
                  << vec.Dim() << " vs " << mat.NumCols();
      }
      mat.AddVecToRows(1.0, vec);
      mat_writer.Write(key, mat);
      num_done++;
    }

    KALDI_LOG << "Added offset to " << num_done << " matrices.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


