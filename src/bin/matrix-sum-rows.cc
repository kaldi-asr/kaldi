// bin/matrix-sum-rows.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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
        "Sum the rows of an input table of matrices and output the corresponding\n"
        "table of vectors\n"
        "\n"
        "Usage: matrix-sum-rows [options] <matrix-rspecifier> <vector-wspecifier>\n"
        "e.g.: matrix-sum-rows ark:- ark:- | vector-sum ark:- sum.vec\n"
        "See also: matrix-sum, vector-sum\n";


    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);
    
    SequentialBaseFloatMatrixReader mat_reader(rspecifier);
    BaseFloatVectorWriter vec_writer(wspecifier);
    
    int32 num_done = 0;
    int64 num_rows_done = 0;
    
    for (; !mat_reader.Done(); mat_reader.Next()) {
      std::string key = mat_reader.Key();
      Matrix<double> mat(mat_reader.Value());
      Vector<double> vec(mat.NumCols());
      vec.AddRowSumMat(1.0, mat, 0.0);
      // Do the summation in double, to minimize roundoff.
      Vector<BaseFloat> float_vec(vec);
      vec_writer.Write(key, float_vec);
      num_done++;
      num_rows_done += mat.NumRows();
    }
    
    KALDI_LOG << "Summed rows " << num_done << " matrices, "
              << num_rows_done << " rows in total.";
    
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


