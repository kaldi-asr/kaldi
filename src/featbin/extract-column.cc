// featbin/extract-column.cc

// Copyright 2015  Vimal Manohar (Johns Hopkins University)

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
    using namespace std;

    const char *usage =
        "Extract a column out of a matrix. \n"
        "This is most useful to extract log-energies \n"
        "from feature files\n"
        "\n"
        "Usage: extract-column [options] --column-index=<col-index> <features-rspecifier> <vector-wspecifier>\n"
        "  e.g. extract-column ark:feats-in.ark ark:energies.ark\n"
        "See also: select-feats, subset-feats, subsample-feats, extract-rows\n";
    
    ParseOptions po(usage);

    int32 column_index = 0;
    
    po.Register("column-index", &column_index,
                "Index of column to extract");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    string feat_rspecifier = po.GetArg(1);
    string vector_wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader reader(feat_rspecifier);
    BaseFloatVectorWriter writer(vector_wspecifier);

    int32 num_done = 0, num_err = 0;

    string line;

    for (; !reader.Done(); reader.Next(), num_done++) {
      const Matrix<BaseFloat>& feats(reader.Value());
      Vector<BaseFloat> col(feats.NumRows());
      if (column_index >= feats.NumCols()) {
        KALDI_ERR << "Column index " << column_index << " is "
                  << "not less than number of columns " << feats.NumCols();
      }
      col.CopyColFromMat(feats, column_index);
      writer.Write(reader.Key(), col);
    }

    KALDI_LOG << "Processed " << num_done << " segments successfully; "
              << "errors on " << num_err;

    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

