// bin/matrix-sum.cc

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
        "Sum (and optionally scale) input archives of the same dimension\n"
        "\n"
        "Usage: matrix-sum [options] <matrix-rspecifier1> <matrix-rspecifier2> <sum-wspecifier>\n"
        " (this version sums two archives to a single archive)\n"
        "Or:  matrix-sum [options] <matrix-rspecifier> <sum-wxfilename>\n"
        " (this version sums up one archive to a single file)\n";

    bool binary = true;
    BaseFloat scale1 = 1.0, scale2 = 1.0;

    ParseOptions po(usage);

    po.Register("scale1", &scale1, "Scale applied to first archive");
    po.Register("scale2", &scale2, "Scale applied to second archive");
    po.Register("binary", &binary, "If true, write output in binary form (for "
                "single-archive form of program");
    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0;
      
    if (po.NumArgs() == 3) {
      std::string rspecifier1 = po.GetArg(1);
      std::string rspecifier2 = po.GetArg(2);
      std::string wspecifier = po.GetArg(3);
    
      SequentialBaseFloatMatrixReader mat1_reader(rspecifier1);

      RandomAccessBaseFloatMatrixReader mat2_reader(rspecifier2);
      BaseFloatMatrixWriter mat_writer(wspecifier);
    
      int32 num_err = 0;
      
      for (; !mat1_reader.Done(); mat1_reader.Next()) {
        std::string key = mat1_reader.Key();
        Matrix<BaseFloat> mat1 (mat1_reader.Value());
        if (!mat2_reader.HasKey(key)) {
          KALDI_WARN << "No such key " << key << " in second table.";
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &mat2 (mat2_reader.Value(key));
        if (!SameDim(mat1, mat2)) {
          KALDI_WARN << "Matrices for key " << key << " have different dims "
                     << mat1.NumRows() << " x " << mat1.NumCols() << " vs. "
                     << mat2.NumRows() << " x " << mat2.NumCols();
          num_err++;
          continue;
        }
        if (scale1 != 1.0) mat1.Scale(scale1);
        mat1.AddMat(scale2, mat2);
        mat_writer.Write(key, mat1);
        num_done++;
      }
      KALDI_LOG << "Added " << num_done << " matrices; " << num_err
                << " had errors.";
    } else { // NumArgs() == 2
      std::string rspecifier = po.GetArg(1),
          wxfilename = po.GetArg(2);
      if (scale1 != 1.0 || scale2 != 1.0)
        KALDI_ERR << "The --scale1 and --scale2 options are not supported in "
                  << "the two-argument form of this program.";
      SequentialBaseFloatMatrixReader mat_reader(rspecifier);
      Matrix<double> sum;
      for (; !mat_reader.Done(); mat_reader.Next(), num_done++) {
        Matrix<double> mat_dbl(mat_reader.Value());
        if (sum.NumRows() == 0) {
          sum = mat_dbl;
        } else {
          sum.AddMat(1.0, mat_dbl);
        }
      }
      WriteKaldiObject(sum, wxfilename, binary);
      KALDI_LOG << "Read " << num_done << " matrices, output sum to "
                << PrintableWxfilename(wxfilename);
    }
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


