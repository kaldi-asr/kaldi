// gmmbin/gmm-transform-means-global.cc

// Copyright 2009-2011  Microsoft Corporation
//           2012-2014  Johns Hopkins University (author: Daniel Povey)

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
#include "gmm/diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "transform/mllt.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Transform GMM means with linear or affine transform\n"
        "This version for a single GMM, e.g. a UBM.\n"
        "Useful when estimating MLLT/STC\n"
        "Usage:  gmm-transform-means-global <transform-matrix> <gmm-in> <gmm-out>\n"
        "e.g.: gmm-transform-means-global 2.mat 2.dubm 3.dubm\n";

    bool binary = true;  // write in binary if true.

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string mat_rxfilename = po.GetArg(1),
        gmm_in_rxfilename = po.GetArg(2),
        gmm_out_wxfilename = po.GetArg(3);

    Matrix<BaseFloat> mat;
    ReadKaldiObject(mat_rxfilename, &mat);

    DiagGmm gmm;
    ReadKaldiObject(gmm_in_rxfilename, &gmm);
    
    int32 dim = gmm.Dim();
    if (mat.NumRows() != dim)
      KALDI_ERR << "Transform matrix has " << mat.NumRows() << " rows but "
          "model has dimension " << gmm.Dim();
    if (mat.NumCols() != dim
       && mat.NumCols()  != dim+1)
      KALDI_ERR << "Transform matrix has " << mat.NumCols() << " columns but "
          "model has dimension " << gmm.Dim() << " (neither a linear nor an "
          "affine transform";

    Matrix<BaseFloat> means;
    gmm.GetMeans(&means);
    Matrix<BaseFloat> new_means(means.NumRows(), means.NumCols());
    if (mat.NumCols() == dim) { // linear case
      // Right-multiply means by mat^T (equivalent to left-multiplying each
      // row by mat).
      new_means.AddMatMat(1.0, means, kNoTrans, mat, kTrans, 0.0);
    } else { // affine case
      Matrix<BaseFloat> means_ext(means.NumRows(), means.NumCols()+1);
      means_ext.Set(1.0);  // set all elems to 1.0
      SubMatrix<BaseFloat> means_part(means_ext, 0, means.NumRows(),
                                      0, means.NumCols());
      means_part.CopyFromMat(means);  // copy old part...
      new_means.AddMatMat(1.0, means_ext, kNoTrans, mat, kTrans, 0.0);
    }
    gmm.SetMeans(new_means);
    gmm.ComputeGconsts();
    
    WriteKaldiObject(gmm, gmm_out_wxfilename, binary);
    KALDI_LOG << "Written model to " << gmm_out_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


