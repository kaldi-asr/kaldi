// gmmbin/gmm-transform-means.cc

// Copyright 2009-2011  Microsoft Corporation
//           2012  Johns Hopkins University (author: Daniel Povey)

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
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "transform/mllt.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Transform GMM means with linear or affine transform\n"
        "Usage:  gmm-transform-means <transform-matrix> <model-in> <model-out>\n"
        "e.g.: gmm-transform-means 2.mat 2.mdl 3.mdl\n";

    bool binary = true;  // write in binary if true.

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string mat_rxfilename = po.GetArg(1),
        model_in_rxfilename = po.GetArg(2),
        model_out_wxfilename = po.GetArg(3);

    Matrix<BaseFloat> mat;
    ReadKaldiObject(mat_rxfilename, &mat);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    int32 dim = am_gmm.Dim();
    if (mat.NumRows() != dim)
      KALDI_ERR << "Transform matrix has " << mat.NumRows() << " rows but "
          "model has dimension " << am_gmm.Dim();
    if (mat.NumCols() != dim
       && mat.NumCols()  != dim+1)
      KALDI_ERR << "Transform matrix has " << mat.NumCols() << " columns but "
          "model has dimension " << am_gmm.Dim() << " (neither a linear nor an "
          "affine transform";

    for (int32 i = 0; i < am_gmm.NumPdfs(); i++) {
      DiagGmm &gmm = am_gmm.GetPdf(i);

      Matrix<BaseFloat> means;
      gmm.GetMeans(&means);
      Matrix<BaseFloat> new_means(means.NumRows(), means.NumCols());
      if (mat.NumCols() == dim) {  // linear case
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
    }

    {
      Output ko(model_out_wxfilename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written model to " << model_out_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


