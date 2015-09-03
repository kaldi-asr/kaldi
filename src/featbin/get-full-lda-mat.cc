// featbin/get-full-lda-mat.cc

// Copyright 2012-2013  Johns Hopkins University (author: Daniel Povey)

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
#include "transform/transform-common.h"

namespace kaldi {
void IncreaseTransformDimension(int32 new_dimension,
                       Matrix<BaseFloat> *mat) {
  int32 d = mat->NumRows();
  if (new_dimension < d)
    KALDI_ERR << "--new-dimension argument invalid or not specified: "
              << new_dimension << " < " << d;
  if (mat->NumCols() == d) { // linear transform d->d
    mat->Resize(new_dimension, new_dimension, kCopyData);
    for (int32 i = d; i < new_dimension; i++)
      (*mat)(i, i) = 1.0; // set new dims to unit matrix.
  } else if (mat->NumCols() == d+1) { // affine transform d->d.
    Vector<BaseFloat> offset(mat->NumRows());
    offset.CopyColFromMat(*mat, d);
    mat->Resize(d, d, kCopyData); // remove offset from mat->
    mat->Resize(new_dimension, new_dimension+1, kCopyData); // extend with zeros.
    for (int32 i = d; i < new_dimension; i++)
      (*mat)(i, i) = 1.0; // set new dims to unit matrix.
    for (int32 i = 0; i < d; i++) // and set offset [last column]
      (*mat)(d, i) = offset(i);          
  } else {
    KALDI_ERR << "Input matrix has unexpected dimension " << d
              << " x " << mat->NumCols();
  }  
}

} // end namespace kaldi


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "This is a special-purpose program to be used in \"predictive SGMMs\".\n"
        "It takes in an LDA+MLLT matrix, and the original \"full\" LDA matrix\n"
        "as output by the --write-full-matrix option of est-lda; and it writes\n"
        "out a \"full\" LDA+MLLT matrix formed by the LDA+MLLT matrix plus the\n"
        "remaining rows of the \"full\" LDA matrix; and also writes out its inverse\n"
        "Usage: get-full-lda-mat [options] <lda-mllt-rxfilename> <full-lda-rxfilename> "
        "<full-lda-mllt-wxfilename> [<inv-full-lda-mllt-wxfilename>]\n"
        "E.g.: get-full-lda-mat final.mat full.mat full_lda_mllt.mat full_lda_mllt_inv.mat\n";
    
    bool binary = true;
    ParseOptions po(usage);

    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string lda_mllt_rxfilename = po.GetArg(1),
        full_lda_rxfilename = po.GetArg(2),
        full_lda_mllt_wxfilename = po.GetArg(3),
        inv_full_lda_mllt_wxfilename = po.GetOptArg(4);

    Matrix<BaseFloat> lda_mllt;
    ReadKaldiObject(lda_mllt_rxfilename, &lda_mllt);
    Matrix<BaseFloat> full_lda;
    ReadKaldiObject(full_lda_rxfilename, &full_lda);
    
    KALDI_ASSERT(full_lda.NumCols() == lda_mllt.NumCols());
    KALDI_ASSERT(full_lda.NumRows() == full_lda.NumCols());

    Matrix<BaseFloat> full_lda_mllt(full_lda);
    full_lda_mllt.Range(0, lda_mllt.NumRows(),
                        0, lda_mllt.NumCols()).CopyFromMat(lda_mllt);

    WriteKaldiObject(full_lda_mllt, full_lda_mllt_wxfilename, binary);

    if (po.NumArgs() != 3) {
      full_lda_mllt.Invert();
      WriteKaldiObject(full_lda_mllt, inv_full_lda_mllt_wxfilename, binary);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


