// featbin/extend-transform-dim.cc

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
      (*mat)(i, new_dimension) = offset(i);          
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
        "Read in transform from dimension d -> d (affine or linear), and output a transform\n"
        "from dimension e -> e (with e >= d, and e controlled by option --new-dimension).\n"
        "This new transform will leave the extra dimension unaffected, and transform the old\n"
        "dimensions in the same way.\n"
        "Usage: extend-transform-dim [options] (transform-A-rspecifier|transform-A-rxfilename) (transform-out-wspecifier|transform-out-wxfilename)\n"
        "E.g.: extend-transform-dim --new-dimension=117 in.mat big.mat\n";
    
    bool binary = true;
    int32 new_dimension = -1;
    ParseOptions po(usage);

    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("new-dimension", &new_dimension,
                "Larger dimension we are changing matrix to");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string transform_in_fn = po.GetArg(1);
    std::string transform_out_fn = po.GetArg(2);
    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(transform_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(transform_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);
    
    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Either none or both of the (input, output) must be a Table.";
    
    if (in_is_rspecifier) {
      SequentialBaseFloatMatrixReader reader(transform_in_fn);
      BaseFloatMatrixWriter writer(transform_out_fn);
      int32 num_done = 0;
      for (; !reader.Done(); reader.Next()) {
        std::string key = reader.Key();
        Matrix<BaseFloat> mat(reader.Value());
        IncreaseTransformDimension(new_dimension, &mat);
        writer.Write(key, mat);
        num_done++;
      }
      KALDI_LOG << "Increased transform dim to " << new_dimension
                << " for " << num_done << " matrices.";
      return (num_done != 0 ? 0 : 1);
    } else {
      Matrix<BaseFloat> mat;
      ReadKaldiObject(transform_in_fn, &mat);
      int32 old_dim = mat.NumRows();
      IncreaseTransformDimension(new_dimension, &mat);
      WriteKaldiObject(mat, transform_out_fn, binary);
      KALDI_LOG << "Increased transform dim from " << old_dim << " to "
                << mat.NumRows() << " and wrote to " << transform_out_fn;
      return 0;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

