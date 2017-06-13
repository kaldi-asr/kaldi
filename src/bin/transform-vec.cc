// bin/transform-vec.cc

// Copyright 2009-2012  Microsoft Corporation
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
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "This program applies a linear or affine transform to individual vectors, e.g.\n"
        "iVectors.  It is transform-feats, except it works on vectors rather than matrices,\n"
        "and expects a single transform matrix rather than possibly a table of matrices\n"
        "\n"
        "Usage: transform-vec [options] <transform-rxfilename> <feats-rspecifier> <feats-wspecifier>\n"
        "See also: transform-feats, est-pca\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string transform_rxfilename = po.GetArg(1);
    std::string vec_rspecifier = po.GetArg(2);
    std::string vec_wspecifier = po.GetArg(3);

    SequentialBaseFloatVectorReader vec_reader(vec_rspecifier);
    BaseFloatVectorWriter vec_writer(vec_wspecifier);
    
    Matrix<BaseFloat> transform;
    ReadKaldiObject(transform_rxfilename, &transform);

    int32 num_done = 0;
    
    for (; !vec_reader.Done(); vec_reader.Next()) {
      std::string key = vec_reader.Key();
      const Vector<BaseFloat> &vec(vec_reader.Value());

      int32 transform_rows = transform.NumRows(),
          transform_cols = transform.NumCols(),
          vec_dim = vec.Dim();
      
      Vector<BaseFloat> vec_out(transform_rows);

      if (transform_cols == vec_dim) {
        vec_out.AddMatVec(1.0, transform, kNoTrans, vec, 0.0);
      } else {
        if (transform_cols != vec_dim + 1) {
          KALDI_ERR << "Dimension mismatch: input vector has dimension "
                    << vec.Dim() << " and transform has " << transform_cols
                    << " columns.";
        }
        vec_out.CopyColFromMat(transform, vec_dim);
        vec_out.AddMatVec(1.0, transform.Range(0, transform.NumRows(),
                                               0, vec_dim), kNoTrans, vec, 1.0);
      }
      vec_writer.Write(key, vec_out);
      num_done++;
    }

    KALDI_LOG << "Applied transform to " << num_done << " vectors.";
    
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
