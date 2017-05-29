// ivectorbin/ivector-transform.cc

// Copyright 2013  Daniel Povey

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
#include "ivector/ivector-extractor.h"
#include "util/kaldi-thread.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Multiplies iVectors (on the left) by a supplied transformation matrix\n"
        "\n"
        "Usage:  ivector-transform [options] <matrix-in> <ivector-rspecifier>"
        "<ivector-wspecifier>\n"
        "e.g.: \n"
        " ivector-transform transform.mat ark:ivectors.ark ark:transformed_ivectors.ark\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string matrix_rxfilename = po.GetArg(1),
        ivector_rspecifier = po.GetArg(2),
        ivector_wspecifier = po.GetArg(3);


    Matrix<BaseFloat> transform;
    ReadKaldiObject(matrix_rxfilename, &transform);

    int32 num_done = 0;

    // The following quantities will be needed if we're doing
    // an affine transform (i.e. linear plus an offset)
    SubMatrix<BaseFloat> linear_term(transform,
                                     0, transform.NumRows(),
                                     0, transform.NumCols() - 1);
    Vector<BaseFloat> constant_term(transform.NumRows());
    constant_term.CopyColFromMat(transform, transform.NumCols() - 1);

    Vector<double> sum(transform.NumRows());
    double sumsq = 0.0;

    SequentialBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    BaseFloatVectorWriter ivector_writer(ivector_wspecifier);

    for (; !ivector_reader.Done(); ivector_reader.Next()) {
      std::string key = ivector_reader.Key();
      const Vector<BaseFloat> &ivector = ivector_reader.Value();

      Vector<BaseFloat> transformed_ivector(transform.NumRows());
      if (ivector.Dim() == transform.NumCols()) {
        transformed_ivector.AddMatVec(1.0, transform, kNoTrans, ivector, 0.0);
      } else {
        KALDI_ASSERT(ivector.Dim() == transform.NumCols() - 1);
        transformed_ivector.CopyFromVec(constant_term);
        transformed_ivector.AddMatVec(1.0, linear_term, kNoTrans, ivector, 1.0);
      }
      sum.AddVec(1.0, transformed_ivector);
      sumsq += VecVec(transformed_ivector, transformed_ivector);
      ivector_writer.Write(key, transformed_ivector);
      num_done++;
    }

    KALDI_LOG << "Processed " << num_done << " iVectors.";
    if (num_done != 0) {
      sum.Scale(1.0 / num_done);
      sumsq /= num_done;
      BaseFloat mean_length = sum.Norm(2.0),
          variance = sumsq - VecVec(sum, sum),
          avg_len = sqrt(variance),
          norm_length = avg_len / sqrt(transform.NumRows());
      KALDI_LOG << "Norm of mean was " << mean_length
                << " (should be close to zero), length divided by sqrt(dim) was "
                << norm_length << " (should probably be close to one)";
    }
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
