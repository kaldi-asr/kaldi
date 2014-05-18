// featbin/compare-feats.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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
        "Computes relative difference between two sets of features\n"
        "Can be used to figure out how different two sets of features are.\n"
        "Inputs must have same dimension.  Prints to stdout a similarity\n"
        "metric that is 1.0 if the features identical, and <1.0 otherwise.\n"
        "\n"
        "Usage: compare-feats [options] <in-rspecifier1> <in-rspecifier2>\n"
        "e.g.: compare-feats ark:1.ark ark:2.ark\n";

    ParseOptions po(usage);

    BaseFloat threshold = 0.99;
    po.Register("threshold", &threshold, "Similarity threshold, affects "
                "return status");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier1 = po.GetArg(1), rspecifier2 = po.GetArg(2);
    
    int32 num_done = 0, num_err = 0;
    double prod1 = 0.0, prod2 = 0.0, cross_prod = 0.0;
    
    SequentialBaseFloatMatrixReader feat_reader1(rspecifier1);
    RandomAccessBaseFloatMatrixReader feat_reader2(rspecifier2);

    for (; !feat_reader1.Done(); feat_reader1.Next()) {
      std::string utt = feat_reader1.Key();
      const Matrix<BaseFloat> &feat1 = feat_reader1.Value();
      if (!feat_reader2.HasKey(utt)) {
        KALDI_WARN << "Second table has no feature for utterance "
                   << utt;
        num_err++;
        continue;
      }
      const Matrix<BaseFloat> &feat2 = feat_reader2.Value(utt);
      if (!SameDim(feat1, feat2)) {
        KALDI_WARN << "Feature dimensions differ for utterance "
                   << utt << ", " << feat1.NumRows() << " by "
                   << feat1.NumCols() << " vs. " << feat2.NumRows()
                   << " by " << feat2.NumCols();
        num_err++;
        continue;
      }
      prod1 += TraceMatMat(feat1, feat1, kTrans);
      prod2 += TraceMatMat(feat2, feat2, kTrans);
      cross_prod += TraceMatMat(feat1, feat2, kTrans);
      num_done++;
    }

    KALDI_LOG << "Self-product of 1st features was " << prod1
              << ", self-product of 2nd features was " << prod2
              << ", cross-product was " << cross_prod;

    double similarity_metric = cross_prod / (0.5*prod1 + 0.5*prod2);
    KALDI_LOG << "Similarity metric is " << similarity_metric
              << " (1.0 means identical, the smaller the more different)";
    
    KALDI_LOG << "Processed " << num_done << " feature files, "
              << num_err << " had errors.";
    bool similar = (similarity_metric >= threshold);

    if (num_done > 0) {
      if (similar) {
        KALDI_LOG << "Features are considered similar since "
                  << similarity_metric << " >= " << threshold;
      } else {
        KALDI_LOG << "Features are considered dissimilar since "
                  << similarity_metric << " < " << threshold;
      }
    }

    std::cout << similarity_metric << std::endl;
    
    return (num_done > 0 && similar) ? 0 : 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


/*
  tested with:
 compare-feats 'ark:echo foo [ 1.0 2.0 ]|' 'ark:echo foo [ 1.0 2.0 ]|'
*/
