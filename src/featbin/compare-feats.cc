// featbin/compare-feats.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2014  Mobvoi Inc. (author: Minhua Wu)

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
#include "matrix/kaldi-vector.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Computes relative difference between two sets of features\n"
        "per dimension and an average difference\n"
        "Can be used to figure out how different two sets of features are.\n"
        "Inputs must have same dimension.  Prints to stdout a similarity\n"
        "metric vector that is 1.0 per dimension if the features identical,\n"
        "and <1.0 otherwise, and an average overall similarity value.\n"
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
    
    int32 num_done = 0, num_err = 0, Dim = 0;
    Vector<double> prod1, prod2, cross_prod, similarity_metric;
    double overall_similarity = 0;

    SequentialBaseFloatMatrixReader feat_reader1(rspecifier1);
    RandomAccessBaseFloatMatrixReader feat_reader2(rspecifier2);

    for (; !feat_reader1.Done(); feat_reader1.Next()) {
      std::string utt = feat_reader1.Key();
      Matrix<BaseFloat> feat1 (feat_reader1.Value());


      if (!feat_reader2.HasKey(utt)) {
        KALDI_WARN << "Second table has no feature for utterance "
                   << utt;
        num_err++;
        continue;
      }
      Matrix<BaseFloat> feat2 (feat_reader2.Value(utt));
      if (feat1.NumCols() != feat2.NumCols()) {
        KALDI_WARN << "Feature dimensions differ for utterance "
                   << utt << ", " << feat1.NumCols() << " vs. "
                   << feat2.NumCols() << ", skipping  utterance."
                   << utt;
        num_err++;
        continue;
      }
      
      if (num_done == 0){
        Dim=feat1.NumCols();
        prod1.Resize(Dim);
        prod2.Resize(Dim);
        cross_prod.Resize(Dim);
        similarity_metric.Resize(Dim);
      }
      
      Vector<BaseFloat> feat1_col(feat1.NumRows()), feat2_col(feat2.NumRows());
      for (MatrixIndexT i = 0; i < feat1.NumCols(); i++){
        feat1_col.CopyColFromMat(feat1, i);
        feat2_col.CopyColFromMat(feat2, i);
        prod1(i) += VecVec(feat1_col, feat1_col);
        prod2(i) += VecVec(feat2_col, feat2_col);
        cross_prod(i) += VecVec(feat1_col, feat2_col);
      }
      num_done++;
    }

    KALDI_LOG << "self-product of 1st features for each column dimension: " << prod1;
    KALDI_LOG << "self-product of 2nd features for each column dimension: " << prod2;
    KALDI_LOG << "cross-product for each column dimension: " << cross_prod;

    prod1.AddVec(1.0, prod2);
    similarity_metric.AddVecDivVec(2.0, cross_prod, prod1, 0.0);
    KALDI_LOG << "Similarity metric for each dimension " << similarity_metric
              << " (1.0 means identical, the smaller the more different)";

    overall_similarity = similarity_metric.Sum() / static_cast<double>(Dim);

    KALDI_LOG << "Overall similarity for the two feats is:" << overall_similarity
              << " (1.0 means identical, the smaller the more different)";

    KALDI_LOG << "Processed " << num_done << " feature files, "
              << num_err << " had errors.";

    bool similar = (overall_similarity >= threshold);

    if (num_done > 0) {
      if (similar) {
        KALDI_LOG << "Features are considered similar since "
                  << overall_similarity << " >= " << threshold;
      } else {
        KALDI_LOG << "Features are considered dissimilar since "
                  << overall_similarity << " < " << threshold;
      }
    }

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
