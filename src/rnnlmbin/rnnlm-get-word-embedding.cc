// rnnlmbin/rnnlm-get-word-embedding.cc

// Copyright 2015-2017  Johns Hopkins University (author: Daniel Povey)

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
#include "rnnlm/rnnlm-training.h"
#include "rnnlm/rnnlm-example-utils.h"
#include "nnet3/nnet-utils.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::rnnlm;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This very simple program multiplies a sparse matrix by a\n"
        "dense matrix compute the word embedding (which is also a dense matrix).\n"
        "The sparse matrix is in a text format specific to the RNNLM tools.\n"
        "Usage:\n"
        " rnnlm-get-word-embedding [options] <sparse-word-features-rxfilename> \\\n"
        "   <feature-embedding-rxfilename> <word-embedding-wxfilename>\n"
        " e.g.:\n"
        " rnnlm-get-word-embedding word_features.txt feat_embedding.mat word_embedding.mat\n"
        "See also: rnnlm-get-egs, rnnlm-train\n";

    ParseOptions po(usage);

    bool binary = true;

    po.Register("binary", &binary, "If true, write output in binary format");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string word_features_rxfilename = po.GetArg(1),
        feature_embedding_rxfilename = po.GetArg(2),
        word_embedding_wxfilename = po.GetArg(3);


    Matrix<BaseFloat> feature_embedding_mat;
    ReadKaldiObject(feature_embedding_rxfilename,
                    &feature_embedding_mat);

    SparseMatrix<BaseFloat> word_feature_mat;
    {
      Input input(word_features_rxfilename);
      int32 feature_dim = feature_embedding_mat.NumRows();
      ReadSparseWordFeatures(input.Stream(), feature_dim,
                             &word_feature_mat);
    }


    Matrix<BaseFloat> word_embedding_mat(word_feature_mat.NumRows(),
                                         feature_embedding_mat.NumCols());

    word_embedding_mat.AddSmatMat(1.0, word_feature_mat, kNoTrans,
                                  feature_embedding_mat, 0.0);

    WriteKaldiObject(word_embedding_mat, word_embedding_wxfilename, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
