// rnnlm/rnnlm-training.h

// Copyright 2017  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_RNNLM_RNNLM_TRAIN_H_
#define KALDI_RNNLM_RNNLM_TRAIN_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "nnet3/nnet-nnet.h"

namespace kaldi {
namespace rnnlm {


class RnnlmTrainOptions {
  std::string rnnlm_rxfilename;

  RnnlmCoreTrainerOptions core_config;
  RnnlmEmbeddingTrainerOptions embedding_config;


  void Register(OptionsItf *po) {
    po->Register("read-rnnlm", &rnnlm_rxfilename,
                 "Read RNNLM from this location (e.g. 0.raw)");
    po->Register("write-rnnlm", &rnnlm_wxfilename,
                 "Write RNNLM to this location (e.g. 1.raw)."
                 "If not supplied, the core RNNLM is not trained "
                 "(but other parts of the model might be.");
    po->Register("read-dense-word-embedding", &word_embedding_rxfilename,
                 "Location to read dense word-embedding matrix, of dimension "
                 "(num-words, including epsilon) by (embedding-dim).  In this "
                 "case, the --read-sparse-word-features, "
                 "--read-feature-embedding and --write-feature-embedding "
                 "options are not expected.");
    po->Register("write-dense-word-embedding", &word_embedding_wxfilename,
                 "Location to write the dense word-embedding matrix after "
                 "training.  (Note: for most applications this won't be "
                 "used because we recommend the combination of "
                 "(sparse word-feature matrix) and (feature-embedding matrix)");
    po->Register("read-sparse-word-features", &word_features_rxfilename,
                 "Location to read sparse word-feature matrix, e.g. "
                 "word_feats.txt.  Format is lines like: '1  30 1.0 516 1.0':"
                 "starting with word-index, then a list of pairs "
                 "(feature-index, value) only including nonzero features. "
                 "This will usually be determined in an ad-hoc way based on "
                 "letters and other hand-built features; it's not trainable.");
    po->Register("read-feature-embedding", &feature_embedding_rxfilename,
                 "To be used only when --read-sparse-word-features is used, "
                 "a location to read the dense feature-embedding matrix of "
                 "dimension num-features by embedding-dim.  This matrix "
                 "is trainable (see --write-feature-embedding)");
    po->Register("write-feature-embedding", &feature_embedding_wxfilename,
                 "Location to write the feature-embedding matrix after training "
                 "it.");
    po->Register("matrix-lrate", &matrix_lrate,
                 "Learning rate for training the word-embedding or "
                 "feature-embedding  matrix, as applicable "
                 "(only matters if the --write-feature-embedding or "
                 "--write-dense-word-embedding option is supplied.)");
    po->Register("matrix-max-change", &matrix_max_change,
                 "Maximum parameter-change per minibatch for training "
                 "the word-embedding or feature-embedding  matrix, as "
                 "applicable (only matters if the --write-feature-embedding or "
                 "--write-dense-word-embedding option is supplied.)");





    // register the core RNNLM training options options with the prefix "rnnlm",
    // so they will appear as --rnnlm.max-change and the like.  This is done
    // with a prefix because later we may add a neural net to transform the word
    // embedding, and it would have options that would have a name conflict with
    // some of these options.
    ParseOptions core_opts("rnnlm", core_config);
    core_config.Register(core_opts);

    ParseOptions embedding_opts("embedding", embedding_config);
    core_config.Register(embedding_opts);
  }
};

/*
  The class Rnnlm represents a trained RNNLM model, but it doesn't contain
  everything you need because the "feature representation" of the vocabulary is
  stored separately, as a SparseMatrix.
*/

class Rnnlm {
 public:
  Rnnlm() { }

  Rnnlm(const Rnnlm &other):
    nnet_(other.nnet_),
    feature_embedding_(other.feature_embedding_) { }

  /**
     This constructor initialized the RNNLM with the given
     neural net and the number of features 'num_features', which
     might for instance be the number of letter trigrams in your data,
     or something like that.

     It initializes the embedding matrix randomly using the Glorot
     rule.  The embedding dimension (normally several hundred) is
     inferred from the input and output dimension of the neural net.
   */
  Rnnlm(const Nnet &nnet, int32 num_features);

  int32 EmbeddingDim() const { return feature_embedding_.NumCols(); }

  int32 NumFeatures() const { return feature_embedding_.NumRows(); }

  // Note: the on-disk format is just an Nnet followed by a SparseMatrix;
  // this ensures that it can be read by programs like nnet3-info that
  // expect just a raw Nnet object.
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  const Nnet &GetNnet() const { return nnet_; }

  Nnet &GetNnet() { return nnet_; }

  const CuMatrix<BaseFloat> &FeatureEmbedding() const { return feature_embedding_; }

  CuMatrix<BaseFloat> &FeatureEmbedding() { return feature_embedding_; }

  std::string Info() const;

 private:

  const Rnnlm &operator = (const Rnnlm &other); // Disallow.

  Nnet nnet_;

  // The feature-level embedding matrix.  The row-index is the "feature" index;
  // the column dimension is the embedding dimension.  A typical dimension might
  // be something like about ten thousand by 600.
  // Features will generally represent things like letter trigrams, but they
  // could be anything; the point is that each word is representable as a sparse
  // set of feature weights.  The word embedding vector is the appropriately
  // weighted combination of feature embedding vectors.
  CuMatrix<BaseFloat> feature_embedding_;
};


} // namespace rnnlm
} // namespace kaldi

#endif //KALDI_RNNLM_RNNLM_H_
