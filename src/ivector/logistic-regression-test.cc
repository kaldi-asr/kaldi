// ivector/logistic-regression-test.cc

// Copyright 2014  David Snyder

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

#include <time.h>
#include "ivector/logistic-regression.h"

namespace kaldi {

void UnitTestPosteriors() {
  int32 n_features = rand() % 600 + 10,
        n_xs = rand() % 200 + 100,
        n_labels = rand() % 20 + 10;

  LogisticRegressionConfig conf;
  conf.max_steps = 20;
  conf.normalizer = 0.001;

  Matrix<BaseFloat> xs(n_xs, n_features);
  xs.SetRandn();
  Matrix<BaseFloat> weights(n_labels, n_features + 1);
  weights.SetRandn();
  LogisticRegression classifier = LogisticRegression();
  std::vector<int32> classes;
  for (int32 i = 0; i < weights.NumRows(); i++) {
    classes.push_back(i);
  }
  classifier.SetWeights(weights, classes);
  
  // Get posteriors for the xs using batch and serial methods.
  Matrix<BaseFloat> batch_log_posteriors;
  classifier.GetLogPosteriors(xs, &batch_log_posteriors);
  Matrix<BaseFloat> log_posteriors(n_xs, n_labels);
  for (int32 i = 0; i < n_xs; i++) {
    Vector<BaseFloat> x(n_features);
    x.CopyRowFromMat(xs, i);
    Vector<BaseFloat> log_post;
    classifier.GetLogPosteriors(x, &log_post);
    
    // Verify that sum_y p(y|x) = 1.0.
    Vector<BaseFloat> post(log_post);
    post.ApplyExp();
    KALDI_ASSERT(ApproxEqual(post.Sum(), 1.0));
    log_posteriors.Row(i).CopyFromVec(log_post);
  }
  
  // Verify equivalence of batch and serial methods.
  float tolerance = 0.01;
  KALDI_ASSERT(log_posteriors.ApproxEqual(batch_log_posteriors, tolerance));
}

void UnitTestTrain() {

  int32 n_features = rand() % 600 + 10,
        n_xs = rand() % 200 + 100,
        n_labels = rand() % 20 + 10;
  double normalizer = 0.01;
  Matrix<BaseFloat> xs(n_xs, n_features);
  xs.SetRandn();

  std::vector<int32> ys;
  for (int32 i = 0; i < n_xs; i++) {
    ys.push_back(rand() % n_labels);
  }

  LogisticRegressionConfig conf;
  conf.max_steps = 20;
  conf.normalizer = normalizer;
  // Train the classifier
  LogisticRegression classifier = LogisticRegression();
  classifier.Train(xs, ys, conf);

  // Internally in LogisticRegression we add an additional element to
  // the x vectors: a 1.0 which handles the prior.
  Matrix<BaseFloat> xs_with_prior(n_xs, n_features + 1);
  for (int32 i = 0; i < n_xs; i++) {
    xs_with_prior(i, n_xs) = 1.0;
  }
  SubMatrix<BaseFloat> sub_xs(xs_with_prior, 0, n_xs, 0, n_features);
  sub_xs.CopyFromMat(xs);

  Matrix<BaseFloat> xw(n_xs, n_labels);
  xw.AddMatMat(1.0, xs_with_prior, kNoTrans, classifier.weights_, 
               kTrans, 0.0);

  Matrix<BaseFloat> grad(classifier.weights_.NumRows(),
                      classifier.weights_.NumCols());

  double objf_trained = classifier.GetObjfAndGrad(xs_with_prior, 
                                                  ys, xw, &grad, normalizer);

  // Calculate objective function using a random weight matrix.
  Matrix<BaseFloat> xw_rand(n_xs, n_labels);
  
  Matrix<BaseFloat> weights_rand(classifier.weights_);
  weights_rand.SetRandn();
  xw.AddMatMat(1.0, xs_with_prior, kNoTrans, weights_rand, 
               kTrans, 0.0);

  // Verify that the objective function after training is better
  // than the objective function with a random weight matrix.
  double objf_rand_w = classifier.GetObjfAndGrad(xs_with_prior, ys, 
                                                 xw_rand, &grad, normalizer);
  KALDI_ASSERT(objf_trained > objf_rand_w);
  KALDI_ASSERT(objf_trained > std::log(1.0 / n_xs));
}
}

int main() {
  using namespace kaldi;
  srand (time(NULL));
  UnitTestTrain();
  UnitTestPosteriors();
  return 0;
}
