// ivector/logistic-regression.h

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

#ifndef KALDI_IVECTOR_LOGISTIC_REGRESSION_H_
#define KALDI_IVECTOR_LOGISTIC_REGRESSION_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include <numeric>


namespace kaldi {

struct LogisticRegressionConfig {
  int32 max_steps,
        mix_up;
  double normalizer,
         power;
  LogisticRegressionConfig(): max_steps(20), mix_up(0),
                              normalizer(0.0025), power(0.15){ }
  void Register(OptionsItf *opts) {
    opts->Register("max-steps", &max_steps,
                   "Maximum steps in L-BFGS.");
    opts->Register("normalizer", &normalizer,
                   "Coefficient for L2 regularization.");
    opts->Register("mix-up", &mix_up,
                   "Target number of mixture components to create, "
                   "if supplied.");
    opts->Register("power", &power,
                   "Power rule for determining the number of mixtures "
                   "to create.");
  }
};

class LogisticRegression {
 public:
  // xs and ys are the training data. Each row of xs is a vector
  // corresponding to the class label in the same row of ys.
  void Train(const Matrix<BaseFloat> &xs, const std::vector<int32> &ys,
             const LogisticRegressionConfig &conf);

  // Calculates the log posterior of the class label given the input xs.
  // The rows of log_posteriors corresponds to the rows of xs: the
  // individual data points to be evaluated. The columns of
  // log_posteriors are the integer class labels.
  void GetLogPosteriors(const Matrix<BaseFloat> &xs,
                        Matrix<BaseFloat> *log_posteriors);

  // Calculates the log posterior of the class label given the input x.
  // The indices of log_posteriors are the class labels.
  void GetLogPosteriors(const Vector<BaseFloat> &x,
                        Vector<BaseFloat> *log_posteriors);

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  void ScalePriors(const Vector<BaseFloat> &prior_scales);

 protected:
  void friend UnitTestTrain();
  void friend UnitTestPosteriors();

 private:
  // Performs a step in the L-BFGS. This is mostly used internally
  // By Train() and for testing.
  BaseFloat DoStep(const Matrix<BaseFloat> &xs,
                Matrix<BaseFloat> *xw, const std::vector<int32> &ys,
                OptimizeLbfgs<BaseFloat> *lbfgs,
                BaseFloat normalizer);

  void TrainParameters(const Matrix<BaseFloat> &xs,
             const std::vector<int32> &ys,
             const LogisticRegressionConfig &conf,
             Matrix<BaseFloat> *xw);

  // Creates the mixture components. Uses conf.mix_up, conf.power,
  // the occupancy of ys and GetSplitTargets() to determin the number
  // of mixture components for each weight index.
  void MixUp(const std::vector<int32> &ys, const int32 &num_classes,
             const LogisticRegressionConfig &conf);

  // Returns the objective function given the training data, xs, ys.
  // The gradient is also calculated, and returned in grad. Uses
  // L2 regularization.
  BaseFloat GetObjfAndGrad(const Matrix<BaseFloat> &xs,
                        const std::vector<int32> &ys,
                        const Matrix<BaseFloat> &xw,
                        Matrix<BaseFloat> *grad,
                        BaseFloat normalizer);

  // Sets the weights and class map. This is generally used for testing.
  void SetWeights(const Matrix<BaseFloat> &weights,
                  const std::vector<int32> classes);
  // Before mixture components or added, or if mix_up <= num_classes
  // each row of weights_ corresponds to a class label.
  // If mix_up > num_classes and after MixUp() is called the rows
  // correspond to the mixture components. In either case each column
  // corresponds to a feature in the input vectors (and the last column
  // is an offset).
  Matrix<BaseFloat> weights_;
  // Maps from the row of weights_ to the class.  Normally the
  // identity mapping, but may not be for multi-mixture logistic
  // regression.
  std::vector<int32> class_;
};

}

#endif
