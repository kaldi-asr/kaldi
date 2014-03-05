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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include <numeric>


namespace kaldi {

struct LogisticRegressionConfig {
  int32 max_steps;
  double normalizer;
  LogisticRegressionConfig(): max_steps(20), normalizer(0.002) { }
  void Register(OptionsItf *po) {
    po->Register("max-steps", &max_steps,
                 "Maximum steps in L-BFGS.");
    po->Register("normalizer", &normalizer,
                 "Coefficient for L2 regularization.");
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


  // Returns the objective function given the training data, xs, ys.
  // The gradient is also calculated, and returned in grad. Uses
  // L2 regularization.
  BaseFloat GetObjfAndGrad(const Matrix<BaseFloat> &xs, 
                        const std::vector<int32> &ys, 
                        const Matrix<BaseFloat> &xw, 
                        Matrix<BaseFloat> *grad,
                        BaseFloat normalizer);
  
  // Sets the weights. This is generally used for testing.
  void SetWeights(const Matrix<BaseFloat> &weights);
 private:
  // Each row of weights_ corresponds to the class labels and each column, a
  // feature in the input vectors [the last column is the offset].
  Matrix<BaseFloat> weights_;    
};

}
