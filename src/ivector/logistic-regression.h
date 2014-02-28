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
//#include "thread/kaldi-task-sequence.h"
#include <numeric>


namespace kaldi {

struct LogisticRegressionConfig {
  int32 max_steps;
  LogisticRegressionConfig(): max_steps(20) { }
  void Register(OptionsItf *po) {
    po->Register("max-steps", &max_steps,
                 "Maximum steps in L-BFGS.");
  }
};

class LogisticRegression {
  public:
    // xs and ys are the training data. Each row of xs is a vector
    // corresponding to the class label in the same row of ys. 
    // weights contains the trained parameters.
  void Train(const Matrix<double> &xs, const std::vector<int32> &ys,
             Matrix<double> *weights, int32 max_steps);
 
    // Calculates the posterior of the class label of input xs.
    // The rows of posteriors corresponds to the rows of xs: the 
    // individual data points to be evaluated. The columns of 
    // posteriors are the integer class labels.
    void GetPosteriors(const Matrix<double> &xs, Matrix<double> *posteriors);
 
    // Performs a step in the L-BFGS. This is mostly used internally
    // By Train() and for testing.   
    double DoStep(const Matrix<double> &xs, 
    Matrix<double> *xw, const std::vector<int32> &ys, 
                OptimizeLbfgs<double> *lbfgs);

    // Returns the objective function given the training data, xs, ys.
    // The gradient is also calculated, and returned in grad.
    double GetObjfAndGrad(const Matrix<double> &xs, 
                         const std::vector<int32> &ys, 
                         const Matrix<double> &xw, 
                         Matrix<double> *grad);

    void Write(std::ostream &os, bool binary) const;
    void Read(std::istream &is, bool binary);
    
    // Sets the weights. This is generalyl used for testing.
    void SetWeights(const Matrix<double> &weights);
  private:
    // Each row of weights_ corresponds to the class labels
    // and each column, a feature in the input vectors. 
    Matrix<double> weights_;
};

}
