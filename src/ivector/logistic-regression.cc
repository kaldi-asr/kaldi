// ivector/logistic-regression.cc

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


#include "ivector/logistic-regression.h"

namespace kaldi {

void LogisticRegression::Train(const Matrix<double> &xs, 
                               const std::vector<int32> &ys,
                               Matrix<double> *weights, int32 max_steps) {

  int32 xs_num_rows = xs.NumRows(), xs_num_cols = xs.NumCols(),
                     w_num_rows = weights->NumRows(), 
                     w_num_cols = weights->NumCols(),
                     num_ys = ys.size();
  KALDI_ASSERT(xs_num_cols == w_num_cols);
  KALDI_ASSERT(xs_num_rows == num_ys);
  
  // Adding on extra column for each x to handle the prior.
  Matrix<double> xs_with_prior(xs_num_rows, xs_num_cols + 1);
  SubMatrix<double> sub_xs(xs_with_prior, 0, xs_num_rows, 0, xs_num_cols);
  sub_xs.CopyFromMat(xs);
  Matrix<double> xw(xs_num_rows, w_num_rows);
  weights_.Resize(w_num_rows, w_num_cols + 1);

  // Adding on extra column for each x to handle the prior.
  for (int32 i = 0; i < xs_num_rows; i++) {
    xs_with_prior(i, xs_num_cols) = 1.0;
  }

  LbfgsOptions lbfgs_opts;
  lbfgs_opts.minimize = false;

  weights_.SetZero();

  // Get initial w vector
  Vector<double> init_w(weights_.NumRows() * weights_.NumCols());
  init_w.CopyRowsFromMat(weights_);
  OptimizeLbfgs<double> lbfgs(init_w, lbfgs_opts);

  for (int32 step = 0; step < max_steps; step++) {
    DoStep(xs_with_prior, &xw, ys, &lbfgs);
  }

  Vector<double> best_w(lbfgs.GetValue());
  weights_.CopyRowsFromVec(best_w);
  SubMatrix<double> trained_weights(weights_, 0, w_num_rows, 
                                    0, xs_num_cols);
  weights->SetZero();
  weights->CopyFromMat(trained_weights);
}

void LogisticRegression::GetPosteriors(const Matrix<double> &xs,
                   Matrix<double> *posteriors) {
  int32 xs_num_rows = xs.NumRows(),
        xs_num_cols = xs.NumCols();
  
  posteriors->Resize(xs_num_rows, weights_.NumRows());

  Matrix<double> xs_with_prior(xs_num_rows, xs_num_cols + 1);
  SubMatrix<double> sub_xs(xs_with_prior, 0, xs_num_rows, 0, xs_num_cols);
  sub_xs.CopyFromMat(xs);
  // Adding on extra column for each x to handle the prior.
  for (int32 i = 0; i < xs_num_rows; i++) {
    xs_with_prior(i, xs_num_cols) = 1.0;
  }
  posteriors->AddMatMat(1.0, xs_with_prior, kNoTrans, weights_, 
                        kTrans, 0.0);
  for (int32 i = 0; i < posteriors->NumRows(); i++) {
    posteriors->Row(i).ApplySoftMax();
    posteriors->Row(i).ApplyLog();
  }
}

double LogisticRegression::DoStep(const Matrix<double> &xs,
    Matrix<double> *xw,
    const std::vector<int32> &ys, OptimizeLbfgs<double> *lbfgs) {
  Matrix<double> gradient(weights_.NumRows(), weights_.NumCols());
  // Vector form of the above matrix
  Vector<double> grad_vec(weights_.NumRows() * weights_.NumCols());
    
  // Calculate XW.T. The rows correspond to the x
  // training examples and the columns to the class labels.
  xw->AddMatMat(1.0, xs, kNoTrans, weights_, kTrans, 0.0);

  // Calculate both the gradient and the objective function.
  double objf = GetObjfAndGrad(xs, ys, *xw, &gradient);

  // Convert gradient (a matrix) into a vector of size
  // gradient.NumCols * gradient.NumRows.
  grad_vec.CopyRowsFromMat(gradient);

  // Compute next step in L-BFGS.
  lbfgs->DoStep(objf, grad_vec);
  
  // Update weights
  Vector<double> new_w(lbfgs->GetProposedValue());
  weights_.CopyRowsFromVec(new_w);
  return objf;
}

double LogisticRegression::GetObjfAndGrad(const Matrix<double> &xs,
  const std::vector<int32> &ys, 
  const Matrix<double> &xw,
  Matrix<double> *grad) {
  double objf = 0.0;
  // For each training example class
  for (int32 i = 0; i < ys.size(); i++) {
    Vector<double> row(xw.NumCols());
    row.CopyFromVec(xw.Row(i));
    row.ApplySoftMax();
    objf += std::log(std::max(row(ys[i]), 1.0e-20));
    SubVector<double> x = xs.Row(i);
    // Iterate over the class labels
    for (int32 k = 0; k < weights_.NumRows(); k++) {
      // p(y = k | x_i)
      double p = row(k);
      if (k == ys[i]) {
        grad->Row(k).AddVec(1.0 - p, x);
      } else {
        grad->Row(k).AddVec(-1.0 * p, x);
      }
    }
  }
  grad->Scale(1.0/ys.size());
  return objf/ys.size();
}

void LogisticRegression::SetWeights(const Matrix<double> &weights) {
  weights_.Resize(weights.NumRows(), weights.NumCols());
  weights_.CopyFromMat(weights);
}

void LogisticRegression::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<LogisticRegression>");
  weights_.Write(os, binary);
  WriteToken(os, binary, "</LogisticRegression>");
}

void LogisticRegression::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<LogisticRegression>");
  weights_.Read(is, binary);
  ExpectToken(is, binary, "</LogisticRegression>");
}

}
