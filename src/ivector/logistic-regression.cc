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
#include "gmm/model-common.h" // For GetSplitTargets()
#include <numeric> // For std::accumulate

namespace kaldi {

void LogisticRegression::Train(const Matrix<BaseFloat> &xs, 
                               const std::vector<int32> &ys,
                               const LogisticRegressionConfig &conf) {
  
  int32 xs_num_rows = xs.NumRows(), xs_num_cols = xs.NumCols(),
                     num_ys = ys.size();
  KALDI_ASSERT(xs_num_rows == num_ys);
  
  // Adding on extra column for each x to handle the prior.
  Matrix<BaseFloat> xs_with_prior(xs_num_rows, xs_num_cols + 1);
  SubMatrix<BaseFloat> sub_xs(xs_with_prior, 0, xs_num_rows, 0, xs_num_cols);
  sub_xs.CopyFromMat(xs);

  int32 num_classes = *std::max_element(ys.begin(), ys.end()) + 1;
  weights_.Resize(num_classes, xs_num_cols + 1);
  Matrix<BaseFloat> xw(xs_num_rows, num_classes);

  // Adding on extra column for each x to handle the prior.
  for (int32 i = 0; i < xs_num_rows; i++) {
    xs_with_prior(i, xs_num_cols) = 1.0;
  }

  // At the beginning of training we have no mixture components,
  // therefore class_ is the "identity" mapping, that is
  // class_[i] = i.
  for (int32 i = 0; i < num_classes; i++) {
    class_.push_back(i);
  }

  weights_.SetZero();
  TrainParameters(xs_with_prior, ys, conf, &xw);
  KALDI_LOG << 
    "Finished training parameters without mixture components." << std::endl;

  // If we are using mixture components, we add those components
  // in MixUp and retrain with the extra weights.
  if (conf.mix_up > num_classes) {
    MixUp(ys, num_classes, conf);
    Matrix<BaseFloat> xw(xs_num_rows, weights_.NumRows());
    TrainParameters(xs_with_prior, ys, conf, &xw);
    KALDI_LOG << 
      "Finished training mixture components." << std::endl;
  }
}


void LogisticRegression::MixUp(const std::vector<int32> &ys,
                               const int32 &num_classes,
                               const LogisticRegressionConfig &conf) {
  
  Vector<BaseFloat> counts(num_classes);
  for (int32 i = 0; i < ys.size(); i++) {
    counts(ys[i]) += 1.0;
  }

  // TODO: Figure out what min_count should be
  int32 min_count = 1;
  std::vector<int32> targets;
  GetSplitTargets(counts, conf.mix_up, conf.power, min_count, &targets);
  int32 new_dim = std::accumulate(targets.begin(), targets.end(),
                                  static_cast<int32>(0));

  KALDI_LOG << "Target number mixture components was " << conf.mix_up 
            << ". Training " << new_dim << " mixture components. " 
            << std::endl;

  int32 old_dim = weights_.NumRows(),
        num_components = old_dim,
        num_feats = weights_.NumCols();
                                       
  Matrix<BaseFloat> old_weights(weights_);
  weights_.Resize(new_dim, num_feats);
  SubMatrix<BaseFloat> sub_weights(weights_, 0, num_classes, 0, num_feats);
  // We need to retain the original weights
  sub_weights.CopyFromMat(old_weights);
  class_.resize(new_dim);
  // For each class i
  for (int32 i = 0; i < targets.size(); i++) {
    int32 mixes = targets[i];
    // We start at j = 1 since one copy of the components already
    // exists in weights_.
    for (int32 j = 1; j < mixes; j++) {
      int32 offset = num_components;
      weights_.Row(offset).CopyRowFromMat(weights_, i);
      Vector<BaseFloat> noise(num_feats);
      noise.SetRandn();
      weights_.Row(offset).AddVec(1.0e-05, noise);
      class_[offset] = i; // The class i maps to the row at offset
      num_components += 1;
    }
  }
}

void LogisticRegression::TrainParameters(const Matrix<BaseFloat> &xs,
    const std::vector<int32> &ys, const LogisticRegressionConfig &conf,
    Matrix<BaseFloat> *xw) {
  int32 max_steps = conf.max_steps;
  BaseFloat normalizer = conf.normalizer;
  LbfgsOptions lbfgs_opts;
  lbfgs_opts.minimize = false;
  // Get initial w vector
  Vector<BaseFloat> init_w(weights_.NumRows() * weights_.NumCols());
  init_w.CopyRowsFromMat(weights_);
  OptimizeLbfgs<BaseFloat> lbfgs(init_w, lbfgs_opts);

  for (int32 step = 0; step < max_steps; step++) {
    DoStep(xs, xw, ys, &lbfgs, normalizer);
  }

  Vector<BaseFloat> best_w(lbfgs.GetValue());
  weights_.CopyRowsFromVec(best_w);
}

void LogisticRegression::GetLogPosteriors(const Matrix<BaseFloat> &xs,
                                          Matrix<BaseFloat> *log_posteriors) {
  int32 xs_num_rows = xs.NumRows(),
      xs_num_cols = xs.NumCols(),
      num_mixes = weights_.NumRows();
  
  int32 num_classes = *std::max_element(class_.begin(), class_.end()) + 1;
  
  log_posteriors->Resize(xs_num_rows, num_classes);
  Matrix<BaseFloat> xw(xs_num_rows, num_mixes);

  Matrix<BaseFloat> xs_with_prior(xs_num_rows, xs_num_cols + 1);
  SubMatrix<BaseFloat> sub_xs(xs_with_prior, 0, xs_num_rows, 0, xs_num_cols);
  sub_xs.CopyFromMat(xs);
  // Adding on extra column for each x to handle the prior.
  for (int32 i = 0; i < xs_num_rows; i++) {
    xs_with_prior(i, xs_num_cols) = 1.0;
  }
  xw.AddMatMat(1.0, xs_with_prior, kNoTrans, weights_, 
               kTrans, 0.0);
  
  log_posteriors->Set(-std::numeric_limits<BaseFloat>::infinity());

  // i is the training example
  for (int32 i = 0; i < xs_num_rows; i++) {
    for (int32 j = 0; j < num_mixes; j++) {
      int32 k = class_[j];
      (*log_posteriors)(i,k) = LogAdd((*log_posteriors)(i,k), xw(i, j));
    }
    // Normalize the row.
    log_posteriors->Row(i).Add(-xw.Row(i).LogSumExp());
  }
}

void LogisticRegression::GetLogPosteriors(const Vector<BaseFloat> &x,
                                          Vector<BaseFloat> *log_posteriors) {
  int32 x_dim = x.Dim();
  int32 num_classes = *std::max_element(class_.begin(), class_.end()) + 1,
      num_mixes = weights_.NumRows();
  log_posteriors->Resize(num_classes);
  Vector<BaseFloat> xw(weights_.NumRows());

  Vector<BaseFloat> x_with_prior(x_dim + 1);
  SubVector<BaseFloat> sub_x(x_with_prior, 0, x_dim);
  sub_x.CopyFromVec(x);
  // Adding on extra element to handle the prior
  x_with_prior(x_dim) = 1.0;
  
  xw.AddMatVec(1.0, weights_, kNoTrans, x_with_prior, kNoTrans);

  log_posteriors->Set(-std::numeric_limits<BaseFloat>::infinity());
  
  for (int32 i = 0; i < num_mixes; i++) {
    int32 j = class_[i];
    (*log_posteriors)(j) = LogAdd((*log_posteriors)(j), xw(i));
  }
  log_posteriors->Add(-log_posteriors->LogSumExp());
}

BaseFloat LogisticRegression::DoStep(const Matrix<BaseFloat> &xs,
    Matrix<BaseFloat> *xw,
    const std::vector<int32> &ys, OptimizeLbfgs<BaseFloat> *lbfgs,
    BaseFloat normalizer) {
  Matrix<BaseFloat> gradient(weights_.NumRows(), weights_.NumCols());
  // Vector form of the above matrix
  Vector<BaseFloat> grad_vec(weights_.NumRows() * weights_.NumCols());
    
  // Calculate XW.T. The rows correspond to the x
  // training examples and the columns to the class labels.
  xw->AddMatMat(1.0, xs, kNoTrans, weights_, kTrans, 0.0);

  // Calculate both the gradient and the objective function.
  BaseFloat objf = GetObjfAndGrad(xs, ys, *xw, &gradient, normalizer);

  // Convert gradient (a matrix) into a vector of size
  // gradient.NumCols * gradient.NumRows.
  grad_vec.CopyRowsFromMat(gradient);

  // Compute next step in L-BFGS.
  lbfgs->DoStep(objf, grad_vec);
  
  // Update weights
  Vector<BaseFloat> new_w(lbfgs->GetProposedValue());
  weights_.CopyRowsFromVec(new_w);
  KALDI_LOG << "Objective function is " << objf;
  return objf;
}

BaseFloat LogisticRegression::GetObjfAndGrad(
    const Matrix<BaseFloat> &xs,
    const std::vector<int32> &ys, const Matrix<BaseFloat> &xw,
    Matrix<BaseFloat> *grad, BaseFloat normalizer) {
  BaseFloat raw_objf = 0.0;
  int32 num_classes = *std::max_element(ys.begin(), ys.end()) + 1;
  std::vector< std::vector<int32> > class_to_cols(num_classes, std::vector<int32>());
  for (int32 i = 0; i < class_.size(); i++) {
    class_to_cols[class_[i]].push_back(i);
  }
  // For each training example class
  for (int32 i = 0; i < ys.size(); i++) {
    Vector<BaseFloat> row(xw.NumCols());
    row.CopyFromVec(xw.Row(i));
    row.ApplySoftMax();
    // Identify the rows of weights_ (which are a set of columns in wx) 
    // which correspond to class ys[i]
    const std::vector<int32> &cols = class_to_cols[ys[i]];
    SubVector<BaseFloat> x = xs.Row(i);
    BaseFloat class_sum = 0.0;
    for (int32 j = 0; j < cols.size(); j++) {
      class_sum += row(cols[j]);
    }
    if (class_sum < 1.0e-20) class_sum = 1.0e-20;
    raw_objf += std::log(class_sum);
    // Iterate over weights for each component. If there are no
    // mixtures each row corresponds to a class.
    for (int32 k = 0; k < weights_.NumRows(); k++) {
      // p(y = k | x_i) where k is a component.
      BaseFloat p = row(k);
      if (class_[k] == ys[i]) {
        // If the classes aren't split into mixture components
        // then p/class_sum = 1.0.
        grad->Row(k).AddVec(p/class_sum - p, x);
      } else {
        grad->Row(k).AddVec(-1.0 * p, x);
      }
    }
  }
  // Scale and add regularization term.
  grad->Scale(1.0/ys.size());
  grad->AddMat(-1.0 * normalizer, weights_);
  raw_objf /= ys.size();
  BaseFloat regularizer = - 0.5 * normalizer 
                          * TraceMatMat(weights_, weights_, kTrans);
  KALDI_VLOG(2) << "Objf is " << raw_objf << " + " << regularizer
                << " = " << (raw_objf + regularizer);
  return raw_objf + regularizer;
}

void LogisticRegression::SetWeights(const Matrix<BaseFloat> &weights,
                                    const std::vector<int32> classes) {
  weights_.Resize(weights.NumRows(), weights.NumCols());
  weights_.CopyFromMat(weights);
  class_.resize(classes.size());
  for (int32 i = 0; i < class_.size(); i++) 
    class_[i] = classes[i];
}

void LogisticRegression::ScalePriors(const Vector<BaseFloat> &scales) {
  Vector<BaseFloat> log_scales(scales);
  log_scales.ApplyLog();

  for (int32 i = 0; i < weights_.NumRows(); i++)
    weights_(i, weights_.NumCols() - 1) += log_scales(class_[i]);
}

void LogisticRegression::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<LogisticRegression>");
  WriteToken(os, binary, "<weights>");
  weights_.Write(os, binary);
  WriteToken(os, binary, "<class>");
  WriteIntegerVector(os, binary, class_);
  WriteToken(os, binary, "</LogisticRegression>");
}

void LogisticRegression::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<LogisticRegression>");
  ExpectToken(is, binary, "<weights>");
  weights_.Read(is, binary);
  std::string token;
  ReadToken(is, binary, &token);
  if (token == "<class>") {
    ReadIntegerVector(is, binary, &class_);
  } else {
    int32 num_classes = weights_.NumRows();
    for (int32 i = 0; i < num_classes; i++) {
      class_.push_back(i);
    }
  }
  ExpectToken(is, binary, "</LogisticRegression>");
}

}
