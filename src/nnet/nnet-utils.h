// nnet/nnet-utils.h

// Copyright 2015  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_UTILS_H_
#define KALDI_NNET_NNET_UTILS_H_

#include <iterator>
#include <algorithm>

#include "base/kaldi-common.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-array.h"
#include "hmm/posterior.h"
#include "hmm/transition-model.h"

namespace kaldi {
namespace nnet1 {


/**
 * Define stream insertion opeartor for 'std::vector', useful for log-prints,
 */
template <typename T> 
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os," "));
  return os;
}

/**
 * Convert basic type to string (try not to overuse as ostringstream creation is slow)
 */
template <typename T> 
std::string ToString(const T& t) { 
  std::ostringstream os; 
  os << t; 
  return os.str(); 
}

/**
 * Get a string with statistics of the data in a vector,
 * so we can print them easily.
 */
template <typename Real>
std::string MomentStatistics(const VectorBase<Real> &vec) {
  // we use an auxiliary vector for the higher order powers
  Vector<Real> vec_aux(vec);
  Vector<Real> vec_no_mean(vec); // vec with mean subtracted
  // mean
  Real mean = vec.Sum() / vec.Dim();
  // variance
  vec_aux.Add(-mean); vec_no_mean = vec_aux;
  vec_aux.MulElements(vec_no_mean); // (vec-mean)^2
  Real variance = vec_aux.Sum() / vec.Dim();
  // skewness 
  // - negative : left tail is longer, 
  // - positive : right tail is longer, 
  // - zero : symmetric
  vec_aux.MulElements(vec_no_mean); // (vec-mean)^3
  Real skewness = vec_aux.Sum() / pow(variance, 3.0/2.0) / vec.Dim();
  // kurtosis (peakedness)
  // - makes sense for symmetric distributions (skewness is zero)
  // - positive : 'sharper peak' than Normal distribution
  // - negative : 'heavier tails' than Normal distribution
  // - zero : same peakedness as the Normal distribution
  vec_aux.MulElements(vec_no_mean); // (vec-mean)^4
  Real kurtosis = vec_aux.Sum() / (variance * variance) / vec.Dim() - 3.0;
  // send the statistics to stream,
  std::ostringstream ostr;
  ostr << " ( min " << vec.Min() << ", max " << vec.Max()
       << ", mean " << mean 
       << ", variance " << variance 
       << ", skewness " << skewness
       << ", kurtosis " << kurtosis
       << " ) ";
  return ostr.str();
}

/**
 * Overload MomentStatistics to MatrixBase<Real>
 */
template <typename Real>
std::string MomentStatistics(const MatrixBase<Real> &mat) {
  Vector<Real> vec(mat.NumRows()*mat.NumCols());
  vec.CopyRowsFromMat(mat);
  return MomentStatistics(vec);
}

/**
 * Overload MomentStatistics to CuVectorBase<Real>
 */
template <typename Real>
std::string MomentStatistics(const CuVectorBase<Real> &vec) {
  Vector<Real> vec_host(vec.Dim());
  vec.CopyToVec(&vec_host);
  return MomentStatistics(vec_host);
}

/**
 * Overload MomentStatistics to CuMatrix<Real>
 */
template <typename Real>
std::string MomentStatistics(const CuMatrixBase<Real> &mat) {
  Matrix<Real> mat_host(mat.NumRows(),mat.NumCols());
  mat.CopyToMat(&mat_host);
  return MomentStatistics(mat_host);
}

/**
 * Check that matrix contains no nan or inf
 */
template <typename Real>
void CheckNanInf(const CuMatrixBase<Real> &mat, const char *msg = "") {
  Real sum = mat.Sum();
  if(KALDI_ISINF(sum)) { KALDI_ERR << "'inf' in " << msg; }
  if(KALDI_ISNAN(sum)) { KALDI_ERR << "'nan' in " << msg; }
}

/**
 * Get the standard deviation of values in the matrix
 */
template <typename Real>
Real ComputeStdDev(const CuMatrixBase<Real> &mat) {
  int32 N = mat.NumRows() * mat.NumCols();
  Real mean = mat.Sum() / N;
  CuMatrix<Real> pow_2(mat);
  pow_2.MulElements(mat);
  Real var = pow_2.Sum() / N - mean * mean;
  if (var < 0.0) {
    KALDI_WARN << "Forcing the variance to be non-negative! " << var << "->0.0";
    var = 0.0;
  }
  return sqrt(var);
}

/**
 * Convert Posterior to CuMatrix, 
 * the Posterior outer-dim defines number of matrix-rows,
 * number of matrix-colmuns is set by 'num_cols'.
 */
template <typename Real>
void PosteriorToMatrix(const Posterior &post, int32 num_cols, CuMatrix<Real> *mat) {
  // Make a host-matrix,
  int32 num_rows = post.size();
  Matrix<Real> m(num_rows, num_cols, kSetZero); // zero-filled
  // Fill from Posterior,
  for (int32 t = 0; t < post.size(); t++) {
    for (int32 i = 0; i < post[t].size(); i++) {
      int32 col = post[t][i].first;
      if (col >= num_cols) {
        KALDI_ERR << "Out-of-bound Posterior element with index " << col 
                  << ", higher than number of columns " << num_cols;
      }
      m(t, col) = post[t][i].second;
    }
  }
  // Copy to output GPU matrix,
  (*mat) = m; 
}

/**
 * Convert Posterior to CuMatrix, while mapping to PDFs. 
 * The Posterior outer-dim defines number of matrix-rows,
 * number of matrix-colmuns is set by 'TransitionModel::NumPdfs'.
 */
template <typename Real>
void PosteriorToMatrixMapped(const Posterior &post, const TransitionModel &model, CuMatrix<Real> *mat) {
  // Make a host-matrix,
  int32 num_rows = post.size(),
        num_cols = model.NumPdfs();
  Matrix<Real> m(num_rows, num_cols, kSetZero); // zero-filled
  // Fill from Posterior,
  for (int32 t = 0; t < post.size(); t++) {
    for (int32 i = 0; i < post[t].size(); i++) {
      int32 col = model.TransitionIdToPdf(post[t][i].first);
      if (col >= num_cols) {
        KALDI_ERR << "Out-of-bound Posterior element with index " << col 
                  << ", higher than number of columns " << num_cols;
      }
      m(t, col) += post[t][i].second; // sum,
    }
  }
  // Copy to output GPU matrix,
  (*mat) = m; 
}


} // namespace nnet1
} // namespace kaldi

#endif
