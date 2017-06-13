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

#include <string>
#include <vector>
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
  std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os, " "));
  return os;
}

/**
 * Convert basic type to a string (please don't overuse),
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
  Vector<Real> vec_no_mean(vec);  // vec with mean subtracted
  // mean
  Real mean = vec.Sum() / vec.Dim();
  // variance
  vec_aux.Add(-mean);
  vec_no_mean = vec_aux;
  vec_aux.MulElements(vec_no_mean);  // (vec-mean)^2
  Real variance = vec_aux.Sum() / vec.Dim();
  // skewness
  // - negative : left tail is longer,
  // - positive : right tail is longer,
  // - zero : symmetric
  vec_aux.MulElements(vec_no_mean);  // (vec-mean)^3
  Real skewness = vec_aux.Sum() / pow(variance, 3.0/2.0) / vec.Dim();
  // kurtosis (peakedness)
  // - makes sense for symmetric distributions (skewness is zero)
  // - positive : 'sharper peak' than Normal distribution
  // - negative : 'heavier tails' than Normal distribution
  // - zero : same peakedness as the Normal distribution
  vec_aux.MulElements(vec_no_mean);  // (vec-mean)^4
  Real kurtosis = vec_aux.Sum() / (variance * variance) / vec.Dim() - 3.0;
  // send the statistics to stream,
  std::ostringstream ostr;
  ostr << " ( min " << vec.Min() << ", max " << vec.Max()
       << ", mean " << mean
       << ", stddev " << sqrt(variance)
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
  Matrix<Real> mat_host(mat.NumRows(), mat.NumCols());
  mat.CopyToMat(&mat_host);
  return MomentStatistics(mat_host);
}

/**
 * Check that matrix contains no nan or inf
 */
template <typename Real>
void CheckNanInf(const CuMatrixBase<Real> &mat, const char *msg = "") {
  Real sum = mat.Sum();
  if (KALDI_ISINF(sum)) { KALDI_ERR << "'inf' in " << msg; }
  if (KALDI_ISNAN(sum)) { KALDI_ERR << "'nan' in " << msg; }
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
 * Fill CuMatrix with random numbers (Gaussian distribution):
 * mu = the mean value,
 * sigma = standard deviation,
 *
 * Using the CPU random generator.
 */
template <typename Real>
void RandGauss(BaseFloat mu, BaseFloat sigma, CuMatrixBase<Real>* mat,
               struct RandomState* state = NULL) {
  // fill temporary matrix with 'Normal' samples,
  Matrix<Real> m(mat->NumRows(), mat->NumCols(), kUndefined);
  for (int32 r = 0; r < m.NumRows(); r++) {
    for (int32 c = 0; c < m.NumCols(); c++) {
      m(r, c) = RandGauss(state);
    }
  }
  // re-shape the distrbution,
  m.Scale(sigma);
  m.Add(mu);
  // export,
  mat->CopyFromMat(m);
}

/**
 * Fill CuMatrix with random numbers (Uniform distribution):
 * mu = the mean value,
 * range = the 'width' of the uniform PDF (spanning mu-range/2 .. mu+range/2)
 *
 * Using the CPU random generator.
 */
template <typename Real>
void RandUniform(BaseFloat mu, BaseFloat range, CuMatrixBase<Real>* mat,
                 struct RandomState* state = NULL) {
  // fill temporary matrix with '0..1' samples,
  Matrix<Real> m(mat->NumRows(), mat->NumCols(), kUndefined);
  for (int32 r = 0; r < m.NumRows(); r++) {
    for (int32 c = 0; c < m.NumCols(); c++) {
      m(r, c) = Rand(state) / static_cast<Real>(RAND_MAX);
    }
  }
  // re-shape the distrbution,
  m.Scale(range);  // 0..range,
  m.Add(mu - (range / 2.0));  // mu-range/2 .. mu+range/2,
  // export,
  mat->CopyFromMat(m);
}

/**
 * Fill CuVector with random numbers (Uniform distribution):
 * mu = the mean value,
 * range = the 'width' of the uniform PDF (spanning mu-range/2 .. mu+range/2)
 *
 * Using the CPU random generator.
 */
template <typename Real>
void RandUniform(BaseFloat mu, BaseFloat range, CuVectorBase<Real>* vec,
                 struct RandomState* state = NULL) {
  // fill temporary vector with '0..1' samples,
  Vector<Real> v(vec->Dim(), kUndefined);
  for (int32 i = 0; i < v.Dim(); i++) {
    v(i) = Rand(state) / static_cast<Real>(RAND_MAX);
  }
  // re-shape the distrbution,
  v.Scale(range);  // 0..range,
  v.Add(mu - (range / 2.0));  // mu-range/2 .. mu+range/2,
  // export,
  vec->CopyFromVec(v);
}


/**
 * Build 'integer vector' out of vector of 'matlab-like' representation:
 * 'b, b:e, b:s:e'
 *
 * b,e,s are integers, where:
 * b = beginning
 * e = end
 * s = step
 *
 * The sequence includes 'end', 1:3 => [ 1 2 3 ].
 * The 'step' has to be positive.
 */
inline void BuildIntegerVector(const std::vector<std::vector<int32> >& in,
                               std::vector<int32>* out) {
  // start with empty vector,
  out->clear();
  // loop over records,
  for (int32 i = 0; i < in.size(); i++) {
    // process i'th record,
    int32 beg = 0, end = 0, step = 1;
    switch (in[i].size()) {
      case 1:
        beg  = in[i][0];
        end  = in[i][0];
        step = 1;
        break;
      case 2:
        beg  = in[i][0];
        end  = in[i][1];
        step = 1;
        break;
      case 3:
        beg  = in[i][0];
        end  = in[i][2];
        step = in[i][1];
        break;
      default:
        KALDI_ERR << "Something is wrong! (should be 1-3) : "
                  << in[i].size();
    }
    // check the inputs,
    KALDI_ASSERT(beg <= end);
    KALDI_ASSERT(step > 0);  // positive,
    // append values to vector,
    for (int32 j = beg; j <= end; j += step) {
      out->push_back(j);
    }
  }
}

/**
 * Wrapper with 'CuArray<int32>' output.
 */
inline void BuildIntegerVector(const std::vector<std::vector<int32> >& in,
                               CuArray<int32>* out) {
  std::vector<int32> v;
  BuildIntegerVector(in, &v);
  (*out) = v;
}


/**
 * Wrapper of PosteriorToMatrix with CuMatrix argument.
 */
template <typename Real>
void PosteriorToMatrix(const Posterior &post,
                       const int32 post_dim, CuMatrix<Real> *mat) {
  Matrix<Real> m;
  PosteriorToMatrix(post, post_dim, &m);
  (*mat) = m;
}


/**
 * Wrapper of PosteriorToMatrixMapped with CuMatrix argument.
 */
template <typename Real>
void PosteriorToPdfMatrix(const Posterior &post,
                          const TransitionModel &model,
                          CuMatrix<Real> *mat) {
  Matrix<BaseFloat> m;
  PosteriorToPdfMatrix(post, model, &m);
  // Copy to output GPU matrix,
  (*mat) = m;
}


}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_UTILS_H_
