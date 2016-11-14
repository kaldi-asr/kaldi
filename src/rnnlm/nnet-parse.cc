// nnet3/nnet-parse.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include <iomanip>
#include "rnnlm/nnet-parse.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-matrix.h"

namespace kaldi {
namespace rnnlm {

void PrintParameterStats(std::ostringstream &os,
                         const std::string &name,
                         const Vector<BaseFloat> &params,
                         bool include_mean) {
  os << std::setprecision(4);
  os << ", " << name << '-';
  if (include_mean) {
    BaseFloat mean = params.Sum() / params.Dim(),
        stddev = std::sqrt(VecVec(params, params) / params.Dim() - mean * mean);
    os << "{mean,stddev}=" << mean << ',' << stddev;
  } else {
    BaseFloat rms = std::sqrt(VecVec(params, params) / params.Dim());
    os << "rms=" << rms;
  }
  os << std::setprecision(6);  // restore the default precision.
}

void PrintParameterStats(std::ostringstream &os,
                         const std::string &name,
                         const Matrix<BaseFloat> &params,
                         bool include_mean) {
  os << std::setprecision(4);
  os << ", " << name << '-';
  int32 dim = params.NumRows() * params.NumCols();
  if (include_mean) {
    BaseFloat mean = params.Sum() / dim,
        stddev = std::sqrt(TraceMatMat(params, params, kTrans) / dim -
                           mean * mean);
    os << "{mean,stddev}=" << mean << ',' << stddev;
  } else {
    BaseFloat rms = std::sqrt(TraceMatMat(params, params, kTrans) / dim);
    os << "rms=" << rms;
  }
  os << std::setprecision(6);  // restore the default precision.
}


} // namespace nnet3
} // namespace kaldi
