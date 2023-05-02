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
#include "nnet3/nnet-parse.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-matrix.h"

namespace kaldi {
namespace nnet3 {

bool DescriptorTokenize(const std::string &input,
                        std::vector<std::string> *tokens) {
  KALDI_ASSERT(tokens != NULL);
  size_t start = input.find_first_not_of(" \t"), size = input.size();
  tokens->clear();
  while (start < size) {
    KALDI_ASSERT(!isspace(input[start]));
    if (input[start] == '(' || input[start] == ')' || input[start] == ',') {
      tokens->push_back(std::string(input, start, 1));
      start = input.find_first_not_of(" \t", start + 1);
    } else {
      size_t found = input.find_first_of(" \t(),", start);
      KALDI_ASSERT(found != start);
      if (found == std::string::npos) {
        std::string str(input, start, input.size() - start);
        BaseFloat tmp;
        if (!IsValidName(str) && !ConvertStringToReal(str, &tmp)) {
          KALDI_WARN << "Could not tokenize line " << ErrorContext(std::string(input, start));
          return false;
        }
        tokens->push_back(str);
        break;
      } else {
        if (input[found] == '(' || input[found] == ')' || input[found] == ',') {
          std::string str(input, start, found - start);
          BaseFloat tmp;
          if (!IsValidName(str) && !ConvertStringToReal(str, &tmp)) {
            KALDI_WARN << "Could not tokenize line " << ErrorContext(std::string(input, start));
            return false;
          }
          tokens->push_back(str);
          start = found;
        } else {
          std::string str(input, start, found - start);
          BaseFloat tmp;
          if (!IsValidName(str) && !ConvertStringToReal(str, &tmp)) {
            KALDI_WARN << "Could not tokenize line " << ErrorContext(std::string(input, start));
            return false;
          }
          tokens->push_back(str);
          start = input.find_first_not_of(" \t", found);
        }
      }
    }
  }
  return true;
}

std::string ErrorContext(std::istream &is) {
  if (!is.good()) return "end of line";
  char buf[21];
  is.read(buf, 21);
  if (is) {
    return (std::string(buf, 20) + "...");
  }
  return std::string(buf, is.gcount());
}

std::string ErrorContext(const std::string &str) {
  if (str.size() == 0) return "end of line";
  if (str.size() <= 20) return str;
  return std::string(str, 0, 20) + "...";
}

static void PrintFloatSuccinctly(std::ostream &os, BaseFloat f) {
  if (fabs(f) < 10000.0 && fabs(f) >= 10.0) {
    os  << std::fixed << std::setprecision(0) << f;
  } else if (fabs(f) >= 0.995) {
    os  << std::fixed << std::setprecision(1) << f;
  } else if (fabs(f) >= 0.01) {
    os  << std::fixed << std::setprecision(2) << f;
  } else {
    os << std::setprecision(1) << f;
  }
  os.unsetf(std::ios_base::floatfield);
  os << std::setprecision(6);  // Restore the default.
}


// Returns a string that summarizes a vector fairly succintly, for
// printing stats in info lines.
std::string SummarizeVector(const VectorBase<float> &vec) {
  std::ostringstream os;
  if (vec.Dim() < 10) {
    os << "[ ";
    for (int32 i = 0; i < vec.Dim(); i++) {
      PrintFloatSuccinctly(os, vec(i));
      os << ' ';
    }
    os << "]";
  } else {
    // print out mean and standard deviation, and some selected values.
    BaseFloat mean = vec.Sum() / vec.Dim(),
        stddev = sqrt(VecVec(vec, vec) / vec.Dim() - mean * mean);

    std::string percentiles_str = "0,1,2,5 10,20,50,80,90 95,98,99,100";
    std::vector<int32> percentiles;
    bool ans = SplitStringToIntegers(percentiles_str, ", ", false,
                                     &percentiles);
    KALDI_ASSERT(ans);
    os << "[percentiles(" << percentiles_str << ")=(";
    Vector<BaseFloat> vec_sorted(vec);
    std::sort(vec_sorted.Data(), vec_sorted.Data() + vec_sorted.Dim());
    int32 n = vec.Dim() - 1;
    for (size_t i = 0; i < percentiles.size(); i++) {
      int32 percentile = percentiles[i];
      BaseFloat value = vec_sorted((n * percentile) / 100);
      PrintFloatSuccinctly(os, value);
      if (i + 1 < percentiles.size())
        os << (i == 3 || i == 8 ? ' ' : ',');
    }
    os << std::setprecision(3);
    os << "), mean=" << mean << ", stddev=" << stddev << "]";
  }
  return os.str();
}

std::string SummarizeVector(const VectorBase<double> &vec) {
  Vector<float> vec_copy(vec);
  return SummarizeVector(vec_copy);
}

std::string SummarizeVector(const CuVectorBase<BaseFloat> &cu_vec) {
  Vector<float> vec(cu_vec);
  return SummarizeVector(vec);
}

void PrintParameterStats(std::ostringstream &os,
                         const std::string &name,
                         const CuVectorBase<BaseFloat> &params,
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
                         const CuMatrix<BaseFloat> &params,
                         bool include_mean,
                         bool include_row_norms,
                         bool include_column_norms,
                         bool include_singular_values) {
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

  if (include_row_norms) {
    CuVector<BaseFloat> row_norms(params.NumRows());
    row_norms.AddDiagMat2(1.0, params, kNoTrans, 0.0);
    row_norms.ApplyPow(0.5);
    Vector<BaseFloat> row_norms_cpu;
    row_norms.Swap(&row_norms_cpu);
    os << ", " << name << "-row-norms="
       << SummarizeVector(row_norms_cpu);
  }
  if (include_column_norms) {
    CuVector<BaseFloat> col_norms(params.NumCols());
    col_norms.AddDiagMat2(1.0, params, kTrans, 0.0);
    col_norms.ApplyPow(0.5);
    Vector<BaseFloat> col_norms_cpu;
    col_norms.Swap(&col_norms_cpu);
    os << ", " << name << "-col-norms="
       << SummarizeVector(col_norms_cpu);
  }
  if (include_singular_values) {
    Matrix<BaseFloat> params_cpu(params);
    Vector<BaseFloat> s(std::min(params.NumRows(), params.NumCols()));
    params_cpu.Svd(&s);
    std::string singular_values_str = SummarizeVector(s);
    os << ", " << name << "-singular-values=" << singular_values_str;
    std::ostringstream name_os;
  }
}


void ParseConfigLines(const std::vector<std::string> &lines,
                      std::vector<ConfigLine> *config_lines) {
  config_lines->resize(lines.size());
  for (size_t i = 0; i < lines.size(); i++) {
    bool ret = (*config_lines)[i].ParseLine(lines[i]);
    if (!ret) {
      KALDI_ERR << "Error parsing config line: " << lines[i];
    }
  }
}

bool NameMatchesPattern(const char *name, const char *pattern) {
  if (*pattern == '*') {
    return NameMatchesPattern(name, pattern + 1) ||
        (*name != '\0' && NameMatchesPattern(name + 1, pattern));
  } else if (*name == *pattern) {
    return (*name == '\0' || NameMatchesPattern(name + 1, pattern + 1));
  } else {
    return false;
  }
}



} // namespace nnet3
} // namespace kaldi
