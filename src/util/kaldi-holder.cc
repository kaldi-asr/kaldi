// util/kaldi-holder.cc

// Copyright 2009-2011     Microsoft Corporation
//                2016     Xiaohui Zhang

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

#include "util/kaldi-holder.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

template<class Real>
bool ExtractObjectRange(const Matrix<Real> &input, const std::string &range,
                        Matrix<Real> *output) {
  if (range.empty()) {
    KALDI_ERR << "Empty range specifier.";
    return false;
  }
  std::vector<std::string> splits;
  SplitStringToVector(range, ",", false, &splits);
  if (!((splits.size() == 1 && !splits[0].empty()) ||
        (splits.size() == 2  && !splits[0].empty() && !splits[1].empty()))) {
    KALDI_ERR << "Invalid range specifier: " << range;
    return false;
  }
  std::vector<int32> row_range, col_range;
  bool status = true;
  if (splits[0] != ":")
    status = SplitStringToIntegers(splits[0], ":", false, &row_range);
  if (splits.size() == 2 && splits[1] != ":") {
    status = status && SplitStringToIntegers(splits[1], ":", false, &col_range);
  }
  if (row_range.size() == 0) {
    row_range.push_back(0);
    row_range.push_back(input.NumRows() - 1);
  }
  if (col_range.size() == 0) {
    col_range.push_back(0);
    col_range.push_back(input.NumCols() - 1);
  }
  if (!(status && row_range.size() == 2 && col_range.size() == 2 &&
        row_range[0] >= 0 && row_range[0] <= row_range[1] &&
        row_range[1] < input.NumRows() && col_range[0] >=0 &&
        col_range[0] <= col_range[1] && col_range[1] < input.NumCols())) {
    KALDI_ERR << "Invalid range specifier: " << range
              << " for matrix of size " << input.NumRows()
              << "x" << input.NumCols();
    return false;
  }
  int32 row_size = row_range[1] - row_range[0] + 1,
        col_size = col_range[1] - col_range[0] + 1;
  output->Resize(row_size, col_size, kUndefined);
  output->CopyFromMat(input.Range(row_range[0], row_size,
                                  col_range[0], col_size));
  return true;
}

// template instantiation
template bool ExtractObjectRange(const Matrix<double> &, const std::string &,
                                 Matrix<double> *);
template bool ExtractObjectRange(const Matrix<float> &, const std::string &,
                                 Matrix<float> *);

bool ExtractRangeSpecifier(const std::string &rxfilename_with_range,
                           std::string *data_rxfilename,
                           std::string *range) {
  if (rxfilename_with_range.empty() ||
      rxfilename_with_range[rxfilename_with_range.size()-1] != ']')
    KALDI_ERR << "ExtractRangeRspecifier called wrongly.";
  std::vector<std::string> splits;
  SplitStringToVector(rxfilename_with_range, "[", false, &splits);
  if (splits.size() == 2 && !splits[0].empty() && splits[1].size() > 1) {
    *data_rxfilename = splits[0];
    range->assign(splits[1], 0, splits[1].size()-1);
    return true;
  }
  return false;
}

}  // end namespace kaldi
