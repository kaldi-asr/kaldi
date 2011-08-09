// gmm/diag-gmm-normal.cc

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Yanmin Qian

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

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "gmm/diag-gmm-normal.h"
#include "gmm/diag-gmm.h"

namespace kaldi {

void DiagGmmNormal::Resize(int32 nmix, int32 dim) {
  KALDI_ASSERT(nmix > 0 && dim > 0);
  if (weights_.Dim() != nmix) weights_.Resize(nmix);
  if (vars_.NumRows() != nmix ||
      vars_.NumCols() != dim) {
    vars_.Resize(nmix, dim);
    vars_.Set(1.0);
    // must be initialized to unit for case of calling SetMeans while having
    // covars/invcovars that are not set yet (i.e. zero)
  }
  if (means_.NumRows() != nmix ||
      means_.NumCols() != dim)
    means_.Resize(nmix, dim);
}

void DiagGmmNormal::CopyFromDiagGmm(const DiagGmm &diaggmm) {
  int32 num_comp = diaggmm.NumGauss(), dim = diaggmm.Dim();
  Resize(num_comp, dim);
  weights_.CopyFromVec(diaggmm.weights());
  vars_.CopyFromMat(diaggmm.inv_vars());
  vars_.InvertElements();
  means_.CopyFromMat(diaggmm.means_invvars());
  means_.MulElements(vars_);
}

void DiagGmmNormal::CopyToDiagGmm(DiagGmm *diaggmm) {
    assert((static_cast<int32>(diaggmm->Dim()) == Dim()) && (static_cast<int32>(diaggmm->NumGauss()) == NumGauss()));
    diaggmm->SetWeights(weights());
    Matrix<double> means(NumGauss(), Dim()), invvars(NumGauss(), Dim());
    means = means_;
    invvars = vars_;
    invvars.InvertElements();
    diaggmm->SetInvVarsAndMeans(invvars, means);
}

void DiagGmmNormal::RemoveComponent(int32 gauss, bool renorm_weights) {
  KALDI_ASSERT(gauss < NumGauss());
  if (NumGauss() == 1)
    KALDI_ERR << "Attempting to remove the only remaining component.";
  weights_.RemoveElement(gauss);
  means_.RemoveRow(gauss);
  vars_.RemoveRow(gauss);
  double sum_weights = weights_.Sum();
  if (renorm_weights) {
    weights_.Scale(1.0/sum_weights);
  }
}

void DiagGmmNormal::RemoveComponents(const std::vector<int32> &gauss_in,
                               bool renorm_weights) {
  std::vector<int32> gauss(gauss_in);
  std::sort(gauss.begin(), gauss.end());
  KALDI_ASSERT(IsSortedAndUniq(gauss));
  // If efficiency is later an issue, will code this specially (unlikely).
  for (size_t i = 0; i < gauss.size(); i++) {
    RemoveComponent(gauss[i], renorm_weights);
    for (size_t j = i + 1; j < gauss.size(); j++)
      gauss[j]--;
  }
}

void DiagGmmNormal::Write(std::ostream &out_stream, bool binary) const {
  WriteMarker(out_stream, binary, "<DiagGMMNormal>");
  if (!binary) out_stream << "\n";
  WriteMarker(out_stream, binary, "<WEIGHTS>");
  weights_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<MEANS>");
  means_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "<VARS>");
  vars_.Write(out_stream, binary);
  WriteMarker(out_stream, binary, "</DiagGMMNormal>");
  if (!binary) out_stream << "\n";
}

std::ostream & operator <<(std::ostream & out_stream,
                           const kaldi::DiagGmmNormal &gmm) {
  gmm.Write(out_stream, false);
  return out_stream;
}

void DiagGmmNormal::Read(std::istream &in_stream, bool binary) {
//  ExpectMarker(in_stream, binary, "<DiagGMMBegin>");
  std::string marker;
  ReadMarker(in_stream, binary, &marker);
  // <DiagGMMBegin> is for compatibility. Will be deleted later
  if (marker != "<DiagGMMBegin>" && marker != "<DiagGMMNormal>")
    KALDI_ERR << "Expected <DiagGMMNormal>, got " << marker;
  ReadMarker(in_stream, binary, &marker);
  ExpectMarker(in_stream, binary, "<WEIGHTS>");
  weights_.Read(in_stream, binary);
  ExpectMarker(in_stream, binary, "<MEANS>");
  means_.Read(in_stream, binary);
  ExpectMarker(in_stream, binary, "<VARS>");
  vars_.Read(in_stream, binary);
//  ExpectMarker(in_stream, binary, "<DiagGMMEnd>");
  ReadMarker(in_stream, binary, &marker);
  // <DiagGMMEnd> is for compatibility. Will be deleted later
  if (marker != "<DiagGMMEnd>" && marker != "</DiagGMMNormal>")
    KALDI_ERR << "Expected </DiagGMMNormal>, got " << marker;
}

std::istream & operator >>(std::istream & rIn, kaldi::DiagGmmNormal &gmm) {
  gmm.Read(rIn, false);  // false == non-binary.
  return rIn;
}

}  // End namespace kaldi
