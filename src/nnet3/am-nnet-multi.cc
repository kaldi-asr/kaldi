// nnet3/am-nnet-simple.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet3/am-nnet-multi.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {

using std::ostringstream;

int32 AmNnetMulti::NumPdfs(int i) const {
  ostringstream os;
  os << i;
  int32 ans = nnet_.OutputDim("output" + os.str());
  KALDI_ASSERT(ans > 0);
  return ans;
}

void AmNnetMulti::Write(std::ostream &os, bool binary) const {
  // We don't write any header or footer like <AmNnetMulti> and </AmNnetMulti> -- we just
  // write the neural net and then the priors.  Who knows, there might be some
  // situation where we want to just read the neural net.
  nnet_.Write(os, binary);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context_);
  WriteToken(os, binary, "<RightContext>");
  WriteBasicType(os, binary, right_context_);
  WriteToken(os, binary, "<Priors>");
  WriteBasicType(os, binary, num_outputs_);
  for (int i = 0; i < num_outputs_; i++) {
    priors_vec_[i].Write(os, binary);
  }
}

void AmNnetMulti::Read(std::istream &is, bool binary) {
  nnet_.Read(is, binary);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context_);
  ExpectToken(is, binary, "<RightContext>");
  ReadBasicType(is, binary, &right_context_);
  ExpectToken(is, binary, "<Priors>");
  ReadBasicType(is, binary, &num_outputs_);
  priors_vec_.resize(num_outputs_);
  for (int i = 0; i < num_outputs_; i++) {
    priors_vec_[i].Read(is, binary);
  }
}

void AmNnetMulti::SetNnet(const Nnet &nnet) {
  // TODO(hxu)
  nnet_ = nnet;
  SetContext();
  int i = 0;
  ostringstream os;
  os << i;
  if (priors_vec_[i].Dim() != 0 &&
      priors_vec_[i].Dim() != nnet_.OutputDim("output" + os.str())) {
    KALDI_WARN << "Removing priors since there is a dimension mismatch after "
               << "changing the nnet: " << priors_vec_[i].Dim() << " vs. "
               << nnet_.OutputDim("output" + os.str());
    priors_vec_[i].Resize(0);
  }
}

void AmNnetMulti::SetPriors(const VectorBase<BaseFloat> &priors,
                            int i) {
  priors_vec_[i] = priors;

  ostringstream os;
  os << i;
  if (priors_vec_[i].Dim() != nnet_.OutputDim("output" + os.str())) {
    KALDI_ERR << "Dimension mismatch when setting priors: priors have dim "
              << priors_vec_[i].Dim() << ", model expects "
              << nnet_.OutputDim("output" + os.str());
  }
}

std::string AmNnetMulti::Info() const {
  // TODO(hxu)
  std::ostringstream ostr;
  ostr << "left-context: " << left_context_ << "\n";
  ostr << "right-context: " << right_context_ << "\n";
  ostr << "input-dim: " << nnet_.InputDim("input") << "\n";
  ostr << "ivector-dim: " << nnet_.InputDim("ivector") << "\n";
  ostr << "num-pdfs: " << nnet_.OutputDim("output") << "\n";
  int i = 0;
  ostr << "prior-dimension: " << priors_vec_[i].Dim() << "\n";
  if (priors_vec_[i].Dim() != 0) {
    int i = 0;
    ostr << "prior-sum: " << priors_vec_[i].Sum() << "\n";
    ostr << "prior-min: " << priors_vec_[i].Min() << "\n";
    ostr << "prior-max: " << priors_vec_[i].Max() << "\n";
  }
  ostr << "# Nnet info follows.\n" << "\n";
  return ostr.str() + nnet_.Info();
}


void AmNnetMulti::SetContext() {
  if (!IsMultiOutputNnet(nnet_)) {
    KALDI_ERR << "Class AmNnetMulti is only intended for a restricted type of "
              << "nnet, and this one does not meet the conditions.";
  }
  ComputeMultiNnetContext(nnet_,
                          num_outputs_,
                          &left_context_,
                          &right_context_);
}


} // namespace nnet3
} // namespace kaldi
