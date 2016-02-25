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

#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {



int32 AmNnetSimple::NumPdfs() const {
  int32 ans = nnet_.OutputDim("output");
  KALDI_ASSERT(ans > 0);
  return ans;
}

void AmNnetSimple::Write(std::ostream &os, bool binary) const {
  // We don't write any header or footer like <AmNnetSimple> and </AmNnetSimple> -- we just
  // write the neural net and then the priors.  Who knows, there might be some
  // situation where we want to just read the neural net.
  nnet_.Write(os, binary);
  WriteToken(os, binary, "<LeftContext>");
  WriteBasicType(os, binary, left_context_);
  WriteToken(os, binary, "<RightContext>");
  WriteBasicType(os, binary, right_context_);
  WriteToken(os, binary, "<Priors>");
  priors_.Write(os, binary);
}

void AmNnetSimple::Read(std::istream &is, bool binary) {
  nnet_.Read(is, binary);
  ExpectToken(is, binary, "<LeftContext>");
  ReadBasicType(is, binary, &left_context_);
  ExpectToken(is, binary, "<RightContext>");
  ReadBasicType(is, binary, &right_context_);
  SetContext();  // temporarily, I'm not trusting the written ones (there was
                 // briefly a bug)
  ExpectToken(is, binary, "<Priors>");
  priors_.Read(is, binary);
}

void AmNnetSimple::SetNnet(const Nnet &nnet) {
  nnet_ = nnet;
  SetContext();
  if (priors_.Dim() != 0 && priors_.Dim() != nnet_.OutputDim("output")) {
    KALDI_WARN << "Removing priors since there is a dimension mismatch after "
               << "changing the nnet: " << priors_.Dim() << " vs. "
               << nnet_.OutputDim("output");
    priors_.Resize(0);
  }
}

void AmNnetSimple::SetPriors(const VectorBase<BaseFloat> &priors) {
  priors_ = priors;
  if (priors_.Dim() != nnet_.OutputDim("output") &&
      priors_.Dim() != 0) {
    KALDI_ERR << "Dimension mismatch when setting priors: priors have dim "
              << priors.Dim() << ", model expects "
              << nnet_.OutputDim("output");
  }
}

std::string AmNnetSimple::Info() const {
  std::ostringstream ostr;
  ostr << "left-context: " << left_context_ << "\n";
  ostr << "right-context: " << right_context_ << "\n";
  ostr << "input-dim: " << nnet_.InputDim("input") << "\n";
  ostr << "ivector-dim: " << nnet_.InputDim("ivector") << "\n";
  ostr << "num-pdfs: " << nnet_.OutputDim("output") << "\n";
  ostr << "prior-dimension: " << priors_.Dim() << "\n";
  if (priors_.Dim() != 0) {
    ostr << "prior-sum: " << priors_.Sum() << "\n";
    ostr << "prior-min: " << priors_.Min() << "\n";
    ostr << "prior-max: " << priors_.Max() << "\n";
  }
  ostr << "# Nnet info follows.\n";
  return ostr.str() + nnet_.Info();
}


void AmNnetSimple::SetContext() {
  if (!IsSimpleNnet(nnet_)) {
    KALDI_ERR << "Class AmNnetSimple is only intended for a restricted type of "
              << "nnet, and this one does not meet the conditions.";
  }
  ComputeSimpleNnetContext(nnet_,
                           &left_context_,
                           &right_context_);
}


} // namespace nnet3
} // namespace kaldi
