// nnet2/am-nnet.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet2/am-nnet.h"

namespace kaldi {
namespace nnet2 {


void AmNnet::Init(std::istream &config_is) {
  nnet_.Init(config_is);
}


void AmNnet::Write(std::ostream &os, bool binary) const {
  // We don't write any header or footer like <AmNnet> and </AmNnet> -- we just
  // write the neural net and then the priors.  Who knows, there might be some
  // situation where we want to just read the neural net.
  nnet_.Write(os, binary);
  priors_.Write(os, binary);
}

void AmNnet::Read(std::istream &is, bool binary) {
  nnet_.Read(is, binary);
  priors_.Read(is, binary);
}

void AmNnet::SetPriors(const VectorBase<BaseFloat> &priors) {
  priors_ = priors;
  if (priors_.Dim() > NumPdfs())    
    KALDI_ERR << "Dimension of priors cannot exceed number of pdfs.";

  if (priors_.Dim() > 0 && priors_.Dim() < NumPdfs()) {
    KALDI_WARN << "Dimension of priors is " << priors_.Dim() << " < "
               << NumPdfs() << ": extending with zeros, in case you had "
               << "unseen pdf's, but this possibly indicates a serious problem.";
    priors_.Resize(NumPdfs(), kCopyData);
  }
}

std::string AmNnet::Info() const {
  std::ostringstream ostr;
  ostr << "prior dimension: " << priors_.Dim();
  if (priors_.Dim() != 0) {
    ostr << ", prior sum: " << priors_.Sum() << ", prior min: " << priors_.Min()
         << "\n";
  }
  return nnet_.Info() + ostr.str();
}

void AmNnet::Init(const Nnet &nnet) {
  nnet_ = nnet;
  if (priors_.Dim() != 0 && priors_.Dim() != nnet.OutputDim()) {
    KALDI_WARN << "Initializing neural net: prior dimension mismatch, "
               << "discarding old priors.";
    priors_.Resize(0);
  }
}


} // namespace nnet2
} // namespace kaldi
