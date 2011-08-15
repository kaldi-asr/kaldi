// nnet/nnet-loss.cc

// Copyright 2011  Karel Vesely

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

#include "nnet/nnet-loss.h"

#include <sstream>

namespace kaldi {


void Xent::Eval(const Matrix<BaseFloat>& net_out, const Matrix<BaseFloat>& target, Matrix<BaseFloat>* diff) {
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  diff->Resize(net_out.NumRows(),net_out.NumCols(),kUndefined);

  //compute derivative wrt. activations of last layer of neurons
  diff->CopyFromMat(net_out);
  diff->AddMat(-1.0,target);

  //we'll not produce per-frame classification accuracy for soft labels
  correct_ = -1;

  //compute xentropy
  BaseFloat val;
  for(int32 r=0; r<net_out.NumRows(); r++) {
    for(int32 c=0; c<net_out.NumCols(); c++) {
      val = -target(r,c)*log(net_out(r,c));
      if(KALDI_ISINF(val)) val = 1e10;
      loss_ += val;
    }
  }

  frames_ += net_out.NumRows();
}


void Xent::Eval(const Matrix<BaseFloat>& net_out, const std::vector<int32>& target, Matrix<BaseFloat>* diff) {
  KALDI_ASSERT(net_out.NumRows() == (int32)target.size());

  //check the labels
  int32 max=0;
  std::vector<int32>::const_iterator it; 
  for(it=target.begin(); it!=target.end(); ++it) {
    if(max < *it) max = *it;
  }
  if(max >= net_out.NumCols()) {
    KALDI_ERR << "Network has " << net_out.NumCols() 
              << " outputs while having " << max+1 << " labels";
  }

  //compute derivative wrt. activations of last layer of neurons
  diff->Resize(net_out.NumRows(),net_out.NumCols(),kUndefined);
  diff->CopyFromMat(net_out);
  for(int32 r=0; r<(int32)target.size(); r++) {
    KALDI_ASSERT(target.at(r) <= diff->NumCols());
    (*diff)(r,target.at(r)) -= 1.0;
  }

  //we'll not produce per-frame classification accuracy for soft labels
  correct_ += Correct(net_out,target);

  //compute xentropy
  BaseFloat val;
  for(int32 r=0; r<net_out.NumRows(); r++) {
    KALDI_ASSERT(target.at(r) <= net_out.NumCols());
    val = -log(net_out(r,target.at(r)));
    if(KALDI_ISINF(val)) val = 1e10;
    loss_ += val;
  }

  frames_ += net_out.NumRows();
}


std::string Xent::Report() {
  std::ostringstream oss;
  oss << "Xent:" << loss_ << " frames:" << frames_ 
      << " err/frm:" << loss_/frames_;
  if(correct_ >= 0.0) {
    oss << " correct[" << 100.0*correct_/frames_ << "%]";
  }
  oss << std::endl;
  return oss.str(); 
}


int32 Xent::Correct(const Matrix<BaseFloat>& net_out, const std::vector<int32>& target) {
  int32 correct = 0;
  for(int32 r=0; r<net_out.NumRows(); r++) {
    BaseFloat max = -1;
    int32 max_id = -1;
    for(int32 c=0; c<net_out.NumCols(); c++) {
      if(max < net_out(r,c)) {
        max = net_out(r,c);
        max_id = c;
      }
    }
    if(target.at(r) == max_id) {
      correct++;
    }
  }
  return correct;
}


void Mse::Eval(const Matrix<BaseFloat>& net_out, const Matrix<BaseFloat>& target, Matrix<BaseFloat>* diff) {
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  diff->Resize(net_out.NumRows(),net_out.NumCols(),kUndefined);

  //compute derivative w.r.t. neural nerwork outputs
  diff->CopyFromMat(net_out);
  diff->AddMat(-1.0,target);

  //compute mean square error
  BaseFloat val;
  for(int32 r=0; r<net_out.NumRows(); r++) {
    for(int32 c=0; c<net_out.NumCols(); c++) {
      val = target(r,c) - net_out(r,c);
      loss_ += val*val;
    }
  }
  
  frames_ += net_out.NumRows();
}


std::string Mse::Report() {
  std::ostringstream oss;
  oss << "Mse:" << loss_ << " frames:" << frames_
      << " err/frm:" << loss_/frames_ 
      << std::endl;
  return oss.str();
}


} // namespace
