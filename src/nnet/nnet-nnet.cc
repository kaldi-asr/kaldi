// nnet/nnet-nnet.cc

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

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-various.h"


namespace kaldi {

void Nnet::Propagate(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);

  if (LayerCount() == 0) { 
    out->Resize(in.NumRows(), in.NumCols());
    out->CopyFromMat(in); 
    return; 
  }

  // we need at least L+1 input buffers
  KALDI_ASSERT((int32)propagate_buf_.size() >= LayerCount()+1);

  
  propagate_buf_[0].Resize(in.NumRows(), in.NumCols());
  propagate_buf_[0].CopyFromMat(in);

  for(int32 i=0; i<(int32)nnet_.size(); i++) {
    nnet_[i]->Propagate(propagate_buf_[i], &propagate_buf_[i+1]);
  }

  CuMatrix<BaseFloat> &mat = propagate_buf_[nnet_.size()];
  out->Resize(mat.NumRows(), mat.NumCols());
  out->CopyFromMat(mat);
}


void Nnet::Backpropagate(const CuMatrix<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {
  if(LayerCount() == 0) { KALDI_ERR << "Cannot backpropagate on empty network"; }

  // we need at least L+1 input bufers
  KALDI_ASSERT((int32)propagate_buf_.size() >= LayerCount()+1);
  // we need at least L-1 error derivative bufers
  KALDI_ASSERT((int32)backpropagate_buf_.size() >= LayerCount()-1);

  //////////////////////////////////////
  // Backpropagation
  //

  // don't copy the out_diff to buffers, use it as is...
  int32 i = nnet_.size()-1;
  nnet_.back()->Backpropagate(propagate_buf_[i], propagate_buf_[i+1], 
                              out_diff, &backpropagate_buf_[i-1]);
  if (nnet_[i]->IsUpdatable()) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(nnet_[i]);
    uc->Update(propagate_buf_[i], out_diff);
  }

  // backpropagate by using buffers
  for(i--; i >= 1; i--) {
    nnet_[i]->Backpropagate(propagate_buf_[i], propagate_buf_[i+1],
                            backpropagate_buf_[i], &backpropagate_buf_[i-1]);
    if (nnet_[i]->IsUpdatable()) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(nnet_[i]);
      uc->Update(propagate_buf_[i], backpropagate_buf_[i]);
    }
  }

  // now backpropagate through first layer, 
  // but only if asked to (by in_diff pointer)
  if (NULL != in_diff) {
    nnet_[0]->Backpropagate(propagate_buf_[0], propagate_buf_[1],
                            backpropagate_buf_[0], in_diff);
  }

  // update the first layer 
  if (nnet_[0]->IsUpdatable()) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(nnet_[0]);
    uc->Update(propagate_buf_[0], backpropagate_buf_[0]);
  }

  //
  // End of Backpropagation
  //////////////////////////////////////
}


void Nnet::Feedforward(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);

  if (LayerCount() == 0) { 
    out->Resize(in.NumRows(), in.NumCols());
    out->CopyFromMat(in); 
    return; 
  }

  if (LayerCount() == 1) {
    nnet_[0]->Propagate(in, out);
    return;
  }

  // we need at least 2 input buffers
  KALDI_ASSERT(propagate_buf_.size() >= 2);

  // propagate by using exactly 2 auxiliary buffers
  int32 L = 0;
  nnet_[L]->Propagate(in, &propagate_buf_[L%2]);
  for(L++; L<=LayerCount()-2; L++) {
    nnet_[L]->Propagate(propagate_buf_[(L-1)%2], &propagate_buf_[L%2]);
  }
  nnet_[L]->Propagate(propagate_buf_[(L-1)%2], out);
}


void Nnet::Read(std::istream &in, bool binary) {
  // get the network layers from a factory
  Component *comp;
  while (NULL != (comp = Component::Read(in, binary, this))) {
    if (LayerCount() > 0 && nnet_.back()->OutputDim() != comp->InputDim()) {
      KALDI_ERR << "Dimensionality mismatch!"
                << " Previous layer output:" << nnet_.back()->OutputDim()
                << " Current layer input:" << comp->InputDim();
    }
    nnet_.push_back(comp);
  }
  // create empty buffers
  propagate_buf_.resize(LayerCount()+1);
  backpropagate_buf_.resize(LayerCount()-1);
  // reset learn rate
  learn_rate_ = 0.0;
}


void Nnet::SetLearnRate(BaseFloat lrate, const char *lrate_factors) {
  // split lrate_factors to a vector
  std::vector<BaseFloat> lrate_factor_vec;
  if (NULL != lrate_factors) {
    char *copy = new char[strlen(lrate_factors)+1];
    strcpy(copy, lrate_factors);
    char *tok = NULL;
    while(NULL != (tok = strtok((tok==NULL?copy:NULL),",:; "))) {
      lrate_factor_vec.push_back(atof(tok));
    }
    delete copy;
  }

  // count trainable layers
  int32 updatable = 0;
  for(int i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) updatable++;
  }
  // check number of factors
  if (lrate_factor_vec.size() > 0 && updatable != (int32)lrate_factor_vec.size()) {
    KALDI_ERR << "Mismatch between number of trainable layers " << updatable
              << " and learn rate factors " << lrate_factor_vec.size();
  }

  // set learn rates
  updatable=0;
  for(int32 i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) {
      BaseFloat lrate_scaled = lrate;
      if (lrate_factor_vec.size() > 0) lrate_scaled *= lrate_factor_vec[updatable++];
      dynamic_cast<UpdatableComponent*>(nnet_[i])->SetLearnRate(lrate_scaled);
    }
  }
  // set global learn rate
  learn_rate_ = lrate;
}


std::string Nnet::GetLearnRateString() {
  std::ostringstream oss;
  oss << "LEARN_RATE global: " << learn_rate_ << " individual: ";
  for(int32 i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) {
      oss << dynamic_cast<UpdatableComponent*>(nnet_[i])->GetLearnRate() << " ";
    }
  }
  return oss.str();
}




  
} // namespace
