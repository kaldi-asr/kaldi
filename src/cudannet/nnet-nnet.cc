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

#include "cudannet/nnet-nnet.h"
#include "cudannet/nnet-component.h"
#include "cudannet/nnet-activation.h"
#include "cudannet/nnet-biasedlinearity.h"

namespace kaldi {

void Nnet::Propagate(const CuMatrix<BaseFloat>& in, CuMatrix<BaseFloat>* out) {
  KALDI_ASSERT(NULL != out);

  if(LayerCount() == 0) { 
    out->Resize(in.NumRows(),in.NumCols());
    out->CopyFrom(in); 
    return; 
  }

  //we need at least L+1 input buffers
  KALDI_ASSERT((int32)propagate_buf_.size() >= LayerCount()+1);

  
  propagate_buf_[0].Resize(in.NumRows(),in.NumCols());
  propagate_buf_[0].CopyFrom(in);

  for(int32 i=0; i<(int32)nnet_.size(); i++) {
    nnet_[i]->Propagate(propagate_buf_[i],&propagate_buf_[i+1]);
  }

  CuMatrix<BaseFloat>& mat = propagate_buf_[nnet_.size()];
  out->Resize(mat.NumRows(),mat.NumCols());
  out->CopyFrom(mat);
}


void Nnet::Backpropagate(const CuMatrix<BaseFloat>& in_err, CuMatrix<BaseFloat>* out_err) {
  if(LayerCount() == 0) { KALDI_ERR << "Cannot backpropagate on empty network"; }

  //we need at least L+1 input bufers
  KALDI_ASSERT((int32)propagate_buf_.size() >= LayerCount()+1);
  //we need at least L-1 error bufers
  KALDI_ASSERT((int32)backpropagate_buf_.size() >= LayerCount()-1);

  //find out when we can stop backprop
  int32 backprop_stop = -1;
  if(NULL == out_err) {
    backprop_stop++;
    while(1) {
      if(nnet_[backprop_stop]->IsUpdatable()) {
        if(0.0 != dynamic_cast<UpdatableComponent*>(nnet_[backprop_stop])->LearnRate()) {
          break;
        }
      }
      backprop_stop++;
      if(backprop_stop == (int32)nnet_.size()) {
        KALDI_ERR << "All layers have zero learning rate!";
        break;
      }
    }
  }
  //disable!
  backprop_stop=-1;

  //////////////////////////////////////
  // Backpropagation
  //

  //don't copy the in_err to buffers, use it as is...
  int32 i = nnet_.size()-1;
  if(nnet_[i]->IsUpdatable()) {
    UpdatableComponent* uc = dynamic_cast<UpdatableComponent*>(nnet_[i]);
    if(uc->LearnRate() > 0.0) {
      uc->Update(propagate_buf_[i],in_err);
    }
  }
  nnet_.back()->Backpropagate(in_err,&backpropagate_buf_[i-1]);

  //backpropagate by using buffers
  for(i--; i >= 1; i--) {
    if(nnet_[i]->IsUpdatable()) {
      UpdatableComponent* uc = dynamic_cast<UpdatableComponent*>(nnet_[i]);
      if(uc->LearnRate() > 0.0) {
        uc->Update(propagate_buf_[i],backpropagate_buf_[i]);
      }
    }
    if(backprop_stop == i) break;
    nnet_[i]->Backpropagate(backpropagate_buf_[i],&backpropagate_buf_[i-1]);
  }

  //update first layer 
  if(nnet_[0]->IsUpdatable()  &&  0 >= backprop_stop) {
    UpdatableComponent* uc = dynamic_cast<UpdatableComponent*>(nnet_[0]);
    if(uc->LearnRate() > 0.0) {
      uc->Update(propagate_buf_[0],backpropagate_buf_[0]);
    }
  }
  //now backpropagate through first layer, but only if asked to (by out_err pointer)
  if(NULL != out_err) {
    nnet_[0]->Backpropagate(backpropagate_buf_[0],out_err);
  }

  //
  // End of Backpropagation
  //////////////////////////////////////
}


void Nnet::Feedforward(const CuMatrix<BaseFloat>& in, CuMatrix<BaseFloat>* out) {
  KALDI_ASSERT(NULL != out);

  if(LayerCount() == 0) { 
    out->Resize(in.NumRows(),in.NumCols());
    out->CopyFrom(in); 
    return; 
  }

  //we need at least 2 input buffers
  KALDI_ASSERT(propagate_buf_.size() >= 2);

  //propagate by using exactly 2 auxiliary buffers
  int32 L = 0;
  nnet_[L]->Propagate(in,&propagate_buf_[L%2]);
  for(L++; L<=LayerCount()-2; L++) {
    nnet_[L]->Propagate(propagate_buf_[(L-1)%2],&propagate_buf_[L%2]);
  }
  nnet_[L]->Propagate(propagate_buf_[(L-1)%2],out);
}


void Nnet::Read(std::istream& in, bool binary) {
  //get the network layers from a factory
  Component *comp;
  while(NULL != (comp = Component::Read(in,binary,this))) {
    if(LayerCount() > 0 && nnet_.back()->OutputDim() != comp->InputDim()) {
      KALDI_ERR << "Dimensionality mismatch!"
                << " Previous layer output:" << nnet_.back()->OutputDim()
                << " Current layer input:" << comp->InputDim();
    }
    nnet_.push_back(comp);
  }
  //create empty buffers
  propagate_buf_.resize(LayerCount()+1);
  backpropagate_buf_.resize(LayerCount()-1);
  //reset learn rate
  learn_rate_ = 0.0;
}


void Nnet::LearnRate(BaseFloat lrate, const char* lrate_factors) {
  //split lrate_factors to a vector
  std::vector<BaseFloat> lrate_factor_vec;
  if(NULL != lrate_factors) {
    char* copy = new char[strlen(lrate_factors)+1];
    strcpy(copy, lrate_factors);
    char* tok = NULL;
    while(NULL != (tok = strtok((tok==NULL?copy:NULL),",:; "))) {
      lrate_factor_vec.push_back(atof(tok));
    }
    delete copy;
  }

  //count trainable layers
  int32 updatable = 0;
  for(int i=0; i<LayerCount(); i++) {
    if(nnet_[i]->IsUpdatable()) updatable++;
  }
  //check number of factors
  if(lrate_factor_vec.size() > 0 && updatable != (int32)lrate_factor_vec.size()) {
    KALDI_ERR << "Mismatch between number of trainable layers " << updatable
              << " and learn rate factors " << lrate_factor_vec.size();
  }

  //set learn rates
  updatable=0;
  for(int32 i=0; i<LayerCount(); i++) {
    if(nnet_[i]->IsUpdatable()) {
      BaseFloat lrate_scaled = lrate;
      if(lrate_factor_vec.size() > 0) lrate_scaled *= lrate_factor_vec[updatable++];
      dynamic_cast<UpdatableComponent*>(nnet_[i])->LearnRate(lrate_scaled);
    }
  }
  //set global learn rate
  learn_rate_ = lrate;
}


std::string Nnet::LearnRateString() {
  std::ostringstream oss;
  oss << "LEARN_RATE global: " << learn_rate_ << " individual: ";
  for(int32 i=0; i<LayerCount(); i++) {
    if(nnet_[i]->IsUpdatable()) {
      oss << dynamic_cast<UpdatableComponent*>(nnet_[i])->LearnRate() << " ";
    }
  }
  return oss.str();
}




  
} // namespace
