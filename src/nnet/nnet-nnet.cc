// nnet/nnet-nnet.cc

// Copyright 2011-2013 Brno University of Technology (Author: Karel Vesely)

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
namespace nnet1 {

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
  // release the buffers we don't need anymore
  propagate_buf_[0].Resize(0,0);
  propagate_buf_[1].Resize(0,0);
}

void Nnet::GetWeights(Vector<BaseFloat>* wei_copy) {
  wei_copy->Resize(NumParams());
  int32 pos = 0;
  //copy the params
  for(int32 n=0; n<nnet_.size(); n++) {
    if(nnet_[n]->IsUpdatable()) {
      switch(nnet_[n]->GetType()) {
        case Component::kAffineTransform : {
          //get the weights from CuMatrix to Matrix
          const CuMatrix<BaseFloat>& cu_mat = 
            dynamic_cast<AffineTransform*>(nnet_[n])->GetLinearity();
          Matrix<BaseFloat> mat(cu_mat.NumRows(),cu_mat.NumCols());
          cu_mat.CopyToMat(&mat);
          //copy the the matrix row-by-row to the vector
          int32 mat_size = mat.NumRows()*mat.NumCols();
          wei_copy->Range(pos,mat_size).CopyRowsFromMat(mat);
          pos += mat_size;
          //get the biases from CuVector to Vector
          const CuVector<BaseFloat>& cu_vec = 
            dynamic_cast<AffineTransform*>(nnet_[n])->GetBias();
          Vector<BaseFloat> vec(cu_vec.Dim());
          cu_vec.CopyToVec(&vec);
          //append biases to the supervector
          wei_copy->Range(pos,vec.Dim()).CopyFromVec(vec);
          pos += vec.Dim();
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(nnet_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}


void Nnet::SetWeights(const Vector<BaseFloat>& wei_src) {
  KALDI_ASSERT(wei_src.Dim() == NumParams());
  int32 pos = 0;
  for(int32 n=0; n<nnet_.size(); n++) {
    if(nnet_[n]->IsUpdatable()) {
      switch(nnet_[n]->GetType()) {
        case Component::kAffineTransform : {
          //get the component
          AffineTransform* aff_t = dynamic_cast<AffineTransform*>(nnet_[n]);
          //we need weight matrix with original dimensions
          const CuMatrix<BaseFloat>& cu_mat = aff_t->GetLinearity();
          Matrix<BaseFloat> mat(cu_mat.NumRows(),cu_mat.NumCols());
          int32 mat_size = mat.NumRows() * mat.NumCols();
          mat.CopyRowsFromVec(wei_src.Range(pos,mat_size));
          pos += mat_size;
          //get the bias vector
          const CuVector<BaseFloat>& cu_vec = aff_t->GetBias();
          Vector<BaseFloat> vec(cu_vec.Dim());
          vec.CopyFromVec(wei_src.Range(pos,vec.Dim()));
          pos += vec.Dim();
          
          //copy the data to CuMatrix/CuVector and assign to the component
          //weights
          aff_t->SetLinearity(CuMatrix<BaseFloat>(mat));
          //bias
          aff_t->SetBias(CuVector<BaseFloat>(vec));
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(nnet_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}

 
void Nnet::GetGradient(Vector<BaseFloat>* grad_copy) {
  grad_copy->Resize(NumParams());
  int32 pos = 0;
  //copy the params
  for(int32 n=0; n<nnet_.size(); n++) {
    if(nnet_[n]->IsUpdatable()) {
      switch(nnet_[n]->GetType()) {
        case Component::kAffineTransform : {
          //get the weights from CuMatrix to Matrix
          const CuMatrix<BaseFloat>& cu_mat = 
            dynamic_cast<AffineTransform*>(nnet_[n])->GetLinearityCorr();
          Matrix<BaseFloat> mat(cu_mat.NumRows(),cu_mat.NumCols());
          cu_mat.CopyToMat(&mat);
          //copy the the matrix row-by-row to the vector
          int32 mat_size = mat.NumRows()*mat.NumCols();
          grad_copy->Range(pos,mat_size).CopyRowsFromMat(mat);
          pos += mat_size;
          //get the biases from CuVector to Vector
          const CuVector<BaseFloat>& cu_vec = 
            dynamic_cast<AffineTransform*>(nnet_[n])->GetBiasCorr();
          Vector<BaseFloat> vec(cu_vec.Dim());
          cu_vec.CopyToVec(&vec);
          //append biases to the supervector
          grad_copy->Range(pos,vec.Dim()).CopyFromVec(vec);
          pos += vec.Dim();
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(nnet_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}


int32 Nnet::NumParams() const {
  int32 n_params = 0;
  for(int32 n=0; n<nnet_.size(); n++) {
    if(nnet_[n]->IsUpdatable()) {
      switch(nnet_[n]->GetType()) {
        case Component::kAffineTransform :
          n_params += (1 + nnet_[n]->InputDim()) * nnet_[n]->OutputDim();
          break;
        default :
          KALDI_WARN << Component::TypeToMarker(nnet_[n]->GetType())
                     << "is updatable, but its parameter count not implemented";
      }
    }
  }
  return n_params;
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
  opts_.learn_rate = 0.0;
}


std::string Nnet::Info() const {
  std::ostringstream ostr;
  ostr << "num-components " << LayerCount() << std::endl;
  ostr << "input-dim " << InputDim() << std::endl;
  ostr << "output-dim " << OutputDim() << std::endl;
  ostr << "number-of-parameters " << static_cast<float>(NumParams())/1e6 
       << " millions" << std::endl;
  for (int32 i = 0; i < LayerCount(); i++)
    ostr << "component " << i+1 << " : " 
         << Component::TypeToMarker(nnet_[i]->GetType()) 
         << ", input-dim " << nnet_[i]->InputDim()
         << ", output-dim " << nnet_[i]->OutputDim()
         << ", " << nnet_[i]->Info() << std::endl;
  return ostr.str();
}
  
} // namespace nnet1
} // namespace kaldi
