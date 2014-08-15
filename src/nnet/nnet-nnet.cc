// nnet/nnet-nnet.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

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

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-various.h"


namespace kaldi {
namespace nnet1 {


Nnet::Nnet(const Nnet& other) {
  // copy the components
  for(int32 i=0; i<other.NumComponents(); i++) {
    components_.push_back(other.GetComponent(i).Copy());
  }
  // create empty buffers
  propagate_buf_.resize(NumComponents()+1);
  backpropagate_buf_.resize(NumComponents()-1);
  // copy train opts
  SetTrainOptions(other.opts_);
  Check(); 
}

Nnet & Nnet::operator = (const Nnet& other) {
  Destroy();
  // copy the components
  for(int32 i=0; i<other.NumComponents(); i++) {
    components_.push_back(other.GetComponent(i).Copy());
  }
  // create empty buffers
  propagate_buf_.resize(NumComponents()+1);
  backpropagate_buf_.resize(NumComponents()-1);
  // copy train opts
  SetTrainOptions(other.opts_); 
  Check();
  return *this;
}


Nnet::~Nnet() {
  Destroy();
}


void Nnet::Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);

  if (NumComponents() == 0) {
    (*out) = in; // copy 
    return; 
  }

  // we need at least L+1 input buffers
  KALDI_ASSERT((int32)propagate_buf_.size() >= NumComponents()+1);
  
  propagate_buf_[0].Resize(in.NumRows(), in.NumCols());
  propagate_buf_[0].CopyFromMat(in);

  for(int32 i=0; i<(int32)components_.size(); i++) {
    components_[i]->Propagate(propagate_buf_[i], &propagate_buf_[i+1]);
  }
  
  (*out) = propagate_buf_[components_.size()];
}


void Nnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {

  //////////////////////////////////////
  // Backpropagation
  //

  // 0 layers
  if(NumComponents() == 0) { 
    (*in_diff) = out_diff; //copy
    return;
  }

  // we need at least L+1 input bufers
  KALDI_ASSERT((int32)propagate_buf_.size() >= NumComponents()+1);
  // we need at least L-1 error derivative bufers
  KALDI_ASSERT((int32)backpropagate_buf_.size() >= NumComponents()-1);

  // 1 layer
  if(NumComponents() == 1) { 
    components_[0]->Backpropagate(propagate_buf_[0], propagate_buf_[1], out_diff, in_diff);
    if (components_[0]->IsUpdatable()) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[0]);
      uc->Update(propagate_buf_[0], out_diff);
    }
    return;
  }

  // >1 layers
  // we don't copy the out_diff to buffers, we use it as it is...
  int32 i = components_.size()-1;
  components_.back()->Backpropagate(propagate_buf_[i], propagate_buf_[i+1], 
                              out_diff, &backpropagate_buf_[i-1]);
  if (components_[i]->IsUpdatable()) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
    uc->Update(propagate_buf_[i], out_diff);
  }

  // backpropagate by using buffers
  for (i--; i >= 1; i--) {
    components_[i]->Backpropagate(propagate_buf_[i], propagate_buf_[i+1],
                            backpropagate_buf_[i], &backpropagate_buf_[i-1]);
    if (components_[i]->IsUpdatable()) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->Update(propagate_buf_[i], backpropagate_buf_[i]);
    }
  }

  // now backpropagate through first layer, 
  components_[0]->Backpropagate(propagate_buf_[0], propagate_buf_[1],
                                backpropagate_buf_[0], in_diff);

  // update the first layer 
  if (components_[0]->IsUpdatable()) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[0]);
    uc->Update(propagate_buf_[0], backpropagate_buf_[0]);
  }

  //
  // End of Backpropagation
  //////////////////////////////////////
}


void Nnet::Feedforward(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);

  if (NumComponents() == 0) { 
    out->Resize(in.NumRows(), in.NumCols());
    out->CopyFromMat(in); 
    return; 
  }

  if (NumComponents() == 1) {
    components_[0]->Propagate(in, out);
    return;
  }

  // we need at least 2 input buffers
  KALDI_ASSERT(propagate_buf_.size() >= 2);

  // propagate by using exactly 2 auxiliary buffers
  int32 L = 0;
  components_[L]->Propagate(in, &propagate_buf_[L%2]);
  for(L++; L<=NumComponents()-2; L++) {
    components_[L]->Propagate(propagate_buf_[(L-1)%2], &propagate_buf_[L%2]);
  }
  components_[L]->Propagate(propagate_buf_[(L-1)%2], out);
  // release the buffers we don't need anymore
  propagate_buf_[0].Resize(0,0);
  propagate_buf_[1].Resize(0,0);
}


int32 Nnet::OutputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.back()->OutputDim();
}

int32 Nnet::InputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.front()->InputDim();
}

const Component& Nnet::GetComponent(int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

Component& Nnet::GetComponent(int32 component) {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

void Nnet::SetComponent(int32 c, Component *component) {
  KALDI_ASSERT(static_cast<size_t>(c) < components_.size());
  delete components_[c];
  components_[c] = component;
  Check(); // Check that all the dimensions still match up.
}

void Nnet::AppendComponent(Component* dynamically_allocated_comp) {
  components_.push_back(dynamically_allocated_comp);
  Check();
}

void Nnet::AppendNnet(const Nnet& nnet_to_append) {
  for(int32 i=0; i<nnet_to_append.NumComponents(); i++) {
    AppendComponent(nnet_to_append.GetComponent(i).Copy());
  }
  Check();
}

void Nnet::RemoveComponent(int32 component) {
  KALDI_ASSERT(component < NumComponents());
  Component* ptr = components_[component];
  components_.erase(components_.begin()+component);
  delete ptr;
  Check();
}


void Nnet::GetParams(Vector<BaseFloat>* wei_copy) const {
  wei_copy->Resize(NumParams());
  int32 pos = 0;
  // copy the params
  for(int32 i=0; i<components_.size(); i++) {
    if(components_[i]->IsUpdatable()) {
      UpdatableComponent& c = dynamic_cast<UpdatableComponent&>(*components_[i]);
      Vector<BaseFloat> c_params; 
      c.GetParams(&c_params);
      wei_copy->Range(pos,c_params.Dim()).CopyFromVec(c_params);
      pos += c_params.Dim();
    }
  }
  KALDI_ASSERT(pos == NumParams());
}


void Nnet::GetWeights(Vector<BaseFloat>* wei_copy) const {
  wei_copy->Resize(NumParams());
  int32 pos = 0;
  // copy the params
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      switch(components_[n]->GetType()) {
        case Component::kAffineTransform : {
          // copy weight matrix row-by-row to the vector
          Matrix<BaseFloat> mat(dynamic_cast<AffineTransform*>(components_[n])->GetLinearity());
          int32 mat_size = mat.NumRows()*mat.NumCols();
          wei_copy->Range(pos,mat_size).CopyRowsFromMat(mat);
          pos += mat_size;
          // append biases
          Vector<BaseFloat> vec(dynamic_cast<AffineTransform*>(components_[n])->GetBias());
          wei_copy->Range(pos,vec.Dim()).CopyFromVec(vec);
          pos += vec.Dim();
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(components_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}


void Nnet::SetWeights(const Vector<BaseFloat>& wei_src) {
  KALDI_ASSERT(wei_src.Dim() == NumParams());
  int32 pos = 0;
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      switch(components_[n]->GetType()) {
        case Component::kAffineTransform : {
          // get the component
          AffineTransform* aff_t = dynamic_cast<AffineTransform*>(components_[n]);
          // we need weight matrix with original dimensions
          Matrix<BaseFloat> mat(aff_t->GetLinearity());
          int32 mat_size = mat.NumRows()*mat.NumCols();
          mat.CopyRowsFromVec(wei_src.Range(pos,mat_size));
          pos += mat_size;
          // get the bias vector
          Vector<BaseFloat> vec(aff_t->GetBias());
          vec.CopyFromVec(wei_src.Range(pos,vec.Dim()));
          pos += vec.Dim();
          // assign to the component
          aff_t->SetLinearity(CuMatrix<BaseFloat>(mat));
          aff_t->SetBias(CuVector<BaseFloat>(vec));
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(components_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}

 
void Nnet::GetGradient(Vector<BaseFloat>* grad_copy) const {
  grad_copy->Resize(NumParams());
  int32 pos = 0;
  // copy the params
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      switch(components_[n]->GetType()) {
        case Component::kAffineTransform : {
          // get the weights from CuMatrix to Matrix
          const CuMatrixBase<BaseFloat>& cu_mat = 
            dynamic_cast<AffineTransform*>(components_[n])->GetLinearityCorr();
          Matrix<BaseFloat> mat(cu_mat.NumRows(),cu_mat.NumCols());
          cu_mat.CopyToMat(&mat);
          // copy the the matrix row-by-row to the vector
          int32 mat_size = mat.NumRows()*mat.NumCols();
          grad_copy->Range(pos,mat_size).CopyRowsFromMat(mat);
          pos += mat_size;
          // get the biases from CuVector to Vector
          const CuVector<BaseFloat>& cu_vec = 
            dynamic_cast<AffineTransform*>(components_[n])->GetBiasCorr();
          Vector<BaseFloat> vec(cu_vec.Dim());
          cu_vec.CopyToVec(&vec);
          // append biases to the supervector
          grad_copy->Range(pos,vec.Dim()).CopyFromVec(vec);
          pos += vec.Dim();
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(components_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}


int32 Nnet::NumParams() const {
  int32 n_params = 0;
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      n_params += dynamic_cast<UpdatableComponent*>(components_[n])->NumParams();
    }
  }
  return n_params;
}


void Nnet::SetDropoutRetention(BaseFloat r)  {
  for (int32 c=0; c < NumComponents(); c++) {
    if (GetComponent(c).GetType() == Component::kDropout) {
      Dropout& comp = dynamic_cast<Dropout&>(GetComponent(c));
      BaseFloat r_old = comp.GetDropoutRetention();
      comp.SetDropoutRetention(r);
      KALDI_LOG << "Setting dropout-retention in component " << c 
                << " from " << r_old << " to " << r;
    }
  }
}


void Nnet::Init(const std::string &file) {
  Input in(file);
  std::istream &is = in.Stream();
  ExpectToken(is, false, "<NnetProto>");
  // do the initialization with config lines
  std::string conf_line;
  while (1) {
    if (is.eof()) KALDI_ERR << "Missing </NnetProto> at the end.";
    KALDI_ASSERT(is.good());
    if ('/' == PeekToken(is, false)) { // end the loop
      ExpectToken(is, false, "</NnetProto>");
      break;
    }
    std::getline(is, conf_line); // get the line in config file
    KALDI_VLOG(1) << conf_line; 
    AppendComponent(Component::Init(conf_line+"\n"));
  }
  // cleanup
  in.Close();
  Check();
}


void Nnet::Read(const std::string &file) {
  bool binary;
  Input in(file, &binary);
  Read(in.Stream(), binary);
  in.Close();
  // Warn if the NN is empty
  if(NumComponents() == 0) {
    KALDI_WARN << "The network '" << file << "' is empty.";
  }
}


void Nnet::Read(std::istream &is, bool binary) {
  // get the network layers from a factory
  Component *comp;
  while (NULL != (comp = Component::Read(is, binary))) {
    if (NumComponents() > 0 && components_.back()->OutputDim() != comp->InputDim()) {
      KALDI_ERR << "Dimensionality mismatch!"
                << " Previous layer output:" << components_.back()->OutputDim()
                << " Current layer input:" << comp->InputDim();
    }
    components_.push_back(comp);
  }
  // create empty buffers
  propagate_buf_.resize(NumComponents()+1);
  backpropagate_buf_.resize(NumComponents()-1);
  // reset learn rate
  opts_.learn_rate = 0.0;
  
  Check(); //check consistency (dims...)
}


void Nnet::Write(const std::string &file, bool binary) const {
  Output out(file, binary, true);
  Write(out.Stream(), binary);
  out.Close();
}


void Nnet::Write(std::ostream &os, bool binary) const {
  Check();
  WriteToken(os, binary, "<Nnet>");
  if(binary == false) os << std::endl;
  for(int32 i=0; i<NumComponents(); i++) {
    components_[i]->Write(os, binary);
  }
  WriteToken(os, binary, "</Nnet>");  
  if(binary == false) os << std::endl;
}


std::string Nnet::Info() const {
  // global info
  std::ostringstream ostr;
  ostr << "num-components " << NumComponents() << std::endl;
  ostr << "input-dim " << InputDim() << std::endl;
  ostr << "output-dim " << OutputDim() << std::endl;
  ostr << "number-of-parameters " << static_cast<float>(NumParams())/1e6 
       << " millions" << std::endl;
  // topology & weight stats
  for (int32 i = 0; i < NumComponents(); i++) {
    ostr << "component " << i+1 << " : " 
         << Component::TypeToMarker(components_[i]->GetType()) 
         << ", input-dim " << components_[i]->InputDim()
         << ", output-dim " << components_[i]->OutputDim()
         << ", " << components_[i]->Info() << std::endl;
  }
  return ostr.str();
}

std::string Nnet::InfoGradient() const {
  std::ostringstream ostr;
  // gradient stats
  ostr << "### Gradient stats :\n";
  for (int32 i = 0; i < NumComponents(); i++) {
    ostr << "Component " << i+1 << " : " 
         << Component::TypeToMarker(components_[i]->GetType()) 
         << ", " << components_[i]->InfoGradient() << std::endl;
  }
  return ostr.str();
}

std::string Nnet::InfoPropagate() const {
  std::ostringstream ostr;
  // forward-pass buffer stats
  ostr << "### Forward propagation buffer content :\n";
  for (int32 i=0; i<propagate_buf_.size(); i++) {
    ostr << "["<<1+i<< "] output of " 
         << (i==0 ? "<Input>" : Component::TypeToMarker(components_[i-1]->GetType()))
         << MomentStatistics(propagate_buf_[i]) << std::endl;
  }
  return ostr.str();
}

std::string Nnet::InfoBackPropagate() const {
  std::ostringstream ostr;
  // forward-pass buffer stats
  ostr << "### Backward propagation buffer content :\n";
  for (int32 i=0; i<backpropagate_buf_.size(); i++) {
    ostr << "["<<1+i<< "] diff-output of " 
         << Component::TypeToMarker(components_[i]->GetType())
         << MomentStatistics(backpropagate_buf_[i]) << std::endl;
  }
  return ostr.str();
}



void Nnet::Check() const {
  for (size_t i = 0; i + 1 < components_.size(); i++) {
    KALDI_ASSERT(components_[i] != NULL);
    int32 output_dim = components_[i]->OutputDim(),
      next_input_dim = components_[i+1]->InputDim();
    KALDI_ASSERT(output_dim == next_input_dim);
  }
  // check for nan/inf in network weights
  Vector<BaseFloat> weights;
  GetParams(&weights);
  BaseFloat sum = weights.Sum();
  if(KALDI_ISINF(sum)) {
    KALDI_ERR << "'inf' in network parameters (weight explosion, try lower learning rate?)";
  }
  if(KALDI_ISNAN(sum)) {
    KALDI_ERR << "'nan' in network parameters (try lower learning rate?)";
  }
}


void Nnet::Destroy() {
  for(int32 i=0; i<NumComponents(); i++) {
    delete components_[i];
  }
  components_.resize(0);
  propagate_buf_.resize(0);
  backpropagate_buf_.resize(0);
}


void Nnet::SetTrainOptions(const NnetTrainOptions& opts) {
  opts_ = opts;
  //set values to individual components
  for (int32 l=0; l<NumComponents(); l++) {
    if(GetComponent(l).IsUpdatable()) {
      dynamic_cast<UpdatableComponent&>(GetComponent(l)).SetTrainOptions(opts_);
    }
  }
}

 
} // namespace nnet1
} // namespace kaldi
