// nnet/nnet-nnet.cc

// Copyright 2011-2016  Brno University of Technology (Author: Karel Vesely)
//           2018 Alibaba.Inc (Author: ShiLiang Zhang)  

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
#include "nnet/nnet-parallel-component.h"
#include "nnet/nnet-multibasis-component.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-various.h"
#include "nnet/nnet-fsmn.h"
#include "nnet/nnet-deep-fsmn.h"
#include "nnet/nnet-uni-fsmn.h"
#include "nnet/nnet-uni-deep-fsmn.h"

namespace kaldi {
namespace nnet1 {

Nnet::Nnet() {
}

Nnet::~Nnet() {
  Destroy();
}

Nnet::Nnet(const Nnet& other) {
  // copy the components
  for (int32 i = 0; i < other.NumComponents(); i++) {
    components_.push_back(other.GetComponent(i).Copy());
  }
  // create empty buffers
  propagate_buf_.resize(NumComponents()+1);
  backpropagate_buf_.resize(NumComponents()+1);
  // copy train opts
  SetTrainOptions(other.opts_);
  Check();
}

Nnet& Nnet::operator= (const Nnet& other) {
  Destroy();
  // copy the components
  for (int32 i = 0; i < other.NumComponents(); i++) {
    components_.push_back(other.GetComponent(i).Copy());
  }
  // create empty buffers
  propagate_buf_.resize(NumComponents()+1);
  backpropagate_buf_.resize(NumComponents()+1);
  // copy train opts
  SetTrainOptions(other.opts_);
  Check();
  return *this;
}

/**
 * Forward propagation through the network,
 * (from first component to last).
 */
void Nnet::Propagate(const CuMatrixBase<BaseFloat> &in,
                     CuMatrix<BaseFloat> *out) {
  // In case of empty network copy input to output,
  if (NumComponents() == 0) {
    (*out) = in;  // copy,
    return;
  }
  // We need C+1 buffers,
  if (propagate_buf_.size() != NumComponents()+1) {
    propagate_buf_.resize(NumComponents()+1);
  }
  // Copy input to first buffer,
  propagate_buf_[0] = in;
  // Propagate through all the components,
  for (int32 i = 0; i < static_cast<int32>(components_.size()); i++) {
    components_[i]->Propagate(propagate_buf_[i], &propagate_buf_[i+1]);
  }
  // Copy the output from the last buffer,
  (*out) = propagate_buf_[NumComponents()];
}


/**
 * Error back-propagation through the network,
 * (from last component to first).
 */
void Nnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff,
                         CuMatrix<BaseFloat> *in_diff) {
  // Copy the derivative in case of empty network,
  if (NumComponents() == 0) {
    (*in_diff) = out_diff;  // copy,
    return;
  }
  // We need C+1 buffers,
  KALDI_ASSERT(static_cast<int32>(propagate_buf_.size()) == NumComponents()+1);
  if (backpropagate_buf_.size() != NumComponents()+1) {
    backpropagate_buf_.resize(NumComponents()+1);
  }
  // Copy 'out_diff' to last buffer,
  backpropagate_buf_[NumComponents()] = out_diff;
  // Loop from last Component to the first,
  for (int32 i = NumComponents()-1; i >= 0; i--) {
    // Backpropagate through 'Component',
    components_[i]->Backpropagate(propagate_buf_[i],
                                  propagate_buf_[i+1],
                                  backpropagate_buf_[i+1],
                                  &backpropagate_buf_[i]);
    // Update 'Component' (if applicable),
    if (components_[i]->IsUpdatable()) {
      UpdatableComponent* uc =
        dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->Update(propagate_buf_[i], backpropagate_buf_[i+1]);
    }
  }
  // Export the derivative (if applicable),
  if (NULL != in_diff) {
    (*in_diff) = backpropagate_buf_[0];
  }
}


void Nnet::Feedforward(const CuMatrixBase<BaseFloat> &in,
                       CuMatrix<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);
  (*out) = in;  // works even with 0 components,
  CuMatrix<BaseFloat> tmp_in;
  for (int32 i = 0; i < NumComponents(); i++) {
    out->Swap(&tmp_in);
    components_[i]->Propagate(tmp_in, out);
  }
}


int32 Nnet::OutputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.back()->OutputDim();
}

int32 Nnet::InputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.front()->InputDim();
}

const Component& Nnet::GetComponent(int32 c) const {
  return *(components_.at(c));
}

Component& Nnet::GetComponent(int32 c) {
  return *(components_.at(c));
}

const Component& Nnet::GetLastComponent() const {
  return *(components_.at(NumComponents()-1));
}

Component& Nnet::GetLastComponent() {
  return *(components_.at(NumComponents()-1));
}

void Nnet::ReplaceComponent(int32 c, const Component& comp) {
  delete components_.at(c);
  components_.at(c) = comp.Copy();  // deep copy,
  Check();
}

void Nnet::SwapComponent(int32 c, Component** comp) {
  Component* tmp = components_.at(c);
  components_.at(c) = *comp;
  (*comp) = tmp;
  Check();
}

void Nnet::AppendComponent(const Component& comp) {
  components_.push_back(comp.Copy());  // append,
  Check();
}

void Nnet::AppendComponentPointer(Component* dynamically_allocated_comp) {
  components_.push_back(dynamically_allocated_comp);  // append,
  Check();
}

void Nnet::AppendNnet(const Nnet& other) {
  for (int32 i = 0; i < other.NumComponents(); i++) {
    AppendComponent(other.GetComponent(i));
  }
  Check();
}

void Nnet::RemoveComponent(int32 c) {
  Component* ptr = components_.at(c);
  components_.erase(components_.begin()+c);
  delete ptr;
  Check();
}

void Nnet::RemoveLastComponent() {
  RemoveComponent(NumComponents()-1);
}

int32 Nnet::NumParams() const {
  int32 n_params = 0;
  for (int32 n = 0; n < components_.size(); n++) {
    if (components_[n]->IsUpdatable()) {
      n_params +=
        dynamic_cast<UpdatableComponent*>(components_[n])->NumParams();
    }
  }
  return n_params;
}

void Nnet::GetGradient(Vector<BaseFloat>* gradient) const {
  gradient->Resize(NumParams());
  int32 pos = 0;
  // loop over Components,
  for (int32 i = 0; i < components_.size(); i++) {
    if (components_[i]->IsUpdatable()) {
      UpdatableComponent& c =
        dynamic_cast<UpdatableComponent&>(*components_[i]);
      SubVector<BaseFloat> grad_range(gradient->Range(pos, c.NumParams()));
      c.GetGradient(&grad_range);  // getting gradient,
      pos += c.NumParams();
    }
  }
  KALDI_ASSERT(pos == NumParams());
}

void Nnet::GetParams(Vector<BaseFloat>* params) const {
  params->Resize(NumParams());
  int32 pos = 0;
  // loop over Components,
  for (int32 i = 0; i < components_.size(); i++) {
    if (components_[i]->IsUpdatable()) {
      UpdatableComponent& c =
        dynamic_cast<UpdatableComponent&>(*components_[i]);
      SubVector<BaseFloat> params_range(params->Range(pos, c.NumParams()));
      c.GetParams(&params_range);  // getting params,
      pos += c.NumParams();
    }
  }
  KALDI_ASSERT(pos == NumParams());
}

void Nnet::SetParams(const VectorBase<BaseFloat>& params) {
  KALDI_ASSERT(params.Dim() == NumParams());
  int32 pos = 0;
  // loop over Components,
  for (int32 i = 0; i < components_.size(); i++) {
    if (components_[i]->IsUpdatable()) {
      UpdatableComponent& c =
        dynamic_cast<UpdatableComponent&>(*components_[i]);
      c.SetParams(params.Range(pos, c.NumParams()));  // setting params,
      pos += c.NumParams();
    }
  }
  KALDI_ASSERT(pos == NumParams());
}

void Nnet::SetDropoutRate(BaseFloat r)  {
  for (int32 c = 0; c < NumComponents(); c++) {
    if (GetComponent(c).GetType() == Component::kDropout) {
      Dropout& comp = dynamic_cast<Dropout&>(GetComponent(c));
      BaseFloat r_old = comp.GetDropoutRate();
      comp.SetDropoutRate(r);
      KALDI_LOG << "Setting dropout-rate in component " << c
                << " from " << r_old << " to " << r;
    }
  }
}


void Nnet::ResetStreams(const std::vector<int32> &stream_reset_flag) {
  for (int32 c = 0; c < NumComponents(); c++) {
    if (GetComponent(c).IsMultistream()) {
      MultistreamComponent& comp =
        dynamic_cast<MultistreamComponent&>(GetComponent(c));
      comp.ResetStreams(stream_reset_flag);
    }
  }
}

void Nnet::SetSeqLengths(const std::vector<int32> &sequence_lengths) {
  for (int32 c = 0; c < NumComponents(); c++) {
    if (GetComponent(c).IsMultistream()) {
      MultistreamComponent& comp =
        dynamic_cast<MultistreamComponent&>(GetComponent(c));
      comp.SetSeqLengths(sequence_lengths);
    }
  }
}

void Nnet::Init(const std::string &proto_file) {
  Input in(proto_file);
  std::istream &is = in.Stream();
  std::string proto_line, token;

  // Initialize from the prototype, where each line
  // contains the description for one component.
  while (is >> std::ws, !is.eof()) {
    KALDI_ASSERT(is.good());

    // get a line from the proto file,
    std::getline(is, proto_line);
    if (proto_line == "") continue;
    KALDI_VLOG(1) << proto_line;

    // get the 1st token from the line,
    std::istringstream(proto_line) >> std::ws >> token;
    // ignore these tokens:
    if (token == "<NnetProto>" || token == "</NnetProto>") continue;

    // create new component, append to Nnet,
    this->AppendComponentPointer(Component::Init(proto_line+"\n"));
  }
  // cleanup
  in.Close();
  Check();
}


/**
 * I/O wrapper for converting 'rxfilename' to 'istream',
 */
void Nnet::Read(const std::string &rxfilename) {
  bool binary;
  Input in(rxfilename, &binary);
  Read(in.Stream(), binary);
  in.Close();
  // Warn if the NN is empty
  if (NumComponents() == 0) {
    KALDI_WARN << "The network '" << rxfilename << "' is empty.";
  }
}


void Nnet::Read(std::istream &is, bool binary) {
  // Read the Components through the 'factory' Component::Read(...),
  Component* comp(NULL);
  while (comp = Component::Read(is, binary), comp != NULL) {
    // Check dims,
    if (NumComponents() > 0) {
      if (components_.back()->OutputDim() != comp->InputDim()) {
        KALDI_ERR << "Dimensionality mismatch!"
                  << " Previous layer output:" << components_.back()->OutputDim()
                  << " Current layer input:" << comp->InputDim();
      }
    }
    // Append to 'this' Nnet,
    AppendComponentPointer(comp);
  }
  Check();
}


/**
 * I/O wrapper for converting 'wxfilename' to 'ostream',
 */
void Nnet::Write(const std::string &wxfilename, bool binary) const {
  Output out(wxfilename, binary, true);
  Write(out.Stream(), binary);
  out.Close();
}


void Nnet::Write(std::ostream &os, bool binary) const {
  Check();
  WriteToken(os, binary, "<Nnet>");
  if (binary == false) os << std::endl;
  for (int32 i = 0; i < NumComponents(); i++) {
    components_[i]->Write(os, binary);
  }
  WriteToken(os, binary, "</Nnet>");
  if (binary == false) os << std::endl;
}


std::string Nnet::Info() const {
  // global info
  std::ostringstream ostr;
  ostr << "num-components " << NumComponents() << std::endl;
  if (NumComponents() == 0)
    return ostr.str();
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

std::string Nnet::InfoGradient(bool header) const {
  std::ostringstream ostr;
  // gradient stats
  if (header) ostr << "\n### GRADIENT STATS :\n";
  for (int32 i = 0; i < NumComponents(); i++) {
    ostr << "Component " << i+1 << " : "
         << Component::TypeToMarker(components_[i]->GetType())
         << ", " << components_[i]->InfoGradient() << std::endl;
  }
  if (header) ostr << "### END GRADIENT\n";
  return ostr.str();
}

std::string Nnet::InfoPropagate(bool header) const {
  std::ostringstream ostr;
  // forward-pass buffer stats
  if (header) ostr << "\n### FORWARD PROPAGATION BUFFER CONTENT :\n";
  ostr << "[0] output of <Input> " << MomentStatistics(propagate_buf_[0])
       << std::endl;
  for (int32 i = 0; i < NumComponents(); i++) {
    ostr << "[" << 1+i << "] output of "
         << Component::TypeToMarker(components_[i]->GetType())
         << MomentStatistics(propagate_buf_[i+1]) << std::endl;
    // nested networks too...
    if (Component::kParallelComponent == components_[i]->GetType()) {
      ostr <<
        dynamic_cast<ParallelComponent*>(components_[i])->InfoPropagate();
    }
    if (Component::kMultiBasisComponent == components_[i]->GetType()) {
      ostr << dynamic_cast<MultiBasisComponent*>(components_[i])->InfoPropagate();
    }
  }
  if (header) ostr << "### END FORWARD\n";
  return ostr.str();
}

std::string Nnet::InfoBackPropagate(bool header) const {
  std::ostringstream ostr;
  // forward-pass buffer stats
  if (header) ostr << "\n### BACKWARD PROPAGATION BUFFER CONTENT :\n";
  ostr << "[0] diff of <Input> " << MomentStatistics(backpropagate_buf_[0])
       << std::endl;
  for (int32 i = 0; i < NumComponents(); i++) {
    ostr << "["<<1+i<< "] diff-output of "
         << Component::TypeToMarker(components_[i]->GetType())
         << MomentStatistics(backpropagate_buf_[i+1]) << std::endl;
    // nested networks too...
    if (Component::kParallelComponent == components_[i]->GetType()) {
      ostr <<
        dynamic_cast<ParallelComponent*>(components_[i])->InfoBackPropagate();
    }
    if (Component::kMultiBasisComponent == components_[i]->GetType()) {
      ostr << dynamic_cast<MultiBasisComponent*>(components_[i])->InfoBackPropagate();
    }
  }
  if (header) ostr << "### END BACKWARD\n\n";
  return ostr.str();
}


void Nnet::Check() const {
  // check dims,
  for (size_t i = 0; i + 1 < components_.size(); i++) {
    KALDI_ASSERT(components_[i] != NULL);
    int32 output_dim = components_[i]->OutputDim(),
      next_input_dim = components_[i+1]->InputDim();
    // show error message,
    if (output_dim != next_input_dim) {
      KALDI_ERR << "Component dimension mismatch!"
                << " Output dim of [" << i << "] "
                << Component::TypeToMarker(components_[i]->GetType())
                << " is " << output_dim << ". "
                << "Input dim of next [" << i+1 << "] "
                << Component::TypeToMarker(components_[i+1]->GetType())
                << " is " << next_input_dim << ".";
    }
  }
  // check for nan/inf in network weights,
  Vector<BaseFloat> weights;
  GetParams(&weights);
  BaseFloat sum = weights.Sum();
  if (KALDI_ISINF(sum)) {
    KALDI_ERR << "'inf' in network parameters "
              << "(weight explosion, need lower learning rate?)";
  }
  if (KALDI_ISNAN(sum)) {
    KALDI_ERR << "'nan' in network parameters (need lower learning rate?)";
  }
}


void Nnet::Destroy() {
  for (int32 i = 0; i < NumComponents(); i++) {
    delete components_[i];
  }
  components_.resize(0);
  propagate_buf_.resize(0);
  backpropagate_buf_.resize(0);
}


void Nnet::SetTrainOptions(const NnetTrainOptions& opts) {
  opts_ = opts;
  // set values to individual components,
  for (int32 l = 0; l < NumComponents(); l++) {
    if (GetComponent(l).IsUpdatable()) {
      dynamic_cast<UpdatableComponent&>(GetComponent(l)).SetTrainOptions(opts_);
    }
  }
}

void Nnet::SetFlags(const Vector<BaseFloat> &flags) {    
  for (int32 c = 0; c < NumComponents(); c++) {
    if (GetComponent(c).GetType() == Component::kFsmn) {
      Fsmn& comp = dynamic_cast<Fsmn&>(GetComponent(c));
      comp.SetFlags(flags);
    }
    if (GetComponent(c).GetType() == Component::kDeepFsmn) {
      DeepFsmn& comp = dynamic_cast<DeepFsmn&>(GetComponent(c));
      comp.SetFlags(flags);
    }
    if (GetComponent(c).GetType() == Component::kUniFsmn) {
      UniFsmn& comp = dynamic_cast<UniFsmn&>(GetComponent(c));
      comp.SetFlags(flags);
    }
    if (GetComponent(c).GetType() == Component::kUniDeepFsmn) {
      UniDeepFsmn& comp = dynamic_cast<UniDeepFsmn&>(GetComponent(c));
      comp.SetFlags(flags);
    }
  }
}

}  // namespace nnet1
}  // namespace kaldi
