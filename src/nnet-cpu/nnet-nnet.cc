// nnet/nnet-nnet.cc

// Copyright 2011-2012  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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

#include "nnet-cpu/nnet-nnet.h"
#include "util/stl-utils.h"

namespace kaldi {


int32 Nnet::OutputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.back()->OutputDim();
}

int32 Nnet::InputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.front()->InputDim();
}


int32 Nnet::LeftContext() const {
  KALDI_ASSERT(!components_.empty());
  int32 ans = 0;
  for (size_t i = 0; i < components_.size(); i++)
    ans += components_[i]->LeftContext();
  return ans;
}

int32 Nnet::RightContext() const {
  KALDI_ASSERT(!components_.empty());
  int32 ans = 0;
  for (size_t i = 0; i < components_.size(); i++)
    ans += components_[i]->RightContext();
  return ans;
}

const Component& Nnet::GetComponent(int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

Component& Nnet::GetComponent(int32 component) {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

void Nnet::SetZero(bool treat_as_gradient) {
  for (size_t i = 0; i < components_.size(); i++) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
    if (uc != NULL) uc->SetZero(treat_as_gradient);
  }
}
  
void Nnet::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Nnet>");
  int32 num_components = components_.size();
  WriteToken(os, binary, "<NumComponents>");
  WriteBasicType(os, binary, num_components);
  WriteToken(os, binary, "<Components>");
  for (int32 c = 0; c < num_components; c++) {
    components_[c]->Write(os, binary);
    if (!binary) os << std::endl;
  }
  WriteToken(os, binary, "</Components>");
  WriteToken(os, binary, "</Nnet>");  
}

void Nnet::Read(std::istream &is, bool binary) {
  Destroy();
  ExpectToken(is, binary, "<Nnet>");
  int32 num_components;
  ExpectToken(is, binary, "<NumComponents>");
  ReadBasicType(is, binary, &num_components);
  ExpectToken(is, binary, "<Components>");
  components_.resize(num_components);
  for (int32 c = 0; c < num_components; c++) 
    components_[c] = Component::ReadNew(is, binary);
  ExpectToken(is, binary, "</Components>");
  ExpectToken(is, binary, "</Nnet>");  
}


void Nnet::ZeroOccupancy() {
  for (size_t i = 0; i < components_.size(); i++) {
    SoftmaxComponent *softmax_component =
        dynamic_cast<SoftmaxComponent*>(components_[i]);
    if (softmax_component != NULL) { // If it was this type
      softmax_component->ZeroOccupancy();
    }
  }
}
void Nnet::Destroy() {
  while (!components_.empty()) {
    delete components_.back();
    components_.pop_back();
  }
}

void Nnet::ComponentDotProducts(
    const Nnet &other,
    VectorBase<BaseFloat> *dot_prod) const {
  KALDI_ASSERT(dot_prod->Dim() == NumComponents());
  for (size_t i = 0; i < components_.size(); i++) {
    UpdatableComponent *uc1 = dynamic_cast<UpdatableComponent*>(components_[i]);
    const UpdatableComponent *uc2 = dynamic_cast<const UpdatableComponent*>(
        &(other.GetComponent(i)));
    KALDI_ASSERT((uc1 != NULL) == (uc2 != NULL));
    if (uc1 != NULL)
      (*dot_prod)(i) = uc1->DotProduct(*uc2);
    else
      (*dot_prod)(i) = 0.0;
  }    
}


Nnet::Nnet(const Nnet &other): components_(other.components_.size()) {
  for (size_t i = 0; i < other.components_.size(); i++)
    components_[i] = other.components_[i]->Copy();
}


Nnet &Nnet::operator = (const Nnet &other) {
  Destroy();
  components_.resize(other.components_.size());
  for (size_t i = 0; i < other.components_.size(); i++)
    components_[i] = other.components_[i]->Copy();
  return *this;
}

std::string Nnet::Info() const {
  std::ostringstream ostr;
  ostr << "num-components " << NumComponents() << std::endl;
  ostr << "left-context " << LeftContext() << std::endl;
  ostr << "right-context " << RightContext() << std::endl;
  ostr << "input-dim " << InputDim() << std::endl;
  ostr << "output-dim " << OutputDim() << std::endl;
  for (int32 i = 0; i < NumComponents(); i++)
    ostr << "component " << i << " : " << components_[i]->Info() << std::endl;
  return ostr.str();
}

void Nnet::Check() const {
  KALDI_ASSERT(!components_.empty());
  for (size_t i = 0; i + 1 < components_.size(); i++) {
    int32 output_dim = components_[i]->OutputDim(),
      next_input_dim = components_[i+1]->InputDim();
    KALDI_ASSERT(output_dim == next_input_dim);
  }
}

void Nnet::Init(std::istream &is) {
  Destroy();
  std::string line;
  /* example config file as follows.  The things in brackets specify the context
     splicing for each layer, and after that is the info about the actual layer.
     Imagine the input dim is 13, and the speaker dim is 40, so (13 x 9) + 40 =  527.
     The config file might be as follows; the lines beginning with # are comments.
     
     # layer-type layer-options
     AffineLayer 0.01 0.001 527 1000 0.04356
  */
  components_.clear();
  while (getline(is, line)) {
    std::istringstream line_is(line);
    line_is >> std::ws; // Eat up whitespace.
    if (line_is.peek() == '#' || line_is.eof()) continue; // Comment or empty.
    Component *c = Component::NewFromString(line);
    KALDI_ASSERT(c != NULL);
    components_.push_back(c);    
  }
  Check();
}

void Nnet::Init(std::vector<Component*> *components) {
  Destroy();
  components_.swap(*components);
  Check();
}

void Nnet::AdjustLearningRatesAndL2Penalties(
    const VectorBase<BaseFloat> &old_model_old_gradient,
    const VectorBase<BaseFloat> &new_model_old_gradient,
    const VectorBase<BaseFloat> &old_model_new_gradient,
    const VectorBase<BaseFloat> &new_model_new_gradient,
    BaseFloat measure_at, // where to measure gradient, on line between old and new model;
                          // 0.5 < measure_at <= 1.0.
    BaseFloat ratio, // e.g. 1.1; ratio by  which we change learning rate.
    BaseFloat max_learning_rate,
    BaseFloat min_l2_penalty,
    BaseFloat max_l2_penalty) {
  std::vector<BaseFloat> new_lrates, new_l2_penalties;
  KALDI_ASSERT(old_model_old_gradient.Dim() == NumComponents() &&
               new_model_old_gradient.Dim() == NumComponents() &&
               old_model_new_gradient.Dim() == NumComponents() &&
               new_model_new_gradient.Dim() == NumComponents());
  KALDI_ASSERT(ratio >= 1.0);
  KALDI_ASSERT(measure_at > 0.5 && measure_at <= 1.0);
  BaseFloat inv_ratio = 1.0 / ratio;
  for (int32 c = 0; c < NumComponents(); c++) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[c]);
    if (uc == NULL) { // Non-updatable component.
      KALDI_ASSERT(old_model_old_gradient(c) == 0.0);
      continue; 
    }
    BaseFloat this_end_dotprod = new_model_new_gradient(c);
    BaseFloat grad_dotprod_at_end =
        new_model_new_gradient(c) - old_model_new_gradient(c),
        grad_dotprod_at_start =
        new_model_old_gradient(c) - old_model_old_gradient(c),
        grad_dotprod_interp =
        measure_at * grad_dotprod_at_end +
        (1.0 - measure_at) * grad_dotprod_at_start;
    // grad_dotprod_interp will be positive if we want more of the gradient term
    // -> faster learning rate for this component

    BaseFloat lrate = uc->LearningRate(),
        l2_penalty = uc->L2Penalty();
    lrate *= (grad_dotprod_interp > 0 ? ratio : inv_ratio);
    l2_penalty *= (this_end_dotprod > 0 ? inv_ratio : ratio);
    if (lrate > max_learning_rate) lrate = max_learning_rate;
    if (l2_penalty > max_l2_penalty) l2_penalty = max_l2_penalty;
    if (l2_penalty < min_l2_penalty) l2_penalty = min_l2_penalty;
    
    new_lrates.push_back(lrate);
    new_l2_penalties.push_back(l2_penalty);
    uc->SetLearningRate(lrate);
    uc->SetL2Penalty(l2_penalty);
  }
  std::ostringstream lrate_str;
  for (size_t i = 0; i < new_lrates.size(); i++)
    lrate_str << new_lrates[i] << ' ';
  KALDI_VLOG(1) << "Learning rates are " << lrate_str.str();
  std::ostringstream l2_penalty_str;
  for (size_t i = 0; i < new_l2_penalties.size(); i++)
    l2_penalty_str << new_l2_penalties[i] << ' ';
  KALDI_VLOG(2) << "L2 penalties are " << l2_penalty_str.str();
}

} // namespace
