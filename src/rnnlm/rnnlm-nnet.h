// Copyright    2016  Hainan Xu

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

#ifndef KALDI_RNNLM_RNNLM_NNET_H
#define KALDI_RNNLM_RNNLM_NNET_H

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet3/nnet-nnet.h"
#include "rnnlm/rnnlm-component.h"
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-utils.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>

namespace kaldi {

//namespace nnet3 {
//  class AffineComponent;
//  class NonlinearComponent;
//}

namespace rnnlm {

using nnet3::Component;
//using nnet3::AffineComponent;
//using nnet3::NonlinearComponent;

class LmNnet {
 public:
  friend class LmNnetTrainer;
//  friend void ComputeObjectiveFunction(const GeneralMatrix &supervision,
//                              nnet3::ObjectiveType objective_type,
//                              const std::string &output_name,
//                              bool supply_deriv,
//                              nnet3::NnetComputer *computer,
//                              BaseFloat *tot_weight,
//                              BaseFloat *tot_objf,
//                              const Component *output_projection_1,
//                              const Component *output_projection_2,
//                              LmNnet *nnet);
  LmNnet() {
    nnet_ = new nnet3::Nnet();
  }

  ~LmNnet() {
    delete input_projection_;
    delete output_projection_;
    delete output_layer_;
    delete nnet_;
  }

  nnet3::Nnet* GetNnet() {
    return nnet_;
  }

  const nnet3::Nnet& Nnet() const {
    return *nnet_;
  }

  std::string Info() const;

  void Read(std::istream &is, bool binary);

  void ReadConfig(std::istream &config_file);

  void Write(std::ostream &os, bool binary) const;

  LmNnet* Copy() {
    LmNnet* other = new LmNnet();
    other->input_projection_ = input_projection_->Copy();
    other->output_projection_ = output_projection_->Copy();
    other->output_layer_ = output_layer_->Copy();
    other->nnet_ = nnet_->Copy();

    return other;
  }

  void Add(const LmNnet &other, BaseFloat scale) {
    nnet3::AddNnet(other.Nnet(), scale, nnet_);
    input_projection_->Add(scale, *other.I());
    output_projection_->Add(scale, *other.O());
    
  }

  void Scale(BaseFloat scale) {
    nnet3::ScaleNnet(scale, nnet_);
    input_projection_->Scale(scale);
    output_projection_->Scale(scale);
    
  }

  void ZeroStats() {
    nnet3::ZeroComponentStats(nnet_);
    input_projection_->ZeroStats();
    output_projection_->ZeroStats();
    output_layer_->ZeroStats();
  }

  void SetZero(bool is_gradient) {
    nnet3::SetZero(is_gradient, nnet_);
    LmUpdatableComponent* p;
    if ((p = dynamic_cast<LmUpdatableComponent*>(input_projection_)) != NULL) {
      p->SetZero(is_gradient);
    }

    nnet3::UpdatableComponent* p2;
    if ((p2 = dynamic_cast<nnet3::UpdatableComponent*>(output_projection_)) != NULL) {
      p2->SetZero(is_gradient);
    }

    if ((p2 = dynamic_cast<nnet3::UpdatableComponent*>(output_layer_)) != NULL) {
      p2->SetZero(is_gradient);
    }
  }

  void SetLearningRate(BaseFloat learning_rate) {
    nnet3::SetLearningRate(learning_rate, nnet_);
    LmUpdatableComponent *p;
    if ((p = dynamic_cast<LmUpdatableComponent*>(input_projection_)) != NULL) {
      p->SetUnderlyingLearningRate(learning_rate);
    }
    nnet3::UpdatableComponent *p2;
    if ((p2 = dynamic_cast<nnet3::UpdatableComponent*>(output_projection_)) != NULL) {
      p2->SetUnderlyingLearningRate(learning_rate);
    }

  }

  const LmComponent* I() const {
    return dynamic_cast<const LmComponent*>(input_projection_);
  }
  const nnet3::AffineComponent* O() const {
    return dynamic_cast<const nnet3::AffineComponent*>(output_projection_);
  }
  const nnet3::NonlinearComponent* N() const {
    return dynamic_cast<const nnet3::NonlinearComponent*>(output_layer_);
  }

 private:
  LmComponent* input_projection_;      // Affine
  nnet3::Component* output_projection_;    // Affine
  nnet3::Component* output_layer_;         // Softmax 
  nnet3::Nnet* nnet_;
};

}
}
#endif
