// tensor/variable-inl.h

// Copyright      2019  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_TENSOR_VARIABLE_INL_H_
#define KALDI_TENSOR_VARIABLE_INL_H_ 1

// Do not include this file directly.  It is to be included from variable.h.

namespace kaldi {
namespace tensor {

bool VariableImpl::Tracked() const {
  if (!base_) {
    return node_ != nullptr;
  } else if (base_->node_ != nullptr) {
    node_ = base_->node_;  // Re-cache it, and the corresponding grad.
    grad_ = node_->GetGradFor(data_);  // Cache the grad too.
    return true;
  } else {
    return false;
  }
}

const std::shared_ptr<Node>& VariableImpl::GetNode() {
  if (node_) {
    return node_;
  } else if (!base_) {
    // This is a base Variable and we need to construct the node.
    node_ = std::make_shared<Node>(data_);
    grad_ = node_->grad;
    return node_;
  } else {
    // This is a view Variable
    if (!base_->node_) {  // make node of base if needed
      base_->node_ = std::make_shared<Node>(base->data_);
      base_->grad_ = node_->grad;
    }
    // cache node in view
    node_ = base_->node_;
    grad_ = node_->GetGradFor(data_);  // Cache the grad too.
    return node_;
  }
}


const std::shared_ptr<Tensor>& VariableImpl::GetGrad() {
  // The code is almost exactly the same as GetNode() above.  Note:
  // We assume that either grad_ and node_ are both NULL, or both
  // non-NULL.
  if (grad_) {
    return grad_;
  } else if (!base_) {
    // This is a base Variable and we need to construct the node.  (Assume it
    // is not allocated if grad_ was not allocated).
    node_ = std::make_shared<Node>(data_);
    grad_ = node_->grad;
    return grad_;
  } else {
    // This is a view Variable
    if (!base_->node_) {  // make node of base if needed
      base_->node_ = std::make_shared<Node>(base->data_);
      base_->grad_ = node_->grad;
    }
    // cache node in view
    node_ = base_->node_;
    grad_ = node_->GetGradFor(data_);
    return grad_;
  }
}

}  // namespace tensor
}  // namespace kaldi


#endif  // KALDI_TENSOR_VARIABLE_INL_H_
