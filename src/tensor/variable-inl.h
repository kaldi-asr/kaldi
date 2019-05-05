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
  if (grad_) {
    return true;
  } else if (!base_) {
    return false;  // This is a base Variable with no grad -> not tracked.
  } else if (base_->grad_ == nullptr) {
    return false;
  } else {
    // We need to obtain and cache the Tensor corresponding to this
    // sub-part of the grad.  (See "Lazy allocation" in glossary in tensor.h
    // for why this won't allocate much memory).
    grad_ = base_->GetGradForView(data_);
    return true;
  }
}

Tensor VariableImpl::GetGradForView(const Tensor &data) {
  // Check that this is a tracked base Variable.
  KALDI_PARANOID_ASSERT(base_ == nullptr && grad_ != nullptr);
  std::shared_ptr<TensorImpl> ans = new TensorImpl(data.Meta(),
                                                   grad_->storage);
  if (!rebase_grad_) {
    // The grad will have exactly the same offset, dims and strides as the data.
    // This is the normal case, which we encounter when the Variable was
    // constructed from a Tensor that is justified and contiguous (see glossary
    // in pattern.h for meanings).
    return Tensor(ans);
  } else {
    if (!aux_)
      aux_ = new VariableImplAux;
    if (!aux_->rebaser)
      aux_->rebaser = new PatternRebaser(pattern_,
                                               grad_->pattern_);
    const PatternRebaser &rebaser = *(aux_->rebaser);
    if (!rebaser->Rebase(&(ans->pattern))) {
      // die.
      KALDI_ERR << "Rebasing failed.  Likely you are using views "
          "in a very strange way.";
    }
    KALDI_PARANOID_ASSERT(ans->IsValid());
    return Tensor(ans);
  }
}


const std::shared_ptr<Tensor>& VariableImpl::GetGrad() {
  if (grad_) {
    return grad_;
  } else if (!base_) {
    CreateGrad();
    return grad_;
  } else {
    if (!base->grad_)
      base->CreateGrad();
    grad_ = base->GetGradForView(data_);
    return grad_;
  }
}


void VariableImpl::CreateGrad() {
  if (ContiguousAndStartsFromZero(data_->Impl())) {
    // the following creates a new TensorImpl with its own new
    // Storage object with the meta-info provided; it will just
    // mirror data_.
    grad_ = new TensorImpl(data_.Meta(), true);
    rebase_grad_ = false;
  } else {
    // Don't allocate the storage yet; we need to fix the pattern to fill in any
    // gaps and move the offset to zero.
    grad_ = new TensorImpl();
    // grad_->pattern will be as the pattern of data_, but with any
    // gaps filled in and the smallest mindex equal to zero.
    MakeContiguousAndJustified(data_.Meta().pattern,
                               &(grad_->pattern));
    rebase_grad_ = true;
  }


    // This is a base Variable and we need to construct the grad.
    //

    // node.  (Assume it
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
