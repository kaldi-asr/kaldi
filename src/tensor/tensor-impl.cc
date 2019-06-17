// tensor/tensor-impl.cc

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

#include "tensor/tensor-impl.h"


namespace kaldi {
namespace tensor {

TensorImpl(const TensorMeta &meta,
           StridePolicy sp):
    dtype(meta.dtype),
    device(meta.device) {
  switch (sp) {
    case kKeepStrideOrder:
      MakeCompactNonnegativeAndJustified(meta.pattern, &pattern);
      break;
    case kNormalized:
      MakeCompactNormalizedAndJustified(meta.pattern, &pattern);
      break;
    case kCopyStrides:
      pattern = meta.pattern;
      MakeJustified(&pattern);
      break;
    default:  // would be code error.
      KALDI_ERR << "Stride policy out of range";
  }
  CreateTensorStorage(this);
  KALDI_PARANOID_ASSERT(this->IsValid());
}

TensorImpl::TensorImpl(const TensorMeta &meta,
                       const std::shared_ptr<Storage> &storage):
    pattern(meta.pattern),
    dtype(meta.dtype),
    device(meta.device),
    storage(storage) {
  KALDI_PARANOID_ASSERT(this->IsValid());
}


TensorImpl::TensorImpl(const TensorMeta &meta,
                       const std::shared_ptr<Storage> &&storage):
    // todo: ask @kkm if this will actually do move construction on the
    // shared_ptr.
    pattern(meta.pattern),
    dtype(meta.dtype),
    device(meta.device),
    storage(storage) {
  KALDI_PARANOID_ASSERT(this->IsValid());
}



}  // namespace kaldi
}  // namespace tensor
