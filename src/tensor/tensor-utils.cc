// tensor/tensor-utils.cc

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

#include "tensor/tensor-utils.cc"

namespace kaldi {
namespace tensor {

void DebugNormalOpInternal(const Tensor &a, TensorUseEnum a_use,
                           const Tensor &b, TensorUseEnum b_use) {
  if (!Broadcastable(a, b))
    KALDI_ERR << "Tensors in Operation do not have broadcastable shapes.";
  if (Overlap(a, b))
    KALDI_ERR << "Tensors in Operation overlap.";
  if (!Broadcastable(a, b))
    KADLDI_ERR << "Tensors in Operation do not have broadcastable shapes.";
  if (a.Dtype() != b.Dtype())
    KALDI_ERR << "Tensors in Operation have different data-types";
  if (a.Device() != b.Device())
    KALDI_ERR << "Tensors in Operation have different device";
  RecordUse(a, a_use);
  RecordUse(b, b_use);
}


void DebugNormalOpInternal(const Tensor &a, TensorUseEnum a_use,
                           const Tensor &b, TensorUseEnum b_use,
                           const Tensor &b, TensorUseEnum c_use) {
  if (!Broadcastable(a, b, c))
    KALDI_ERR << "Tensors in Operation do not have broadcastable shapes.";
  bool a_written = (a_use == kWrite || a_use == kReadWrite);
  bool b_written = (b_use == kWrite || b_use == kReadWrite);
  bool c_written = (b_use == kWrite || b_use == kReadWrite);

  if ((a_written || b_written) && Overlap(a, b))
    KALDI_ERR << "Tensors a and b in Operation overlap.";
  if ((b_written || c_written) && Overlap(b, c))
    KALDI_ERR << "Tensors b and c in Operation overlap.";
  if ((a_written || c_written) && Overlap(a, c))
    KALDI_ERR << "Tensors a and c in Operation overlap.";

  if (a.Dtype() != b.Dtype())
    KALDI_ERR << "Tensors in Operation have different data-types";
  if (a.Device() != b.Device())
    KALDI_ERR << "Tensors in Operation have different device";
  RecordUse(a, a_use);
  RecordUse(b, b_use);
  RecordUse(c, c_use);
}




}  // namespace tensor
}  // namespace kaldi
