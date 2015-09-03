// nnet2/nnet-functions.cc

// Copyright 2011-2012  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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

#include "nnet2/nnet-nnet.h"
#include "util/stl-utils.h"

namespace kaldi {
namespace nnet2 {

int32 IndexOfSoftmaxLayer(const Nnet &nnet) {
  int32 index = -1, nc = nnet.NumComponents();
  for (int32 c = 0; c < nc; c++) {
    const Component *component = &(nnet.GetComponent(c));
    if (dynamic_cast<const SoftmaxComponent*>(component) != NULL) {
      if (index != -1) return -1; // >1 softmax components.
      else index = c;
    }
  }
  return index;
}

void InsertComponents(const Nnet &src_nnet,
                      int32 c_to_insert, // component-index before which to insert.
                      Nnet *dest_nnet) {
  KALDI_ASSERT(c_to_insert >= 0 && c_to_insert <= dest_nnet->NumComponents());
  int32 c_tot = dest_nnet->NumComponents() + src_nnet.NumComponents();
  std::vector<Component*> components(c_tot);
  for (int32 c = 0; c < c_to_insert; c++)
    components[c] = dest_nnet->GetComponent(c).Copy();
  for (int32 c = 0; c < src_nnet.NumComponents(); c++)
    components[c + c_to_insert] = src_nnet.GetComponent(c).Copy();
  for (int32 c = c_to_insert; c < dest_nnet->NumComponents(); c++)
    components[c + src_nnet.NumComponents()] = dest_nnet->GetComponent(c).Copy();
  // Re-initialize "dest_nnet" from the resulting list of components.

  // The Init method will take ownership of the pointers in the vector:
  dest_nnet->Init(&components);
}


void ReplaceLastComponents(const Nnet &src_nnet,
                           int32 num_to_remove,
                           Nnet *dest_nnet) {
  KALDI_ASSERT(num_to_remove >= 0 && num_to_remove <= dest_nnet->NumComponents());
  int32 c_orig = dest_nnet->NumComponents() - num_to_remove;

  std::vector<Component*> components;
  for (int32 c = 0; c < c_orig; c++)
    components.push_back(dest_nnet->GetComponent(c).Copy());
  for (int32 c = 0; c < src_nnet.NumComponents(); c++)
    components.push_back(src_nnet.GetComponent(c).Copy());

  // Re-initialize "dest_nnet" from the resulting list of components.
  // The Init method will take ownership of the pointers in the vector:
  dest_nnet->Init(&components);
}



} // namespace nnet2
} // namespace kaldi
