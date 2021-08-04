// itf/transition-information.h

// Copyright 2021 NVIDIA (author: Daniel Galvez)

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

#ifndef KALDI_ITF_TRANSITION_INFORMATION_H_
#define KALDI_ITF_TRANSITION_INFORMATION_H_

#include <stdint.h>

namespace kaldi {

/**
 * Class that abstracts out TransitionModel's methods originally used
 * in the lat/ directory. By instantiating a subclass of this abstract
 * class other than TransitionModel, you can use kaldi's lattice tools
 * without a dependency on the hmm/ and tree/ directories. For
 * example, you can consider creating a subclass that implements these
 * via extracting information from Eesen's CTC T.fst object.
 */
class TransitionInformation {
 public:
  virtual ~TransitionInformation() {};
  virtual int32_t TransitionIdToTransitionState(int32_t trans_id) const = 0;
    /**
     * Phone should really be Token here, where Token could be a word
     * piece, phone, or character. However, because the original
     * class, TransitionModel, used Phone, we retain it for the sake
     * of not changing all the callers of this method.
     */
  virtual int32_t TransitionIdToPhone(int32_t trans_id) const = 0;
  virtual bool IsFinal(int32_t trans_id) const = 0;
  virtual bool IsSelfLoop(int32_t trans_id) const = 0;
  virtual int32_t TransitionIdToPdf(int32_t trans_id) const = 0;
  virtual int32_t NumTransitionIds() const = 0;
  virtual int32_t TransitionIdToHmmState(int32_t trans_id) const = 0;
};

}

#endif // KALDI_HMM_TRANSITION_MODEL_H_
