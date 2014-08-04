// nnet2/nnet-functions.h

// Copyright  2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET2_NNET_FUNCTIONS_H_
#define KALDI_NNET2_NNET_FUNCTIONS_H_

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet2/nnet-component.h"
#include "nnet2/nnet-nnet.h"

#include <iostream>
#include <sstream>
#include <vector>


namespace kaldi {
namespace nnet2 {

// Here we declare various functions for manipulating the neural net,
// such as adding new hidden layers; we'll add things like "mixing up"
// to here.


/// If "nnet" has exactly one softmax layer, this function will return
/// its index; otherwise it will return -1.
int32 IndexOfSoftmaxLayer(const Nnet &nnet);

/**
   Inserts the components of one neural network into a particular place in the
   other one.  This is useful for adding hidden layers to a neural net.  Inserts
   the components of "src_nnet" before component index c of "dest_nnet".
*/
void InsertComponents(const Nnet &src_nnet,
                      int32 c,
                      Nnet *dest_nnet);

/**
   Removes the last "num_to_remove" components and
   adds the components from "src_nnet".
 */
void ReplaceLastComponents(const Nnet &src_nnet,
                           int32 num_to_remove,
                           Nnet *dest_nnet);



} // namespace nnet2
} // namespace kaldi

#endif


