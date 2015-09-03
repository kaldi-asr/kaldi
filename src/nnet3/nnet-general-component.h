// nnet3/nnet-general-component.h

// Copyright 2011-2013  Karel Vesely
//           2012-2015  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang    
//                2014  Vijayaditya Peddinti
//                2014  Guoguo Chen

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

#ifndef KALDI_NNET3_NNET_GENERAL_COMPONENT_H_
#define KALDI_NNET3_NNET_GENERAL_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

/// @file  This file contains declarations of components that are not "simple",
///   meaning they care about the indexes they are operating on, don't return
///   the kSimpleComponent flag in their Properties(), and may return a different
///   number of outputs than inputs.


// We don't have any general components yet.


} // namespace nnet3
} // namespace kaldi


#endif
