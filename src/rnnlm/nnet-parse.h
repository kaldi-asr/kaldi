// nnet3/nnet-parse.h

// Copyright 2015    Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_RNNLM_NNET_PARSE_H_
#define KALDI_RNNLM_NNET_PARSE_H_

#include "util/text-utils.h"
#include "matrix/kaldi-vector.h"

namespace kaldi {
namespace rnnlm {

void PrintParameterStats(std::ostringstream &os,
                         const std::string &name,
                         const Vector<BaseFloat> &params,
                         bool include_mean = false);

/** Print to 'os' some information about the mean and standard deviation of
    some parameters, used in Info() functions in nnet-simple-component.cc.
    For example:
     PrintParameterStats(os, "linear-params", linear_params_;
    would print to 'os' something like the string 
     ", linear-params-rms=0.239".
    If you set include_mean to true, it will print something like
    ", linear-params-{mean-stddev}=0.103,0.183".
 */
void PrintParameterStats(std::ostringstream &os,
                         const std::string &name,
                         const Matrix<BaseFloat> &params,
                         bool include_mean = false);


} // namespace nnet3
} // namespace kaldi


#endif

