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

#ifndef KALDI_NNET3_NNET_PARSE_H_
#define KALDI_NNET3_NNET_PARSE_H_

#include "util/text-utils.h"

namespace kaldi {
namespace nnet3 {


/// \file This header contains a few parsing-related functions that are used
///    while reading parsing neural network files and config files.

/// Function used in Init routines.  Suppose name=="foo", if "string" has a
/// field like foo=12, this function will set "param" to 12 and remove that
/// element from "string".  It returns true if the parameter was read.
bool ParseFromString(const std::string &name, std::string *string,
                     int32 *param);

/// This version of ParseFromString is for parameters of type BaseFloat.
bool ParseFromString(const std::string &name, std::string *string,
                     BaseFloat *param);

/// This version of ParseFromString is for parameters of type std::vector<int32>; it expects
/// them as a colon-separated list, without spaces.
bool ParseFromString(const std::string &name, std::string *string,
                     std::vector<int32> *param);

/// This version of ParseFromString is for parameters of type bool, which can
/// appear as any string beginning with f, F, t or T.
bool ParseFromString(const std::string &name, std::string *string,
                     bool *param);

/// This version of ParseFromString is for parsing strings.  (these
/// should not contain space).
bool ParseFromString(const std::string &name, std::string *string,
                     std::string *param);

/// This version of ParseFromString handles colon-separated or comma-separated
/// lists of integers.
bool ParseFromString(const std::string &name, std::string *string,
                     std::vector<int32> *param);


/// This function is like ExpectToken but for two tokens, and it will either
/// accept token1 and then token2, or just token2.  This is useful in Read
/// functions where the first token may already have been consumed.
void ExpectOneOrTwoTokens(std::istream &is, bool binary,
                          const std::string &token1,
                          const std::string &token2);

} // namespace nnet3
} // namespace kaldi


#endif

