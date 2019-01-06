// nnet3/nnet-chaina-utils-test.cc

// Copyright 2018  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3a/nnet-chaina-utils.h"

namespace kaldi {
namespace nnet3 {

void UnitTestParseFromQueryString(){
  std::string value;
  KALDI_ASSERT(ParseFromQueryString("abc", "d", &value) == false);
  KALDI_ASSERT(ParseFromQueryString("abc?e=f", "d", &value) == false);
  KALDI_ASSERT(ParseFromQueryString("abc?d=f", "d", &value) == true &&
               value == "f");
  KALDI_ASSERT(ParseFromQueryString("abc?dd=f", "d", &value) == false);
  KALDI_ASSERT(ParseFromQueryString("abc?dd=f&d=gab", "d", &value) == true &&
               value == "gab");
  KALDI_ASSERT(ParseFromQueryString("abc?d=f&dd=gab", "d", &value) == true &&
               value == "f");
  KALDI_ASSERT(ParseFromQueryString("abc?d=f&ex=fda&dd=gab", "ex", &value) == true &&
               value == "fda");


  BaseFloat f;
  KALDI_ASSERT(ParseFromQueryString("abc?d=f&ex=1.0&dd=gab", "ex", &f) == true &&
               f == 1.0);
  KALDI_ASSERT(ParseFromQueryString("abc?d=f&ex=1.0&dd=gab", "e", &f) == false);
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  SetVerboseLevel(2);
  UnitTestParseFromQueryString();
  KALDI_LOG << "Tests succeeded.";

  return 0;
}
