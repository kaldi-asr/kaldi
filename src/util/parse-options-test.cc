// util/parse-options-test.cc

// Copyright 2009-2011  Microsoft Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#include "util/parse-options.h"

namespace kaldi {

void UnitTestParseOptions() {
  int argc = 4;
  std::string str;
  const char* argv[4] = { "--i=boo", "a", "b", "c" };
  ParseOptions po("my usage msg");
  po.Register("i", &str, "My variable");
  po.Read(argc, argv);
  assert(po.NumArgs() == 3);
  assert(po.GetArg(1) == "a");
  assert(po.GetArg(2) == "b");
  assert(po.GetArg(3) == "c");
}


}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  UnitTestParseOptions();
  return 0;
}


