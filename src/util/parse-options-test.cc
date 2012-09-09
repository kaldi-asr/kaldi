// util/parse-options-test.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Frantisek Skala

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
  int argc = 7;
  std::string str="default_for_str";
  int32 num = 1;
  uint32 unum = 2;
  const char *argv[7] = { "program_name", "--unum=5", "--num=3", "--i=boo",
    "a", "b", "c" };
  ParseOptions po("my usage msg");
  po.Register("i", &str, "My variable");
  po.Register("num", &num, "My int32 variable");
  po.Register("unum", &unum, "My uint32 variable");
  po.Read(argc, argv);
  assert(po.NumArgs() == 3);
  assert(po.GetArg(1) == "a");
  assert(po.GetArg(2) == "b");
  assert(po.GetArg(3) == "c");
  assert(unum == 5);
  assert(num == 3);
  assert(str == "boo");

  ParseOptions po2("my another msg");
  int argc2 = 4;
  const char *argv2[4] = { "program_name", "--i=foo",
    "--to-be-NORMALIZED=test", "c" };
  std::string str2 = "default_for_str2";
  po2.Register("To_Be_Normalized", &str2,
               "My variable (name has to be normalized)");
  po2.Register("i", &str, "My variable");
  po2.Read(argc2, argv2);
  assert(po2.NumArgs() == 1);
  assert(po2.GetArg(1) == "c");
  assert(str2 == "test");
  assert(str == "foo");
}


}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  UnitTestParseOptions();
  return 0;
}


