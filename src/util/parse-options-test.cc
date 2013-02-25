// util/parse-options-test.cc

// Copyright 2009-2011  Microsoft Corporation
// Copyright 2012-2013  Frantisek Skala;  Arnab Ghoshal

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

struct DummyOptions {
  int32 my_int;
  bool my_bool;
  std::string my_string;

  DummyOptions() {
    my_int = 0;
    my_bool = true;
    my_string = "default dummy string";
  }

  void Register(ParseOptions *po) {
    po->Register("my-int", &my_int, "An int32 variable in DummyOptions.");
    po->Register("my-bool", &my_bool, "A Boolean varaible in DummyOptions.");
    po->Register("my-str", &my_string, "A string varaible in DummyOptions.");
  }
};

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
  KALDI_ASSERT(po.NumArgs() == 3);
  KALDI_ASSERT(po.GetArg(1) == "a");
  KALDI_ASSERT(po.GetArg(2) == "b");
  KALDI_ASSERT(po.GetArg(3) == "c");
  KALDI_ASSERT(unum == 5);
  KALDI_ASSERT(num == 3);
  KALDI_ASSERT(str == "boo");

  ParseOptions po2("my another msg");
  int argc2 = 4;
  const char *argv2[4] = { "program_name", "--i=foo",
    "--to-be-NORMALIZED=test", "c" };
  std::string str2 = "default_for_str2";
  po2.Register("To_Be_Normalized", &str2,
               "My variable (name has to be normalized)");
  po2.Register("i", &str, "My variable");
  po2.Read(argc2, argv2);
  KALDI_ASSERT(po2.NumArgs() == 1);
  KALDI_ASSERT(po2.GetArg(1) == "c");
  KALDI_ASSERT(str2 == "test");
  KALDI_ASSERT(str == "foo");

  ParseOptions po3("now checking options with prefix");
  ParseOptions ro3("prefix", &po3);  // to register with prefix
  DummyOptions dummy_opts;
  int argc3 = 9;
  const char *argv3[9] = { "program_name", "--prefix.unum=5", "--num=3",
      "--prefix.str=foo", "--str=bar", "--prefix.my-bool=false",
      "--prefix.my-str=baz", "a", "b" };
  po3.Register("str", &str, "My string variable");
  po3.Register("num", &num, "My int32 variable");
  // Now register with prefix
  ro3.Register("unum", &unum, "My uint32 variable");
  ro3.Register("str", &str2, "My other string variable");
  dummy_opts.Register(&ro3);
  po3.PrintUsage(false);

  po3.Read(argc3, argv3);
  KALDI_ASSERT(po3.NumArgs() == 2);
  KALDI_ASSERT(po3.GetArg(1) == "a");
  KALDI_ASSERT(po3.GetArg(2) == "b");
  KALDI_ASSERT(unum == 5);
  KALDI_ASSERT(num == 3);
  KALDI_ASSERT(str2 == "foo");
  KALDI_ASSERT(str == "bar");
  KALDI_ASSERT(dummy_opts.my_bool == false);
  KALDI_ASSERT(dummy_opts.my_string == "baz");
}


}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  UnitTestParseOptions();
  return 0;
}


