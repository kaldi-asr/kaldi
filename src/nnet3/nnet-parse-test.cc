// nnet3/nnet-parse-test.cc

// Copyright 2015  Vimal Manohar (Johns Hopkins University)

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

#include "nnet3/nnet-parse.h"


namespace kaldi {
namespace nnet3 {


void UnitTestDescriptorTokenize() {
  std::vector<std::string> lines;

  std::string str = "(,test )";
  KALDI_ASSERT(DescriptorTokenize(str, &lines));
  KALDI_ASSERT(lines[0] == "(" && lines[1] == "," && lines[2] == "test" && lines[3] == ")");

  str = "(,1test )";
  KALDI_ASSERT(!DescriptorTokenize(str, &lines));

  str = "t (,-1 )";
  KALDI_ASSERT(DescriptorTokenize(str, &lines));
  KALDI_ASSERT(lines.size() == 5 && lines[0] == "t" && lines[3] == "-1");

  str = "   sd , -112 )";
  KALDI_ASSERT(DescriptorTokenize(str, &lines));
  KALDI_ASSERT(lines.size() == 4 && lines[0] == "sd" && lines[2] == "-112");

  str = "   sd , +112 )";
  KALDI_ASSERT(DescriptorTokenize(str, &lines));
  KALDI_ASSERT(lines.size() == 4 && lines[0] == "sd" && lines[2] == "+112");

  str = "foo";
  KALDI_ASSERT(DescriptorTokenize(str, &lines));
  KALDI_ASSERT(lines.size() == 1 && lines[0] == "foo");

}

void UnitTestSummarizeVector() {
  // will be eyeballed by a human.
  Vector<BaseFloat> vec(9);
  vec.SetRandn();
  vec(0) = 1024.2343;
  vec(1) = 0.01;
  vec(2) = 0.001234;
  vec(3) = 0.000198;
  vec(3) = 1.98e-09;
  vec(4) = 153.0;
  vec(5) = 0.154;
  vec(6) = 1.2;
  vec(7) = 9.2;
  vec(8) = 10.8;

  KALDI_LOG << "vec = " << vec << " -> " << SummarizeVector(vec);

  vec.Resize(20, kCopyData);
  KALDI_LOG << "vec = " << vec << " -> " << SummarizeVector(vec);
}

void  UnitTestNameMatchesPattern() {
  KALDI_ASSERT(NameMatchesPattern("hello", "hello"));
  KALDI_ASSERT(!NameMatchesPattern("hello", "hellox"));
  KALDI_ASSERT(!NameMatchesPattern("hellox", "hello"));
  KALDI_ASSERT(NameMatchesPattern("hellox", "hello*"));
  KALDI_ASSERT(NameMatchesPattern("hello", "hello*"));
  KALDI_ASSERT(NameMatchesPattern("", "*"));
  KALDI_ASSERT(NameMatchesPattern("x", "*"));
  KALDI_ASSERT(NameMatchesPattern("foo12bar", "foo*bar"));
  KALDI_ASSERT(NameMatchesPattern("foo12bar", "foo*"));
  KALDI_ASSERT(NameMatchesPattern("foo12bar", "*bar"));
}

} // namespace nnet3

} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;

  UnitTestDescriptorTokenize();
  UnitTestSummarizeVector();
  UnitTestNameMatchesPattern();

  KALDI_LOG << "Parse tests succeeded.";

  return 0;
}
