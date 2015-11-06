// decision-tree/boolean-io-funcs-test.cc

// Copyright 2015   Vimal Manohar

// See ../../COPYING for clarification regarding multiple authors
//
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
#include "decision-tree/boolean-io-funcs.h"
#include "base/io-funcs.h"
#include "base/kaldi-math.h"

namespace kaldi {

void UnitTestBooleanIo(bool binary) {
  {
    const char *filename = "tmpf";
    std::ofstream outfile(filename, std::ios_base::out | std::ios_base::binary);
    InitKaldiOutputStream(outfile, binary);
    if (!binary) outfile << "\t";
    std::vector<bool> vec1;
    for (size_t i = 0; i < 10; i++) vec1.push_back(RandUniform() > 0.5);
    WriteBooleanVector(outfile, binary, vec1);
    if (!binary && Rand()%2 == 0) outfile << " \n";
    const char *token1 = "<Vector2>";
    WriteToken(outfile, binary, token1);
    std::vector<bool> vec2;
    for (size_t i = 0; i < 10; i++) vec2.push_back(RandUniform() > 0.5);
    WriteBooleanVector(outfile, binary, vec2);
    if (!binary) outfile << " \n";
    if (!binary && Rand()%2 == 0) outfile << " \n";
    if (!binary && Rand()%2 == 0) outfile << "\t";
    outfile.close();

    {
      std::ifstream infile(filename, std::ios_base::in | std::ios_base::binary);
      bool binary_in;
      InitKaldiInputStream(infile, &binary_in);
      std::vector<bool> vec1_in;
      ReadBooleanVector(infile, binary_in, &vec1_in);
      KALDI_ASSERT(vec1_in == vec1);
      ExpectToken(infile, binary_in, "<Vector2>");
      std::vector<bool> vec2_in;
      ReadBooleanVector(infile, binary_in, &vec2_in);
      KALDI_ASSERT(vec2_in == vec2);
    }
    unlink(filename);
  }
}



}  // end namespace kaldi.

int main() {
  using namespace kaldi;
  for (size_t i = 0; i < 10; i++) {
    UnitTestBooleanIo(false);
    UnitTestBooleanIo(true);
  }
  KALDI_ASSERT(1);  // just to check that KALDI_ASSERT does not fail for 1.
  return 0;
}


