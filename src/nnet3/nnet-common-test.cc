// nnet3/nnet-common-test.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2016  Xiaohui Zhang

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

#include "nnet3/nnet-common.h"

namespace kaldi {
namespace nnet3 {



void UnitTestIndexIo() {
  std::vector<Index> indexes(RandInt(0, 10));

  for (int32 i = 0; i < indexes.size(); i++) {
    if (i == 0 || RandInt(0, 1) == 0) {
      indexes[i].n = RandInt(-1, 2);
      indexes[i].t = RandInt(-150, 150);
      indexes[i].x = RandInt(-1, 1);
    } else {
      // this case gets optimized while writing. (if abs(diff-in-t) < 125).
      indexes[i].n = indexes[i-1].n;
      indexes[i].t = indexes[i-1].t + RandInt(-127, 127);
      indexes[i].x = indexes[i-1].x;
    }
  }

  std::ostringstream os;
  bool binary = (RandInt(0, 1) == 0);
  WriteIndexVector(os, binary, indexes);

  std::vector<Index> indexes2;
  if (RandInt(0, 1) == 0)
    indexes2 = indexes;
  std::istringstream is(os.str());
  ReadIndexVector(is, binary, &indexes2);
  if (indexes != indexes2) {
    WriteIndexVector(std::cerr, false, indexes);
    std::cerr << "  vs. \n";
    WriteIndexVector(std::cerr, false, indexes2);
    std::cerr << "\n";
    KALDI_ERR << "Indexes differ.";
  }
}

void UnitTestCindexIo() {
  std::vector<Cindex> cindexes(RandInt(0, 15));

  for (int32 i = 0; i < cindexes.size(); i++) {
    if (i == 0 || RandInt(0, 4) == 0) {
      cindexes[i].first = RandInt(-256, 256);
    } else {
      cindexes[i].first = cindexes[i-1].first;
    }
    Index &index = cindexes[i].second;
    if (i == 0) {
      if (RandInt(0, 3) == 0) {
        index.n = 0;
        index.x = 0;
        if (RandInt(0, 1) == 0)
          index.t = RandInt(-5, 5);
        else if (RandInt(0, 1) == 0) {
          index.t = 124;
        } else if (RandInt(0, 1) == 0) {
          index.t = -124;
        } else if (RandInt(0, 1) == 0) {
          index.t = std::numeric_limits<int32>::min();
        } else {
          index.t = 0;
        }
      } else if (RandInt(0, 1) == 0) {
        index.t = 0;
        index.x = 0;
        index.n = RandInt(0, 1);
      } else {
        index.t = RandInt(-3, 3);
        if (RandInt(0, 1) == 0)
          index.t = std::numeric_limits<int32>::min();
        index.x = RandInt(-1,1);
        index.n = RandInt(-1,1);
      }
    } else {
      if (RandInt(0, 3) == 0) {
        cindexes[i].second.n = cindexes[i-1].second.n;
        cindexes[i].second.x = cindexes[i-1].second.x;
        if (RandInt(0, 1) == 0) {
          cindexes[i].second.t = cindexes[i-1].second.t + RandInt(-127, 127);
        } else if (RandInt(0, 1) == 0) {
          cindexes[i].second.t = cindexes[i-1].second.t + 124;
        } else if (RandInt(0, 1) == 0) {
          cindexes[i].second.t = cindexes[i-1].second.t + -124;
        } else if (RandInt(0, 1) == 0) {
          cindexes[i].second.t = std::numeric_limits<int32>::min();
        } else {
          cindexes[i].second.t = RandInt(-2, 2);
        }
      } else if (RandInt(0, 1) == 0) {
        cindexes[i].second.t = cindexes[i-1].second.t;
        cindexes[i].second.x = cindexes[i-1].second.x;
        cindexes[i].second.n = cindexes[i-1].second.n + RandInt(-2,2);
      } else if (RandInt(0, 1) == 0) {
        cindexes[i].second.t = cindexes[i-1].second.t + RandInt(-2, 2);
        cindexes[i].second.x = cindexes[i-1].second.x + RandInt(-2, 2);
        cindexes[i].second.n = cindexes[i-1].second.n + RandInt(-2,2);
        if (RandInt(0, 3) == 0)
          cindexes[i].second.t = std::numeric_limits<int32>::min();

      } else {
        cindexes[i].second.t = RandInt(-128, 128);
        cindexes[i].second.x = RandInt(-128, 128);
        cindexes[i].second.n = RandInt(-128, 128);
        if (RandInt(0, 3) == 0)
          cindexes[i].second.t = std::numeric_limits<int32>::min();
      }
    }
  }

  if (RandInt(0, 10) == 0) {
    // trying to reproduce a failure
    Cindex temp(0, Index(0, 0, 0));
    cindexes.clear();
    cindexes.resize(4, temp);
    cindexes[RandInt(0, 3)].second.t = std::numeric_limits<int32>::min();
  }

  std::ostringstream os;
  bool binary = (RandInt(0, 1) == 0);
  WriteCindexVector(os, binary, cindexes);
  std::vector<Cindex> cindexes2;
  if (RandInt(0, 1) == 0)
    cindexes2 = cindexes;
  std::istringstream is(os.str());
  ReadCindexVector(is, binary, &cindexes2);

  std::ostringstream os2;
  WriteCindexVector(os2, binary, cindexes2);

  if (cindexes != cindexes2 || os.str() != os2.str()) {
    WriteCindexVector(std::cerr, false, cindexes);
    std::cerr << "  vs. \n";
    WriteCindexVector(std::cerr, false, cindexes2);
    std::cerr << "\n";
    KALDI_ERR << "Indexes differ.";
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;

  for (int32 i = 0; i < 100; i++) {
    UnitTestIndexIo();
    UnitTestCindexIo();
  }

  KALDI_LOG << "Nnet-common tests succeeded.";

  return 0;
}
