// hmm/hmm-topology-test.cc

// Copyright 2009-2011  Microsoft Corporation
//                2015  Johns Hopkins University (author: Daniel Povey)
//                2019  Hossein Hadian

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

#include "hmm/topology.h"
#include "hmm/hmm-test-utils.h"

namespace kaldi {


void TestTopology() {
  bool binary = (Rand() % 2 == 0);

  std::string input_str = "<Topology>\n"
      "<TopologyEntry>\n"
      "<ForPhones> 1 2 3 4 5 6 7 8 9 </ForPhones>\n"
      " 0  1  1  0.0\n"
      " 1  1  1  0.693\n"
      " 1  2  2  0.693\n"
      " 2  2  2  0.693\n"
      " 2  3  3  0.693\n"
      " 3  3  3  0.693\n"
      " 3  0.693\n\n"
      " </TopologyEntry>\n"

      "<TopologyEntry>\n"
      "<ForPhones> 10 11 13 </ForPhones>\n"
      " 0  0  1  0.693\n"
      " 0  1  1  0.693\n"
      " 1  1  2  0.693\n"
      " 1  2  2  0.693\n"
      " 2 \n\n"
      "</TopologyEntry>\n"
      "</Topology>\n";

  std::string chain_input_str = "<Topology>\n"
      "<TopologyEntry>\n"
      "<ForPhones> 1 2 3 4 5 6 7 8 9 </ForPhones>\n"
      " 0  1  1  0.0\n"
      " 1  1  2  0.693\n"
      " 1  0.693\n\n"
      "</TopologyEntry>\n"
      "</Topology>\n";

  Topology topo;

  if (RandInt(0, 1) == 0) {
    topo = GenRandTopology();
  } else {
    std::istringstream iss(input_str);
    topo.Read(iss, false);
    KALDI_ASSERT(topo.MinLength(3) == 3);
    KALDI_ASSERT(topo.MinLength(11) == 2);
  }

  std::ostringstream oss;
  topo.Write(oss, binary);

  Topology topo2;
  std::istringstream iss2(oss.str());
  topo2.Read(iss2, binary);

  {  // test equality.
    std::ostringstream oss1, oss2;
    topo.Write(oss1, false);
    topo2.Write(oss2, false);
    KALDI_ASSERT(oss1.str() == oss2.str());
  }

  {  // test chain topology
    Topology chain_topo;
    std::istringstream chain_iss(chain_input_str);
    chain_topo.Read(chain_iss, false);
    KALDI_ASSERT(chain_topo.MinLength(3) == 1);
  }

  {  // make sure GetDefaultTopology does not crash.
    std::vector<int32> phones;
    phones.push_back(1);
    phones.push_back(2);
    GetDefaultTopology(phones);
  }
}


}

int main() {
  // repeat the test ten times
  for (int i = 0; i < 10; i++) {
    kaldi::TestTopology();
  }
  std::cout << "Test OK.\n";
}
