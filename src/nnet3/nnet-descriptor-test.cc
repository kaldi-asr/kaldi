// nnet3/nnet-descriptor-test.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-descriptor.h"

namespace kaldi {
namespace nnet3 {

ForwardingDescriptor *GenRandForwardingDescriptor(int32 num_nodes) {
  if (Rand() % 2 != 0) {
    return new SimpleForwardingDescriptor(Rand() % num_nodes);
  } else {
    int32 r = Rand() % 4;
    if (r == 0) {
      Index offset;
      offset.t = Rand() % 5;
      offset.x = Rand() % 2;
      return
          new OffsetForwardingDescriptor(GenRandForwardingDescriptor(num_nodes),
                                         offset);
    } else if (r == 1) {
      std::vector<ForwardingDescriptor*> vec;
      int32 n = 1 + Rand() % 3;
      for (int32 i = 0; i < n; i++)
        vec.push_back(GenRandForwardingDescriptor(num_nodes));
      return new SwitchingForwardingDescriptor(vec);
    } else if (r == 2) {
      return new RoundingForwardingDescriptor(
          GenRandForwardingDescriptor(num_nodes), 1 + Rand() % 4);
    } else {
      return new ReplaceIndexForwardingDescriptor(
          GenRandForwardingDescriptor(num_nodes),
          (Rand() % 2 == 0 ? ReplaceIndexForwardingDescriptor::kT :
           ReplaceIndexForwardingDescriptor::kX),
          -2 + Rand() % 4);
    }
  }
}

// generates a random descriptor.
SumDescriptor *GenRandSumDescriptor(
    int32 num_nodes) {
  if (Rand() % 3 != 0) {
    bool required = (Rand() % 2 == 0);
    return new UnarySumDescriptor(GenRandForwardingDescriptor(num_nodes),
                                  required);
  } else {
    return new BinarySumDescriptor((Rand() % 2 == 0 ? BinarySumDescriptor::kSum:
                                    BinarySumDescriptor::kFailover),
                                   GenRandSumDescriptor(num_nodes),
                                   GenRandSumDescriptor(num_nodes));
  }
}


// generates a random descriptor.
void GenRandDescriptor(int32 num_nodes,
                       Descriptor *desc) {
  int32 num_parts = 1 + Rand() % 3;
  std::vector<SumDescriptor*> parts;
  for (int32 part = 0; part < num_parts; part++)
    parts.push_back(GenRandSumDescriptor(num_nodes));
  *desc = Descriptor(parts);                    

}


// This function tests both the I/O for the descriptors, and the
// Copy() function.
void UnitTestDescriptorIo() {
  for (int32 i = 0; i < 100; i++) {
    int32 num_nodes = 1 + Rand() % 5;
    std::vector<std::string> node_names(num_nodes);
    for (int32 i = 0; i < node_names.size(); i++) {
      std::ostringstream ostr;
      ostr << "a" << (i+1);
      node_names[i] = ostr.str();
    }
    Descriptor desc;
    std::ostringstream ostr;
    GenRandDescriptor(num_nodes, &desc);
    desc.WriteConfig(ostr, node_names);

    Descriptor desc2(desc), desc3, desc4;
    desc3 = desc;
    std::vector<std::string> tokens;
    DescriptorTokenize(ostr.str(), &tokens);
    tokens.push_back("end of input");
    std::istringstream istr(ostr.str());
    const std::string *next_token = &(tokens[0]);
    bool ans = desc4.Parse(node_names, &next_token);
    KALDI_ASSERT(ans);
    
    std::ostringstream ostr2;
    desc2.WriteConfig(ostr2, node_names);
    std::ostringstream ostr3;
    desc3.WriteConfig(ostr3, node_names);
    std::ostringstream ostr4;
    desc4.WriteConfig(ostr4, node_names);    

    KALDI_ASSERT(ostr.str() == ostr2.str());
    KALDI_ASSERT(ostr.str() == ostr3.str());
    KALDI_LOG << "x = " << ostr.str();
    KALDI_LOG << "y = " << ostr4.str();
    KALDI_ASSERT(ostr.str() == ostr4.str());
  }
}


} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;

  UnitTestDescriptorIo();

  KALDI_LOG << "Nnet descriptor tests succeeded.";

  return 0;
}
