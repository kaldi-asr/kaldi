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
    bool not_required = (Rand() % 5 == 0);
    if (not_required)
      return new OptionalSumDescriptor(GenRandSumDescriptor(num_nodes));
    else
      return new SimpleSumDescriptor(GenRandForwardingDescriptor(num_nodes));
  } else {
    return new BinarySumDescriptor(
        (Rand() % 2 == 0 ? BinarySumDescriptor::kSumOperation:
         BinarySumDescriptor::kFailoverOperation),
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
    if (ostr.str() != ostr4.str()) {
      KALDI_WARN << "x and y differ: checking that it's due to Offset normalization.";
      KALDI_ASSERT(ostr.str().find("Offset(Offset") != std::string::npos ||
                   (ostr.str().find("Offset(") != std::string::npos &&
                    ostr.str().find(", 0)") != std::string::npos));
    }
  }
}


// This function tests GeneralDescriptor, but only for correctly-normalized input.
void UnitTestGeneralDescriptor() {
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

    Descriptor desc2(desc), desc3;
    desc3 = desc;
    std::vector<std::string> tokens;
    DescriptorTokenize(ostr.str(), &tokens);
    tokens.push_back("end of input");
    std::istringstream istr(ostr.str());
    const std::string *next_token = &(tokens[0]);


    GeneralDescriptor *gen_desc = GeneralDescriptor::Parse(node_names,
                                                           &next_token);

    if (*next_token != "end of input")
      KALDI_ERR << "Parsing Descriptor, expected end of input but got "
                << "'" <<  *next_token << "'";

    Descriptor *desc4 = gen_desc->ConvertToDescriptor();
    std::ostringstream ostr2;
    desc4->WriteConfig(ostr2, node_names);
    KALDI_LOG << "Original descriptor was: " << ostr.str();
    KALDI_LOG << "Parsed descriptor was: " << ostr2.str();
    if (ostr2.str() != ostr.str())
      KALDI_WARN << "Strings differed.  Check manually.";

    delete gen_desc;
    delete desc4;
  }
}


// normalizes the text form of a descriptor.
std::string NormalizeTextDescriptor(const std::vector<std::string> &node_names,
                                    const std::string &desc_str) {
  std::vector<std::string> tokens;
  DescriptorTokenize(desc_str, &tokens);
  tokens.push_back("end of input");
  const std::string *next_token = &(tokens[0]);
  GeneralDescriptor *gen_desc = GeneralDescriptor::Parse(node_names,
                                                         &next_token);
  if (*next_token != "end of input")
    KALDI_ERR << "Parsing Descriptor, expected end of input but got "
              << "'" <<  *next_token << "'";
  Descriptor *desc = gen_desc->ConvertToDescriptor();
  std::ostringstream ostr;
  desc->WriteConfig(ostr, node_names);
  delete gen_desc;
  delete desc;
  KALDI_LOG << "Result of normalizing " << desc_str << " is: " << ostr.str();
  return ostr.str();
}

void UnitTestGeneralDescriptorSpecial() {
  std::vector<std::string> names;
  names.push_back("a");
  names.push_back("b");
  names.push_back("c");
  names.push_back("d");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "a") == "a");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "Scale(-1.0, a)") == "Scale(-1, a)");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "Scale(-1.0, Scale(-2.0, a))") == "Scale(2, a)");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "Scale(2.0, Sum(Scale(2.0, a), b, c))") ==
               "Sum(Scale(4, a), Sum(Scale(2, b), Scale(2, c)))");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "Const(1.0, 512)") == "Const(1, 512)");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "Sum(Const(1.0, 512), Scale(-1.0, a))") ==
               "Sum(Const(1, 512), Scale(-1, a))");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "Offset(Offset(a, 3, 5), 2, 1)")
               == "Offset(a, 5, 6)");

  KALDI_ASSERT(NormalizeTextDescriptor(names, "Offset(Sum(a, b), 2, 1)") ==
               "Sum(Offset(a, 2, 1), Offset(b, 2, 1))");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "Sum(Append(a, b), Append(c, d))") ==
               "Append(Sum(a, c), Sum(b, d))");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "Append(Append(a, b), Append(c, d))") ==
               "Append(a, b, c, d)");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "Sum(a, b, c, d)") ==
               "Sum(a, Sum(b, Sum(c, d)))");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "Sum(a)") == "a");
  KALDI_ASSERT(NormalizeTextDescriptor(names, "Offset(a, 0)") == "a");
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;


  UnitTestGeneralDescriptorSpecial();
  UnitTestGeneralDescriptor();
  UnitTestDescriptorIo();


  KALDI_LOG << "Nnet descriptor tests succeeded.";

  return 0;
}
