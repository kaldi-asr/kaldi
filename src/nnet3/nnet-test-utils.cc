// nnet3/nnet-test-utils.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include "nnet3/nnet-test-utils.h"

namespace kaldi {
namespace nnet3 {

// A simple case, just to get started.
// Generate a single config with one input, splicing, and one hidden layer.
void GenerateConfigSequenceSimple(
    const NnetGenerationConfig &opts,
    std::vector<std::string> *configs) {
  std::ostringstream os;

  std::vector<int32> splice_context;
  for (int32 i = -5; i < 4; i++)
    if (rand() % 3 == 0)
      splice_context.push_back(i);
  if (splice_context.empty())
    splice_context.push_back(0);
  
  int32 input_dim = 10 + rand() % 20,
      spliced_dim = input_dim * splice_context.size(),
      output_dim = 100 + rand() % 200,
      hidden_dim = 40 + rand() % 50;
  os << "component name=affine1 type=NaturalGradientAffineComponent input-dim="
     << spliced_dim << " output-dim=" << hidden_dim << std::endl;
  os << "component name=affine2 type=NaturalGradientAffineComponent input-dim="
     << hidden_dim << " output-dim=" << output_dim << std::endl;
  os << "component name=logsoftmax type=LogSoftmaxComponent dim="
     << output_dim << std::endl;
  os << "input-node name=input dim=" << input_dim << std::endl;
  
  os << "component-node name=affine1_node component=affine1 input=Append(";
  for (size_t i = 0; i < splice_context.size(); i++) {
    int32 offset = splice_context[i];
    os << "Offset(input, " << offset << ") ";
    if (i + 1 < splice_context.size())
      os << ", ";
  }
  os << ")\n";
  os << "component-node name=affine2 component=affine2 input=affine1_node\n";
  os << "component-node name=output_nonlin component=logsoftmax input=affine2\n";
  os << "output-node name=output input=output_nonlin\n";
  configs->push_back(os.str());
}


void GenerateConfigSequence(
    const NnetGenerationConfig &opts,
    std::vector<std::string> *configs) {
  int32 network_type = RandInt(0, 0);
  switch(network_type) {
    case 0:
      GenerateConfigSequenceSimple(opts, configs);
    default:
      ;
  }
}

void ComputeExampleComputationRequestSimple(
    const Nnet &nnet,
    std::vector<Matrix<BaseFloat> > *inputs) {
  KALDI_ASSERT(IsSimpleNnet(nnet));
  
  int32 num_output_frames = 1 + rand() % 10;
  // TODO.
}




} // namespace nnet3
} // namespace kaldi
