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
#include "nnet3/nnet-utils.h"

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
  os << "component name=relu1 type=RectifiedLinearComponent dim="
     << hidden_dim << std::endl;
  os << "component name=affine2 type=NaturalGradientAffineComponent input-dim="
     << hidden_dim << " output-dim=" << output_dim << std::endl;
  os << "component name=logsoftmax type=LogSoftmaxComponent dim="
     << output_dim << std::endl;
  os << "input-node name=input dim=" << input_dim << std::endl;
  
  os << "component-node name=affine1_node component=affine1 input=Append(";
  for (size_t i = 0; i < splice_context.size(); i++) {
    int32 offset = splice_context[i];
    os << "Offset(input, " << offset << ")";
    if (i + 1 < splice_context.size())
      os << ", ";
  }
  os << ")\n";
  os << "component-node name=nonlin1 component=relu1 input=affine1_node\n";
  os << "component-node name=affine2 component=affine2 input=nonlin1\n";
  os << "component-node name=output_nonlin component=logsoftmax input=affine2\n";
  os << "output-node name=output input=output_nonlin\n";
  configs->push_back(os.str());
}


// This generates a single config corresponding to an RNN.
void GenerateConfigSequenceRnn(
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
  os << "component name=nonlin1 type=RectifiedLinearComponent dim="
     << hidden_dim << std::endl;
  os << "component name=recurrent_affine1 type=NaturalGradientAffineComponent input-dim="
     << hidden_dim << " output-dim=" << hidden_dim << std::endl;
  os << "component name=affine2 type=NaturalGradientAffineComponent input-dim="
     << hidden_dim << " output-dim=" << output_dim << std::endl;
  os << "component name=logsoftmax type=LogSoftmaxComponent dim="
     << output_dim << std::endl;
  os << "input-node name=input dim=" << input_dim << std::endl;
  
  os << "component-node name=affine1_node component=affine1 input=Append(";
  for (size_t i = 0; i < splice_context.size(); i++) {
    int32 offset = splice_context[i];
    os << "Offset(input, " << offset << ")";
    if (i + 1 < splice_context.size())
      os << ", ";
  }
  os << ")\n";
  os << "component-node name=recurrent_affine1 component=recurrent_affine1 "
        "input=Offset(nonlin1, -1)\n";
  os << "component-node name=nonlin1 component=nonlin1 "
        "input=Sum(affine1_node, IfDefined(recurrent_affine1))\n";
  os << "component-node name=affine2 component=affine2 input=nonlin1\n";
  os << "component-node name=output_nonlin component=logsoftmax input=affine2\n";
  os << "output-node name=output input=output_nonlin\n";
  configs->push_back(os.str());
}


void GenerateConfigSequence(
    const NnetGenerationConfig &opts,
    std::vector<std::string> *configs) {
start:
  int32 network_type = RandInt(0, 1);
  switch(network_type) {
    case 0:
      GenerateConfigSequenceSimple(opts, configs);
    case 1:
      if (!opts.allow_recursion)  // disallow.
        goto start;
      GenerateConfigSequenceRnn(opts, configs);
    default:
      ;
  }
}

void ComputeExampleComputationRequestSimple(
    const Nnet &nnet,
    ComputationRequest *request,
    std::vector<Matrix<BaseFloat> > *inputs) {
  KALDI_ASSERT(IsSimpleNnet(nnet));

  int32 left_context, right_context;
  ComputeSimpleNnetContext(nnet, &left_context, &right_context);

  int32 num_output_frames = 1 + rand() % 10,
      output_start_frame = rand() % 10,
      num_examples = 1 + rand() % 10,
      output_end_frame = output_start_frame + num_output_frames,
      input_start_frame = output_start_frame - left_context - (rand() % 3),
      input_end_frame = output_end_frame + right_context + (rand() % 3),
      n_offset = rand() % 2;


  request->inputs.clear();
  request->outputs.clear();
  inputs->clear();
  
  std::vector<Index> input_indexes, ivector_indexes, output_indexes;
  for (int32 n = n_offset; n < n_offset + num_examples; n++) {
    for (int32 t = input_start_frame; t < input_end_frame; t++)
      input_indexes.push_back(Index(n, t, 0));
    for (int32 t = output_start_frame; t < output_end_frame; t++)
      output_indexes.push_back(Index(n, t, 0));
    ivector_indexes.push_back(Index(n, 0, 0));
  }
  request->outputs.push_back(IoSpecification("output", output_indexes));
  request->inputs.push_back(IoSpecification("input", input_indexes));
  int32 input_dim = nnet.InputDim("input");
  KALDI_ASSERT(input_dim > 0);
  inputs->push_back(
      Matrix<BaseFloat>((input_end_frame - input_start_frame) * num_examples,
                        input_dim));
  inputs->back().SetRandn();
  int32 ivector_dim = nnet.InputDim("ivector");  // may not exist.
  if (ivector_dim != -1) {
    request->inputs.push_back(IoSpecification("ivector", ivector_indexes));
    inputs->push_back(Matrix<BaseFloat>(num_examples, ivector_dim));
    inputs->back().SetRandn();
  }
}




} // namespace nnet3
} // namespace kaldi
