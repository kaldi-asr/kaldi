// nnet3/nnet-utils-test.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)
//           2016  Daniel Galvez

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

#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-test-utils.h"

namespace kaldi {
namespace nnet3 {


void UnitTestNnetContext() {
  for (int32 n = 0; n < 20; n++) {
    struct NnetGenerationOptions gen_config;
    
    std::vector<std::string> configs;
    GenerateConfigSequence(gen_config, &configs);
    Nnet nnet;
    std::istringstream is(configs[0]);
    nnet.ReadConfig(is);

    // this test doesn't really test anything except that it runs;
    // we manually inspect the output.
    int32 left_context, right_context;
    ComputeSimpleNnetContext(nnet, &left_context, &right_context);
    KALDI_LOG << "Left,right-context= " << left_context << ","
              << right_context << " for config: " << configs[0];

    KALDI_LOG << "Info for nnet is: " << NnetInfo(nnet);
  }
}

void UnitTestConvertRepeatedToBlockAffine() {
  // a test without a composite component.
  std::string config =
    "component name=repeated-affine1 type=RepeatedAffineComponent "
    "input-dim=100 output-dim=200 num-repeats=20\n"
    "component name=relu1 type=RectifiedLinearComponent dim=200\n"
    "component name=block-affine1 type=BlockAffineComponent "
    "input-dim=200 output-dim=100 num-blocks=10\n"
    "component name=relu2 type=RectifiedLinearComponent dim=100\n"
    "component name=repeated-affine2 type=NaturalGradientRepeatedAffineComponent "
    "input-dim=100 output-dim=200 num-repeats=10\n"
    "\n"
    "input-node name=input dim=100\n"
    "component-node name=repeated-affine1 component=repeated-affine1 input=input\n"
    "component-node name=relu1 component=relu1 input=repeated-affine1\n"
    "component-node name=block-affine1 component=block-affine1 input=relu1\n"
    "component-node name=relu2 component=relu2 component=relu2 input=block-affine1\n"
    "component-node name=repeated-affine2 component=repeated-affine2 input=relu2\n"
    "output-node name=output input=repeated-affine2\n";

  Nnet nnet;
  std::istringstream is(config);
  nnet.ReadConfig(is);
  ConvertRepeatedToBlockAffine(&nnet);

  for(int i = 0; i < nnet.NumComponents(); i++) {
    Component *c = nnet.GetComponent(i);
    KALDI_ASSERT(c->Type() != "RepeatedAffineComponent"
                 && c->Type() != "NaturalGradientRepeatedAffineComponent");
  }
}

void UnitTestConvertRepeatedToBlockAffineComposite() {
  // test that repeated affine components nested within a CompositeComponent
  // are converted.
  struct NnetGenerationOptions gen_config;
  gen_config.output_dim = 0;
  std::vector<std::string> configs;
  // this function generates a neural net with one component:
  // a composite component.
  GenerateConfigSequenceCompositeBlock(gen_config, &configs);
  Nnet nnet;
  std::istringstream is(configs[0]);
  nnet.ReadConfig(is);
  KALDI_ASSERT(nnet.NumComponents() == 1);
  ConvertRepeatedToBlockAffine(&nnet);
  CompositeComponent *cc = dynamic_cast<CompositeComponent*>(nnet.GetComponent(0));
  for(int i = 0; i < cc->NumComponents(); i++) {
    const Component *c = cc->GetComponent(i);
    KALDI_ASSERT(c->Type() == "BlockAffineComponent");
  }
}

} // namespace nnet3
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  SetVerboseLevel(2);

  UnitTestNnetContext();
  UnitTestConvertRepeatedToBlockAffine();
  UnitTestConvertRepeatedToBlockAffineComposite();

  KALDI_LOG << "Nnet tests succeeded.";

  return 0;
}
