// nnet3bin/nnet3-info.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-nnet.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Print some text information about 'raw' nnet3 neural network, to\n"
        "standard output\n"
        "\n"
        "Usage:  nnet3-info [options] <raw-nnet>\n"
        "e.g.:\n"
        " nnet3-info 0.raw\n"
        "See also: nnet3-am-info\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet_rxfilename = po.GetArg(1);
    
    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);

    std::cout << nnet.Info();

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

/*
Test script:

cat <<EOF | nnet3-init --binary=false - - | nnet3-info -
component name=affine1 type=NaturalGradientAffineComponent input-dim=72 output-dim=59
component name=relu1 type=RectifiedLinearComponent dim=59
component name=final_affine type=NaturalGradientAffineComponent input-dim=59 output-dim=298
component name=logsoftmax type=SoftmaxComponent dim=298
input-node name=input dim=18
component-node name=affine1_node component=affine1 input=Append(Offset(input, -4), Offset(input, -3), Offset(input, -2), Offset(input, 0))
component-node name=nonlin1 component=relu1 input=affine1_node
component-node name=final_affine component=final_affine input=nonlin1
component-node name=output_nonlin component=logsoftmax input=final_affine
output-node name=output input=output_nonlin  
EOF
*/
