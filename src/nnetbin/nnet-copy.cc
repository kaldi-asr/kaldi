// nnetbin/nnet-copy.cc

// Copyright 2012-2015  Brno University of Technology (author: Karel Vesely)

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
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-parallel-component.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    const char *usage =
      "Copy Neural Network model (and possibly change binary/text format)\n"
      "Usage:  nnet-copy [options] <model-in> <model-out>\n"
      "e.g.:\n"
      " nnet-copy --binary=false nnet.mdl nnet_txt.mdl\n";

    bool binary_write = true;
    int32 remove_first_components = 0;
    int32 remove_last_components = 0;
    BaseFloat dropout_rate = -1.0;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Register("remove-first-layers", &remove_first_components,
        "Deprecated, please use --remove-first-components");
    po.Register("remove-last-layers", &remove_last_components,
        "Deprecated, please use --remove-last-components");

    po.Register("remove-first-components", &remove_first_components,
        "Remove N first Components from the Nnet");
    po.Register("remove-last-components", &remove_last_components,
        "Remove N last layers Components from the Nnet");

    po.Register("dropout-rate", &dropout_rate,
        "Probability that neuron is dropped"
        "(-1.0 keeps original value).");

    std::string from_parallel_component;
    po.Register("from-parallel-component", &from_parallel_component,
        "Extract nested network from parallel component (two possibilities: "
        "'3' = search for ParallelComponent and get its 3rd network; "
        "'1:3' = get 3nd network from 1st component; ID = 1..N).");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // load the network
    Nnet nnet;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    // eventually replace 'nnet' by nested network from <ParallelComponent>,
    if (from_parallel_component != "") {
      std::vector<int32> component_id_nested_id;
      kaldi::SplitStringToIntegers(from_parallel_component, ":", false,
                                   &component_id_nested_id);
      // parse the argument,
      int32 component_id = -1, nested_id = 0;
      switch (component_id_nested_id.size()) {
        case 1:
          nested_id = component_id_nested_id[0];
          break;
        case 2:
          component_id = component_id_nested_id[0];
          nested_id = component_id_nested_id[1];
          break;
        default:
          KALDI_ERR << "Check the csl '--from-parallel-component='"
                    << from_parallel_component
                    << " There must be 1 or 2 elements.";
      }
      // search for first <ParallelComponent> (we don't know component_id yet),
      if (component_id == -1) {
        for (int32 i = 0; i < nnet.NumComponents(); i++) {
          if (nnet.GetComponent(i).GetType() == Component::kParallelComponent) {
            component_id = i+1;
            break;
          }
        }
      }
      // replace the nnet,
      KALDI_ASSERT(nnet.GetComponent(component_id-1).GetType() ==
                   Component::kParallelComponent);
      ParallelComponent& parallel_comp =
        dynamic_cast<ParallelComponent&>(nnet.GetComponent(component_id-1));
      nnet = parallel_comp.GetNestedNnet(nested_id-1);  // replace!
    }

    // optionally remove N first components,
    if (remove_first_components > 0) {
      for (int32 i = 0; i < remove_first_components; i++) {
        nnet.RemoveComponent(0);
      }
    }

    // optionally remove N last components,
    if (remove_last_components > 0) {
      for (int32 i = 0; i < remove_last_components; i++) {
        nnet.RemoveLastComponent();
      }
    }

    // dropout,
    if (dropout_rate != -1.0) {
      nnet.SetDropoutRate(dropout_rate);
    }

    // store the network,
    {
      Output ko(model_out_filename, binary_write);
      nnet.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written 'nnet1' to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
