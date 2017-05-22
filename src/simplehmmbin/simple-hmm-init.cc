// bin/simple-hmm-init.cc

// Copyright 2016   Vimal Manohar (Johns Hopkins University)

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
#include "hmm/hmm-topology.h"
#include "simplehmm/simple-hmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize simple HMM from topology.\n"
        "Usage:  simple-hmm-init <topology-in> <model-out>\n"
        "e.g.: \n"
        " simple-hmm-init topo init.mdl\n";

    bool binary = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string topo_filename = po.GetArg(1);
    std::string model_filename = po.GetArg(2);
    
    HmmTopology topo;
    {
      bool binary_in;
      Input ki(topo_filename, &binary_in);
      topo.Read(ki.Stream(), binary_in);
    }

    SimpleHmm model(topo);
    {
      Output ko(model_filename, binary);
      model.Write(ko.Stream(), binary);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


