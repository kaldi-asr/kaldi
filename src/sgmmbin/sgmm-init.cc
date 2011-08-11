// sgmmbin/sgmm-init.cc

// Copyright 2009-2011   Saarland University
// Author:  Arnab Ghoshal

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

#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize an SGMM from a trained full-covariance UBM and a specified"
        " model topology.\n"
        "Usage: sgmm-init [options] <topology-in> <tree-in> <ubm-in> <sgmm-out>\n";
    
    bool binary = false;
    int32 phn_space_dim = 0, spk_space_dim = 0;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("phn-space-dim", &phn_space_dim, "Phonetic space dimension.");
    po.Register("spk-space-dim", &spk_space_dim, "Speaker space dimension.");


    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string topo_in_filename = po.GetArg(1),
        tree_in_filename = po.GetArg(2),
        ubm_in_filename = po.GetArg(3),
        sgmm_out_filename = po.GetArg(4);

    ContextDependency ctx_dep;
    {
      bool binary_in;
      Input ki(tree_in_filename.c_str(), &binary_in);
      ctx_dep.Read(ki.Stream(), binary_in);
    }


    HmmTopology topo;
    {
      bool binary_in;
      Input ki(topo_in_filename, &binary_in);
      topo.Read(ki.Stream(), binary_in);
    }
    
    TransitionModel trans_model(ctx_dep, topo);    

    kaldi::FullGmm ubm;
    {
      bool binary_read;
      kaldi::Input is(ubm_in_filename, &binary_read);
      ubm.Read(is.Stream(), binary_read);
    }

    kaldi::AmSgmm sgmm;
    sgmm.InitializeFromFullGmm(ubm, trans_model.NumPdfs(), phn_space_dim,
                               spk_space_dim);
    sgmm.ComputeNormalizers();

    {
      kaldi::Output os(sgmm_out_filename, binary);
      trans_model.Write(os.Stream(), binary);
      sgmm.Write(os.Stream(), binary, kaldi::kSgmmWriteAll);
    }

    KALDI_LOG << "Written model to " << sgmm_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


