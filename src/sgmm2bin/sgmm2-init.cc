// sgmm2bin/sgmm2-init.cc

// Copyright 2012   Arnab Ghoshal  Johns Hopkins University (author: Daniel Povey)
// Copyright 2009-2011   Saarland University

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

#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "sgmm2/am-sgmm2.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize an SGMM from a trained full-covariance UBM and a specified"
        " model topology.\n"
        "Usage: sgmm2-init [options] <topology> <tree> <init-model> <sgmm-out>\n"
        "The <init-model> argument can be a UBM (the default case) or another\n"
        "SGMM (if the --init-from-sgmm flag is used).\n"
        "For systems with two-level tree, use --pdf-map argument.";
    
    bool binary = true, init_from_sgmm = false, spk_dep_weights = false; // will
    // make it true later.
    int32 phn_space_dim = 0, spk_space_dim = 0;
    std::string pdf_map_rxfilename;
    double self_weight = 1.0;
    
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("phn-space-dim", &phn_space_dim, "Phonetic space dimension.");
    po.Register("spk-space-dim", &spk_space_dim, "Speaker space dimension.");
    po.Register("spk-dep-weights", &spk_dep_weights, "If true, have speaker-"
                "dependent weights (symmetric SGMM)");
    po.Register("init-from-sgmm", &init_from_sgmm,
                "Initialize from another SGMM (instead of a UBM).");
    po.Register("self-weight", &self_weight,
                "If < 1.0, will be the weight of a pdf with its \"own\" mixture, "
                "where we initialize each group with a number of mixtures.  If"
                "1.0, we initialize each group with just one mixture component.");
    po.Register("pdf-map", &pdf_map_rxfilename,
                "For systems with 2-level trees [SCTM systems], the file that "
                "maps from pdfs to groups (from build-tree-two-level)");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string topo_in_filename = po.GetArg(1),
        tree_in_filename = po.GetArg(2),
        init_model_filename = po.GetArg(3),
        sgmm_out_filename = po.GetArg(4);

    ContextDependency ctx_dep;
    {
      bool binary_in;
      Input ki(tree_in_filename.c_str(), &binary_in);
      ctx_dep.Read(ki.Stream(), binary_in);
    }

    std::vector<int32> pdf2group;
    if (pdf_map_rxfilename != "") {
      bool binary_in;
      Input ki(pdf_map_rxfilename, &binary_in);
      ReadIntegerVector(ki.Stream(), binary_in, &pdf2group);
    } else {
      for (int32 i = 0; i < ctx_dep.NumPdfs(); i++) pdf2group.push_back(i);
    }

    
    HmmTopology topo;
    ReadKaldiObject(topo_in_filename, &topo);

    TransitionModel trans_model(ctx_dep, topo);
    
    kaldi::AmSgmm2 sgmm;
    if (init_from_sgmm) {
      kaldi::AmSgmm2 init_sgmm;
      {
        bool binary_read;
        kaldi::Input ki(init_model_filename, &binary_read);
        init_sgmm.Read(ki.Stream(), binary_read);
      }
      sgmm.CopyGlobalsInitVecs(init_sgmm, pdf2group, self_weight);
    } else {
      kaldi::FullGmm ubm;
      {
        bool binary_read;
        kaldi::Input ki(init_model_filename, &binary_read);
        ubm.Read(ki.Stream(), binary_read);
      }
      sgmm.InitializeFromFullGmm(ubm, pdf2group, phn_space_dim,
                                 spk_space_dim, spk_dep_weights,
                                 self_weight);
    }
    sgmm.ComputeNormalizers();

    {
      kaldi::Output ko(sgmm_out_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      sgmm.Write(ko.Stream(), binary, kaldi::kSgmmWriteAll);
    }

    KALDI_LOG << "Written model to " << sgmm_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


