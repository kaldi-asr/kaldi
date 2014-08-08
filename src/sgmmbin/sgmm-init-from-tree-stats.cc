// sgmmbin/sgmm-init-from-tree-stats.cc

// Copyright 2012   Arnab Ghoshal  Johns Hopkins University (Author: Daniel Povey)
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
#include "sgmm/am-sgmm.h"
#include "sgmm/sgmm-clusterable.h"
#include "sgmm/estimate-am-sgmm.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"
#include "tree/build-tree-utils.h"



namespace kaldi {
void InitAndOutputSgmm(const HmmTopology &topo,
                       const AmSgmm &am_sgmm,
                       const ContextDependency &ctx_dep,
                       const std::vector<SpMatrix<double> > &H,
                       const BuildTreeStatsType &stats,
                       const std::string &sgmm_wxfilename,
                       bool binary) {
  int32 num_pdfs = ctx_dep.NumPdfs();
  AmSgmm am_sgmm_out;
  am_sgmm_out.CopyGlobalsInitVecs(am_sgmm, am_sgmm.PhoneSpaceDim(),
                                  am_sgmm.SpkSpaceDim(), num_pdfs);
  MleAmSgmmOptions opts; // Use default options; we can change this later
  // if we need to use any non-default options.
  MleAmSgmmUpdater updater(opts);
      
  std::vector<BuildTreeStatsType> split_stats;      
  SplitStatsByMap(stats, ctx_dep.ToPdfMap(), &split_stats);
  // Make sure each leaf has stats.
  for (size_t i = 0; i < split_stats.size(); i++)
    KALDI_ASSERT(! split_stats[i].empty() && "Tree has leaves with no stats."
                 "  Modify your roots file as necessary to fix this.");
  std::vector<Clusterable*> summed_stats;
  SumStatsVec(split_stats, &summed_stats);

  std::vector<SgmmClusterable*> &summed_sgmm_stats =
      *(reinterpret_cast<std::vector<SgmmClusterable*>*> (&summed_stats));

  for (int32 iter = 0; iter < 5; iter++) { // Update for
    // several iterations; we're starting from zero so we won't
    // converge exactly on the first iteration.
    updater.UpdatePhoneVectorsCheckedFromClusterable(summed_sgmm_stats,
                                                     H,
                                                     &am_sgmm_out);
  }
  DeletePointers(&summed_stats);

  TransitionModel trans_model_out(ctx_dep, topo);
  {
    Output ko(sgmm_wxfilename, binary);
    am_sgmm_out.ComputeNormalizers();
    trans_model_out.Write(ko.Stream(), binary);
    am_sgmm_out.Write(ko.Stream(), binary, kSgmmWriteAll);
  }
}

}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize an SGMM from a previously built SGMM, a tree, \n"
        "and SGMM-type tree stats\n"
        "Usage: sgmm-init-from-tree-stats [options] <old-sgmm> <tree> <sgmm-tree-stats> <sgmm-out>\n";

    bool binary = true;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string sgmm_in_filename = po.GetArg(1),
        tree_in_filename = po.GetArg(2),
        tree_stats_filename = po.GetArg(3),
        sgmm_out_filename = po.GetArg(4);

    AmSgmm am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(sgmm_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    const HmmTopology &topo = trans_model.GetTopo();
    std::vector<SpMatrix<double> > H;
    am_sgmm.ComputeH(&H);

    ContextDependency ctx_dep;
    {
      bool binary_in;
      Input ki(tree_in_filename.c_str(), &binary_in);
      ctx_dep.Read(ki.Stream(), binary_in);
    }

    BuildTreeStatsType stats;
    {
      bool binary_in;
      SgmmClusterable sc(am_sgmm, H);  // dummy stats needed to provide
      // type info, and access to am_sgmm and H.
      Input ki(tree_stats_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, sc, &stats);
    }
    KALDI_LOG << "Number of separate statistics is " << stats.size();
    
    InitAndOutputSgmm(topo, am_sgmm, ctx_dep, H, stats,
                      sgmm_out_filename, binary);
    
    KALDI_LOG << "Written model to " << sgmm_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


