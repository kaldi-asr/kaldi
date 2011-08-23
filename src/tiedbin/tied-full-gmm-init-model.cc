// tiedbin/tied-full-gmm-init-model.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "gmm/full-gmm.h"
#include "tied/am-tied-full-gmm.h"
#include "tied/mle-am-tied-full-gmm.h"
#include "hmm/transition-model.h"
#include "tree/build-tree-utils.h"
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

namespace kaldi {

/// InitAmGmm initializes the GMM with one Gaussian per state.
void InitAmTiedFullGmm(const BuildTreeStatsType &stats,
               const EventMap &to_pdf_map,
               AmTiedFullGmm *am_gmm) {
  // Get stats split by tree-leaf ( == pdf):
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, to_pdf_map, &split_stats);
  // Make sure each leaf has stats.
  for (size_t i = 0; i < split_stats.size(); i++)
    KALDI_ASSERT(! split_stats[i].empty() && "Tree has leaves with no stats."
                 "  Modify your roots file as necessary to fix this.");
  
  KALDI_ASSERT(static_cast<int32>(split_stats.size()-1) == to_pdf_map.MaxResult()
               && "Tree may have final leaf with no stats.  "
               "Modify your roots file as necessary to fix this.");
  std::vector<Clusterable*> summed_stats;
  SumStatsVec(split_stats, &summed_stats);

  // very basic for now, just initialize all states on the same codebook
  TiedGmm *tied = new TiedGmm();
  tied->Setup(0, am_gmm->GetPdf(0).NumGauss());
  for (size_t i = 0; i < summed_stats.size(); i++) {
    KALDI_ASSERT(summed_stats[i] != NULL);
    am_gmm->AddTiedPdf(*tied);
  }
  delete tied;
  am_gmm->ComputeGconsts();
  DeletePointers(&summed_stats);
}

/// Get state occupation counts.
void GetOccs(const BuildTreeStatsType &stats,
             const EventMap &to_pdf_map,
             Vector<BaseFloat> *occs) {

    // Get stats split by tree-leaf ( == pdf):
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, to_pdf_map, &split_stats);
  // Make sure each leaf has stats.
  for (size_t i = 0; i < split_stats.size(); i++)
    KALDI_ASSERT(! split_stats[i].empty() && "Tree has leaves with no stats."
                 "  Modify your roots file as necessary to fix this.");
  KALDI_ASSERT(static_cast<int32>(split_stats.size()-1) == to_pdf_map.MaxResult()
               && "Tree may have final leaf with no stats.  "
               "Modify your roots file as necessary to fix this.");
  occs->Resize(split_stats.size());
  for (int32 pdf = 0; pdf < occs->Dim(); pdf++)
    (*occs)(pdf) = SumNormalizer(split_stats[pdf]);
}

}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Train decision tree\n"
        "Usage:  gmm-init-model [options] <tree-in> <tree-stats-in> <topo-file> <full-gmm> <model-out>\n"
        "e.g.: \n"
        "  gmm-init-model tree treeacc topo tree full.ubm 1.mdl\n";

    bool binary = false;
    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("write-occs", &occs_out_filename, "File to write state "
                "occupancies to.");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        tree_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        topo_filename = po.GetArg(3),
        cb_filename = po.GetArg(4),
        model_out_filename = po.GetArg(5);

    ContextDependency ctx_dep;
    {
      bool binary_in;
      Input ki(tree_filename.c_str(), &binary_in);
      ctx_dep.Read(ki.Stream(), binary_in);
    }

    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input is(stats_filename, &binary_in);
      ReadBuildTreeStats(is.Stream(), binary_in, gc, &stats);
    }
    KALDI_LOG << "Number of separate statistics is " << stats.size();

    HmmTopology topo;
    {
      bool binary_in;
      Input ki(topo_filename, &binary_in);
      topo.Read(ki.Stream(), binary_in);
    }


    std::vector<int32> phone2num_pdf_classes;
    topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);

    const EventMap &to_pdf = ctx_dep.ToPdfMap();  // not owned here.

    FullGmm cb;
    {
      bool binary_in;
      Input ki(cb_filename, &binary_in);
      cb.Read(ki.Stream(), binary_in);
    }

    // Now, the summed_stats will be used to initialize the GMM.
    AmTiedFullGmm am_gmm;
    am_gmm.Init(cb);
    InitAmTiedFullGmm(stats, to_pdf, &am_gmm);  // Normal case: initialize 1 Gauss/model from tree stats.

    if (!occs_out_filename.empty()) {  // write state occs
      Vector<BaseFloat> occs;
      GetOccs(stats, to_pdf, &occs);
      Output ko(occs_out_filename, binary);
      occs.Write(ko.Stream(), binary);
    }

    TransitionModel trans_model(ctx_dep, topo);

    {
      Output os(model_out_filename, binary);
      trans_model.Write(os.Stream(), binary);
      am_gmm.Write(os.Stream(), binary);
    }
    KALDI_LOG << "Wrote tree and model.";

    DeleteBuildTreeStats(&stats);
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
