// gmmbin/gmm-init-model.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
//                     Johns Hopkins University  (author: Guoguo Chen)

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/mle-am-diag-gmm.h"
#include "tree/build-tree-utils.h"
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

namespace kaldi {

/// InitAmGmm initializes the GMM with one Gaussian per state.
void InitAmGmm(const BuildTreeStatsType &stats,
               const EventMap &to_pdf_map,
               AmDiagGmm *am_gmm,
               const TransitionModel &trans_model,
               BaseFloat var_floor) {
  // Get stats split by tree-leaf ( == pdf):
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, to_pdf_map, &split_stats);

  split_stats.resize(to_pdf_map.MaxResult() + 1); // ensure that
  // if the last leaf had no stats, this vector still has the right size.
  
  // Make sure each leaf has stats.
  for (size_t i = 0; i < split_stats.size(); i++) {
    if (split_stats[i].empty()) {
      std::vector<int32> bad_pdfs(1, i), bad_phones;
      GetPhonesForPdfs(trans_model, bad_pdfs, &bad_phones);
      std::ostringstream ss;
      for (int32 idx = 0; idx < bad_phones.size(); idx ++)
        ss << bad_phones[idx] << ' ';
      KALDI_WARN << "Tree has pdf-id " << i 
          << " with no stats; corresponding phone list: " << ss.str();
      /*
        This probably means you have phones that were unseen in training 
        and were not shared with other phones in the roots file. 
        You should modify your roots file as necessary to fix this.
        (i.e. share that phone with a similar but seen phone on one line 
        of the roots file). Be sure to regenerate roots.int from roots.txt, 
        if using s5 scripts. To work out the phone, search for 
        pdf-id  i  in the output of show-transitions (for this model). */
    }
  }
  std::vector<Clusterable*> summed_stats;
  SumStatsVec(split_stats, &summed_stats);
  Clusterable *avg_stats = SumClusterable(summed_stats);
  KALDI_ASSERT(avg_stats != NULL && "No stats available in gmm-init-model.");
  for (size_t i = 0; i < summed_stats.size(); i++) {
    GaussClusterable *c =
        static_cast<GaussClusterable*>(summed_stats[i] != NULL ? summed_stats[i] : avg_stats);
    DiagGmm gmm;
    Vector<BaseFloat> x (c->x_stats());
    Vector<BaseFloat> x2 (c->x2_stats());
    BaseFloat count =  c->count();
    gmm.Resize(1, x.Dim());
    if (count < 100) {
      std::vector<int32> bad_pdfs(1, i), bad_phones;
      GetPhonesForPdfs(trans_model, bad_pdfs, &bad_phones);
      std::ostringstream ss;
      for (int32 idx = 0; idx < bad_phones.size(); idx ++)
        ss << bad_phones[idx] << ' ';
      KALDI_WARN << "Very small count for state " << i << ": " 
          << count << "; corresponding phone list: " << ss.str();
    }
    x.Scale(1.0/count);
    x2.Scale(1.0/count);
    x2.AddVec2(-1.0, x);  // subtract mean^2.
    x2.ApplyFloor(var_floor);
    x2.InvertElements();  // get inv-var.
    KALDI_ASSERT(x2.Min() > 0);

    Matrix<BaseFloat> mean(1, x.Dim());
    mean.Row(0).CopyFromVec(x);
    Matrix<BaseFloat> inv_var(1, x.Dim());
    inv_var.Row(0).CopyFromVec(x2);

    gmm.SetInvVarsAndMeans(inv_var, mean);
    Vector<BaseFloat> weights(1);
    weights(0) = 1.0;
    gmm.SetWeights(weights);
    gmm.ComputeGconsts();
    am_gmm->AddPdf(gmm);
  }
  DeletePointers(&summed_stats);
  delete avg_stats;
}

/// Get state occupation counts.
void GetOccs(const BuildTreeStatsType &stats,
             const EventMap &to_pdf_map,
             Vector<BaseFloat> *occs) {

    // Get stats split by tree-leaf ( == pdf):
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, to_pdf_map, &split_stats);
  if (split_stats.size() != to_pdf_map.MaxResult()+1) {
    KALDI_ASSERT(split_stats.size() < to_pdf_map.MaxResult()+1);
    split_stats.resize(to_pdf_map.MaxResult()+1);
  }
  occs->Resize(split_stats.size());
  for (int32 pdf = 0; pdf < occs->Dim(); pdf++)
    (*occs)(pdf) = SumNormalizer(split_stats[pdf]);
}



/// InitAmGmmFromOld initializes the GMM based on a previously trained
/// model and tree, which must require no more phonetic context than
/// the current tree.  It does this by finding the closest PDF in the
/// old model, to any given PDF in the new model.  Here, "closest" is
/// defined as: has the largest count in common from the tree stats.

void InitAmGmmFromOld(const BuildTreeStatsType &stats,
                      const EventMap &to_pdf_map,
                      int32 N,  // context-width
                      int32 P,  // central-position
                      const std::string &old_tree_rxfilename,
                      const std::string &old_model_rxfilename,
                      AmDiagGmm *am_gmm) {

  AmDiagGmm old_am_gmm;
  ContextDependency old_tree;
  {  // Read old_gm_gmm
    bool binary_in;
    TransitionModel old_trans_model;
    Input ki(old_model_rxfilename, &binary_in);
    old_trans_model.Read(ki.Stream(), binary_in);
    old_am_gmm.Read(ki.Stream(), binary_in);
  }
  {  // Read tree.
    bool binary_in;
    Input ki(old_tree_rxfilename, &binary_in);
    old_tree.Read(ki.Stream(), binary_in);
  }


  // Get stats split by (new) tree-leaf ( == pdf):
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, to_pdf_map, &split_stats);
  // Make sure each leaf has stats.
  for (size_t i = 0; i < split_stats.size(); i++)
    KALDI_ASSERT(! split_stats[i].empty() && "Tree has leaves with no stats."
                 "  Modify your roots file as necessary to fix this.");
  KALDI_ASSERT(static_cast<int32>(split_stats.size()-1) == to_pdf_map.MaxResult()
               && "Tree may have final leaf with no stats.  "
               "Modify your roots file as necessary to fix this.");

  int32 oldN = old_tree.ContextWidth(), oldP = old_tree.CentralPosition();

  const EventMap &old_map = old_tree.ToPdfMap();

  KALDI_ASSERT(am_gmm->NumPdfs() == 0);
  int32 num_pdfs = static_cast<int32>(split_stats.size());
  for (int32 pdf = 0; pdf < num_pdfs; pdf++) {
    BuildTreeStatsType &my_stats = split_stats[pdf];
    // The next statement converts the stats to a possibly narrower older
    // context-width (e.g. triphone -> monophone).
    // note: don't get confused by the "old" and "new" in the parameters
    // to ConvertStats.  The next line is correct.
    bool ret = ConvertStats(N, P, oldN, oldP, &my_stats);
    if (!ret)
      KALDI_ERR << "InitAmGmmFromOld: old system has wider context "
          "so cannot convert stats.";
    // oldpdf_to_count works out a map from old pdf-id to count (for stats
    // that align to this "new" pdf... we'll use it to work out the old pdf-id
    // that's "closest" in stats overlap to this new pdf ("pdf").
    std::map<int32, BaseFloat> oldpdf_to_count;
    KALDI_ASSERT(!my_stats.empty());  // would be code error; checked already.
    for (size_t i = 0; i < my_stats.size(); i++) {
      EventType evec = my_stats[i].first;
      EventAnswerType ans;
      bool ret = old_map.Map(evec, &ans);
      if (!ret) { KALDI_ERR << "Could not map context using old tree."; }
      KALDI_ASSERT(my_stats[i].second != NULL);
      BaseFloat stats_count = my_stats[i].second->Normalizer();
      if (oldpdf_to_count.count(ans) == 0) oldpdf_to_count[ans] = stats_count;
      else oldpdf_to_count[ans] += stats_count;
    }
    KALDI_ASSERT(!oldpdf_to_count.empty());
    BaseFloat max_count = 0; int32 max_old_pdf = -1;
    for (std::map<int32, BaseFloat>::const_iterator iter = oldpdf_to_count.begin();
        iter != oldpdf_to_count.end();
        ++iter) {
      if (iter->second > max_count) {
        max_count = iter->second;
        max_old_pdf = iter->first;
      }
    }
    KALDI_ASSERT(max_count != 0 && max_old_pdf != -1);

    am_gmm->AddPdf(old_am_gmm.GetPdf(max_old_pdf));  // Here is where we copy the relevant old PDF.
  }
}



}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize GMM from decision tree and tree stats\n"
        "Usage:  gmm-init-model [options] <tree-in> <tree-stats-in> <topo-file> <model-out> [<old-tree> <old-model>]\n"
        "e.g.: \n"
        "  gmm-init-model tree treeacc topo 1.mdl\n"
        "or (initializing GMMs with old model):\n"
        "  gmm-init-model tree treeacc topo 1.mdl prev/tree prev/30.mdl\n";

    bool binary = true;
    double var_floor = 0.01;
    std::string occs_out_filename;


    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("write-occs", &occs_out_filename, "File to write state "
                "occupancies to.");
    po.Register("var-floor", &var_floor, "Variance floor used while "
                "initializing Gaussians");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4 && po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        tree_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        topo_filename = po.GetArg(3),
        model_out_filename = po.GetArg(4),
        old_tree_filename = po.GetOptArg(5),
        old_model_filename = po.GetOptArg(6);

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);

    BuildTreeStatsType stats;
    {
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(stats_filename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }
    KALDI_LOG << "Number of separate statistics is " << stats.size();

    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    const EventMap &to_pdf = ctx_dep.ToPdfMap();  // not owned here.

    TransitionModel trans_model(ctx_dep, topo);
    
    // Now, the summed_stats will be used to initialize the GMM.
    AmDiagGmm am_gmm;
    if (old_tree_filename.empty())
      InitAmGmm(stats, to_pdf, &am_gmm, trans_model, var_floor);  // Normal case: initialize 1 Gauss/model from tree stats.
    else {
      InitAmGmmFromOld(stats, to_pdf,
                       ctx_dep.ContextWidth(),
                       ctx_dep.CentralPosition(),
                       old_tree_filename,
                       old_model_filename,
                       &am_gmm);
    }

    if (!occs_out_filename.empty()) {  // write state occs
      Vector<BaseFloat> occs;
      GetOccs(stats, to_pdf, &occs);
      Output ko(occs_out_filename, binary);
      occs.Write(ko.Stream(), binary);
    }

    {
      Output ko(model_out_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Wrote model.";
    
    DeleteBuildTreeStats(&stats);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
