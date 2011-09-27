// tiedbin/init-tied-codebooks.cc

// Copyright 2011 Univ. Erlangen Nuremberg, Korbinian Riedhammer

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

#include <vector>
#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "gmm/diag-gmm-normal.h"
#include "gmm/full-gmm-normal.h"
#include "tree/build-tree-utils.h"
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

namespace kaldi {

/// Allocate one Gaussian per state/dectreeleaf, will be merged to large ubms
/// later. Note that the weights will be set to the counts to allow a later
/// adjustment of the resulting UBM size
void AllocateDiagGmms(const BuildTreeStatsType &stats,
                      const EventMap &to_pdf_map,
                      std::vector<DiagGmm *> *leafs) {
  KALDI_ASSERT(leafs != NULL);

  // Get stats split by tree-leaf ( == pdf):
  std::vector<BuildTreeStatsType> split_stats;
  SplitStatsByMap(stats, to_pdf_map, &split_stats);

  // Make sure each leaf has stats.
  for (size_t i = 0; i < split_stats.size(); i++)
    KALDI_ASSERT(!split_stats[i].empty() && "Tree has leaves with no stats."
                 "  Modify your roots file as necessary to fix this.");

  KALDI_ASSERT(static_cast<int32>(split_stats.size()-1)
               == to_pdf_map.MaxResult()
               && "Tree may have final leaf with no stats.  "
               "Modify your roots file as necessary to fix this.");

  std::vector<Clusterable*> summed_stats;
  SumStatsVec(split_stats, &summed_stats);

  for (size_t i = 0; i < summed_stats.size(); i++) {
    KALDI_ASSERT(summed_stats[i] != NULL);
    DiagGmm *gmm = new DiagGmm();
    Vector<double> x(static_cast<GaussClusterable*>(
                       summed_stats[i])->x_stats());
    Vector<double> x2(static_cast<GaussClusterable*>(
                       summed_stats[i])->x2_stats());
    BaseFloat count =  static_cast<GaussClusterable*>(summed_stats[i])->count();
    gmm->Resize(1, x.Dim());

    if (count < 100)
      KALDI_VLOG(1) << "Very small count for state "<< i << ": " << count;

    x.Scale(1.0 / count);
    x2.Scale(1.0 / count);
    x2.AddVec2(-1.0, x);  // subtract mean^2.
    KALDI_ASSERT(x2.Min() > 0);

    DiagGmmNormal ngmm(*gmm);
    ngmm.means_.CopyRowFromVec(x, 0);
    ngmm.vars_.CopyRowFromVec(x2, 0);
    ngmm.weights_(0) = count;
    ngmm.CopyToDiagGmm(gmm);

    gmm->ComputeGconsts();
    leafs->push_back(gmm);
  }
  DeletePointers(&summed_stats);
}
}

using namespace kaldi;

using std::vector;

int main(int argc, char *argv[]) {
  try {
    const char *usage =
        "Generate codebooks for a tied mixture system based on the accumulated "
        "tree stats and optional two-level tree. Will write to "
          "<codebook-out>[.num]\n"
        "Usage:  init-tied-codebooks [options] <tree-in> <tree-stats-in> "
          "<codebook-out> [tree.map]\n"
        "e.g.: \n"
        "  init-tied-codebooks tree tree.acc ubm-full tree.map\n";

    bool binary = false;
    int max_num_gaussians = 512;
    bool split_gaussians = false;
    BaseFloat perturb = 0.01;
    int min_num_gaussians = 3;
    bool full = false;
	BaseFloat power = 1.0;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("max-gauss", &max_num_gaussians, "Maximum number of total "
                "gaussians to allocate (including all codebooks.)");
    po.Register("split-gaussians", &split_gaussians, "If the resulting "
                "codebook(s) have a total number of gaussians less then the max"
                " split the components");
    po.Register("perturb", &perturb, "Perturbation factor for gaussian "
                "splitting.");
    po.Register("min-gauss", &min_num_gaussians, "Minimum number of gaussians "
                "per codebook");
    po.Register("full", &full, "Write full covariance models with covariances "
                "set to the min variance");
	po.Register("power", &power, "Power to allocate Gaussians to codebooks");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        tree_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filebase = po.GetArg(3),
        tied_to_pdf_file = (po.NumArgs() == 4 ? po.GetArg(4) : "");

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

    const EventMap &to_pdf = ctx_dep.ToPdfMap();  // not owned here.

    // step 1: build single gaussians per decision tree leaf
    vector<DiagGmm *> leafs;
    AllocateDiagGmms(stats, to_pdf, &leafs);

    KALDI_ASSERT(leafs.size() > 0);

    KALDI_LOG << "Allocated " << leafs.size() << " 1-comp Gmms, now merging";
    int32 dim = leafs[0]->Dim();

    // step 2: walk up the tree see which gmms to merge
    // read in tied to pdf map
    vector<int32> tied_to_pdf;
    {
      if (tied_to_pdf_file.length() > 0) {
        bool binary_in;
        Input ki(tied_to_pdf_file, &binary_in);
        ReadIntegerVector(ki.Stream(), binary_in, &tied_to_pdf);
      } else {
        // allocate dummy map by putting each leaf in the same codebook
        tied_to_pdf.resize(leafs.size(), 0);
      }
    }

    KALDI_ASSERT(tied_to_pdf.size() == leafs.size());

    // determine the number of codebooks from the map: max() + 1 due to indexing
    int32 num_pdf = 0;
    for (vector<int32>::iterator it = tied_to_pdf.begin(),
         end = tied_to_pdf.end(); it != end; ++it)
      if (*it > num_pdf) num_pdf = *it;
    num_pdf++;

    // query the tree structure
    int32 num_tied;
    vector<int32> p;
    GetTreeStructure(to_pdf, &num_tied, &p);

    KALDI_LOG << "Allocating num-pdf=" << num_pdf << " num-tied=" << num_tied;

    // go through the leaves, sum up occupancies and get number of initial comp
    vector<vector<int32> > comp(num_pdf);
    vector<double> occs(num_pdf, 0.);
    double tot_occ = 0.;
    {
      int32 i = 0;
      for (std::vector<int32>::iterator it = tied_to_pdf.begin(),
           end = tied_to_pdf.end(); it != end; ++it, ++i) {
        comp[*it].push_back(i);
        occs[*it] += leafs[i]->weights()(0);
      }

      // we will attribute the Gaussians according to a power law
	  for (int32 i = 0; i < num_pdf; ++i) {
        occs[i] = pow(occs[i], power);
		tot_occ += occs[i];
      }
    }

    // compute target sizes of the codebooks by distributing the number of
    // gaussians according to their share of occupancies
    vector<DiagGmm *> pdfs;
    {
      double slack = 0.;
      for (int32 i = 0; i < num_pdf; ++i) {
        // build initial GMM
        int c = comp[i].size();
        KALDI_ASSERT(c > 0);

        // resize the codebook and get normal form
        pdfs.push_back(new DiagGmm());
        pdfs[i]->Resize(c, dim);
        DiagGmmNormal npdfsi(*(pdfs[i]));

        // add components
        for (int32 j = 0; j < comp[i].size(); ++j) {
          DiagGmmNormal n(*(leafs[comp[i][j]]));
          npdfsi.weights_(j) = n.weights_(0);
          npdfsi.means_.CopyRowFromVec(n.means_.Row(0), j);
          npdfsi.vars_.CopyRowFromVec(n.vars_.Row(0), j);
        }

        // transfer back
        npdfsi.weights_.Scale(1.0 / npdfsi.weights_.Sum());
        npdfsi.CopyToDiagGmm(pdfs[i]);

        // now resize to target number of gaussians
        // we do this by assigning at least min_num gaussians to each codebook
        // and distribute the rest according to their share of total occupancy
        double sd = static_cast<double> (max_num_gaussians -
                                         num_pdf*min_num_gaussians)
                    * occs[i] / tot_occ;
        int32 si = round(sd + slack);

        // update slack
        slack = sd + slack - si;

        // assign at least min_num_gaussians
        si += min_num_gaussians;

        // resize the codebook
        KALDI_LOG << "pdf-id=" << i << " <== init-c=" << c << " final=" << si;
        if (si > c)
          pdfs[i]->Split(si, perturb);
        else if (si < c)
          pdfs[i]->Merge(si);
      }
    }

    // write out codebooks
    KALDI_LOG << "Writing out " << model_out_filebase << ".*";
    for (int32 i = 0; i < pdfs.size(); ++i) {
      std::ostringstream str;
      str << model_out_filebase << "." << i;

      if (full) {
        // Init full covariance Gmms by using var.Min() as covariance
        FullGmm fgmm;
        fgmm.Resize(pdfs[i]->NumGauss(), dim);

        FullGmmNormal fgn(fgmm);
        DiagGmmNormal dgn(*(pdfs[i]));

        fgn.weights_.CopyFromVec(dgn.weights_);
        fgn.means_.CopyFromMat(dgn.means_);

        BaseFloat var = dgn.vars_.Min();

        for (int32 j = 0; j < pdfs[i]->NumGauss(); ++j) {
          for (int32 k = 0; k < dim; ++k) {
            fgn.vars_[j](k, k) = dgn.vars_.Row(j)(k);
            for (int32 l = k+1; l < dim; ++l) {
              fgn.vars_[j](k, l) = var;
              fgn.vars_[j](l, k) = var;
            }
          }
        }

        fgn.CopyToFullGmm(&fgmm);
        fgmm.ComputeGconsts();

        Output ko(str.str(), binary);
        fgmm.Write(ko.Stream(), binary);
      } else {
        Output ko(str.str(), binary);
        pdfs[i]->ComputeGconsts();
        pdfs[i]->Write(ko.Stream(), binary);
      }
    }

    // be nice
    DeletePointers(&pdfs);
    DeletePointers(&leafs);
    DeleteBuildTreeStats(&stats);

    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
