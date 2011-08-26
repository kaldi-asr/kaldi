// tiedbin/init-full-codebooks.cc

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
    KALDI_ASSERT(! split_stats[i].empty() && "Tree has leaves with no stats."
                 "  Modify your roots file as necessary to fix this.");
                 
  KALDI_ASSERT(static_cast<int32>(split_stats.size()-1) == to_pdf_map.MaxResult()
               && "Tree may have final leaf with no stats.  "
               "Modify your roots file as necessary to fix this.");
  
  std::vector<Clusterable*> summed_stats;
  SumStatsVec(split_stats, &summed_stats);

  for (size_t i = 0; i < summed_stats.size(); i++) {
    KALDI_ASSERT(summed_stats[i] != NULL);
    DiagGmm *gmm = new DiagGmm();
    Vector<double> x (static_cast<GaussClusterable*>(summed_stats[i])->x_stats());
    Vector<double> x2 (static_cast<GaussClusterable*>(summed_stats[i])->x2_stats());
    BaseFloat count =  static_cast<GaussClusterable*>(summed_stats[i])->count();
    gmm->Resize(1, x.Dim());
    
    if (count < 100)
      KALDI_VLOG(1) << "Very small count for state "<< i << ": " << count;
 
    x.Scale(1. / count);
    x2.Scale(1. / count);
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

int main(int argc, char *argv[]) {
  try {
    const char *usage =
        "Generate codebooks for a tied mixture system based on the accumulated "
        "tree stats and given two-level tree. Will write to <codebook-out-base>.(num)\n"
        "Usage:  init-full-codebooks [options] <tree-in> <tree-stats-in> <tree-map> <codebook-out-base>\n"
        "e.g.: \n"
        "  init-full-codebooks tree tree.acc tree.map ubm-full\n";

    bool binary = false;
    int max_num_gaussians = 512;
    bool split_gaussians = false;
    BaseFloat perturb = 0.01;
    int min_num_gaussians = 1;
    bool full = false;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("max-gauss", &max_num_gaussians, "Maximum number of total "
                "gaussians to allocate (including all codebooks.)");
    po.Register("split-gaussians", &split_gaussians, "If the resulting "
                "codebook(s) have a total number of gaussians less then the max "
                "split the components");
    po.Register("perturb", &perturb, "Perturbation factor for gaussian splitting.");
    po.Register("min-gauss", &min_num_gaussians, "Minimum number of gaussians per codebook");
    po.Register("full", &full, "Write full covariance models with cov-floor covariances.");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        tree_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        tied_to_pdf_file = po.GetArg(3),
        model_out_filebase = po.GetArg(4);

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
    std::vector<DiagGmm *> leafs;
    AllocateDiagGmms(stats, to_pdf, &leafs);

    KALDI_ASSERT(leafs.size() > 0);
    
    KALDI_LOG << "Allocated " << leafs.size() << " 1-comp Gmms, now merging";
    int32 dim = leafs[0]->Dim();
    
    // step 2: walk up the tree see which gmms to merge
    // read in tied to pdf map
    std::vector<int32> tied_to_pdf;
    {
      bool binary_in;
      Input ki(tied_to_pdf_file, &binary_in);
      ReadIntegerVector(ki.Stream(), binary_in, &tied_to_pdf);
    }
    
    KALDI_ASSERT(tied_to_pdf.size() == leafs.size());
    
    // determine the number of codebooks from the map: max() + 1 due to indexing
    int32 num_pdf = 0;
    for (std::vector<int32>::iterator it = tied_to_pdf.begin(),
         end = tied_to_pdf.end(); it != end; ++it)
      if (*it > num_pdf) num_pdf = *it;
    num_pdf++;
    
    // query the tree structure
    int32 num_tied;
    std::vector<int32> p;
    GetTreeStructure(to_pdf, &num_tied, &p);
    
    KALDI_LOG << "Allocating num-pdf=" << num_pdf << " num-tied=" << num_tied;
      
    // distribute the leafs to the codebooks
    std::vector<std::vector<int32> > comp(num_pdf);
        
    int32 k = 0;
    for (std::vector<int32>::iterator it = tied_to_pdf.begin(),
         end = tied_to_pdf.end(); it != end; ++it, ++k)
      comp[*it].push_back(k); 
  
    // allocate the codebooks
    std::vector<DiagGmm *> pdfs;
    for (int32 i = 0; i < num_pdf; ++i) {
      int c = comp[i].size();
      KALDI_ASSERT(c > 0);
      KALDI_LOG << "pdf-id=" << i << " <== " << c << " components";
        
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
      npdfsi.weights_.Scale(1. / npdfsi.weights_.Sum());
      npdfsi.CopyToDiagGmm(pdfs[i]);
    }
    
    // resize the codebooks to match the requested number of gaussians
    int32 totg = 0;
    int32 ming = 0;
    int32 actual = 0; // actual components, might differ from requested due to rounding
    for (int32 i = 0; i < num_pdf; ++i) {
      int32 ng = pdfs[i]->NumGauss();
      totg += ng;
      if (ng < min_num_gaussians)
        ming += ng;
    }

    if (totg != max_num_gaussians) {
      // compute target number of gaussians per codebook
      BaseFloat r = (BaseFloat) (max_num_gaussians-ming) / (totg-ming); // larger than one: split!
        
      if (r > 1. && !split_gaussians) {
        KALDI_WARN << "Have less than target gaussians, but split-gaussians=false!";
      } else {
        KALDI_LOG << "Resizing from " << totg << " to " << max_num_gaussians 
                  << " r=" << r << " ==> " << (r > 1. ? "split" : "merge");
        
        // merge/split codebook gaussians to achieve desired size; use slack
        // variable to get a straight fit in the end!
        BaseFloat slack = 0.f;
        int32 k = 0;
        for (std::vector<DiagGmm *>::iterator it = pdfs.begin(), end = pdfs.end();
             it != end; ++it, ++k) {
         // approx target number of gaussians
         BaseFloat tf = r * (*it)->NumGauss();
         
         // don't touch gaussians that would get too
         if (tf < min_num_gaussians) {
           actual += (*it)->NumGauss();
           continue;
         }
         
         // actual number of target gaussians w.r.t. slack from prior merge/split
         int32 ti = round(tf + slack);
         slack = tf + slack - ti;
            
         actual += ti;
           
         KALDI_VLOG(1) << "pdf-id=" << k << " " << (*it)->NumGauss()
                       << " >> " << ti << " slack=" << slack;
           
         if (r > 1.)
           (*it)->Split(ti, perturb);
         else
           (*it)->Merge(ti);
        }
      }
    } // end resize
    
    if (actual != max_num_gaussians) {
      KALDI_LOG << "Actual number of gaussians after resize: " << actual;
      int32 n = abs(max_num_gaussians - actual);
      
      if (n > num_pdf) {
        KALDI_ERR << "Could not automatically balance for min-gaussians, select"
                     " smaller value!";
        return 1;
      }
      
      KALDI_LOG << "Resizing by " << n << " gaussians...";
        
      // split in top n largest gmms
      int *ndx = new int [n];
      int *val = new int [n];
      
      int i = 0;
      for (i = 0; i < n; ++i) { ndx[i] = val[i] = 0; }
      i = 0;
      for (std::vector<DiagGmm *>::iterator it = pdfs.begin(), end = pdfs.end();
           it != end; ++it, ++i) {
        int32 s = (*it)->NumGauss();
							
        // do we need to consider this density score at all?
        if (s < val[0])
          continue;
								
			  // locate the insert position
        int32 ptr = 0;
        while (ptr < n - 1 && s > val[ptr + 1])
          ptr++;
								
        // shift the old values and indices
        for (int j = 1; j <= ptr; ++j) { 
          ndx[j-1] = ndx[j]; 
          val[j-1] = val[j];
        }
								
        // insert
        ndx[ptr] = i;
        val[ptr] = s;
      }
        
        // split top n largest gmms
      for (int32 i = 0; i < n; ++i) {
        if (max_num_gaussians > actual) {
          KALDI_VLOG(1) << "Splitting pdf-id=" << ndx[i];
          pdfs[ndx[i]]->Split(val[i] + 1, perturb);
        } else {
          KALDI_VLOG(1) << "Splitting pdf-id=" << ndx[i];
          pdfs[ndx[i]]->Merge(val[i] -1);
        }
      }
      
      delete [] val;
      delete [] ndx;
    }
    
    // write out codebooks
    for (int32 i = 0; i < pdfs.size(); ++i) {
      KALDI_LOG << "Writing out " << model_out_filebase << "." << i;
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
        
        for (int32 i = 0; i < pdfs[i]->NumGauss(); ++i) {
          for (int32 j = 0; j < dim; ++j) {
            fgn.vars_[i](j, j) = dgn.vars_.Row(i)(j);
            for (int32 k = j+1; k < dim; ++k) {
              fgn.vars_[i](j, k) = var;
              fgn.vars_[i](k, j) = var;
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
