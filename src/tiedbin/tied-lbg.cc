// tiedbin/tied-lbg.cc

// Copyright 2011 Univ. Erlangen Nuremberg, Korbinian Riedhammer

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

#include <vector>
#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/kaldi-io.h"
#include "util/text-utils.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "gmm/diag-gmm-normal.h"
#include "gmm/mle-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"

namespace kaldi {

using std::vector;


/// Perform an LBG-like initialization for each codebook by sequences of
/// splitting and EM. The mapping distributes the feature vectors to the 
/// codebooks; the Gmms in the output vector will be allocated. Input parameters
/// are: minimum number of Gaussians per codebook, total number of Gaussians, 
/// the sample->codebook mapping, feature matrix, the number of interim EM 
/// iterations and the desired MleFullGmmOptions
void Lbg(int32 min_n, int32 total_n, vector<int32> &mapping, 
         vector<Vector<BaseFloat> > &features, int32 em_it, 
         BaseFloat perturb, const MleDiagGmmOptions &config, 
         vector<DiagGmm *> &gmms) {
  // make sure the mapping is complete
  KALDI_ASSERT(features.size() == mapping.size());
  
  KALDI_LOG << "features.size() = " << features.size();

  // determine number of codebooks to allocate via the mapped ids
  int32 numc = 0;
  for (vector<int32>::iterator it = mapping.begin(), end = mapping.end();
       it != end; ++it) {
    if ((*it + 1) > numc) numc = (*it + 1);
  }

  // allocate Gmms
  KALDI_LOG << "Estimating " << numc << " single Gaussians";
  vector<int32> target_size(numc, 0);
  gmms.resize(numc, NULL);
  for (int32 i = 0; i < numc; i++) {
    // ML estimate for single component
    Vector<double> x;
    Vector<double> x2;
    
    x.Resize(features[0].Dim());
    x2.Resize(features[0].Dim());
    
    int32 n = 0;
    vector<int32>::iterator mit, mend;
    vector<Vector<BaseFloat> >::iterator fit, fend;
    for (mit = mapping.begin(), mend = mapping.end(), fit = features.begin(),
         fend = features.end(); mit != mend && fit != fend; ++mit, ++fit) {
      // accumulate only for target Gmm
      if (*mit != i) continue;

      x.AddVec(1.0, *fit);
      x2.AddVec2(1.0, Vector<double>(*fit));
      ++n;
    }
    
    KALDI_ASSERT(n > 0 
                 && "No observations for codebook, check tree and alignment!");
    
    // normalize parameters
    x.Scale(1.0 / n);
    x2.Scale(1.0 / n);
    x2.AddVec2(-1.0, x);
    
    // transfer to normal gmm...
    DiagGmm *diag = new DiagGmm();
    diag->Resize(1, x.Dim());
    DiagGmmNormal ngmm(*diag);
    ngmm.weights_(0) = 1.0;
    ngmm.means_.CopyRowFromVec(x, 0);
    ngmm.vars_.CopyRowFromVec(x2, 0);
    ngmm.CopyToDiagGmm(diag);
    diag->ComputeGconsts();
    
    // save params
    gmms[i] = diag;
    target_size[i] = n; // we keep the occupancy for now
  }
  
  // determine target number of Gaussians for each codebook
  double slack = 0.;
  for (int32 i = 0; i < numc; i++) {
    // this codebook's share of the freely distributable Gaussians
    double sd = static_cast<double> (total_n - numc * min_n) 
                * target_size[i] / features.size();
    int32 si = round(sd + slack);

    // update slack
    slack = sd + slack - si;

    // assign at least min_num_gaussians
    si += min_n;
    
    target_size[i] = si;
  }
  
  // now for each codebook that hasn't reached target size, do the split and 
  // re-estimate loop
  for (int32 i = 0; i < numc; i++) {
    KALDI_LOG << "Initializing codebook " << i << " with " << target_size[i]
              << " Gaussians";
    DiagGmm *diag = gmms[i];
    while (diag->NumGauss() < target_size[i]) {
      // split
      diag->Split(diag->NumGauss() + 1, perturb);
      
      // iterate acc/est
      for (int32 j = 0; j < em_it; j++) {
        AccumDiagGmm acc(*diag, kGmmAll);
        vector<int32>::iterator mit, mend;
        vector<Vector<BaseFloat> >::iterator fit, fend;
        for (mit = mapping.begin(), mend = mapping.end(), fit = features.begin(),
             fend = features.end(); mit != mend && fit != fend; ++mit, ++fit) {
          if (*mit == i)
            acc.AccumulateFromDiag(*diag, *fit, 1.0);
        }

        MleDiagGmmUpdate(config, acc, kGmmAll, diag, NULL, NULL);
      }
    }
  }
}

}

using namespace kaldi;
using std::vector;
using kaldi::int32;

int main(int argc, char **argv) {
try {
    const char *usage =
        "Generate codebooks for a tied mixture system based on an existing\n"
        "tree and alignment, and write them to codebook-out[.num]\n"
        "If no tree.map is given, a single codebook generated.\n"
        "Usage:  tied-lbg [options] tree-old tree-tied topo features-rspecifier"
        " alignments-rspecifier codebook-out [tree.map]\n"
        "e.g.: \n"
        "  tied-lbg tree-old tree-tied topo scp:train.scp ark:ali ubm-full "
        "tree.map\n";

    bool binary = true;
    bool full = true;
    
    BaseFloat perturb = 0.01;
    
    int32 max_num_gaussians = 512;
    int32 min_num_gaussians = 3;
    int32 interim_em = 5;
    
    MleDiagGmmOptions config;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("max-gauss", &max_num_gaussians, "Maximum number of total "
                "gaussians to allocate (including all codebooks.)");
    po.Register("perturb", &perturb, "Perturbation factor for gaussian "
                "splitting.");
    po.Register("min-gauss", &min_num_gaussians, "Minimum number of gaussians "
                "per codebook");
    po.Register("full", &full, "Estimate full-covariance models");
    po.Register("interim-em", &interim_em, "Number of interim EM iterations "
                "between codebook splits");

    config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() < 6 || po.NumArgs() > 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        oldt_filename = po.GetArg(1),
        tree_filename = po.GetArg(2),
        topo_filename = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        alignment_rspecifier = po.GetArg(5),
        outfile_base = po.GetArg(6),
        tied_to_pdf_file = (po.NumArgs() == 7 ? po.GetArg(7) : "");

    // load trees
    ContextDependency ctx_dep;
    {
      bool binary_in;
      Input ki(tree_filename.c_str(), &binary_in);
      ctx_dep.Read(ki.Stream(), binary_in);
    }

    ContextDependency ctx_dep_old;
    {
      bool binary_in;
      Input ki(oldt_filename.c_str(), &binary_in);
      ctx_dep_old.Read(ki.Stream(), binary_in);
    }
    
    // load topo
    HmmTopology topo;
    ReadKaldiObject(topo_filename, &topo);

    // build transition models
    TransitionModel trans_model(ctx_dep, topo);
    TransitionModel trans_model_old(ctx_dep_old, topo);
    
    // read in tied to pdf map
    vector<int32> tied_to_pdf;
    {
      if (tied_to_pdf_file.length() > 0) {
        bool binary_in;
        Input ki(tied_to_pdf_file, &binary_in);
        ReadIntegerVector(ki.Stream(), binary_in, &tied_to_pdf);
      }
    }

    vector<int32> mapping;
    vector<Vector<BaseFloat> > features;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignment_reader(alignment_rspecifier);
    
    int num_done = 0, num_no_alignment = 0, num_other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!alignment_reader.HasKey(key)) {
        num_no_alignment++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignment_reader.Value(key);
        
        if (alignment.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (alignment.size())
                     << " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        // convert alignment
        std::vector<int32> newalg;
        if (!ConvertAlignment(trans_model_old,
                             trans_model,
                             ctx_dep,
                             alignment,
                             NULL,
                             &newalg)) {
          KALDI_WARN << "Could not convert alignment";
          num_other_error++;
          continue;                    
        }

        // translate alignment into codebook ids and append data
        for (int32 i = 0; i < newalg.size(); i++) {
          // get the pdf associated with this transition
          int32 pdfid = trans_model.TransitionIdToPdf(newalg[i]);
          
          // see if we have multiple codebooks
          if (tied_to_pdf.size() == 0)
            pdfid = 0;
          else
            pdfid = tied_to_pdf[pdfid];

          mapping.push_back(pdfid);
          features.push_back(Vector<BaseFloat>(mat.Row(i)));
        }

        num_done++;
      }
    }

    KALDI_LOG << "Processed " << num_done << " missing alignment for " 
              << num_no_alignment << " " << num_other_error << " other errors";

    vector<DiagGmm *> pdfs;
      
    // initialize the codebooks
    Lbg(min_num_gaussians, max_num_gaussians, mapping, features, interim_em, 
        perturb, config, pdfs);

    // write out codebooks
    if (pdfs.size() > 1) {
      KALDI_LOG << "Writing out " << outfile_base << ".[0.." 
                << (pdfs.size() - 1) << "]";
      for (int32 i = 0; i < pdfs.size(); i++) {
        std::ostringstream str;
        str << outfile_base << "." << i;

        Output ko(str.str(), binary);
        if (full) {
          FullGmm fgmm;
          fgmm.CopyFromDiagGmm(*pdfs[i]);
          fgmm.Write(ko.Stream(), binary);
        } else {
          pdfs[i]->ComputeGconsts();
          pdfs[i]->Write(ko.Stream(), binary);
        }
      }
    } else {
      KALDI_LOG << "Writing out " <<  outfile_base;
      Output ko(outfile_base, binary);
      
      if (full) {
        FullGmm fgmm;
        fgmm.CopyFromDiagGmm(*pdfs[0]);
        fgmm.Write(ko.Stream(), binary);
      } else {
        pdfs[0]->ComputeGconsts();
        pdfs[0]->Write(ko.Stream(), binary);
      }
    }
      
    DeletePointers(&pdfs);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

