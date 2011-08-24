// gmmbin/gmm-init-model.cc

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

#include <vector>

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

using std::map;
using std::vector;

/// Initialize the tied densities of the AmTiedFull using the
/// AmTiedFullGmm (with set codebooks) and a list of codebook ids for the stats
void InitAmTiedFullGmm(AmTiedFullGmm *am_gmm, const vector<int32> *tied_to_pdf) {
  TiedGmm *tied = new TiedGmm();

  // initialize for ever leaf
  for (int32 i = 0; i < tied_to_pdf->size() ; i++) {
    int32 pdfid = (*tied_to_pdf)[i];

    // make sure we have this codebook
    KALDI_ASSERT(pdfid < am_gmm->NumPdfs());

    // link to codebook and allocate uniform weights
    tied->Setup(pdfid, am_gmm->GetPdf(pdfid).NumGauss());
    am_gmm->AddTiedPdf(*tied);
  }

  delete tied;
  
  am_gmm->ComputeGconsts();
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Train decision tree\n"
        "Usage:  tied-full-gmm-init-model [options] <tree> <topo> <tied_to_pdf_map> <full-ubm0> [full-ubm0 ...] <model-out>\n"
        "e.g.: \n"
        "  tied-full-gmm-init-model tree topo tiedmap diag0.ubm diag1.ubm 1.mdl\n";

    bool binary = false;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() < 5) {
      po.PrintUsage();
      exit(1);
    }

    // use the last argument as output
    std::string 
      tree_filename = po.GetArg(1),
      topo_filename = po.GetArg(2),
      tied_to_pdf_file = po.GetArg(3),
      model_out_filename = po.GetArg(po.NumArgs());

    ContextDependency ctx_dep;
    {
      bool binary_in;
      Input ki(tree_filename.c_str(), &binary_in);
      ctx_dep.Read(ki.Stream(), binary_in);
    }

    HmmTopology topo;
    {
      bool binary_in;
      Input ki(topo_filename, &binary_in);
      topo.Read(ki.Stream(), binary_in);
    }

    // read in tied->pdf map
    std::vector<int32> tied_to_pdf;
    {
      bool binary_in;
      Input ki(tied_to_pdf_file, &binary_in);
      ReadIntegerVector(ki.Stream(), binary_in, &tied_to_pdf);
    }

    KALDI_ASSERT(tied_to_pdf.size() > 0);

    // subsequently add the codebooks
    AmTiedFullGmm am_gmm;
    for (int32 i = 4; i < po.NumArgs() - 1; ++i) {
      FullGmm cb;
      bool binary_in;
      Input ki(po.GetArg(i), &binary_in);
      cb.Read(ki.Stream(), binary_in);
      
      if (i == 0)
        am_gmm.Init(cb);
      else
        am_gmm.AddPdf(cb);
    } 

    // Init the model by allocating the tied mixtures
    InitAmTiedFullGmm(&am_gmm, &tied_to_pdf);  

    TransitionModel trans_model(ctx_dep, topo);
    {
      Output os(model_out_filename, binary);
      trans_model.Write(os.Stream(), binary);
      am_gmm.Write(os.Stream(), binary);
    }
    KALDI_LOG << "Wrote tree and model.";

    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
