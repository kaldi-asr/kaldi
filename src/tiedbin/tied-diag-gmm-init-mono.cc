// tiedbin/tied-diag-gmm-init-mono.cc

// Copyright 2011 Univ. Erlangen-Nuremberg, Korbinian Riedhammer

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
#include "gmm/diag-gmm.h"
#include "tied/am-tied-diag-gmm.h"
#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"

namespace kaldi {
// This function reads a file like:
// 1 2 3
// 4 5
// 6 7 8
// where each line is a list of integer id's of phones (that should have their pdfs shared).
void ReadSharedPhonesList(std::string rxfilename, std::vector<std::vector<int32> > *list_out) {
  list_out->clear();
  Input input(rxfilename);
  std::istream &is = input.Stream();
  std::string line;
  while (std::getline(is, line)) {
    list_out->push_back(std::vector<int32>());
    if (!SplitStringToIntegers(line, " \t\r", true, &(list_out->back())))
      KALDI_ERR << "Bad line in shared phones list: " << line << " (reading "
                << PrintableRxfilename(rxfilename) << ")";
  }
}

} // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize monophone GMM with tied mixtures.\n"
        "Usage:  tied-diag-gmm-init-mono <topology-in> <diag-codebook> <model-out> <tree-out> \n"
        "e.g.: \n"
        " tied-diag-gmm-init-mono topo cb.pdf mono.mdl mono.tree\n";

    bool binary = false;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string topo_filename = po.GetArg(1);
    std::string cb_filename = po.GetArg(2);
    std::string model_filename = po.GetArg(3);
    std::string tree_filename = po.GetArg(4);

    DiagGmm gmm;
    bool binary_in;
    Input ki1(cb_filename, &binary_in);
    gmm.Read(ki1.Stream(), binary_in);

    HmmTopology topo;
    Input ki2(topo_filename, &binary_in);
    topo.Read(ki2.Stream(), binary_in);

    const std::vector<int32> &phones = topo.GetPhones();

    std::vector<int32> phone2num_pdf_classes (1+phones.back());
    for (size_t i = 0; i < phones.size(); i++)
      phone2num_pdf_classes[phones[i]] = topo.NumPdfClasses(phones[i]);

    // Now the tree [not really a tree at this point]:
    ContextDependency *ctx_dep = MonophoneContextDependency(phones, phone2num_pdf_classes);
    
    int32 num_pdfs = ctx_dep->NumPdfs();

    // init the tied model with a single gmm
    AmTiedDiagGmm am;
    am.Init(gmm);
    
    // setup the prototype tied density
    TiedGmm tied;
    tied.Setup(0, gmm.NumGauss());

    for (int i = 0; i < num_pdfs; i++)
      am.AddTiedPdf(tied);
    
    am.ComputeGconsts();

    // Now the transition model:
    TransitionModel trans_model(*ctx_dep, topo);

    {
      Output ko(model_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am.Write(ko.Stream(), binary);
    }

    // Now write the tree.
    ctx_dep->Write(Output(tree_filename, binary).Stream(),
                   binary);

    delete ctx_dep;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

