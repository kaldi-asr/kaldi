// sgmmbin/sgmm-calc-distances.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

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
#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compute matrix of approximated K-L divergences between states\n"
        "Only works properly if a single substate per state.\n"
        "Usage: sgmm-calc-distances [options] model-in occs-in distances-out\n";

    bool binary = true;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        occs_in_filename = po.GetArg(2),
        distances_out_filename = po.GetArg(3);
    

    AmSgmm am_sgmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      TransitionModel trans_model;
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }
    
    Vector<BaseFloat> occs;
    ReadKaldiObject(occs_in_filename, &occs);

    Matrix<BaseFloat> dists(am_sgmm.NumPdfs(), am_sgmm.NumPdfs());
    AmSgmmFunctions::ComputeDistances(am_sgmm, occs, &dists);

    Output ko(distances_out_filename, binary);
    dists.Write(ko.Stream(), binary);

    KALDI_LOG << "Wrote distances to " << distances_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


