// sgmmbin/sgmm-info.cc

// Copyright 2012  Arnab Ghoshal

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

#include <iomanip>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Print various information about an SGMM.\n"
        "Usage: sgmm-info [options] <model-in> [model-in2 ... ]\n";

    bool sgmm_detailed = false;
    bool trans_detailed = false;

    ParseOptions po(usage);
    po.Register("sgmm-detailed", &sgmm_detailed,
                "Print detailed information about substates.");
    po.Register("trans-detailed", &trans_detailed,
                "Print detailed information about transition model.");

    po.Read(argc, argv);
    if (po.NumArgs() < 1) {
      po.PrintUsage();
      exit(1);
    }

    for (int i = 1, max = po.NumArgs(); i <= max; ++i) {
      std::string model_in_filename = po.GetArg(i);
      AmSgmm am_sgmm;
      TransitionModel trans_model;
      {
        bool binary;
        Input ki(model_in_filename, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_sgmm.Read(ki.Stream(), binary);
      }

      {
        using namespace std;
        cout.setf(ios::left);
        cout << "\nModel file: " << model_in_filename << endl;
        cout << " SGMM information:\n"
          << setw(40) << "  # of HMM states" << am_sgmm.NumPdfs() << endl
          << setw(40) << "  # of Gaussians per state" << am_sgmm.NumGauss() << endl
          << setw(40) << "  Dimension of phone vector space"
          << am_sgmm.PhoneSpaceDim() << endl
          << setw(40) << "  Dimension of speaker vector space"
          << am_sgmm.SpkSpaceDim() << endl
          << setw(40) << "  Dimension of feature vectors"
          << am_sgmm.FeatureDim() << endl;
        int32 total_substates = 0;
        for (int32 j = 0; j < am_sgmm.NumPdfs(); j++) {
          total_substates += am_sgmm.NumSubstates(j);
          if (sgmm_detailed) {
            cout << "  # of substates for state " << setw(13) << j
                 << am_sgmm.NumSubstates(j) << endl;
          }
        }
        cout << setw(40) << "  Total # of substates " << total_substates << endl;

        cout << "\nTransition model information:\n"
             << setw(40) << " # of HMM states" << trans_model.NumPdfs() << endl
             << setw(40) << " # of transition states"
             << trans_model.NumTransitionStates() << endl;
          int32 total_indices = 0;
          for (int32 s = 0; s < trans_model.NumTransitionStates(); s++) {
            total_indices += trans_model.NumTransitionIndices(s);
            if (trans_detailed) {
              cout << "  # of transition ids for state " << setw(8) << s
                   << trans_model.NumTransitionIndices(s) << endl;
            }
          }
          cout << setw(40) << "  Total # of transition ids " << total_indices
               << endl;
      }
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


