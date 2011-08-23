// tiedbin/smooth-stats-diag.cc

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
#include "matrix/kaldi-vector.h"
#include "tree/context-dep.h"
#include "tied/mle-am-tied-diag-gmm.h"
#include "tied/mle-tied-gmm.h"

using namespace kaldi;
using std::vector;

int main(int argc, char *argv[]) {
  try {

    const char *usage =
        "Smooth the sufficient statistics of tied models by propagating them up and\n"
        "interpolating them down in the tree hierarchy.\n"
        "Usage:   smooth-stats-diag [options] tree acc-in acc-out\n"
        "e.g.: \n"
        " smooth-stats-diag tri1-tied/tree tri1-tied/3.acc tri1-tied/3.acc.smoothed\n";
    ParseOptions po(usage);

    bool propagate = true;
    bool interpolate = true;

    BaseFloat rho = 10.;

    po.Register("rho", &rho, "Interpolation factor, rho(i) = rho / (rho + gamma(i)");
    po.Register("propagate", &propagate, "Propagate the sufficient statistics");
    po.Register("interpolate", &interpolate, "Interpolate the sufficient statistics");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_in_filename = po.GetArg(1);
    std::string acc_in_filename = po.GetArg(2);
    std::string acc_out_filename = po.GetArg(3);

    ContextDependency ctx_dep;  // the tree.
    {
      bool binary;
      Input is(tree_in_filename, &binary);
      ctx_dep.Read(is.Stream(), binary);
    }

    Vector<double> trans_accs;
    AccumAmTiedDiagGmm acc;
    {
      bool binary;
      Input is(acc_in_filename, &binary);
      trans_accs.Read(is.Stream(), binary);
      acc.Read(is.Stream(), binary, false);
    }

    int32 num_leaves;
    vector<int32> p;
    if (!GetTreeStructure(ctx_dep.ToPdfMap(), &num_leaves, &p)) {
      KALDI_ERR << "Could not flatten tree!";
      return 1;
    }

    // validate all leaf accumulators
    std::cout << "Supposedly " << num_leaves << " leaves in the tree. Querying accumulators..." << std::endl;
    for (int32 i = 0; i < num_leaves; ++i) {
        const AccumTiedGmm &a = acc.GetTiedAcc(i);
        std::cout << "model(" << i << ") p=" << p[i] << " gamma(i)=" << a.occupancy().Sum() << std::endl 
                  << "  tracing:";
        int32 pp = p[i];
        while (pp != p[pp]) {
          std::cout << " " << pp;
          pp = p[pp];
        }
        std::cout << std::endl;
    }

    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


