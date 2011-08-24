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

#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-vector.h"
#include "tree/context-dep.h"
#include "tied/mle-tied-gmm.h"
#include "tied/mle-am-tied-diag-gmm.h"

using namespace kaldi;
using std::vector;

int main(int argc, char *argv[]) {
  try {
    const char *usage =
        "Smooth the sufficient statistics of tied models by propagating them up and\n"
        "interpolating them down in the tree hierarchy (w.r.t. their codebook)\n"
        "Usage:   smooth-stats-diag [options] tree tiedmap acc-in acc-out\n"
        "e.g.: \n"
        " smooth-stats-diag tri1-tied/tree tri1-tied/tiedmap tri1-tied/3.acc tri1-tied/3.acc.smoothed\n";
    ParseOptions po(usage);

    bool propagate = true;
    bool interpolate = true;

    BaseFloat rho = 10.;

    po.Register("rho", &rho, "Interpolation factor, rho(i) = rho / (rho + gamma(i)");
    po.Register("propagate", &propagate, "Propagate the sufficient statistics");
    po.Register("interpolate", &interpolate, "Interpolate the sufficient statistics");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string 
      tree_in_filename = po.GetArg(1),
      tied_to_pdf_file = po.GetArg(2),
      acc_in_filename = po.GetArg(3),
      acc_out_filename = po.GetArg(4);

    if (interpolate && !propagate) {
      propagate = true;
      std::cout << "Activating propagation due to requested interpolation." << std::endl;
    }

    if (!propagate) {
      std::cout << "Nothing to do. Bye." << std::endl;
      return 0;
    }

    ContextDependency ctx_dep;  // the tree.
    {
      bool binary;
      Input is(tree_in_filename, &binary);
      ctx_dep.Read(is.Stream(), binary);
    }

    // read in tied->pdf map
    std::vector<int32> tied_to_pdf;
    {
      bool binary_in;
      Input ki(tied_to_pdf_file, &binary_in);
      ReadIntegerVector(ki.Stream(), binary_in, &tied_to_pdf);
    }

    Vector<double> trans_accs;
    bool binary_accu;
    AccumAmTiedDiagGmm acc;
    {
      Input is(acc_in_filename, &binary_accu);
      trans_accs.Read(is.Stream(), binary_accu);
      acc.Read(is.Stream(), binary_accu, false);
    }

    int32 num_leaves;
    vector<int32> p;
    if (!GetTreeStructure(ctx_dep.ToPdfMap(), &num_leaves, &p)) {
      KALDI_ERR << "Could not flatten tree!";
      return 1;
    }

    KALDI_ASSERT(num_leaves == tied_to_pdf.size());

    // walk up the tree, validate accumulators, allocate interim allocators 
    // and remember trace
    std::vector<std::vector<int32> > trace(p.size() - num_leaves);
    std::vector<AccumTiedGmm *> interim(p.size() - num_leaves, NULL);

    std::cout << "Propagating " << num_leaves << " leaves" << std::endl;
    for (int32 i = 0; i < num_leaves; ++i) {
        AccumTiedGmm &a = acc.GetTiedAcc(i);
        std::cout << "model-id=" << i << " p=" << p[i] << " occ=" << a.occupancy().Sum() << " ==>"; 
        int32 cur = i, par = p[i] ;

        // walk up, as long as the parent is not the root (aka selfloop)
        while (par != p[par]) {
          int32 k = par - num_leaves;

          // check if the current node is already listed as child
          bool present = false;
          for (std::vector<int32>::iterator it = trace[k].begin(),
               end = trace[k].end(); it != end && !present; ++it)
            present |= (*it == cur);
          if (!present)
            trace[k].push_back(cur);

          // add accumulator
          if (interim[k] == NULL) {
            std::cout << " alloc:" << par;
            interim[k] = new AccumTiedGmm(a);
          } else {
            std::cout << " " << par;
            a.Propagate(interim[k]);
          }

          cur = par;
          par = p[cur];
        }
        // add the current node to the root node
        // check if the current node is already listed as child
        bool present = false;
        for (std::vector<int32>::iterator it = trace[par-num_leaves].begin(),
             end = trace[par-num_leaves].end(); it != end && !present; ++it)
          present |= (*it == cur);
        if (!present)
          trace[trace.size()-1].push_back(cur);
        std::cout << std::endl;
    }

    if (interpolate) {
      std::cout << "Interpolating, rho=" << rho << std::endl;

      // interpolate down, beginning from the top
      int32 k = trace.size() - 1;
      for (std::vector<std::vector<int32> >::reverse_iterator rit = trace.rbegin(), rend = trace.rend();
           rit != rend; ++rit, --k) {
        std::cout << (k + num_leaves) << " <==";

        // the root will have some trace, but we're not propagating over the root
        if (interim[k] == NULL) {
          std::cout<< " (null -- skipping; sizeof trace = " << rit->size() << ")" << std::endl;
          continue;
        }

        for (std::vector<int32>::iterator it = (*rit).begin(), end = (*rit).end();
             it != end; ++it) {
          int32 t = *it;
          if (t < num_leaves) {
            // this will be a pdf accumulator
            std::cout << " " << t;
            acc.GetTiedAcc(t).Interpolate(rho, interim[k]);
          } else {
            // this will be an interim accumulator
            std::cout << " interim:" << t;
            interim[t-num_leaves]->Interpolate(rho, interim[k]);
          }
        }

        std::cout << std::endl;
      }
    }

    {
      Output os(acc_out_filename, binary_accu);
      trans_accs.Write(os.Stream(), binary_accu);
      acc.Write(os.Stream(), binary_accu);
    }

    std::cout << "Wrote " << acc_out_filename << std::endl;

    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


