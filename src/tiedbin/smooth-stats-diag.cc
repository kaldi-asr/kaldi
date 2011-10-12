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
#include <sstream>

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
        "Usage:   smooth-stats-diag [options] <tree> <tree.map> <acc-in> <acc-out>\n"
        "e.g.: \n"
        " smooth-stats-diag tree tree.map 12.acc 12.acc.s\n";
    ParseOptions po(usage);

    bool preserve_counts = false;
    bool print_diversity = false;

    BaseFloat tau = 10.;

    po.Register("tau", &tau, "Interpolation factor (see tied-gmm.h: Interpolate{1,2})");
    po.Register("preserve-counts", &preserve_counts, "Preserve the counts, uses Interpolate2");
    po.Register("print-diversity", &print_diversity, "Print the diversity of the nodes, i.e. the childrens' codebooks");

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

    ContextDependency ctx_dep;  // the tree.
    {
      bool binary;
      Input is(tree_in_filename, &binary);
      ctx_dep.Read(is.Stream(), binary);
    }

    // read in tied->pdf map
    vector<int32> tied_to_pdf;
    {
      bool binary_in;
      Input ki(tied_to_pdf_file, &binary_in);
      ReadIntegerVector(ki.Stream(), binary_in, &tied_to_pdf);
    }

    // determine the number of codebooks from the map: max() + 1 due to indexing
    int32 num_pdfs = 0;
    for (vector<int32>::iterator it = tied_to_pdf.begin(),
         end = tied_to_pdf.end(); it != end; ++it) {
      if (*it > num_pdfs)
        num_pdfs = *it;
    }
    num_pdfs++;

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

    // make sure we're at home
    KALDI_ASSERT(num_leaves == tied_to_pdf.size());
    KALDI_ASSERT(num_leaves == acc.NumTiedAccs());
    KALDI_ASSERT(num_pdfs   == acc.NumDiagAccs());

    // step 1: determine the codebooks under each node
    vector<std::set<int32> > diversity(p.size() - num_leaves);
    for (int32 i = 0; i < num_leaves; ++i) {
      int32 cur = i, par = p[i];
      int32 pdf = tied_to_pdf[i];
      while (cur != par) {
        int32 offs = par-num_leaves;
        if (diversity[offs].find(pdf) == diversity[offs].end())
          diversity[offs].insert(pdf);
        cur = par;
        par = p[cur];
      }
    }

    // print the diversity of each node, i.e. the codebook ids related to its
    // children
    if (print_diversity) {
      int32 k = diversity.size() - 1;
      for (vector<std::set<int32> >::reverse_iterator rit = diversity.rbegin(), rend = diversity.rend();
           rit != rend; ++rit, --k) {
        std::stringstream sstr;
        sstr << "node=" << (k+num_leaves) << " pdf-ids [ ";
        for (std::set<int32>::iterator si = (*rit).begin(), se = (*rit).end();
             si != se; ++si)
          sstr << *si << " ";
        sstr << "]";
        KALDI_LOG << sstr.str();
      }
    }

    // walk up the tree, validate accumulators, allocate interim allocators
    // and remember trace; terminate propagation on diverse nodes
    vector<std::set<int32> > trace(p.size() - num_leaves);
    vector<AccumTiedGmm *> interim(p.size() - num_leaves, NULL);

    KALDI_LOG << "Propagating " << num_leaves << " leaves";
    for (int32 i = 0; i < num_leaves; ++i) {
      AccumTiedGmm &a = acc.GetTiedAcc(i);
      std::stringstream sstr;
      sstr << "tied-id=" << i << " occ=" << a.occupancy().Sum() << " ==>";
      int32 cur = i, par = p[i] ;

      // walk up, as long as the parent is a diverse node or the root node
      while (cur != par) {
        int32 k = par - num_leaves;
        if (diversity[k].size() != 1) {
          sstr << " stop -- div = [";
          for (std::set<int32>::iterator si = diversity[k].begin(),
               se = diversity[k].end(); si != se; ++si)
            sstr << " " << (*si);
          sstr << " ]";
          break;
        }

        // add the node is already listed as child
        if (trace[k].find(cur) == trace[k].end())
          trace[k].insert(cur);

        // add accumulator
        if (interim[k] == NULL) {
          sstr << " alloc:" << par;
          interim[k] = new AccumTiedGmm(a);
        } else {
          sstr << " " << par;
          a.Propagate(interim[k]);
        }

        cur = par;
        par = p[cur];
      }

      KALDI_LOG << sstr.str();
    }

    KALDI_LOG << "Interpolating, tau=" << tau;

    // interpolate down, beginning from the top
    int32 k = trace.size() - 1;
    for (vector<std::set<int32> >::reverse_iterator rit = trace.rbegin(), rend = trace.rend();
         rit != rend; ++rit, --k) {
      std::stringstream sstr;
      sstr << (k + num_leaves) << " <==";

      // no interpolation on diverse nodes
      if (diversity[k].size() > 1) {
        sstr << " (null -- diverse node, sizeof diversity = " << diversity[k].size();
        KALDI_LOG << sstr.str();
        continue;
      }

      // the root will have some trace, but we're not propagating over the root
      if (interim[k] == NULL) {
        sstr << " (null -- skipping; sizeof trace = " << rit->size() << ")";
        KALDI_LOG << sstr.str();
        continue;
      }

      for (std::set<int32>::iterator it = (*rit).begin(), end = (*rit).end();
           it != end; ++it) {
        int32 t = *it;
        if (t < num_leaves) {
          // this will be a pdf accumulator
          sstr << " " << t;
          if (preserve_counts)
            acc.GetTiedAcc(t).Interpolate2(tau, *interim[k]);
           else
             acc.GetTiedAcc(t).Interpolate1(tau, *interim[k]);
        } else {
          // this will be an interim accumulator
          sstr << " interim:" << t;
          if (preserve_counts)
            interim[t-num_leaves]->Interpolate2(tau, *interim[k]);
          else
            interim[t-num_leaves]->Interpolate1(tau, *interim[k]);
        }
      }
      
      KALDI_LOG << sstr.str();
    }

    {
      Output os(acc_out_filename, binary_accu);
      trans_accs.Write(os.Stream(), binary_accu);
      acc.Write(os.Stream(), binary_accu);
    }

    KALDI_LOG << "Wrote " << acc_out_filename;

    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


