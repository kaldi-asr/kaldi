// fstbin/fstreorder.cc

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


#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "util/parse-options.h"
#include "util/text-utils.h"
#include "fst/fstlib.h"
#include "fstext/reorder.h"


/* Test:
  while true; do
      fstrand > /tmp/tmp.fst;
      fstreorder /tmp/tmp.fst | fstequivalent --random - /tmp/tmp.fst    || echo 'Error!';
  done
  Running this program on the fully-expanded HCLG, in a WSJ setup, on merlin@fit.vutbr.cz,
  did not affect decoding time in what seemed like a statistically significant way.
*/

int main(int argc, char *argv[])
{
  try {
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Reorder FST states for greater search efficiency [sort arcs by weight then dfs order states]\n"
        "\n"
        "Usage:  fstreorder [in.fst [out.fst] ]\n";

    bool do_arc_sort = true;
    kaldi::ParseOptions po(usage);
    po.Register("arc-sort", &do_arc_sort, "If true, sort arcs in decreasing order by weight, prior to dfs ordering");
    po.Read(argc, argv);

    if (po.NumArgs() > 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst_in_filename;
    fst_in_filename = po.GetOptArg(1);
    if (fst_in_filename == "-") fst_in_filename = "";

    std::string fst_out_filename;
    fst_out_filename = po.GetOptArg(2);
    if (fst_out_filename == "-") fst_out_filename = "";

    VectorFst<StdArc> *fst = VectorFst<StdArc>::Read(fst_in_filename);
    if (!fst) {
      std::cerr << "fstdeterminizestar: could not read input fst from " << fst_in_filename << '\n';
      return 1;
    }

    if (do_arc_sort)
      WeightArcSort(fst);  // sort arcs by weight.
    VectorFst<StdArc> ordered;
    DfsReorder(*fst, &ordered);  // do depth-first-search ordering of fst nodes.
    *fst = ordered;

    if (! fst->Write(fst_out_filename) ) {
      std::cerr << "fstminimizeencoded: error writing the output to "<<fst_out_filename << '\n';
      return 1;
    }
    delete fst;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}

