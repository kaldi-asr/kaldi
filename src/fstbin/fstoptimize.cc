// fstbin/fstoptimize.cc

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
#include "fstext/determinize-star.h"
#include "fstext/fstext-utils.h"


/* some test examples:
 ( echo "0 0 0 0"; echo "0 0" ) | fstcompile | fstoptimize | fstprint
 ( echo "0 1 0 0"; echo " 0 2 0 0"; echo "1 0"; echo "2 0"; ) | fstcompile | fstoptimize | fstprint
*/

int main(int argc, char *argv[])
{
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Optimizes FST using predeterminization, determinization, optional weight and symbol-pushing, and encoded minimization"
        "\n"
        "Usage:  fstoptimize [in.fst [out.fst] ]\n";

    OptimizeConfig cfg;

    ParseOptions po(usage);
    po.Register("delta", &cfg.delta, "Delta likelihood");
    po.Register("use-log", &cfg.maintain_log_stochasticity, "If true, maintain stochasticity in log semiring (but still equivalence in tropical)");
    po.Register("push-weights", &cfg.push_weights, "If true, push weights after determinization [dangerous and unnecessary]");
    po.Register("push-in-log", &cfg.push_in_log, "If true [and push-weights == true], push in log semiring.");
    po.Register("push-labels", &cfg.push_labels, "If true [and push-weights == true], push in log semiring.");

    po.Read(argc, argv);

    if (po.NumArgs() > 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst_in_filename = po.GetOptArg(1);
    if (fst_in_filename == "-") fst_in_filename = "";

    std::string fst_out_filename = po.GetOptArg(2);
    if (fst_out_filename == "-") fst_out_filename = "";

    VectorFst<StdArc> *fst = VectorFst<StdArc>::Read(fst_in_filename);
    if (!fst) {
      std::cerr << "fstoptimize: could not read input fst from " << fst_in_filename << '\n';
      return 1;
    }

    Optimize(fst, cfg);

    if (! fst->Write(fst_out_filename) ) {
      std::cerr << "fstoptimize: error writing the output to "<<fst_out_filename << '\n';
      return 1;
    }
    delete fst;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}

