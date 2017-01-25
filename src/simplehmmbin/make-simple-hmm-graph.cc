// simplehmmbin/make-simple-hmm-graph.cc

// Copyright 2016  Vimal Manohar

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

#include "simplehmm/simple-hmm.h"
#include "simplehmm/simple-hmm-utils.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"
#include "fstext/context-fst.h"
#include "decoder/simple-hmm-graph-compiler.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Make graph to decode with simple HMM. It is an FST from "
        "transition-ids to pdf-ids + 1, \n"
        "Usage:   make-simple-hmm-graph <simple-hmm-model> [<H-fst-out>]\n"
        "e.g.: \n"
        " make-simple-hmm-graph 1.mdl > HCLG.fst\n";
    ParseOptions po(usage);

    SimpleHmmGraphCompilerOptions gopts;
    gopts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1);
    std::string fst_out_filename;
    if (po.NumArgs() >= 2) fst_out_filename = po.GetArg(2);
    if (fst_out_filename == "-") fst_out_filename = "";

    SimpleHmm trans_model;
    ReadKaldiObject(model_filename, &trans_model);

    // The work gets done here.
    fst::VectorFst<fst::StdArc> *H = GetHTransducer (trans_model,
                                                     gopts.transition_scale,
                                                     gopts.self_loop_scale);

#if _MSC_VER
    if (fst_out_filename == "")
      _setmode(_fileno(stdout),  _O_BINARY);
#endif

    if (! H->Write(fst_out_filename) )
      KALDI_ERR << "make-simple-hmm-graph: error writing FST to "
                 << (fst_out_filename == "" ?
                     "standard output" : fst_out_filename);

    delete H;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


