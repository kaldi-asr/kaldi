// bin/make-h-transducer.cc
// Copyright 2009-2011 Microsoft Corporation

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

#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"
#include "fstext/context-fst.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Make H transducer from transition-ids to context-dependent phones, \n"
        " without self-loops [use add-self-loops to add them]\n"
        "Usage:   make-h-transducer <ilabel-info-file> <tree-file> <transition-gmm/acoustic-model> [<H-fst-out>]\n"
        "e.g.: \n"
        " make-h-transducer ilabel_info  1.tree 1.mdl > H.fst\n";
    ParseOptions po(usage);

    HTransducerConfig hcfg;
    std::string disambig_out_filename;
    hcfg.Register(&po);
    po.Register("disambig-syms-out", &disambig_out_filename, "List of disambiguation symbols on input of H [to be output from this program]");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string ilabel_info_filename = po.GetArg(1);
    std::string tree_filename = po.GetArg(2);
    std::string model_filename = po.GetArg(3);
    std::string fst_out_filename;
    if (po.NumArgs() >= 4) fst_out_filename = po.GetArg(4);
    if (fst_out_filename == "-") fst_out_filename = "";

    std::vector<std::vector<int32> > ilabel_info;
    {
      bool binary_in;
      Input ki(ilabel_info_filename, &binary_in);
      fst::ReadILabelInfo(ki.Stream(), binary_in, &ilabel_info);
    }

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_filename, &trans_model);

    std::vector<int32> disambig_syms_out;

    // The work gets done here.
    fst::VectorFst<fst::StdArc> *H = GetHTransducer (ilabel_info,
                                                     ctx_dep,
                                                     trans_model,
                                                     hcfg,
                                                     &disambig_syms_out);
#if _MSC_VER
    if (fst_out_filename == "")
      _setmode(_fileno(stdout),  _O_BINARY);
#endif

    if (disambig_out_filename != "") {  // if option specified..
      if (disambig_out_filename == "-")
        disambig_out_filename = "";
      if (! WriteIntegerVectorSimple(disambig_out_filename, disambig_syms_out))
        KALDI_ERR << "Could not write disambiguation symbols to "
                   << (disambig_out_filename == "" ?
                       "standard output" : disambig_out_filename);
    }

    if (! H->Write(fst_out_filename) )
      KALDI_ERR << "make-h-transducer: error writing FST to "
                 << (fst_out_filename == "" ?
                     "standard output" : fst_out_filename);

    delete H;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

