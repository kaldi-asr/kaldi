// bin/make-ilabel-transducer.cc
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
        "Make transducer that de-duplicates context-dependent ilabels that map to the same state\n"
        "Usage:   make-ilabel-transducer ilabel-info-right tree-file transition-gmm/model ilabel-info-left [mapping-fst-out]\n"
        "e.g.: \n"
        " make-ilabel-transducer old_ilabel_info 1.tree 1.mdl new_ilabel_info > convert.fst\n";
    ParseOptions po(usage);

    bool binary = true;
    std::string disambig_wxfilename;
    std::string old2new_map_wxfilename;
    po.Register("write-disambig-syms", &disambig_wxfilename, "List of disambiguation symbols after the remapping");
    po.Register("old-to-new-mapping", &old2new_map_wxfilename, "Mapping from old to new symbols (wxfilename)");
    po.Register("binary", &binary, "Write output ilabels in binary format");
    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string ilabel_info_rxfilename = po.GetArg(1),
        tree_filename = po.GetArg(2),
        model_filename = po.GetArg(3),
        ilabel_info_wxfilename = po.GetArg(4),
        fst_out_filename = po.GetOptArg(5);
    if (fst_out_filename == "-") fst_out_filename = "";

    std::vector<std::vector<int32> > old_ilabels;
    {
      bool binary_in;
      Input ki(ilabel_info_rxfilename, &binary_in);
      fst::ReadILabelInfo(ki.Stream(), binary_in, &old_ilabels);
    }

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_filename, &trans_model);


    std::vector<int32> old2new_mapping;

    // Most of the work gets done here.
    GetIlabelMapping(old_ilabels,
                     ctx_dep,
                     trans_model,
                     &old2new_mapping);

    if (old2new_map_wxfilename != "")
      if (!WriteIntegerVectorSimple(old2new_map_wxfilename, old2new_mapping))
        KALDI_ERR << "Error writing map from old to new symbols to "
                   << PrintableWxfilename(old2new_map_wxfilename);

    std::vector<std::vector<int32> > new_ilabels;
    KALDI_ASSERT(old2new_mapping.size() != 0);
    new_ilabels.resize(1 + *std::max_element(old2new_mapping.begin(),
                                             old2new_mapping.end()));
    for (size_t old_idx = 0; old_idx < old2new_mapping.size(); old_idx++) {
      int32 new_idx = old2new_mapping[old_idx];
      if (new_ilabels[new_idx].empty())  // select the 1st one we come across..
        new_ilabels[new_idx] = old_ilabels[old_idx];
    }

    // Output the ilabels.
    fst::WriteILabelInfo(Output(ilabel_info_wxfilename, binary).Stream(),
                         binary, new_ilabels);

    // Output the disambig symbols, if requested.
    if (disambig_wxfilename != "") {
      std::vector<int32> new_disambig;
      for (size_t new_idx = 0; new_idx < new_ilabels.size(); new_idx++) {
        if (new_ilabels[new_idx].size() == 1 && new_ilabels[new_idx][0] <= 0) {
          new_disambig.push_back(new_idx);
        }
      }
      if (! WriteIntegerVectorSimple(disambig_wxfilename, new_disambig)) {
        KALDI_ERR << "Could not write disambiguation symbols to "
                   << kaldi::PrintableWxfilename(disambig_wxfilename);
      }
    }

    // Create the mapping FST.
    VectorFst<StdArc> map_fst;
    CreateMapFst(old2new_mapping, &map_fst);

#if _MSC_VER
    if (fst_out_filename == "")
      _setmode(_fileno(stdout),  _O_BINARY);
#endif

    if (!map_fst.Write(fst_out_filename)) {
      KALDI_ERR << "Error writing output fst to "
                 << (fst_out_filename == "" ? " standard output "
                     : fst_out_filename);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

