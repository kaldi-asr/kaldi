// bin/compile-graph.cc

// Copyright 2018     Johns Hopkins University (Author: Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "fstext/push-special.h"
#include "fstext/grammar-context-fst.h"
#include "decoder/grammar-fst.h"



int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;


    const char *usage =
        "Creates HCLG decoding graph.  Similar to mkgraph.sh but done in code.\n"
        "\n"
        "Usage:   compile-graph [options] <tree-in> <model-in> <lexicon-fst-in> "
        " <gammar-rspecifier> <hclg-wspecifier>\n"
        "e.g.: \n"
        " compile-train-graphs-fsts tree 1.mdl L_disambig.fst G.fst HCLG.fst\n";
    ParseOptions po(usage);


    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;  // Caution: the script default is 0.1.
    int32 nonterm_phones_offset = -1;
    std::string disambig_rxfilename;


    po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
                "list of disambiguation symbols in phone symbol table");
    po.Register("transition-scale", &transition_scale, "Scale of transition "
                "probabilities (excluding self-loops).");
    po.Register("self-loop-scale", &self_loop_scale, "Scale of self-loop vs. "
                "non-self-loop probability mass.  Caution: the default of "
                "mkgraph.sh is 0.1, but this defaults to 1.0.");
    po.Register("nonterm-phones-offset", &nonterm_phones_offset, "Integer "
                "value of symbol #nonterm_bos in phones.txt, if present. "
                "(Only relevant for grammar decoding).");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_rxfilename = po.GetArg(1),
        model_rxfilename = po.GetArg(2),
        lex_rxfilename = po.GetArg(3),
        grammar_rxfilename = po.GetArg(4),
        hclg_wxfilename = po.GetArg(5);

    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);

    VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_rxfilename),
        *grammar_fst = fst::ReadFstKaldi(grammar_rxfilename);

    std::vector<int32> disambig_syms;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "Could not read disambiguation symbols from "
                  << disambig_rxfilename;
    if (disambig_syms.empty())
      KALDI_WARN << "You supplied no disambiguation symbols; note, these are "
                 << "typically necessary when compiling graphs from FSTs (i.e. "
                 << "supply L_disambig.fst and the list of disambig syms with\n"
                 << "--read-disambig-syms)";

    const std::vector<int32> &phone_syms = trans_model.GetPhones();
    SortAndUniq(&disambig_syms);
    for (int32 i = 0; i < disambig_syms.size(); i++)
      if (std::binary_search(phone_syms.begin(), phone_syms.end(),
                             disambig_syms[i]))
        KALDI_ERR << "Disambiguation symbol " << disambig_syms[i]
                  << " is also a phone.";

    VectorFst<StdArc> lg_fst;
    TableCompose(*lex_fst, *grammar_fst, &lg_fst);

    DeterminizeStarInLog(&lg_fst, fst::kDelta);

    MinimizeEncoded(&lg_fst, fst::kDelta);

    fst::PushSpecial(&lg_fst, fst::kDelta);

    delete grammar_fst;
    delete lex_fst;

    VectorFst<StdArc> clg_fst;

    std::vector<std::vector<int32> > ilabels;

    int32 context_width = ctx_dep.ContextWidth(),
        central_position = ctx_dep.CentralPosition();

    if (nonterm_phones_offset < 0) {
      // The normal case.
      ComposeContext(disambig_syms, context_width, central_position,
                     &lg_fst, &clg_fst, &ilabels);
    } else {
      // The grammar-FST case. See ../doc/grammar.dox for an intro.
      if (context_width != 2 || central_position != 1) {
        KALDI_ERR << "Grammar-fst graph creation only supports models with left-"
            "biphone context.  (--nonterm-phones-offset option was supplied).";
      }
      ComposeContextLeftBiphone(nonterm_phones_offset,  disambig_syms,
                                lg_fst, &clg_fst, &ilabels);
    }
    lg_fst.DeleteStates();

    HTransducerConfig h_cfg;
    h_cfg.transition_scale = transition_scale;
    h_cfg.nonterm_phones_offset = nonterm_phones_offset;
    std::vector<int32> disambig_syms_h; // disambiguation symbols on
                                        // input side of H.
    VectorFst<StdArc> *h_fst = GetHTransducer(ilabels,
                                              ctx_dep,
                                              trans_model,
                                              h_cfg,
                                              &disambig_syms_h);

    VectorFst<StdArc> hclg_fst;  // transition-id to word.
    TableCompose(*h_fst, clg_fst, &hclg_fst);
    clg_fst.DeleteStates();
    delete h_fst;

    KALDI_ASSERT(hclg_fst.Start() != fst::kNoStateId);

    // Epsilon-removal and determinization combined. This will fail if not determinizable.
    DeterminizeStarInLog(&hclg_fst);

    if (!disambig_syms_h.empty()) {
      RemoveSomeInputSymbols(disambig_syms_h, &hclg_fst);
      RemoveEpsLocal(&hclg_fst);
    }

    // Encoded minimization.
    MinimizeEncoded(&hclg_fst);

    std::vector<int32> disambig;
    bool check_no_self_loops = true,
        reorder = true;
    AddSelfLoops(trans_model,
                 disambig,
                 self_loop_scale,
                 reorder,
                 check_no_self_loops,
                 &hclg_fst);

    if (nonterm_phones_offset >= 0)
      PrepareForGrammarFst(nonterm_phones_offset, &hclg_fst);

    {  // convert 'hclg' to ConstFst and write.
      fst::ConstFst<StdArc> const_hclg(hclg_fst);
      bool binary = true, write_binary_header = false;  // suppress the ^@B
      Output ko(hclg_wxfilename, binary, write_binary_header);
      fst::FstWriteOptions wopts(PrintableWxfilename(hclg_wxfilename));
      const_hclg.Write(ko.Stream(), wopts);
    }

    KALDI_LOG << "Wrote graph with " << hclg_fst.NumStates()
              << " states to " << hclg_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
