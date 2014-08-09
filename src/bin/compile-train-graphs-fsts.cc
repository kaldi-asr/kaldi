// bin/compile-train-graphs-fsts.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2013  Johns Hopkins University (Author: Daniel Povey)

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
#include "fstext/fstext-lib.h"
#include "decoder/training-graph-compiler.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Creates training graphs (without transition-probabilities, by default)\n"
        "This version takes FSTs as inputs (e.g., representing a separate weighted\n"
        "grammar for each utterance)\n"
        "Note: the lexicon should contain disambiguation symbols and you should\n"
        "supply the --read-disambig-syms option which is the filename of a list\n"
        "of disambiguation symbols.\n"
        "Warning: you probably want to set the --transition-scale and --self-loop-scale\n"
        "options; the defaults (zero) are probably not appropriate.\n"
        "Usage:   compile-train-graphs-fsts [options] <tree-in> <model-in> <lexicon-fst-in> "
        " <graphs-rspecifier> <graphs-wspecifier>\n"
        "e.g.: \n"
        " compile-train-graphs-fsts --read-disambig-syms=disambig.list\\\n"
        "   tree 1.mdl lex.fst ark:train.fsts ark:graphs.fsts\n";
    ParseOptions po(usage);

    TrainingGraphCompilerOptions gopts;
    int32 batch_size = 250;
    gopts.transition_scale = 0.0;  // Change the default to 0.0 since we will generally add the
    // transition probs in the alignment phase (since they change each time)
    gopts.self_loop_scale = 0.0;  // Ditto for self-loop probs.
    std::string disambig_rxfilename;
    gopts.Register(&po);

    po.Register("batch-size", &batch_size,
                "Number of FSTs to compile at a time (more -> faster but uses "
                "more memory.  E.g. 500");
    po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
                "list of disambiguation symbols in phone symbol table");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_rxfilename = po.GetArg(1);
    std::string model_rxfilename = po.GetArg(2);
    std::string lex_rxfilename = po.GetArg(3);
    std::string fsts_rspecifier = po.GetArg(4);
    std::string fsts_wspecifier = po.GetArg(5);

    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      // AmDiagGmm am_gmm;
      // am_gmm.Read(ki.Stream(), binary);
    }

    // need VectorFst because we will change it by adding subseq symbol.
    // ownership will be taken by gc.
    VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_rxfilename);

    std::vector<int32> disambig_syms;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << disambig_rxfilename;
    if (disambig_syms.empty())
      KALDI_WARN << "You supplied no disambiguation symbols; note, these are "
                 << "typically necessary when compiling graphs from FSTs (i.e. "
                 << "supply L_disambig.fst and the list of disambig syms with\n"
                 << "--read-disambig-syms)";
    TrainingGraphCompiler gc(trans_model, ctx_dep, lex_fst, disambig_syms, gopts);

    lex_fst = NULL;  // we gave ownership to gc.

    SequentialTableReader<fst::VectorFstHolder> fst_reader(fsts_rspecifier);
    TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);
    
    int num_succeed = 0, num_fail = 0;

    if (batch_size == 1) {  // We treat batch_size of 1 as a special case in order
      // to test more parts of the code.
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string key = fst_reader.Key();
        const VectorFst<StdArc> &grammar = fst_reader.Value(); // weighted
        // grammar for this utterance.
        VectorFst<StdArc> decode_fst;

        if (!gc.CompileGraph(grammar, &decode_fst)) {
          decode_fst.DeleteStates();  // Just make it empty.
        }
        if (decode_fst.Start() != fst::kNoStateId) {
          num_succeed++;
          fst_writer.Write(key, decode_fst);
        } else {
          KALDI_WARN << "Empty decoding graph for utterance "
                     << key;
          num_fail++;
        }
      }
    } else {
      std::vector<std::string> keys;
      std::vector<const VectorFst<StdArc>*> grammars; // word grammars.
      while (!fst_reader.Done()) {
        keys.clear();
        grammars.clear();
        for (; !fst_reader.Done() &&
                static_cast<int32>(grammars.size()) < batch_size;
            fst_reader.Next()) {
          keys.push_back(fst_reader.Key());
          grammars.push_back(new VectorFst<StdArc>(fst_reader.Value()));
        }
        std::vector<fst::VectorFst<fst::StdArc>* > fsts;
        if (!gc.CompileGraphs(grammars, &fsts))
          KALDI_ERR << "Not expecting CompileGraphs to fail.";
        KALDI_ASSERT(fsts.size() == keys.size());

        for (size_t i = 0; i < fsts.size(); i++) {
          delete grammars[i];
          if (fsts[i]->Start() != fst::kNoStateId) {
            num_succeed++;
            fst_writer.Write(keys[i], *(fsts[i]));
          } else {
            KALDI_WARN << "Empty decoding graph for utterance "
                       << keys[i];
            num_fail++;
          }
        }
        DeletePointers(&fsts);
      }
    }
    KALDI_LOG << "compile-train-graphs: succeeded for " << num_succeed
              << " graphs, failed for " << num_fail;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
