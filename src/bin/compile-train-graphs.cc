// bin/compile-train-graphs.cc

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
        "\n"
        "Usage:   compile-train-graphs [options] tree-in model-in lexicon-fst-in transcriptions-rspecifier graphs-wspecifier\n"
        "e.g.: \n"
        " compile-train-graphs tree 1.mdl lex.fst ark:train.tra ark:graphs.fsts\n";
    ParseOptions po(usage);

    TrainingGraphCompilerOptions gopts;
    int32 batch_size = 250;
    gopts.trans_prob_scale = 0.0;  // Change the default to 0.0 since we will generally add the
    // transition probs in the alignment phase (since they change eacm time)
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
    std::string transcript_rspecifier = po.GetArg(4);
    std::string fsts_wspecifier = po.GetArg(5);

    ContextDependency ctx_dep;  // the tree.
    {
      bool binary;
      Input is(tree_rxfilename, &binary);
      ctx_dep.Read(is.Stream(), binary);
    }

    TransitionModel trans_model;
    {
      bool binary;
      Input is(model_rxfilename, &binary);
      trans_model.Read(is.Stream(), binary);
      // AmDiagGmm am_gmm;
      // am_gmm.Read(is.Stream(), binary);
    }

    // need VectorFst because we will change it by adding subseq symbol.
    VectorFst<StdArc> *lex_fst = NULL;  // ownership will be taken by gc.
    {
      std::ifstream is(lex_rxfilename.c_str());
      if (!is.good()) KALDI_ERR << "Could not open lexicon FST " << (std::string)lex_rxfilename;
      lex_fst =
          VectorFst<StdArc>::Read(is, fst::FstReadOptions(lex_rxfilename));
      if (lex_fst == NULL)
        exit(1);
    }

    std::vector<int32> disambig_syms;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << disambig_rxfilename;
    
    TrainingGraphCompiler gc(trans_model, ctx_dep, lex_fst, disambig_syms, gopts);

    lex_fst = NULL;  // we gave ownership to gc.

    SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
    TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);

    int num_succeed = 0, num_fail = 0;

    if (batch_size == 1) {  // We treat batch_size of 1 as a special case in order
      // to test more parts of the code.
      for (; !transcript_reader.Done(); transcript_reader.Next()) {
        std::string key = transcript_reader.Key();
        const std::vector<int32> &transcript = transcript_reader.Value();
        VectorFst<StdArc> decode_fst;

        if (!gc.CompileGraphFromText(transcript, &decode_fst)) {
          KALDI_WARN << "Problem creating decoding graph for utterance "
                     << key << " [serious error]";
          decode_fst.DeleteStates();  // Just make it empty.
        }
        if (decode_fst.Start() != fst::kNoStateId) num_succeed++;
        else num_fail++;
        fst_writer.Write(key, decode_fst);
      }
    } else {
      std::vector<std::string> keys;
      std::vector<std::vector<int32> > transcripts;
      while (!transcript_reader.Done()) {
        keys.clear();
        transcripts.clear();
        for (; !transcript_reader.Done() &&
                static_cast<int32>(transcripts.size()) < batch_size;
            transcript_reader.Next()) {
          keys.push_back(transcript_reader.Key());
          transcripts.push_back(transcript_reader.Value());
        }
        std::vector<fst::VectorFst<fst::StdArc>* > fsts;
        if (!gc.CompileGraphsFromText(transcripts, &fsts)) {
          KALDI_ERR << "Not expecting CompileGraphs to fail.";
        }
        assert(fsts.size() == keys.size());
        for (size_t i = 0; i < fsts.size(); i++)
          fst_writer.Write(keys[i], *(fsts[i]));
        num_succeed += fsts.size();
        DeletePointers(&fsts);
      }
    }
    KALDI_LOG << "compile-train-graphs: succeeded for " << num_succeed
              << " graphs, failed for " << num_fail;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
