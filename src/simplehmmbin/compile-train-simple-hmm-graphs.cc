// bin/compile-train-simple-hmm-graphs.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2015  Johns Hopkins University (Author: Daniel Povey)
//                2016  Vimal Manohar

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
#include "simplehmm/simple-hmm.h"
#include "fstext/fstext-lib.h"
#include "decoder/simple-hmm-graph-compiler.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Creates training graphs (without transition-probabilities, by default)\n"
        "for training SimpleHmm models using alignments of pdf-ids.\n"
        "Usage:   compile-train-simple-hmm-graphs [options] <simple-hmm-in> "
        " <pdf-ali-rspecifier> <graphs-wspecifier>\n"
        "e.g.: \n"
        " compile-train-simple-hmm-graphs 1.mdl ark:train.tra ark:graphs.fsts\n";
    ParseOptions po(usage);

    SimpleHmmGraphCompilerOptions gopts;
    int32 batch_size = 250;
    gopts.transition_scale = 0.0;  // Change the default to 0.0 since we will generally add the
    // transition probs in the alignment phase (since they change eacm time)
    gopts.self_loop_scale = 0.0;  // Ditto for self-loop probs.
    std::string disambig_rxfilename;
    gopts.Register(&po);

    po.Register("batch-size", &batch_size,
                "Number of FSTs to compile at a time (more -> faster but uses "
                "more memory.  E.g. 500");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1);
    std::string alignment_rspecifier = po.GetArg(2);
    std::string fsts_wspecifier = po.GetArg(3);

    SimpleHmm model;
    ReadKaldiObject(model_rxfilename, &model);

    SimpleHmmGraphCompiler gc(model, gopts);

    SequentialInt32VectorReader alignment_reader(alignment_rspecifier);
    TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);

    int32 num_succeed = 0, num_fail = 0;

    if (batch_size == 1) {  // We treat batch_size of 1 as a special case in order
      // to test more parts of the code.
      for (; !alignment_reader.Done(); alignment_reader.Next()) {
        const std::string &key = alignment_reader.Key();
        std::vector<int32> alignment = alignment_reader.Value();

        for (std::vector<int32>::iterator it = alignment.begin();
             it != alignment.end(); ++it) {
          KALDI_ASSERT(*it < model.NumPdfs());
          ++(*it);
        }

        VectorFst<StdArc> decode_fst;

        if (!gc.CompileGraphFromAlignment(alignment, &decode_fst)) {
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
      std::vector<std::vector<int32> > alignments;
      while (!alignment_reader.Done()) {
        keys.clear();
        alignments.clear();
        for (; !alignment_reader.Done() &&
                static_cast<int32>(alignments.size()) < batch_size;
             alignment_reader.Next()) {
          keys.push_back(alignment_reader.Key());
          alignments.push_back(alignment_reader.Value());

          for (std::vector<int32>::iterator it = alignments.back().begin();
               it != alignments.back().end(); ++it) {
            KALDI_ASSERT(*it < model.NumPdfs());
            ++(*it);
          }
        }
        std::vector<fst::VectorFst<fst::StdArc>* > fsts;
        if (!gc.CompileGraphsFromAlignments(alignments, &fsts)) {
          KALDI_ERR << "Not expecting CompileGraphs to fail.";
        }
        KALDI_ASSERT(fsts.size() == keys.size());
        for (size_t i = 0; i < fsts.size(); i++) {
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
    KALDI_LOG << "compile-train--simple-hmm-graphs: succeeded for " 
              << num_succeed << " graphs, failed for " << num_fail;
    return (num_succeed != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


