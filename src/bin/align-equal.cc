// bin/align-equal.cc

// Copyright 2009-2013  Microsoft Corporation
//                      Johns Hopkins University (Author: Daniel Povey)

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


/** @brief Write equally spaced alignments of utterances (to get training started).
*/
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage = "Write equally spaced alignments of utterances "
        "(to get training started)\n"
        "Usage:  align-equal <tree-in> <model-in> <lexicon-fst-in> "
        "<features-rspecifier> <transcriptions-rspecifier> <alignments-wspecifier>\n"
        "e.g.: \n"
        " align-equal 1.tree 1.mdl lex.fst scp:train.scp "
        "'ark:sym2int.pl -f 2- words.txt text|' ark:equal.ali\n";

    ParseOptions po(usage);
    std::string disambig_rxfilename;
    po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
                "list of disambiguation symbols in phone symbol table");
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_in_filename = po.GetArg(1);
    std::string model_in_filename = po.GetArg(2);
    std::string lex_in_filename = po.GetArg(3);
    std::string feature_rspecifier = po.GetArg(4);
    std::string transcript_rspecifier = po.GetArg(5);
    std::string alignment_wspecifier = po.GetArg(6);

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_in_filename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_in_filename, &trans_model);

    // need VectorFst because we will change it by adding subseq symbol.
    VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_in_filename);

    TrainingGraphCompilerOptions gc_opts(1.0, true);  // true -> Dan style graph.

    std::vector<int32> disambig_syms;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << disambig_rxfilename;
    
    TrainingGraphCompiler gc(trans_model,
                             ctx_dep,
                             lex_fst,
                             disambig_syms,
                             gc_opts);

    lex_fst = NULL;  // we gave ownership to gc.


    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    RandomAccessInt32VectorReader transcript_reader(transcript_rspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    int32 done = 0, no_transcript = 0, other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (transcript_reader.HasKey(key)) {
        const std::vector<int32> &transcript = transcript_reader.Value(key);
        int32 num_frames = feature_reader.Value().NumRows();
        if (num_frames == 0) {
          KALDI_WARN << "Zero-length utterance for key " << key;
          other_error++;
          continue;
        }
        VectorFst<StdArc> decode_fst;
        if (!gc.CompileGraphFromText(transcript, &decode_fst)) {
          KALDI_WARN << "Problem creating decoding graph for utterance "
                     << key <<" [serious error]";
          other_error++;
          continue;
        }
        VectorFst<StdArc> path;
        int32 rand_seed = StringHasher()(key); // StringHasher() produces new anonymous
        // object of type StringHasher; we then call operator () on it, with "key".
        if (EqualAlign(decode_fst, num_frames, rand_seed, &path) ) {
          std::vector<int32> aligned_seq, words;
          StdArc::Weight w;
          GetLinearSymbolSequence(path, &aligned_seq, &words, &w);
          KALDI_ASSERT(aligned_seq.size() == num_frames);
          KALDI_ASSERT(words == transcript);
          alignment_writer.Write(key, aligned_seq);
          done++;
        } else {
          KALDI_WARN << "AlignEqual: did not align utterence " << key;
          other_error++;
        }
      } else {
        KALDI_WARN << "AlignEqual: no transcript for utterance " << key;
        no_transcript++;
      }
    }
    if (done != 0 && no_transcript == 0 && other_error == 0) {
      KALDI_LOG << "Success: done " << done << " utterances.";
    } else {
      KALDI_WARN << "Computed " << done << " alignments; " << no_transcript
                 << " lacked transcripts, " << other_error
                 << " had other errors.";
    }
    if (done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


