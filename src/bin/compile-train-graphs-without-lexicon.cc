// bin/compile-train-graphs-without-lexicon.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2015  Johns Hopkins University (Author: Daniel Povey)
//                2021  Xiaomi Corporation (Author: Junbo Zhang)

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


namespace kaldi {

typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;

void MakeLinearLG(const std::vector<int32> &transcripts,
                  const std::string utt_id,
                  const std::map<std::string, std::vector<int32> > &linear_lexicon,
                  int32 silence_id,
                  int32 disambig_id,
                  VectorFst<StdArc> *ofst) {
  typedef typename StdArc::StateId StateId;
  typedef typename StdArc::Weight Weight;

  ofst->DeleteStates();
  StateId cur_state = ofst->AddState();
  ofst->SetStart(cur_state);

  // the header silences
  StateId next_state = ofst->AddState();
  StateId sil_state = ofst->AddState();
  ofst->AddArc(cur_state, StdArc(0, 0, Weight::One(), next_state));
  ofst->AddArc(cur_state, StdArc(0, 0, Weight::One(), sil_state));
  ofst->AddArc(sil_state, StdArc(silence_id, 0, Weight::One(), next_state));
  cur_state = next_state;

  // words
  for (size_t i = 0; i < transcripts.size(); i++) {
    std::string lex_id = utt_id + "." + std::to_string(i);
    if (linear_lexicon.find(lex_id) == linear_lexicon.end()) {
      KALDI_ERR << "No lexicon item: " << lex_id;
    }
    auto &phone_seq = linear_lexicon.at(lex_id);

    for (size_t j = 0; j < phone_seq.size() - 1; j++) {
      next_state = ofst->AddState();
      int32 ilabel = phone_seq[j];
      int32 olabel = (j == 0) ? transcripts[i] : 0;
      ofst->AddArc(cur_state, StdArc(ilabel, olabel, Weight::One(), next_state));
      cur_state = next_state;
    }
    
    // the last phone with an optional silence
    int32 ilabel = phone_seq.back();
    next_state = ofst->AddState();
    StateId sil_state = ofst->AddState();
    ofst->AddArc(cur_state, StdArc(ilabel, 0, Weight::One(), next_state));
    ofst->AddArc(cur_state, StdArc(ilabel, 0, Weight::One(), sil_state));
    ofst->AddArc(sil_state, StdArc(silence_id, 0, Weight::One(), next_state));
    cur_state = next_state;
  }

  // the tail
  ofst->SetFinal(cur_state, Weight::One());
  next_state = ofst->AddState();
  ofst->AddArc(cur_state, StdArc(disambig_id, 0, Weight::One(), next_state));
  ofst->AddArc(next_state, StdArc(disambig_id, 0, Weight::One(), next_state));
  ofst->SetFinal(next_state, Weight::One());
}

} // end namespace kaldi.

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Creates training graphs without lexicon, using phone sequence insead.\n"
        "\n"
        "Usage:   compile-train-graphs-without-lexicon [options] <tree-in> <model-in> "
        "<word-reference-rspecifier> <phone-reference-rspecifier> <graphs-wspecifier>\n"
        "e.g.: \n"
        " compile-train-graphs tree 1.mdl"
        " 'ark:sym2int.pl -f 2- words.txt text|'"
        " 'ark:sym2int.pl -f 2- phones.txt text-phone|'"
        " ark:graphs.fsts\n";
    ParseOptions po(usage);

    TrainingGraphCompilerOptions gopts;
    gopts.transition_scale = 0.0;  // Change the default to 0.0 since we will generally add the
    // transition probs in the alignment phase (since they change eacm time)
    gopts.self_loop_scale = 0.0;  // Ditto for self-loop probs.
    std::string disambig_rxfilename;
    std::string phone_syms_rxfilename;
    std::string word_syms_rxfilename;
    gopts.Register(&po);

    po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
                "list of disambiguation symbols in phone symbol table");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    int32 batch_size = 1;  // TODO: support batch_size > 1

    std::string tree_rxfilename = po.GetArg(1);
    std::string model_rxfilename = po.GetArg(2);
    std::string transcript_rspecifier = po.GetArg(3);
    std::string transcript_phone_rspecifier = po.GetArg(4);
    std::string fsts_wspecifier = po.GetArg(5);

    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);

    // need VectorFst because we will change it by adding subseq symbol.
    std::vector<int32> disambig_syms;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << disambig_rxfilename;

    TrainingGraphCompiler gc(trans_model, ctx_dep, NULL, disambig_syms, gopts);

    SequentialInt32VectorReader word_transcript_reader(transcript_rspecifier);
    SequentialInt32VectorReader phone_transcript_reader(transcript_phone_rspecifier);
    TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);

    // Make the phone-level transcript map
    std::map<std::string, std::vector<int32> > linear_lexicon;
    for (; !phone_transcript_reader.Done(); phone_transcript_reader.Next()) {
      std::string key = phone_transcript_reader.Key();
      const std::vector<int32> phone_seq = phone_transcript_reader.Value();
      linear_lexicon[key] = phone_seq;
    }

    int num_succeed = 0, num_fail = 0;

    if (batch_size == 1) {
      for (; !word_transcript_reader.Done(); word_transcript_reader.Next()) {
        std::string key = word_transcript_reader.Key();
        const std::vector<int32> &word_transcript = word_transcript_reader.Value();

        VectorFst<StdArc> *phone2word_fst = new VectorFst<StdArc>;
        int32 disambig_id = disambig_syms.back() + 1;
        MakeLinearLG(word_transcript, key, linear_lexicon, 1, disambig_id, phone2word_fst);

        VectorFst<StdArc> decode_fst;
        if (!gc.CompileGraphFromLG(*phone2word_fst, &decode_fst)) {
          decode_fst.DeleteStates();  // Just make it empty.
        }
        delete phone2word_fst;
        phone2word_fst = NULL;

        if (decode_fst.Start() != fst::kNoStateId) {
          num_succeed++;
          fst_writer.Write(key, decode_fst);
        } else {
          KALDI_WARN << "Empty decoding graph for utterance "
                     << key;
          num_fail++;
        }
      }
    } else KALDI_ERR << "We only supports the case that batch_size == 1 for now.";
    KALDI_LOG << "compile-train-graphs: succeeded for " << num_succeed
              << " graphs, failed for " << num_fail;
    return (num_succeed != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
