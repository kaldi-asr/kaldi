// decoder/training-graph-compiler.h

// Copyright 2009-2011 Microsoft Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef KALDI_DECODER_TRAINING_GRAPH_COMPILER_H_
#define KALDI_DECODER_TRAINING_GRAPH_COMPILER_H_

#include "base/kaldi-common.h"
#include "hmm/transition-model.h"
#include "fst/fstlib.h"
#include "fstext/fstext-lib.h"


namespace kaldi {

struct TrainingGraphCompilerOptions {

  BaseFloat trans_prob_scale;
  BaseFloat self_loop_scale;
  bool reorder;  // (Dan-style)
  explicit TrainingGraphCompilerOptions(BaseFloat trans_prob_scale = 1.0,
                                        BaseFloat self_loop_scale = 1.0,
                                        bool b = true) :
      trans_prob_scale(trans_prob_scale),
      self_loop_scale(self_loop_scale),
      reorder(b) {}

  void Register(ParseOptions *po) {
    po->Register("transition-scale", &trans_prob_scale, "Scale of transition probabilities relative to LM");
    po->Register("self-loop-scale", &self_loop_scale, "Scale of self-loop vs. non-self-loop prob. mas versus LM");
    po->Register("reorder", &reorder, "Reorder transition ids for greater decoding efficiency.");
  }
};


class TrainingGraphCompiler {
 public:
  TrainingGraphCompiler(const TransitionModel &trans_model,  // Maintains reference to this object.
                        const ContextDependency &ctx_dep,  // And this.
                        fst::VectorFst<fst::StdArc> *lex_fst,  // Takes ownership of this object.
                        // It should not contain disambiguation symbols or subsequential symbol,
                        // but it should contain optional silence.
                        const TrainingGraphCompilerOptions &opts);

  // not const for technical reasons involving the cache.
  // if not for "table_compose" could make it const.
  bool CompileGraph(const std::vector<int32> &transcript,
                    fst::VectorFst<fst::StdArc> *out_fst);

  // CompileGraphs allows you to compile a number of graphs at the same
  // time.  This consumes more memory but is faster.
  bool CompileGraphs(const std::vector<std::vector<int32> > &transcripts,
                     std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts);

  ~TrainingGraphCompiler() { delete lex_fst_; }
 private:
  const TransitionModel &trans_model_;
  const ContextDependency &ctx_dep_;
  fst::VectorFst<fst::StdArc> *lex_fst_;
  fst::TableComposeCache<fst::Fst<fst::StdArc> > lex_cache_;  // stores matcher..
  // this is one of Dan's extensions.

  TrainingGraphCompilerOptions opts_;
};



}  // end namespace kaldi.

#endif
