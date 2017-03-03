// decoder/simple-hmm-graph-compiler.h

// Copyright 2009-2011  Microsoft Corporation
//                2016  Vimal Manohar

// See ../../COPYING for clarification regarding multiple authors
//
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

#ifndef KALDI_DECODER_SIMPLE_HMM_GRAPH_COMPILER_H_
#define KALDI_DECODER_SIMPLE_HMM_GRAPH_COMPILER_H_

#include "base/kaldi-common.h"
#include "simplehmm/simple-hmm.h"
#include "fst/fstlib.h"
#include "fstext/fstext-lib.h"


// This header provides functionality to compile a graph directly from the 
// alignment where the alignment is of classes that are simple mappings 
// of 'pdf-ids' (same as pdf classes for SimpleHmm).

namespace kaldi {

struct SimpleHmmGraphCompilerOptions {
  BaseFloat transition_scale;
  BaseFloat self_loop_scale;
  bool rm_eps;

  explicit SimpleHmmGraphCompilerOptions(BaseFloat transition_scale = 1.0,
                                         BaseFloat self_loop_scale = 1.0):
      transition_scale(transition_scale),
      self_loop_scale(self_loop_scale),
      rm_eps(true) { }

  void Register(OptionsItf *opts) {
    opts->Register("transition-scale", &transition_scale, "Scale of transition "
                   "probabilities (excluding self-loops)");
    opts->Register("self-loop-scale", &self_loop_scale, "Scale of self-loop vs. "
                   "non-self-loop probability mass ");
    opts->Register("rm-eps", &rm_eps,  "Remove [most] epsilons before minimization (only applicable "
                   "if disambig symbols present)");
  }
};


class SimpleHmmGraphCompiler {
 public:
  SimpleHmmGraphCompiler(const SimpleHmm &model,  // Maintains reference to this object.
                         const SimpleHmmGraphCompilerOptions &opts):
    model_(model), opts_(opts) { }


  /// CompileGraph compiles a single training graph its input is a
  /// weighted acceptor (G) at the class level, its output is HCLG-type graph.
  /// Note: G could actually be an acceptor, it would also work.
  /// This function is not const for technical reasons involving the cache.
  /// if not for "table_compose" we could make it const.
  bool CompileGraph(const fst::VectorFst<fst::StdArc> &class_fst,
                    fst::VectorFst<fst::StdArc> *out_fst);
  
  // CompileGraphs allows you to compile a number of graphs at the same
  // time.  This consumes more memory but is faster.
  bool CompileGraphs(
      const std::vector<const fst::VectorFst<fst::StdArc> *> &class_fsts,
      std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts);

  // This version creates an FST from the per-frame alignment and calls 
  // CompileGraph.
  bool CompileGraphFromAlignment(const std::vector<int32> &alignment,
                                 fst::VectorFst<fst::StdArc> *out_fst);

  // This function creates FSTs from the per-frame alignment and calls 
  // CompileGraphs.
  bool CompileGraphsFromAlignments(
      const std::vector<std::vector<int32> >  &alignments,
      std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts);
  
  ~SimpleHmmGraphCompiler() { }
 private:
  const SimpleHmm &model_;

  SimpleHmmGraphCompilerOptions opts_;
};


}  // end namespace kaldi.

#endif  // KALDI_DECODER_SIMPLE_HMM_GRAPH_COMPILER_H_
