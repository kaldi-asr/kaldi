// decoder/training-graph-compiler.cc
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

#include "decoder/training-graph-compiler.h"
#include "hmm/hmm-utils.h" // for GetHTransducer

namespace kaldi {


TrainingGraphCompiler::TrainingGraphCompiler(const TransitionModel &trans_model,
                                               const ContextDependency &ctx_dep,  // Does not maintain reference to this.
                                               fst::VectorFst<fst::StdArc> *lex_fst,
                                               const TrainingGraphCompilerOptions &opts):
    trans_model_(trans_model), ctx_dep_(ctx_dep), lex_fst_(lex_fst), opts_(opts) {
  using namespace fst;

  const std::vector<int32> &phone_syms = trans_model_.GetPhones();  // needed to create context fst.

  assert(!phone_syms.empty());
  assert(IsSortedAndUniq(phone_syms));

  int32 subseq_symbol = 1 + phone_syms.back();

  {
    int32 N = ctx_dep.ContextWidth(),
        P = ctx_dep.CentralPosition();
    if (P != N-1)
      AddSubsequentialLoop(subseq_symbol, lex_fst_);  // This is needed for
    // systems with right-context or we will not successfully compose
    // with C.
  }

  {  // make sure lexicon is olabel sorted.
    fst::OLabelCompare<fst::StdArc> olabel_comp;
    fst::ArcSort(lex_fst_, olabel_comp);
  }
}


bool TrainingGraphCompiler::CompileGraph(const std::vector<int32> &transcript,
                                         fst::VectorFst<fst::StdArc> *out_fst) {
  using namespace fst;
  assert(lex_fst_ !=NULL);
  assert(out_fst != NULL);

  VectorFst<StdArc> word_fst;
  MakeLinearAcceptor(transcript, &word_fst);


  VectorFst<StdArc> phone2word_fst;
  // TableCompose more efficient than compose.
  TableCompose(*lex_fst_, word_fst, &phone2word_fst, &lex_cache_);

  assert(phone2word_fst.Start() != kNoStateId);

  ContextFst<StdArc> *cfst = NULL;
  {  // make cfst [ it's expanded on the fly ]
    std::vector<int32> disambig_syms;  // empty.
    std::vector<int32> phones_disambig_syms;  // also empty.
    const std::vector<int32> &phone_syms = trans_model_.GetPhones();  // needed to create context fst.
    int32 subseq_symbol = phone_syms.back() + 1;

    cfst = new ContextFst<StdArc>(subseq_symbol,
                                  phone_syms,
                                  phones_disambig_syms,
                                  ctx_dep_.ContextWidth(),
                                  ctx_dep_.CentralPosition());
  }

  VectorFst<StdArc> ctx2word_fst;
  ComposeContextFst(*cfst, phone2word_fst, &ctx2word_fst);
  // ComposeContextFst is like Compose but faster for this particular Fst type.
  // [and doesn't expand too many arcs in the ContextFst.]

  assert(ctx2word_fst.Start() != kNoStateId);

  HTransducerConfig h_cfg;
  h_cfg.trans_prob_scale = opts_.trans_prob_scale;

  std::vector<int32> disambig_syms_out;
  VectorFst<StdArc> *H = GetHTransducer(cfst->ILabelInfo(),
                                        ctx_dep_,
                                        trans_model_,
                                        h_cfg,
                                        &disambig_syms_out);
  assert(disambig_syms_out.empty());

  VectorFst<StdArc> &trans2word_fst = *out_fst;  // transition-id to word.
  TableCompose(*H, ctx2word_fst, &trans2word_fst);

  assert(trans2word_fst.Start() != kNoStateId);

  // Remove-eps local;  maintain tropical equivalence but
  // log-semiring stochasticity.  This isn't mentioned in the documentation
  // as it's really an optimization that's not important (DeterminizeStar
  // will remove any remaining epsilons).
  RemoveEpsLocalSpecial(&trans2word_fst);

  // Epsilon-removal and determinization combined. This will fail if not determinizable.
  DeterminizeStarInLog(&trans2word_fst);

  // Encoded minimization.
  MinimizeEncoded(&trans2word_fst);

  std::vector<int32> disambig;
  AddSelfLoops(trans_model_,
               disambig,
               opts_.self_loop_scale,
               opts_.reorder,
               &trans2word_fst);

  delete H;
  delete cfst;
  return true;
}


bool TrainingGraphCompiler::CompileGraphs(const std::vector<std::vector<int32> > &transcripts,
                                          std::vector<fst::VectorFst<fst::StdArc>* > *out_fsts) {

  using namespace fst;
  assert(lex_fst_ !=NULL);
  assert(out_fsts != NULL && out_fsts->empty());
  if (transcripts.empty()) return true;
  out_fsts->resize(transcripts.size(), NULL);

  ContextFst<StdArc> *cfst = NULL;
  {  // make cfst [ it's expanded on the fly ]
    std::vector<int32> disambig_syms;  // empty.
    std::vector<int32> phones_disambig_syms;  // also empty.
    const std::vector<int32> &phone_syms = trans_model_.GetPhones();  // needed to create context fst.
    int32 subseq_symbol = phone_syms.back() + 1;

    cfst = new ContextFst<StdArc>(subseq_symbol,
                                  phone_syms,
                                  phones_disambig_syms,
                                  ctx_dep_.ContextWidth(),
                                  ctx_dep_.CentralPosition());
  }

  for (size_t i = 0; i < transcripts.size(); i++) {
    const std::vector<int32> &transcript = transcripts[i];
    VectorFst<StdArc> word_fst;
    MakeLinearAcceptor(transcript, &word_fst);

    VectorFst<StdArc> phone2word_fst;
    // TableCompose more efficient than compose.
    TableCompose(*lex_fst_, word_fst, &phone2word_fst, &lex_cache_);

    assert(phone2word_fst.Start() != kNoStateId);

    VectorFst<StdArc> ctx2word_fst;
    ComposeContextFst(*cfst, phone2word_fst, &ctx2word_fst);
    // ComposeContextFst is like Compose but faster for this particular Fst type.
    // [and doesn't expand too many arcs in the ContextFst.]

    assert(ctx2word_fst.Start() != kNoStateId);

    (*out_fsts)[i] = ctx2word_fst.Copy();  // For now this contains the FST with symbols
    // representing phones-in-context.
  }

  HTransducerConfig h_cfg;
  h_cfg.trans_prob_scale = opts_.trans_prob_scale;

  std::vector<int32> disambig_syms_out;
  VectorFst<StdArc> *H = GetHTransducer(cfst->ILabelInfo(),
                                        ctx_dep_,
                                        trans_model_,
                                        h_cfg,
                                        &disambig_syms_out);
  assert(disambig_syms_out.empty());

  for (size_t i = 0; i < out_fsts->size(); i++) {
    VectorFst<StdArc> &ctx2word_fst = *((*out_fsts)[i]);
    VectorFst<StdArc> trans2word_fst;
    TableCompose(*H, ctx2word_fst, &trans2word_fst);

    // This step doesn't affect the final answer but may speed it up
    // in certain cases (or may slow it down)...
    RemoveEpsLocalSpecial(&trans2word_fst);

    DeterminizeStarInLog(&trans2word_fst);
    // Encoded minimization.
    MinimizeEncoded(&trans2word_fst);

    std::vector<int32> disambig;
    AddSelfLoops(trans_model_,
                 disambig,
                 opts_.self_loop_scale,
                 opts_.reorder,
                 &trans2word_fst);

    assert(trans2word_fst.Start() != kNoStateId);

    *((*out_fsts)[i]) = trans2word_fst;
  }

  delete H;
  delete cfst;
  return true;
}


}  // end namespace kaldi
