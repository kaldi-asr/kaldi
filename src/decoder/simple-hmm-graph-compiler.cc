// decoder/simple-hmm-graph-compiler.cc

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

#include "decoder/simple-hmm-graph-compiler.h"
#include "simplehmm/simple-hmm-utils.h" // for GetHTransducer

namespace kaldi {

bool SimpleHmmGraphCompiler::CompileGraphFromAlignment(
    const std::vector<int32> &alignment,
    fst::VectorFst<fst::StdArc> *out_fst) {
  using namespace fst;
  VectorFst<StdArc> class_fst;
  MakeLinearAcceptor(alignment, &class_fst);
  return CompileGraph(class_fst, out_fst);
}

bool SimpleHmmGraphCompiler::CompileGraph(
    const fst::VectorFst<fst::StdArc> &class_fst,
    fst::VectorFst<fst::StdArc> *out_fst) {
  using namespace fst;
  KALDI_ASSERT(out_fst);
  KALDI_ASSERT(class_fst.Start() != kNoStateId);

  if (GetVerboseLevel() >= 4) {
    KALDI_VLOG(4) << "Classes FST: ";
    WriteFstKaldi(KALDI_LOG, false, class_fst);
  }

  VectorFst<StdArc> *H = GetHTransducer(model_, opts_.transition_scale, 
                                        opts_.self_loop_scale);
 
  if (GetVerboseLevel() >= 4) {
    KALDI_VLOG(4) << "HTransducer:";
    WriteFstKaldi(KALDI_LOG, false, *H);
  }

  // Epsilon-removal and determinization combined. 
  // This will fail if not determinizable.
  DeterminizeStarInLog(H);
  
  if (GetVerboseLevel() >= 4) {
    KALDI_VLOG(4) << "HTransducer determinized:";
    WriteFstKaldi(KALDI_LOG, false, *H);
  }
  
  VectorFst<StdArc> &trans2class_fst = *out_fst;  // transition-id to class.
  TableCompose(*H, class_fst, &trans2class_fst);
  
  KALDI_ASSERT(trans2class_fst.Start() != kNoStateId);
  
  if (GetVerboseLevel() >= 4) {
    KALDI_VLOG(4) << "trans2class_fst:";
    WriteFstKaldi(KALDI_LOG, false, trans2class_fst);
  }

  // Epsilon-removal and determinization combined. 
  // This will fail if not determinizable.
  DeterminizeStarInLog(&trans2class_fst);

  // we elect not to remove epsilons after this phase, as it is
  // a little slow.
  if (opts_.rm_eps)
    RemoveEpsLocal(&trans2class_fst);
  
  // Encoded minimization.
  MinimizeEncoded(&trans2class_fst);

  delete H;
  return true;
}

bool SimpleHmmGraphCompiler::CompileGraphsFromAlignments(
    const std::vector<std::vector<int32> > &alignments,
    std::vector<fst::VectorFst<fst::StdArc>*> *out_fsts) {
  using namespace fst;
  std::vector<const VectorFst<StdArc>* > class_fsts(alignments.size());
  for (size_t i = 0; i < alignments.size(); i++) {
    VectorFst<StdArc> *class_fst = new VectorFst<StdArc>();
    MakeLinearAcceptor(alignments[i], class_fst);
    class_fsts[i] = class_fst;
  }    
  bool ans = CompileGraphs(class_fsts, out_fsts);
  for (size_t i = 0; i < alignments.size(); i++)
    delete class_fsts[i];
  return ans;
}

bool SimpleHmmGraphCompiler::CompileGraphs(
    const std::vector<const fst::VectorFst<fst::StdArc>* > &class_fsts,
    std::vector<fst::VectorFst<fst::StdArc>* > *out_fsts) {

  using namespace fst;
  KALDI_ASSERT(out_fsts && out_fsts->empty());
  out_fsts->resize(class_fsts.size(), NULL);
  if (class_fsts.empty()) return true;

  for (size_t i = 0; i < class_fsts.size(); i++) {
    const VectorFst<StdArc> *class_fst = class_fsts[i];
    VectorFst<StdArc> out_fst;

    CompileGraph(*class_fst, &out_fst);

    (*out_fsts)[i] = out_fst.Copy();  
  }

  return true;
}


}  // end namespace kaldi
