// chain/chain-supervision-splitter-test.cc

// Copyright      2015  Johns Hopkins University (author:  Daniel Povey)
//                2017  Vimal Manohar

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

#include "chain/chain-supervision-splitter.h"
#include "chain/chain-supervision.h"
#include "fstext/fstext-lib.h"
#include "hmm/hmm-test-utils.h"
#include "hmm/hmm-utils.h"
#include <iostream>
#include "fstext/kaldi-fst-io.h"
#include "lat/lattice-functions.h"

namespace kaldi {
namespace chain {


void FstToLabels(const fst::StdVectorFst &fst,
                 std::vector<ConstIntegerSet<int32> > *labels) {
  std::vector<int32> state_times;
  int32 num_frames = ComputeFstStateTimes(fst, &state_times);

  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;
  
  std::vector<std::set<int32> > temp_labels(num_frames);
  labels->clear();
  labels->resize(num_frames);

  for (StateId s = 0; s < fst.NumStates(); s++) {
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, s);
          !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();

      int32 t = state_times[s];
      KALDI_ASSERT(arc.ilabel == arc.olabel && arc.ilabel != 0);

      temp_labels[t].insert(arc.olabel);
    }
  }

  int32 t = 0;
  for (std::vector<std::set<int32> >::const_iterator it = temp_labels.begin();
       it != temp_labels.end(); ++it, t++) {
    (*labels)[t].Init(*it);
  }
}

void TestSupervisionLatticeSplitting(
    const SupervisionOptions &sup_opts,
    const TransitionModel &trans_model,
    Lattice &lat) {
  
  fst::TopSort(&lat);

  chain::SupervisionLatticeSplitterOptions opts;
  chain::SupervisionLatticeSplitter sup_lat_splitter(
      opts, sup_opts, trans_model);
  sup_lat_splitter.LoadLattice(lat);
  
  std::vector<int32> state_times;
  int32 num_frames_lat = LatticeStateTimes(lat, &state_times);

  Posterior post;
  LatticeForwardBackward(lat, &post);
    
  KALDI_ASSERT(num_frames_lat == post.size());
  
  std::vector<ConstIntegerSet<int32> > pdfs(post.size());
  for (size_t i = 0; i < post.size(); i++) {
    std::vector<int32> this_pdfs;
    for (size_t j = 0; j < post[i].size(); j++) {
      this_pdfs.push_back(trans_model.TransitionIdToPdf(post[i][j].first) + 1);
    }
    pdfs[i].Init(this_pdfs);
  }

  for (int32 i = 0; i < 3; i++) {
    int32 start_frame = RandInt(0, num_frames_lat - 1),
        num_frames = RandInt(1,10);

    if (start_frame + num_frames > num_frames_lat) {
      num_frames = num_frames_lat - start_frame;
    }

    chain::Supervision supervision_part;
    sup_lat_splitter.GetFrameRangeSupervision(
        start_frame, num_frames, &supervision_part);
    
    std::vector<ConstIntegerSet<int32> > labels;
    FstToLabels(supervision_part.fst, &labels);

    KALDI_ASSERT(labels.size() == num_frames);

    for (int32 t = 0; t < labels.size(); t++) {
      for (ConstIntegerSet<int32>::iterator it = labels[t].begin();
           it != labels[t].end(); ++it) {
        // To check that each label is a pdf (1-indexed) within the tolerance 
        // in the original
        bool label_in_original = false;
        for (int32 n = std::max(start_frame + t - sup_opts.left_tolerance, 0);
             n <= std::min(start_frame + t + sup_opts.right_tolerance, num_frames_lat - 1);
             n++) {
          if (pdfs[n].count(*it)) {
            label_in_original = true;
            break;
          }
        }
        KALDI_ASSERT(label_in_original);
      }
    }

    std::vector<int32> self_loop_pdfs_list;
    for (int32 tid = 1; tid <= trans_model.NumTransitionIds(); tid++) {
      if (trans_model.IsSelfLoop(tid)) {
        int32 tstate = trans_model.TransitionIdToTransitionState(tid);
        int32 pdf = trans_model.TransitionStateToSelfLoopPdf(tstate);
        self_loop_pdfs_list.push_back(pdf);
      }
    }

    ConstIntegerSet<int32> self_loop_pdfs(self_loop_pdfs_list);

    // To check that each self-loop pdf in the original is contained as a label 
    // in at least 2 of the tolerance values of the split lattices.
    for (int32 n = start_frame; n < start_frame + num_frames; n++) {
      for (ConstIntegerSet<int32>::iterator it = pdfs[n].begin();
            it != pdfs[n].end(); ++it) {
        if (!self_loop_pdfs.count(*it - 1)) continue; // Ignore forward pdfs
        int32 pdf_count = 0;
        for (int32 t = std::max(n - start_frame - sup_opts.left_tolerance, 0);
             t <= std::min(n - start_frame + sup_opts.right_tolerance, num_frames - 1); t++) {
          pdf_count += labels[t].count(*it); 
        }
        //KALDI_ASSERT(pdf_count > 1);
      }
    }
  }
}

TransitionModel* GetSimpleChainTransitionModel(
    ContextDependency **ctx_dep, int32 num_phones) {

  std::ostringstream oss;

  oss << "<Topology>\n"
      "<TopologyEntry>\n"
      "<ForPhones> ";
  for (int32 i = 1; i <= num_phones; i++) {
    oss << i << " ";
  }
  oss << "</ForPhones>\n"
      " <State> 0 <ForwardPdfClass> 0 <SelfLoopPdfClass> 1\n"
      "  <Transition> 0 0.5\n"
      "  <Transition> 1 0.5\n"
      " </State> \n"
      " <State> 1 </State>\n"
      "</TopologyEntry>\n"
      "</Topology>\n";
  
  std::string chain_input_str = oss.str();

  HmmTopology topo;
  std::istringstream iss(chain_input_str);
  topo.Read(iss, false);
  
  const std::vector<int32> &phones = topo.GetPhones();

  std::vector<int32> phone2num_pdf_classes (1+phones.back());
  for (size_t i = 0; i < phones.size(); i++)
    phone2num_pdf_classes[phones[i]] = topo.NumPdfClasses(phones[i]);

  *ctx_dep = MonophoneContextDependency(phones, phone2num_pdf_classes);

  return new TransitionModel(**ctx_dep, topo);
}

void ChainSupervisionSplitterTest(int32 index) {
  ContextDependency *ctx_dep;
  TransitionModel *trans_model;
  
  if (Rand())
    trans_model = GenRandTransitionModel(&ctx_dep, 2);
  else
    trans_model = GetSimpleChainTransitionModel(&ctx_dep, 2);

  const std::vector<int32> &phones = trans_model->GetPhones();
  
  int32 subsample_factor = 1;
  
  int32 phone_sequence_length = RandInt(1, 10);
  
  CompactLattice clat;
  int32 cur_state = clat.AddState();
  clat.SetStart(cur_state);

  bool reorder = true;

  int32 num_frames_subsampled = 0;
  for (int32 i = 0; i < phone_sequence_length; i++) {
    int32 phone = phones[RandInt(0, phones.size() - 1)];
    int32 next_state = clat.AddState();

    std::vector<int32> tids;
    GenerateRandomAlignment(*ctx_dep, *trans_model, reorder, 
                            std::vector<int32>(1, phone), &tids); 
    clat.AddArc(cur_state,
                CompactLatticeArc(phone, phone,
                                  CompactLatticeWeight(LatticeWeight::One(),
                                                       tids), next_state));
    cur_state = next_state;
    num_frames_subsampled += tids.size();
  }
  clat.SetFinal(cur_state, CompactLatticeWeight::One());

  Lattice lat;
  fst::ConvertLattice(clat, &lat);

  chain::SupervisionOptions sup_opts;
  sup_opts.left_tolerance = 1;
  sup_opts.right_tolerance = 1;
  sup_opts.frame_subsampling_factor = subsample_factor;
  sup_opts.lm_scale = 0.5;

  fst::StdVectorFst tolerance_fst;
  GetToleranceEnforcerFst(sup_opts, *trans_model, &tolerance_fst);
  WriteFstKaldi(std::cerr, false, tolerance_fst);

  TestSupervisionLatticeSplitting(sup_opts, *trans_model, lat);

  delete ctx_dep;
  delete trans_model;
}

void TestToleranceFst() {
  ContextDependency *ctx_dep;
  TransitionModel *trans_model = GetSimpleChainTransitionModel(&ctx_dep, 2);

  chain::SupervisionOptions sup_opts;
  sup_opts.left_tolerance = 1;
  sup_opts.right_tolerance = 1;
  sup_opts.frame_subsampling_factor = 1;
  sup_opts.lm_scale = 0.5;

  fst::StdVectorFst tolerance_fst;
  GetToleranceEnforcerFst(sup_opts, *trans_model, &tolerance_fst);
  WriteFstKaldi(std::cerr, false, tolerance_fst);
  
  fst::ArcSort(&tolerance_fst, fst::ILabelCompare<fst::StdArc>());
  
  delete ctx_dep;
  delete trans_model;
}

} // namespace chain
} // namespace kaldi

int main() {
  using namespace kaldi;
  SetVerboseLevel(2);

  kaldi::chain::TestToleranceFst();
  return 0;

  for (int32 i = 0; i < 10; i++) {
    kaldi::chain::ChainSupervisionSplitterTest(i);
  }
  //kaldi::chain::TestRanges();
}
