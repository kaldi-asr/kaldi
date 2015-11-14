// chain/chain-supervision-test.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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

#include "chain/chain-supervision.h"
#include "chain/chain-numerator.h"
#include "fstext/fstext-lib.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-vector.h"
#include "hmm/hmm-test-utils.h"
#include "chain/chain-den-graph.h"
#include "hmm/hmm-utils.h"



namespace kaldi {
namespace chain {

// computes a phone language-model FST, which has only monophone context.
void ComputeExamplePhoneLanguageModel(const std::vector<int32> &phones,
                                      fst::StdVectorFst *g_fst) {

  g_fst->DeleteStates();
  int32 state = g_fst->AddState();
  g_fst->SetStart(state);

  Vector<BaseFloat> probs(phones.size() + 1);
  probs.SetRandn();
  probs.ApplyPow(2.0);
  probs.Add(0.01);
  probs.Scale(1.0 / probs.Sum());

  for (size_t i = 0; i < phones.size(); i++) {
    int32 phone = phones[i];
    fst::StdArc arc(phone, phone,
                    fst::TropicalWeight(-log(probs(i))), state);
    g_fst->AddArc(state, arc);
  }
  g_fst->SetFinal(state, fst::TropicalWeight(-log(probs(phones.size()))));
}


void ComputeExampleDenFst(const ContextDependency &ctx_dep,
                            const TransitionModel &trans_model,
                            fst::StdVectorFst *den_graph) {
  using fst::StdVectorFst;
  using fst::StdArc;
  StdVectorFst phone_lm;
  ComputeExamplePhoneLanguageModel(trans_model.GetPhones(), &phone_lm);

  CreateDenominatorFst(ctx_dep, trans_model, phone_lm, den_graph);
}


void TestSupervisionIo(const Supervision &supervision) {
  bool binary = (RandInt(0, 1) == 0);
  std::ostringstream os;
  supervision.Write(os, binary);
  std::istringstream is(os.str());
  Supervision supervision2;
  if (RandInt(0, 1) == 0)
    supervision2 = supervision;  // test reading already-existing object.
  supervision2.Read(is, binary);
  std::ostringstream os2;
  supervision2.Write(os2, binary);
  KALDI_ASSERT(os.str() == os2.str());
  if (binary) {
    KALDI_ASSERT(supervision == supervision2);
  }
  // also test swap and constructor
  Supervision supervision3(supervision), supervision4;
  supervision3.Swap(&supervision4);
  KALDI_ASSERT(supervision == supervision4);
}

void TestSupervisionNumerator(const Supervision &supervision) {

  CuMatrix<BaseFloat> nnet_output(supervision.num_sequences *
                                  supervision.frames_per_sequence,
                                  supervision.label_dim);
  nnet_output.SetRandn();

  NumeratorComputation num(supervision, nnet_output);

  // Test that derivs are accurate.

  if (RandInt(0, 1) == 0) {
    std::vector<std::vector<int32> > allowed_initial_symbols,
        allowed_final_symbols;
    num.GetAllowedInitialAndFinalSymbols(&allowed_initial_symbols,
                                         &allowed_final_symbols);
  }
  BaseFloat forward_prob = num.Forward();

  CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                        nnet_output.NumCols());
  num.Backward(&nnet_output_deriv);

  int32 dim = 3;
  Vector<BaseFloat> predicted_objf_changes(dim),
      measured_objf_changes(dim);
  BaseFloat delta = 1.0e-04;
  for (int32 p = 0; p < dim; p++) {
    CuMatrix<BaseFloat> new_nnet_output(nnet_output.NumRows(),
                                        nnet_output.NumCols());
    new_nnet_output.SetRandn();
    new_nnet_output.Scale(delta);
    predicted_objf_changes(p) = TraceMatMat(nnet_output_deriv, new_nnet_output,
                                            kTrans);
    new_nnet_output.AddMat(1.0, nnet_output);
    NumeratorComputation num2(supervision, new_nnet_output);
    measured_objf_changes(p) = num2.Forward() - forward_prob;
  }
  KALDI_LOG << "Predicted objf changes are: "
            << predicted_objf_changes;
  KALDI_LOG << "Measured objf changes are: "
            << measured_objf_changes;
  KALDI_ASSERT(predicted_objf_changes.ApproxEqual(measured_objf_changes, 0.1));

  {
    CuVector<BaseFloat> rand(nnet_output.NumRows());
    rand.SetRandn();
    CuMatrix<BaseFloat> nnet_output_mod(nnet_output);
    nnet_output_mod.AddVecToCols(1.0, rand);
    NumeratorComputation num_mod(supervision, nnet_output_mod);
    BaseFloat forward_prob_mod = num_mod.Forward();
    BaseFloat predicted_change = rand.Sum(),
        observed_change = forward_prob_mod - forward_prob;
    KALDI_ASSERT(fabs(predicted_change - observed_change)  < 0.1);
  }


}

void TestSupervisionAppend(const TransitionModel &trans_model,
                           const Supervision &supervision) {
  int32 num_append = RandInt(1,5);
  std::vector<const Supervision*> input(num_append);
  for (int32 i = 0; i < num_append; i++)
    input[i] = &supervision;
  std::vector<Supervision> output;
  bool compactify = (RandInt(0, 1) == 0);
  AppendSupervision(input, compactify, &output);
  if (compactify) {
    KALDI_ASSERT(output.size() == 1 &&
                 output[0].frames_per_sequence ==
                 supervision.frames_per_sequence &&
                 output[0].num_sequences == num_append);
  } else {
    KALDI_ASSERT(output.size() == input.size());
  }
  int32 tot_sequences_in = 0, tot_sequences_out = 0,
      tot_frames_in = 0, tot_frames_out = 0;
  for (int32 i = 0; i < num_append; i++) {
    tot_sequences_in += input[i]->num_sequences;
    tot_frames_in += input[i]->num_sequences *
        input[i]->frames_per_sequence;
  }
  for (int32 i = 0; i < output.size(); i++) {
    tot_sequences_out += output[i].num_sequences;
    tot_frames_out += output[i].num_sequences *
        output[i].frames_per_sequence;
  }
  KALDI_ASSERT(tot_sequences_out == tot_sequences_in &&
               tot_frames_out == tot_frames_in);

  TestSupervisionIo(output[0]);
  TestSupervisionNumerator(output[0]);
  output[0].Check(trans_model);
}

void TestSupervisionReattached(const TransitionModel &trans_model,
                               const Supervision &supervision,
                               const Supervision &reattached_supervision) {
  using namespace fst;
  KALDI_LOG << "testing reattached";
  KALDI_ASSERT(reattached_supervision.frames_per_sequence *
               reattached_supervision.num_sequences ==
               supervision.frames_per_sequence * supervision.num_sequences &&
               reattached_supervision.weight == supervision.weight &&
               reattached_supervision.label_dim == supervision.label_dim);
  UniformArcSelector<StdArc> selector;
  RandGenOptions<UniformArcSelector<StdArc> > randgen_opts(selector);
  StdVectorFst fst_path;
  RandGen(supervision.fst, &fst_path, randgen_opts);
  StdVectorFst composed;
  Compose(fst_path, reattached_supervision.fst, &composed);
  Connect(&composed);
  KALDI_ASSERT(composed.NumStates() != 0);
  supervision.Check(trans_model);
  reattached_supervision.Check(trans_model);
}


void TestSupervisionFrames(const Supervision &supervision) {
  using namespace fst;
  UniformArcSelector<StdArc> selector;
  RandGenOptions<UniformArcSelector<StdArc> > randgen_opts(selector);
  VectorFst<StdArc> rand_path;
  RandGen(supervision.fst, &rand_path, randgen_opts);
  std::vector<int32> isymbols_out, osymbols_out;
  fst::TropicalWeight weight_out;
  bool ans = GetLinearSymbolSequence(rand_path, &isymbols_out, &osymbols_out,
                                     &weight_out);
  KALDI_ASSERT(ans);
  KALDI_ASSERT(isymbols_out == osymbols_out);
  KALDI_ASSERT(isymbols_out.size() ==
               static_cast<size_t>(supervision.num_sequences *
                                   supervision.frames_per_sequence));
  KALDI_ASSERT(weight_out == fst::TropicalWeight::One());

  bool test = true;
  // make sure epsilon free
  KALDI_ASSERT(supervision.fst.Properties(fst::kNoEpsilons, test) != 0);
  // make sure acceptor
  KALDI_ASSERT(supervision.fst.Properties(fst::kAcceptor, test) != 0);
}


void TestSupervisionSplitting(const ContextDependency &ctx_dep,
                              const TransitionModel &trans_model,
                              const Supervision &supervision) {
  fst::StdVectorFst den_fst, normalization_fst;
  ComputeExampleDenFst(ctx_dep, trans_model, &den_fst);
  DenominatorGraph den_graph(den_fst, trans_model.NumPdfs());
  den_graph.GetNormalizationFst(den_fst, &normalization_fst);

  SupervisionSplitter splitter(supervision);
  int32 num_frames = supervision.num_sequences * supervision.frames_per_sequence,
      frames_per_range = RandInt(3, 10);
  std::vector<int32> range_starts;
  SplitIntoRanges(num_frames, frames_per_range, &range_starts);
  int32 num_ranges = range_starts.size();
  std::vector<Supervision> split_supervision(num_ranges);
  for (int32 i = 0; i < num_ranges; i++) {
    splitter.GetFrameRange(range_starts[i], frames_per_range,
                           &split_supervision[i]);
    bool ans = AddWeightToSupervisionFst(normalization_fst,
                                         &split_supervision[i]);
    KALDI_ASSERT(ans);
    split_supervision[i].Check(trans_model);
  }
  if (num_ranges > 0) {
    TestSupervisionIo(split_supervision[RandInt(0, num_ranges - 1)]);
    TestSupervisionFrames(split_supervision[RandInt(0, num_ranges - 1)]);
    if (num_frames % frames_per_range == 0) {
      // co-test with Append.
      std::vector<Supervision> reattached_supervision;
      std::vector<const Supervision*> to_append(num_ranges);
      for (int32 i = 0; i < num_ranges; i++)
        to_append[i] = &(split_supervision[i]);
      bool compactify = true;
      AppendSupervision(to_append, compactify, &reattached_supervision);
      KALDI_ASSERT(reattached_supervision.size() == 1);
      TestSupervisionReattached(trans_model,
                                supervision,
                                reattached_supervision[0]);
    }
  }
}

void ChainSupervisionTest() {
  ContextDependency *ctx_dep;
  TransitionModel *trans_model = GenRandTransitionModel(&ctx_dep);
  const std::vector<int32> &phones = trans_model->GetPhones();

  int32 subsample_factor = RandInt(1, 3);

  int32 phone_sequence_length = RandInt(1, 10);
  std::vector<std::pair<int32, int32> > phones_durations(phone_sequence_length);

  CompactLattice clat;
  int32 cur_state = clat.AddState();
  clat.SetStart(cur_state);

  for (int32 i = 0; i < phone_sequence_length; i++) {
    int32 phone = phones[RandInt(0, phones.size() - 1)];
    int32 min_length = trans_model->GetTopo().MinLength(phone),
        headroom = 5,
        duration = RandInt(subsample_factor * min_length,
                           subsample_factor * min_length + headroom);
    phones_durations[i].first = phone;
    phones_durations[i].second = duration;
    int32 next_state = clat.AddState();
    std::vector<int32> ones(duration, 1);
    clat.AddArc(cur_state,
                CompactLatticeArc(phone, phone,
                                  CompactLatticeWeight(LatticeWeight::One(),
                                                       ones), next_state));
    cur_state = next_state;
  }
  clat.SetFinal(cur_state, CompactLatticeWeight::One());
  ProtoSupervision proto_sup1, proto_sup2;
  SupervisionOptions opts;
  opts.frame_subsampling_factor = subsample_factor;
  bool ans1 = AlignmentToProtoSupervision(opts, phones_durations, &proto_sup1),
      ans2 = PhoneLatticeToProtoSupervision(opts, clat, &proto_sup2);
  KALDI_ASSERT(ans1 && ans2);
  KALDI_ASSERT(proto_sup1 == proto_sup2);

  Supervision supervision;
  if (!ProtoSupervisionToSupervision(*ctx_dep, *trans_model,
                                     proto_sup1, &supervision)) {
    // we shouldn't fail because we multiplied by
    // 'subsample_factor' when creating the duration.
    KALDI_ERR << "Failed creating supervision.";
  }
  supervision.Check(*trans_model);
  TestSupervisionIo(supervision);
  TestSupervisionSplitting(*ctx_dep, *trans_model, supervision);
  TestSupervisionAppend(*trans_model, supervision);

  {
    fst::StdVectorFst den_fst;
    ComputeExampleDenFst(*ctx_dep, *trans_model, &den_fst);
    DenominatorGraph den_graph(den_fst, trans_model->NumPdfs());
  }

  // HERE
  delete ctx_dep;
  delete trans_model;
}

void AddArc(int32 from, int32 to,
            fst::StdVectorFst *fst) {
  fst->AddArc(from, fst::StdArc(0, 0, fst::TropicalWeight::One(), to));
}

void BreadthFirstTest() {
  using namespace fst;
  StdVectorFst fst;
  for (int32 i = 0; i < 6; i++)
    fst.AddState();
  fst.SetStart(0);
  fst.SetFinal(2, TropicalWeight::One());
  AddArc(0, 3, &fst);
  AddArc(0, 4, &fst);
  AddArc(4, 5, &fst);
  AddArc(3, 5, &fst);
  AddArc(5, 1, &fst);
  AddArc(1, 2, &fst);
  SortBreadthFirstSearch(&fst);

  KALDI_ASSERT(fst.Properties(fst::kTopSorted, true) != 0);

}

}  // namespace chain
}  // namespace kaldi

int main() {
  using namespace kaldi;
  for (int32 i = 0; i < 20; i++) {
    kaldi::chain::ChainSupervisionTest();
    kaldi::chain::BreadthFirstTest();
  }
}
