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
#include "chain/chain-denominator.h"
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

  BaseFloat forward_prob = num.Forward();

  CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                        nnet_output.NumCols());
  num.Backward(&nnet_output_deriv);

  int32 dim = 3;
  Vector<BaseFloat> predicted_objf_changes(dim),
      observed_objf_changes(dim);
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
    observed_objf_changes(p) = num2.Forward() - forward_prob;
  }
  KALDI_LOG << "Predicted objf changes are: "
            << predicted_objf_changes;
  KALDI_LOG << "Observed objf changes are: "
            << observed_objf_changes;

  {
    BaseFloat correction = (predicted_objf_changes.Sum() - observed_objf_changes.Sum()) /
        predicted_objf_changes.Dim();
    observed_objf_changes.Add(correction);
    KALDI_LOG << "Correcting observed objf changes for statistical effects, to "
              << observed_objf_changes;
    KALDI_ASSERT(predicted_objf_changes.ApproxEqual(observed_objf_changes, 0.1));
  }


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
  Supervision output;
  AppendSupervision(input, &output);
  KALDI_ASSERT(output.frames_per_sequence ==
               supervision.frames_per_sequence &&
               output.num_sequences == num_append);
  int32 tot_sequences_in = 0, tot_sequences_out = 0,
      tot_frames_in = 0, tot_frames_out = 0;
  for (int32 i = 0; i < num_append; i++) {
    tot_sequences_in += input[i]->num_sequences;
    tot_frames_in += input[i]->num_sequences *
        input[i]->frames_per_sequence;
  }
  tot_sequences_out += output.num_sequences;
  tot_frames_out += output.num_sequences *
      output.frames_per_sequence;
  KALDI_ASSERT(tot_sequences_out == tot_sequences_in &&
               tot_frames_out == tot_frames_in);

  TestSupervisionIo(output);
  TestSupervisionNumerator(output);
  output.Check(trans_model);
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


void ChainTrainingTest(const DenominatorGraph &den_graph,
                       const Supervision &supervision) {
  int32 num_sequences = supervision.num_sequences,
      frames_per_sequence = supervision.frames_per_sequence;
  if (frames_per_sequence == 1)  // this will break some code.
    return;

  CuMatrix<BaseFloat> nnet_output(num_sequences * frames_per_sequence,
                                  den_graph.NumPdfs());

  bool zero_output = (RandInt(0, 3) == 0);
  if (!zero_output)
    nnet_output.SetRandn();

  ChainTrainingOptions opts;
  if (RandInt(0, 1) == 1)
    opts.leaky_hmm_coefficient = 0.2;

  CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                        nnet_output.NumCols(),
                                        kUndefined);

  BaseFloat objf, l2_term, weight;

  ComputeChainObjfAndDeriv(opts, den_graph, supervision,
                           nnet_output, &objf, &l2_term, &weight,
                           &nnet_output_deriv);

  {
    // make sure each row of nnet_output_deriv sums to one (shift invariance of
    // the nnet output).
    CuVector<BaseFloat> nnet_output_deriv_row_sums(nnet_output_deriv.NumRows());
    nnet_output_deriv_row_sums.AddColSumMat(1.0, nnet_output_deriv, 0.0);
    KALDI_ASSERT(nnet_output_deriv_row_sums.Norm(2.0) < 0.1);
  }

  KALDI_LOG << "Chain objf per frame is " << (objf / weight)
            << " over " << weight << " frames (weighted)";

  { // a check
    BaseFloat output_deriv_sum = nnet_output_deriv.Sum();
    KALDI_LOG << "Sum of nnet-output-deriv is " << output_deriv_sum
              << " vs. expected 0.";
    KALDI_ASSERT(output_deriv_sum < 0.2);
  }

  KALDI_ASSERT(objf <= 0.0);

  int32 num_tries = 5;
  BaseFloat epsilon = 1.0e-04;
  Vector<BaseFloat> predicted_objf_changes(num_tries),
      observed_objf_changes(num_tries);
  for (int32 p = 0; p < num_tries; p++) {
    CuMatrix<BaseFloat> nnet_delta_output(nnet_output.NumRows(),
                                          nnet_output.NumCols());
    nnet_delta_output.SetRandn();
    nnet_delta_output.Scale(epsilon);
    predicted_objf_changes(p) = TraceMatMat(nnet_output_deriv,
                                            nnet_delta_output, kTrans);
    CuMatrix<BaseFloat> nnet_output_perturbed(nnet_delta_output);
    nnet_output_perturbed.AddMat(1.0, nnet_output);

    BaseFloat objf_modified, l2_term_modified, weight_modified;

    ComputeChainObjfAndDeriv(opts, den_graph, supervision,
                             nnet_output_perturbed,
                             &objf_modified, &l2_term_modified,
                             &weight_modified,
                             NULL);

    observed_objf_changes(p) = objf_modified - objf;
  }
  KALDI_LOG << "Predicted objf changes are " << predicted_objf_changes;
  KALDI_LOG << "Observed objf changes are " << observed_objf_changes;
  {
    Vector<BaseFloat> error(predicted_objf_changes);
    error.AddVec(-1.0, observed_objf_changes);
    KALDI_LOG << "num-sequences = " << num_sequences << ", frames-per-sequence = "
              << frames_per_sequence << ", relative accuracy is "
              << (error.Norm(2.0) / predicted_objf_changes.Norm(2.0));
  }

  {
    // we get inaccuracy for long segments, I think because there is a bias when we
    // add random noise for it to increase the likelihood (for winner-take-all reasons)
    // and for long utterances this bias adds up over the frames and tends to
    // outweigh the random component that the gradient predicts (which will tend to
    // cancel).  Try to correct for this...
    BaseFloat correction = (predicted_objf_changes.Sum() - observed_objf_changes.Sum()) /
        predicted_objf_changes.Dim();
    observed_objf_changes.Add(correction);
    KALDI_LOG << "Correcting observed objf changes for statistical effects, to "
              << observed_objf_changes;
    if (frames_per_sequence > 2 &&
        predicted_objf_changes.Norm(2.0) > 0.1 * epsilon) {
      // if we only have the initial and final frames, due to the scaling-down
      // of pdfs not in the numerator sequence the derivative might be zero,
      // which would cause problems doing the comparison.
      // note, epsilon = 1.0e-04.
      KALDI_ASSERT(predicted_objf_changes.ApproxEqual(observed_objf_changes, 0.25));
    }
  }
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

    Supervision reattached_supervision;
    std::vector<const Supervision*> to_append(num_ranges);
    for (int32 i = 0; i < num_ranges; i++)
      to_append[i] = &(split_supervision[i]);
    AppendSupervision(to_append, &reattached_supervision);
    ChainTrainingTest(den_graph, reattached_supervision);
    if (num_frames % frames_per_range == 0) {
      TestSupervisionReattached(trans_model,
                                supervision,
                                reattached_supervision);
    }
  }
}


void ChainDenominatorTest(const DenominatorGraph &den_graph) {

  int32 num_sequences = RandInt(1, 5),
      frames_per_sequence = RandInt(10, 20);
  if (RandInt(0, 3) == 0)
    frames_per_sequence *= 30;  // test how it works on long sequences
  CuMatrix<BaseFloat> nnet_output(num_sequences * frames_per_sequence,
                                  den_graph.NumPdfs());

  bool zero_output = (RandInt(0, 3) == 0);
  if (!zero_output)
    nnet_output.SetRandn();

  ChainTrainingOptions opts;

  DenominatorComputation denominator_computation(opts, den_graph,
                                                 num_sequences, nnet_output);

  BaseFloat forward_prob = denominator_computation.Forward(),
      per_frame = forward_prob / (num_sequences * frames_per_sequence);
  KALDI_LOG << "Forward prob is " << forward_prob
            << " = " << per_frame << " per frame.";

  CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                        nnet_output.NumCols());

  denominator_computation.Backward(1.0, &nnet_output_deriv);


  { // a check
    BaseFloat output_deriv_sum = nnet_output_deriv.Sum();
    KALDI_LOG << "Sum of nnet-output-deriv is " << output_deriv_sum
              << " vs. expected " << (num_sequences * frames_per_sequence);
    KALDI_ASSERT(output_deriv_sum - BaseFloat(num_sequences * frames_per_sequence) <
                 10.0);
  }

  int32 num_tries = 5;
  BaseFloat epsilon = 1.0e-04;
  Vector<BaseFloat> predicted_objf_changes(num_tries),
      observed_objf_changes(num_tries);
  for (int32 p = 0; p < num_tries; p++) {
    CuMatrix<BaseFloat> nnet_delta_output(nnet_output.NumRows(),
                                          nnet_output.NumCols());
    nnet_delta_output.SetRandn();
    nnet_delta_output.Scale(epsilon);
    predicted_objf_changes(p) = TraceMatMat(nnet_output_deriv,
                                            nnet_delta_output, kTrans);
    CuMatrix<BaseFloat> nnet_output_perturbed(nnet_delta_output);
    nnet_output_perturbed.AddMat(1.0, nnet_output);

    DenominatorComputation denominator_computation_perturbed(opts, den_graph,
                                                             num_sequences,
                                                             nnet_output_perturbed);

    BaseFloat forward_prob_perturbed = denominator_computation_perturbed.Forward();
    observed_objf_changes(p) = forward_prob_perturbed - forward_prob;
  }
  KALDI_LOG << "Predicted objf changes are " << predicted_objf_changes;
  KALDI_LOG << "Observed objf changes are " << observed_objf_changes;
  {
    Vector<BaseFloat> error(predicted_objf_changes);
    error.AddVec(-1.0, observed_objf_changes);
    KALDI_LOG << "num-sequences = " << num_sequences << ", frames-per-sequence = "
              << frames_per_sequence << ", relative error is "
              << (error.Norm(2.0) / predicted_objf_changes.Norm(2.0));
  }
  if (frames_per_sequence < 50) {
    // we get inaccuracy for long segments, I think because there is a bias when we
    // add random noise for it to increase the likelihood (for winner-take-all reasons)
    // and for long utterances this bias adds up over the frames and tends to
    // outweigh the random component that the gradient predicts (which will tend to
    // cancel).
    KALDI_ASSERT(predicted_objf_changes.ApproxEqual(observed_objf_changes, 0.25));
  }
}



void ChainSupervisionTest() {
  ContextDependency *ctx_dep;
  TransitionModel *trans_model = GenRandTransitionModel(&ctx_dep);
  const std::vector<int32> &phones = trans_model->GetPhones();

  int32 subsample_factor = RandInt(1, 3);

  int32 phone_sequence_length = RandInt(1, 20);
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
    ChainDenominatorTest(den_graph);
    if (RandInt(0, 1) == 0)
      supervision.weight = 0.5;
    fst::StdVectorFst normalization_fst;
    den_graph.GetNormalizationFst(den_fst, &normalization_fst);
    // add the weight to the numerator FST so we can assert objf <= 0.
    bool ans = AddWeightToSupervisionFst(normalization_fst, &supervision);
    KALDI_ASSERT(ans);
    // TODO: still have to test for appended sequences.
    ChainTrainingTest(den_graph, supervision);
  }

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

// this function tests SplitIntoRanges() and GetWeightsForRanges().
void TestRanges() {
  int32 frames_per_range = RandInt(20, 100),
                 overlap = RandInt(0, 10),
              num_frames = RandInt(15, 500);
  std::vector<int32> range_starts;
  SplitIntoRanges(num_frames - overlap, frames_per_range - overlap,
                  &range_starts);
  Vector<BaseFloat> weights_orig(num_frames),
      weights_new(num_frames);
  int32 num_ranges = range_starts.size();
  for (int32 i = 0; i < num_ranges; i++) {
    int32 start_t = range_starts[i];
    for (int32 j = 0; j < frames_per_range; j++) {
      int32 t = start_t + j;
      weights_orig(t) += 1.0;
    }
  }
  std::vector<Vector<BaseFloat> > weights;
  GetWeightsForRanges(frames_per_range,
                      range_starts, &weights);
  for (int32 i = 0; i < num_ranges; i++) {
    KALDI_LOG << "weights[" << i << "] = "
              << weights[i];
    int32 start_t = range_starts[i];
    for (int32 j = 0; j < frames_per_range; j++) {
      int32 t = start_t + j;
      weights_new(t) += weights[i](j);
    }
  }
  KALDI_LOG << "Orig weights are " << weights_orig;
  KALDI_LOG << "New weights are " << weights_new;
  for (int32 t = 0; t < num_frames; t++) {
    if (weights_orig(t) != 0.0) {
      KALDI_ASSERT(fabs(weights_new(t) - 1.0) < 0.001);
    } else {
      KALDI_ASSERT(weights_new(t) == 0.0);
    }
  }
}


}  // namespace chain
}  // namespace kaldi

int main() {
  using namespace kaldi;
  SetVerboseLevel(1);
#if HAVE_CUDA == 1
  int32 loop = 0;
  for (loop = 0; loop < 2; loop++) {
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif
    for (int32 i = 0; i < 3; i++) {
      kaldi::chain::ChainSupervisionTest();
      kaldi::chain::BreadthFirstTest();
    }
    kaldi::chain::TestRanges();
#if HAVE_CUDA == 1
  }
  CuDevice::Instantiate().PrintProfile();
#endif
}
