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
#include "fstext/fstext-lib.h"
#include "cudamatrix/cu-device.h"
#include "hmm/hmm-test-utils.h"

// This test program tests things declared in chain-supervision.h

namespace kaldi {
namespace ctc {

/*
// This function creates a compact lattice with phones as the labels,
// and strings (which would normally be transition-ids) containing
// repetitions of '1', for the specified duration.  It's used for
// testing the CCTC supervsion code.
void CreateCompactLatticeFromPhonesAndDurations(
    const std::vector<int32> &phones,
    const std::vector<int32> &durations,
    CompactLattice *lat_out) {
  KALDI_ASSERT(phones.size() == durations.size());
  lat_out->DeleteStates();
  int32 current_state = lat_out->AddState();
  lat_out->SetStart(current_state);
  for (size_t i = 0; i < phones.size(); i++) {
    int32 next_state = lat_out->AddState(),
        phone = phones[i], duration = durations[i];
    std::vector<int32> repeated_ones(duration, 1);
    CompactLatticeWeight weight(LatticeWeight(RandUniform(), RandUniform()),
                                repeated_ones);
    CompactLatticeArc arc(phone, phone, weight, next_state);
    lat_out->AddArc(current_state, arc);
    current_state = next_state;
  }
  CompactLatticeWeight weight(LatticeWeight(RandUniform(), RandUniform()),
                              std::vector<int32>());
  lat_out->SetFinal(current_state, weight);
}

std::ostream &operator << (std::ostream &os, const CctcProtoSupervision &p) {
  p.Write(os, false);
  return os;
}
std::ostream &operator << (std::ostream &os, const CctcSupervision &s) {
  s.Write(os, false);
  return os;
}


void ExcludeRangeFromVector(const std::vector<int32> &in,
                            int32 first, int32 last,
                            std::vector<int32> *out) {
  out->clear();
  out->reserve(in.size());
  for (std::vector<int32>::const_iterator iter = in.begin(),
           end = in.end(); iter != end; ++iter) {
    int32 i = *iter;
    if (i < first || i > last)
      out->push_back(i);
  }
}

void TestCctcSupervisionTraining(const CctcTransitionModel &trans_model,
                                 const CctcSupervision &supervision,
                                 int32 num_sequences) {
  BaseFloat delta = 2.0e-04;
  int32 num_frames = supervision.num_frames,
      nnet_output_dim = trans_model.NumOutputIndexes();
  CuMatrix<BaseFloat> nnet_output(num_frames, nnet_output_dim);
  nnet_output.SetRandn();
  CuMatrix<BaseFloat> cu_weights;
  trans_model.ComputeWeights(&cu_weights);
  CctcHmm hmm(trans_model);
  CctcTrainingOptions opts;
  CctcCommonComputation computation(opts, trans_model, hmm, cu_weights,
                                    supervision, num_sequences, nnet_output);
  CuMatrix<BaseFloat> nnet_output_deriv(num_frames, nnet_output_dim,
                                        kUndefined);

  BaseFloat log_like, den_term, normalizer;
  computation.Forward(&log_like, &den_term, &normalizer);
  KALDI_LOG << "log-like of CCTC computation is " << log_like << " + " << den_term
            << " = " << (log_like + den_term);
  computation.Backward(&nnet_output_deriv);

  int32 num_offsets = 3;
  Vector<BaseFloat> predicted_objf_changes(num_offsets),
      measured_objf_changes(num_offsets);
  for (int32 i = 0; i < num_offsets; i++) {
    CuMatrix<BaseFloat> modified_output(num_frames, nnet_output_dim);
    modified_output.SetRandn();
    modified_output.Scale(delta);
    predicted_objf_changes(i) = TraceMatMat(modified_output,
                                            nnet_output_deriv, kTrans);
    modified_output.AddMat(1.0, nnet_output);
    CctcCommonComputation computation(opts, trans_model, hmm, cu_weights,
                                      supervision, num_sequences,
                                      modified_output);
    BaseFloat modified_log_like, modified_den_term, modified_normalizer;
    computation.Forward(&modified_log_like, &modified_den_term,
                        &modified_normalizer);
    // no need to do backward.
    measured_objf_changes(i) =
        (modified_log_like + modified_den_term) - (log_like + den_term);
  }
  KALDI_LOG << "predicted_objf_changes = " << predicted_objf_changes;
  KALDI_LOG << "measured_objf_changes = " << measured_objf_changes;
  KALDI_ASSERT(predicted_objf_changes.ApproxEqual(measured_objf_changes, 0.1));
}

void TestCctcSupervisionFrames(const CctcSupervision &supervision) {
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
  KALDI_ASSERT(isymbols_out.size() == static_cast<size_t>(supervision.num_frames));
  KALDI_ASSERT(weight_out == fst::TropicalWeight::One());

  bool test = true;
  // make sure epsilon free
  KALDI_ASSERT(supervision.fst.Properties(fst::kNoEpsilons, test) != 0);
  // make sure unweighted
  KALDI_ASSERT(supervision.fst.Properties(fst::kUnweighted, test) != 0);
}

void TestCctcSupervisionIo(const CctcSupervision &supervision) {
  bool binary = (RandInt(0, 1) == 0);
  std::ostringstream os;
  supervision.Write(os, binary);
  std::istringstream is(os.str());
  CctcSupervision supervision2;
  if (RandInt(0, 1) == 0)
    supervision2 = supervision;  // test reading already-existing object.
  supervision2.Read(is, binary);
  std::ostringstream os2;
  supervision2.Write(os2, binary);
  KALDI_ASSERT(os.str() == os2.str());
  if (binary) {
    KALDI_ASSERT(supervision == supervision2);
  }
}

void TestCctcSupervisionAppend(const CctcTransitionModel &trans_model,
                               const CctcSupervision &supervision) {
  int32 num_append = RandInt(1,5);
  std::vector<const CctcSupervision*> input(num_append);
  for (int32 i = 0; i < num_append; i++)
    input[i] = &supervision;
  std::vector<CctcSupervision> output;
  bool compactify = (RandInt(0, 1) == 0);
  AppendCctcSupervision(input, compactify, &output);
  if (compactify) {
    KALDI_ASSERT(output.size() == 1 &&
                 output[0].num_frames == num_append * supervision.num_frames);
  } else {
    KALDI_ASSERT(output.size() == input.size());
  }
  TestCctcSupervisionIo(output[0]);
  int32 num_append_0 = output[0].num_frames / supervision.num_frames;
  TestCctcSupervisionTraining(trans_model, output[0], num_append_0);
  output[0].Check(trans_model);
}


// make sure that reattached_supervision is plausibly the result of splitting
// and then reattaching 'supervision'.
void TestCctcSupervisionReattached(const CctcTransitionModel &trans_model,
                                   const CctcSupervision &supervision,
                                   const CctcSupervision &reattached_supervision) {
  using namespace fst;
  KALDI_LOG << "testing reattached";
  KALDI_ASSERT(reattached_supervision.num_frames == supervision.num_frames &&
               reattached_supervision.weight == supervision.weight &&
               reattached_supervision.label_dim == supervision.label_dim);
  UniformArcSelector<StdArc> selector;
  RandGenOptions<UniformArcSelector<StdArc> > randgen_opts(selector);
  VectorFst<StdArc> fst_path;
  RandGen(supervision.fst, &fst_path, randgen_opts);
  VectorFst<StdArc> composed;
  Compose(fst_path, reattached_supervision.fst, &composed);
  Connect(&composed);
  KALDI_ASSERT(composed.NumStates() != 0);
  supervision.Check(trans_model);
  reattached_supervision.Check(trans_model);
}

void TestCctcSupervisionSplitting(const CctcTransitionModel &trans_model,
                                  const CctcSupervision &supervision) {
  CctcSupervisionSplitter splitter(supervision);
  int32 num_frames = supervision.num_frames,
      frames_per_range = RandInt(3, 10);
  std::vector<int32> range_starts;
  SplitIntoRanges(num_frames, frames_per_range, &range_starts);
  int32 num_ranges = range_starts.size();
  std::vector<CctcSupervision> split_supervision(num_ranges);
  for (int32 i = 0; i < num_ranges; i++) {
    splitter.GetFrameRange(range_starts[i], frames_per_range,
                           &split_supervision[i]);
    split_supervision[i].Check(trans_model);
  }
  if (num_ranges > 0) {
    TestCctcSupervisionIo(split_supervision[RandInt(0, num_ranges - 1)]);
    TestCctcSupervisionFrames(split_supervision[RandInt(0, num_ranges - 1)]);
    if (num_frames % frames_per_range == 0) {
      // co-test with Append.
      std::vector<CctcSupervision> reattached_supervision;
      std::vector<const CctcSupervision*> to_append(num_ranges);
      for (int32 i = 0; i < num_ranges; i++)
        to_append[i] = &(split_supervision[i]);
      bool compactify = true;
      AppendCctcSupervision(to_append, compactify, &reattached_supervision);
      KALDI_ASSERT(reattached_supervision.size() == 1);
      TestCctcSupervisionReattached(trans_model,
                                    supervision,
                                    reattached_supervision[0]);

    }
  }
}


void TestCctcSupervision(const CctcTransitionModel &trans_model) {
  int32 num_phones = trans_model.NumPhones(),
      tot_frames = 0, subsample_factor = RandInt(1, 3);
  std::vector<int32> phones, durations;
  int32 sequence_length = RandInt(1, 20);
  for (int32 i = 0; i < sequence_length; i++) {
    int32 phone = RandInt(1, num_phones),
        duration = RandInt(1, 6);
    phones.push_back(phone);
    durations.push_back(duration);
    tot_frames += duration;
  }
  if (tot_frames < subsample_factor) {
    // Don't finish the test because it would fail.  we run this multiple times.
    return;
  }
  CctcSupervisionOptions sup_opts;
  sup_opts.frame_subsampling_factor = subsample_factor;
  // keep the following two lines in sync and keep the silence
  // range contiguous for the test to work.
  int32 first_silence_phone = 1, last_silence_phone = 3;
  sup_opts.silence_phones = "1:2:3";
  bool start_from_lattice = (RandInt(0, 1) == 0);
  CctcProtoSupervision proto_supervision;
  if (start_from_lattice) {
    CompactLattice clat;
    CreateCompactLatticeFromPhonesAndDurations(phones, durations, &clat);
    PhoneLatticeToProtoSupervision(clat, &proto_supervision);
  } else {
    if (RandInt(0, 1) == 0) {
      std::vector<std::pair<int32, int32> > pairs(phones.size());
      for (size_t i = 0; i < pairs.size(); i++) {
        pairs[i].first = phones[i];
        pairs[i].second = durations[i];
      }
      AlignmentToProtoSupervision(pairs, &proto_supervision);
    } else {
      AlignmentToProtoSupervision(phones, durations, &proto_supervision);
    }
  }
  KALDI_LOG << "Original proto-supervision is: " << proto_supervision;
  MakeSilencesOptional(sup_opts, &proto_supervision);
  KALDI_LOG << "Proto-supervision after making silences optional is: " << proto_supervision;
  ModifyProtoSupervisionTimes(sup_opts, &proto_supervision);
  KALDI_LOG << "Proto-supervision after modifying times is: " << proto_supervision;
  AddBlanksToProtoSupervision(&proto_supervision);
  KALDI_LOG << "Proto-supervision after adding blanks is: " << proto_supervision;
  CctcSupervision supervision;
  if (!MakeCctcSupervisionNoContext(proto_supervision, num_phones,
                                   &supervision)) {
    // the only way this should fail is if we had too many phones for
    // the number of subsampled frames.
    KALDI_ASSERT(sequence_length > tot_frames / subsample_factor);
    KALDI_LOG << "Failed to create CctcSupervision because too many "
              << "phones for too few frames.";
    return;
  }
  KALDI_LOG << "Supervision without context is: " << supervision;
  AddContextToCctcSupervision(trans_model, &supervision);
  KALDI_LOG << "Supervision after adding context is: " << supervision;


  fst::StdVectorFst one_path;
  // ShortestPath effectively chooses an arbitrary path, because all paths have
  // unit weight / zero cost.
  ShortestPath(supervision.fst, &one_path);


  std::vector<int32> graph_label_seq_in, graph_label_seq_out;
  fst::TropicalWeight tot_weight;
  GetLinearSymbolSequence(one_path, &graph_label_seq_in,
                          &graph_label_seq_out, &tot_weight);
  KALDI_ASSERT(tot_weight == fst::TropicalWeight::One() &&
               graph_label_seq_in == graph_label_seq_out);

  { // basic testing of ComputeFstStateTimes (it has a lot of asserts).
    std::vector<int32> state_times;
    int32 length = ComputeFstStateTimes(supervision.fst, &state_times);
    KALDI_ASSERT(static_cast<size_t>(length) == graph_label_seq_out.size());
    for (size_t i = 0; i + 1 < state_times.size(); i++)
      KALDI_ASSERT(state_times[i] <= state_times[i+1]);
  }


  std::vector<int32> phones_from_graph;
  for (size_t i = 0; i < graph_label_seq_in.size(); i++) {
    int32 this_phone = trans_model.GraphLabelToPhone(graph_label_seq_in[i]);
    if (this_phone != 0)
      phones_from_graph.push_back(this_phone);
  }
  // phones_from_graph should equal 'phones', except that silences may be
  // deleted.  Check this.
  std::vector<int32> phone_nosil, phones_from_graph_nosil;
  ExcludeRangeFromVector(phones, first_silence_phone, last_silence_phone,
                         &phone_nosil);
  ExcludeRangeFromVector(phones_from_graph, first_silence_phone,
                         last_silence_phone, &phones_from_graph_nosil);
  KALDI_ASSERT(phone_nosil == phones_from_graph_nosil);
  supervision.Check(trans_model);
  TestCctcSupervisionIo(supervision);
  TestCctcSupervisionFrames(supervision);
  TestCctcSupervisionSplitting(trans_model, supervision);
  TestCctcSupervisionAppend(trans_model, supervision);
  TestCctcSupervisionTraining(trans_model, supervision, 1);
}

void CctcSupervisionTest() {
  CctcTransitionModel trans_model;
  GenerateCctcTransitionModel(&trans_model);
  TestCctcSupervision(trans_model);
}

void AddArc(int32 from, int32 to,
            fst::StdVectorFst *fst) {
  fst->AddArc(from, fst::StdArc(0, 0, fst::TropicalWeight::One(), to));
}

*/

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

  // KALDI_ASSERT(output
  TestSupervisionIo(output[0]);
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
  VectorFst<StdArc> fst_path;
  RandGen(supervision.fst, &fst_path, randgen_opts);
  VectorFst<StdArc> composed;
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
  // make sure unweighted
  KALDI_ASSERT(supervision.fst.Properties(fst::kUnweighted, test) != 0);
}


void TestSupervisionSplitting(const TransitionModel &trans_model,
                              const Supervision &supervision) {
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
  TestSupervisionSplitting(*trans_model, supervision);
  TestSupervisionAppend(*trans_model, supervision);

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

}  // namespace ctc
}  // namespace kaldi

int main() {
  using namespace kaldi;
  for (int32 i = 0; i < 20; i++) {
    kaldi::ctc::ChainSupervisionTest();
    kaldi::ctc::BreadthFirstTest();
  }
}
