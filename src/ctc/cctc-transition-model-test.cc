// ctc/cctc-transition-model-test.cc

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

#include "ctc/cctc-transition-model.h"
#include "ctc/cctc-graph.h"
#include "ctc/language-model.h"
#include "ctc/ctc-supervision.h"
#include "tree/build-tree.h"
#include "tree/build-tree-utils.h"


namespace kaldi {
namespace ctc {

static void GetTestingData(int32 *vocab_size,
                    std::vector<std::vector<int32> > *data,
                    std::vector<std::vector<int32> > *validation_data) {
  // read the code of a C++ file as training data.
  bool binary;
  Input input("language-model.cc", &binary);
  KALDI_ASSERT(!binary);
  std::istream &is = input.Stream();
  std::string line;
  *vocab_size = 127;
  int32 line_count = 0;
  for (; getline(is, line); line_count++) {
    std::vector<int32> int_line(line.size());
    for (size_t i = 0; i < line.size(); i++) {
      int32 this_char = line[i];
      if (this_char == 0) {
        this_char = 1;  // should never happen, but just make sure, as 0 is
                        // treated as BOS/EOS in the language modeling code.
      }
      int_line[i] = std::min<int32>(127, this_char);
    }
    if (line_count % 10 != 0)
      data->push_back(int_line);
    else
      validation_data->push_back(int_line);
  }
  KALDI_ASSERT(line_count > 0);
}

// This function, modified from GenRandContextDependency(), generates a random
// context-dependency tree that only has left-context, and ensures that all
// pdf-classes are numbered zero (as required for the CCTC code).
static ContextDependency *GenRandContextDependencySpecial(
    const std::vector<int32> &phone_ids) {
  bool ensure_all_covered = true;
  KALDI_ASSERT(IsSortedAndUniq(phone_ids));
  int32 num_stats = 1 + (Rand() % 15) * (Rand() % 15);  // up to 14^2 + 1 separate stats.
  int32 N = 1 + Rand() % 2;  // 1, 2 or 3.  So 0, 1 or 2 phones of left context.
                             //  The transition-model creation code blows up if
                             //  we have more, as it's based on enumerating all
                             //  phone contexts and then merging identical
                             //  history-states.
  int32 P = N - 1;  // Ensure tree left-context only.
  float ctx_dep_prob = 0.7 + 0.3*RandUniform();
  int32 max_phone = *std::max_element(phone_ids.begin(), phone_ids.end());

  std::vector<bool> is_ctx_dep(max_phone + 1);

  std::vector<int32> hmm_lengths(max_phone + 1, -1);

  // I'm guessing the values for i==0 will never be accessed.
  for (int32 i = 1; i <= max_phone; i++) {
    hmm_lengths[i] = 1;
    is_ctx_dep[i] = (RandUniform() < ctx_dep_prob);  // true w.p. ctx_dep_prob.
  }

  // Generate rand stats.
  BuildTreeStatsType stats;
  size_t dim = 3 + Rand() % 20;
  GenRandStats(dim, num_stats, N, P, phone_ids, hmm_lengths,
               is_ctx_dep, ensure_all_covered, &stats);

  // Now build the tree.

  Questions qopts;
  int32 num_quest = Rand() % 10, num_iters = rand () % 5;
  qopts.InitRand(stats, num_quest, num_iters, kAllKeysUnion);  // This was tested in build-tree-utils-test.cc

  float thresh = 100.0 * RandUniform();

  EventMap *tree = NULL;
  std::vector<std::vector<int32> > phone_sets(phone_ids.size());
  for (size_t i = 0; i < phone_ids.size(); i++)
    phone_sets[i].push_back(phone_ids[i]);
  std::vector<bool> share_roots(phone_sets.size(), true),
      do_split(phone_sets.size(), true);

  tree = BuildTree(qopts, phone_sets, hmm_lengths, share_roots,
                   do_split, stats, thresh, 1000, 0.0, P);
  DeleteBuildTreeStats(&stats);
  return new ContextDependency(N, P, tree);
}

void TestCctcTransitionModelIo(const CctcTransitionModel &trans_model) {
  bool binary = (RandInt(0, 1) == 0);
  std::ostringstream os;
  trans_model.Write(os, binary);
  CctcTransitionModel trans_model2;
  std::istringstream is(os.str());
  trans_model2.Read(is, binary);
  std::ostringstream os2;
  trans_model2.Write(os2, binary);
  if (binary)
    KALDI_ASSERT(os.str() == os2.str());
}

void TestCctcTransitionModelProbs(const CctcTransitionModel &trans_model,
                                  const LanguageModel &lm) {
  int32 num_phones = trans_model.NumPhones(),
      ngram_order = lm.NgramOrder(),
      sequence_length = RandInt(1, 20);
  std::vector<int32> history;
  history.push_back(0);  // Beginning-of-sentence history.
  int32 current_history_state = trans_model.InitialHistoryState();
  for (int32 i = 0; i < sequence_length; i++) {
    int32 next_phone = RandInt(1, num_phones);
    std::vector<int32> history_plus_eos(history);
    history.push_back(next_phone);
    history_plus_eos.push_back(0);  // add end-of-sentence to the old history
    BaseFloat lm_prob = lm.GetProb(history),
        lm_prob_renormalized = lm_prob / (1.0 - lm.GetProb(history_plus_eos));
    BaseFloat lm_prob_from_trans_model =
        trans_model.GetLmProb(current_history_state, next_phone);
    AssertEqual(lm_prob_renormalized, lm_prob_from_trans_model);
    current_history_state =
        trans_model.GetNextHistoryState(current_history_state, next_phone);
    if (history.size() > ngram_order - 1)
      history.erase(history.begin());
  }
}

int32 GetOutputIndex(int32 num_non_blank_indexes,
                     const ContextDependency &ctx_dep,
                     const LmHistoryStateMap &history_state_map,
                     const std::vector<int32> &hist,
                     int32 phone) {
  if (phone == 0) {  // blank.
    return num_non_blank_indexes + history_state_map.GetLmHistoryState(hist);
  } else {
    std::vector<int32> ngram(hist);
    ngram.push_back(phone);
    int32 context_width = ctx_dep.ContextWidth();
    while (ngram.size() < static_cast<size_t>(context_width))
      ngram.insert(ngram.begin(), 0);  // pad with 0s to left.
    while (ngram.size() > static_cast<size_t>(context_width))
      ngram.erase(ngram.begin());  // shift left.
    int32 pdf_class = 0;  // we always set pdf_class to 0 in the CCTC code
                          // (make it as if each phone has one state).
    int32 pdf_id;
    bool ans = ctx_dep.Compute(ngram, pdf_class, &pdf_id);
    KALDI_ASSERT(ans && "Failure computing from tree.");
    KALDI_ASSERT(pdf_id >= 0 && pdf_id < num_non_blank_indexes);
    return pdf_id;
  }
}

void TestCctcTransitionModelIndexes(const CctcTransitionModel &trans_model,
                                    const ContextDependency &ctx_dep,
                                    const LmHistoryStateMap &history_state_map) {
  int32 num_phones = trans_model.NumPhones(),
      left_context = trans_model.PhoneLeftContext(),
      sequence_length = RandInt(1, 20),
      num_non_blank_indexes = trans_model.NumNonBlankIndexes();
  KALDI_ASSERT(num_non_blank_indexes == ctx_dep.NumPdfs());
  std::vector<int32> history;
  history.push_back(0);  // Beginning-of-sentence history.
  int32 current_history_state = trans_model.InitialHistoryState();
  for (int32 i = 0; i < sequence_length; i++) {
    // test_phone is the phone whose output index we will test, which may be
    // zero (blank)
    int32 test_phone = RandInt(0, num_phones); 
    if (RandInt(0, 3) == 0)  // Boost probability of seeing zero (blank phone).
      test_phone = 0;
    
    int32 trans_model_output_index = trans_model.GetOutputIndex(
        current_history_state, test_phone),
        output_index = GetOutputIndex(num_non_blank_indexes, ctx_dep,
                                      history_state_map, history, test_phone);
    KALDI_ASSERT(trans_model_output_index == output_index);

    // Now advance the history-state using a "real" (non-blank) phone.
    int32 next_phone = RandInt(1, num_phones);
    history.push_back(next_phone);
    current_history_state =
        trans_model.GetNextHistoryState(current_history_state, next_phone);
    if (history.size() > left_context)
      history.erase(history.begin());
  }
}


void TestCctcTransitionModelGraph(const CctcTransitionModel &trans_model) {
  int32 num_phones = trans_model.NumPhones(),
      sequence_length = RandInt(1, 20);

  std::vector<int32> phones;
  std::vector<int32> phones_plus_blanks;
  for (int32 i = 0; i < sequence_length; i++) {
    while (RandInt(0, 1) == 0)
      phones_plus_blanks.push_back(0);
    int32 phone = RandInt(1, num_phones);
    phones.push_back(phone);
    phones_plus_blanks.push_back(phone);
  }
  while (RandInt(0, 1) == 0)
    phones_plus_blanks.push_back(0);

  // get the sequence of GraphIds that corresponds to the randomly chosen
  // phones_plus_blanks sequence.
  std::vector<int32> graph_ids;

  int32 current_history_state = trans_model.InitialHistoryState();
  BaseFloat tot_log_prob = 0.0;
  for (size_t i = 0; i < phones_plus_blanks.size(); i++) {
    int32 phone_or_blank = phones_plus_blanks[i];
    graph_ids.push_back(trans_model.GetGraphLabel(current_history_state,
                                                  phone_or_blank));
    tot_log_prob += log(trans_model.GetLmProb(current_history_state,
                                              phone_or_blank));
    current_history_state =
        trans_model.GetNextHistoryState(current_history_state,
                                        phone_or_blank);
  }

  typedef fst::VectorFst<fst::StdArc> Fst;
  Fst phone_acceptor;
  MakeLinearAcceptor(phones, &phone_acceptor);
  // adds one to just the input side of phone_acceptor, and
  // then adds self-loops with the blank-plus-one (=1) on
  // the input side and eps on the output side.
  ShiftPhonesAndAddBlanks(&phone_acceptor);
  BaseFloat phone_language_model_weight = RandUniform();
  
  Fst ctc_fst;
  CreateCctcDecodingFst(trans_model, phone_language_model_weight,
                        phone_acceptor,
                        &ctc_fst);
  /*
  for the following to work, need to #include "fst/script/print-impl.h"

    {
#ifdef HAVE_OPENFST_GE_10400
    fst::FstPrinter<fst::StdArc> fstprinter(ctc_fst, NULL, NULL, NULL,
                                            false, true, "\t");
#else
    fst::FstPrinter<fst::StdArc> fstprinter(ctc_fst, NULL, NULL, NULL,
                                            false, true);
#endif
    fstprinter.Print(&std::cout, "standard output");
    } */
  // make linear acceptor with the symbols we generated manually
  // from the transition-model, corresponding to the phone-seq
  // with random blanks.  We'll be checking that this sequence
  // appears in the generated FST.
  Fst graph_id_acceptor;
  MakeLinearAcceptor(graph_ids, &graph_id_acceptor);

  // WriteIntegerVector(std::cerr, false, graph_ids);
  
  Fst composed_fst;
  // note: ctc_fst has the graph-labels on its input side.
  Compose(graph_id_acceptor, ctc_fst, &composed_fst);

  if (composed_fst.NumStates() == 0)  // empty FST
    KALDI_ERR << "Did not find the expected symbol sequence in CCTC graph.";
  
  // we expect that the output FST will have a linear structure.
  std::vector<int32> input_seq, output_seq;
  fst::StdArc::Weight tot_weight;
  bool is_linear = GetLinearSymbolSequence(composed_fst,
                                           &input_seq, &output_seq,
                                           &tot_weight);
  if (!is_linear)
    KALDI_ERR << "Expected composed FST to have linear structure.";
  KALDI_ASSERT(input_seq == graph_ids);
  KALDI_ASSERT(output_seq == phones);

  BaseFloat tot_cost = tot_weight.Value(),
      expected_tot_cost = -tot_log_prob * phone_language_model_weight;
  if (!ApproxEqual(tot_cost, expected_tot_cost))
    KALDI_ERR << "Total cost of FST is not what we expected";
}  


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

std::ostream &operator << (std::ostream &os, const CtcProtoSupervision &p) {
  p.Write(os, false);
  return os;
}
std::ostream &operator << (std::ostream &os, const CtcSupervision &s) {
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

void TestCtcSupervisionIo(const CtcSupervision &supervision) {
  bool binary = (RandInt(0, 1) == 0);
  std::ostringstream os;
  supervision.Write(os, binary);
  std::istringstream is(os.str());
  CtcSupervision supervision2;
  if (RandInt(0, 1) == 0)
    supervision2 = supervision;  // test reading already-existing object.
  supervision2.Read(is, binary);
  std::ostringstream os2;
  supervision2.Write(os2, binary);
  KALDI_ASSERT(os.str() == os2.str());
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
  CtcSupervisionOptions sup_opts;
  sup_opts.frame_subsampling_factor = subsample_factor;
  // keep the following two lines in sync and keep the silence
  // range contiguous for the test to work.
  int32 first_silence_phone = 1, last_silence_phone = 3;
  sup_opts.silence_phones = "1:2:3";
  bool start_from_lattice = (RandInt(0, 1) == 0);
  CtcProtoSupervision proto_supervision;
  if (start_from_lattice) {
    CompactLattice clat;
    CreateCompactLatticeFromPhonesAndDurations(phones, durations, &clat);
    PhoneLatticeToProtoSupervision(clat, &proto_supervision);
  } else {
    AlignmentToProtoSupervision(phones, durations, &proto_supervision);
  }
  KALDI_LOG << "Original proto-supervision is: " << proto_supervision;
  MakeSilencesOptional(sup_opts, &proto_supervision);
  KALDI_LOG << "Proto-supervision after making silences optional is: " << proto_supervision;
  ModifyProtoSupervisionTimes(sup_opts, &proto_supervision);
  KALDI_LOG << "Proto-supervision after modifying times is: " << proto_supervision;
  AddBlanksToProtoSupervision(&proto_supervision);
  KALDI_LOG << "Proto-supervision after adding blanks is: " << proto_supervision;
  CtcSupervision supervision;
  if (!MakeCtcSupervisionNoContext(proto_supervision, num_phones,
                                   &supervision)) {
    // the only way this should fail is if we had too many phones for
    // the number of subsampled frames.    
    KALDI_ASSERT(sequence_length > tot_frames / subsample_factor);
    KALDI_LOG << "Failed to create CtcSupervision because too many "
              << "phones for too few frames.";
    return;
  }
  KALDI_LOG << "Supervision without context is: " << supervision;
  AddContextToCtcSupervision(trans_model, &supervision);
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
  TestCtcSupervisionIo(supervision);
}

void CctcTransitionModelTest() {
  int32 order = RandInt(1, 4);
  int32 vocab_size;
  std::vector<std::vector<int32> > data, validation_data;

  GetTestingData(&vocab_size, &data, &validation_data);
  
  LanguageModelOptions opts;
  opts.ngram_order = order;
  if (RandInt(0,3) == 0)
    opts.state_count_cutoff1 = 100.0;
  if (RandInt(0,3) == 0) {
    opts.state_count_cutoff1 = 10.0;
    opts.state_count_cutoff2plus = 10.0;
  }
  if (RandInt(0,5) == 0) {
    opts.state_count_cutoff1 = 0.0;
    opts.state_count_cutoff2plus = 0.0;
  }
  
  
  LanguageModelEstimator estimator(opts, vocab_size);
  for (size_t i = 0; i < data.size(); i++) {
    std::vector<int32> &sentence = data[i];
    estimator.AddCounts(sentence);
  }
  estimator.Discount();
  LanguageModel lm;
  estimator.Output(&lm);

  KALDI_LOG << "For order " << order << ", cutoffs "
            << opts.state_count_cutoff1 << ","
            << opts.state_count_cutoff2plus << ", perplexity is "
            << ComputePerplexity(lm, validation_data) << "[valid]"
            << " and " << ComputePerplexity(lm, data) << "[train].";

  std::vector<int32> phones;
  for (int32 p = 1; p <= 127; p++)
    phones.push_back(p);
  ContextDependency *dep = GenRandContextDependencySpecial(phones);

  CctcTransitionModelCreator creator(*dep, lm);
  CctcTransitionModel trans_model;
  creator.InitCctcTransitionModel(&trans_model);

  TestCctcTransitionModelIo(trans_model);
  TestCctcTransitionModelProbs(trans_model, lm);
  LmHistoryStateMap history_state_map;
  history_state_map.Init(lm);
  TestCctcTransitionModelIndexes(trans_model, *dep, history_state_map);
  TestCctcTransitionModelGraph(trans_model);
  TestCctcSupervision(trans_model);

  {
    // each row sum of the weights should be 2 (1 for element 0, 1 for
    // the sum of the rest).
    Matrix<BaseFloat> weights;
    trans_model.ComputeWeights(&weights);
    AssertEqual(weights.Sum(), 2.0 * trans_model.NumHistoryStates());
  }

  KALDI_ASSERT(trans_model.NumGraphLabels() ==
               trans_model.NumHistoryStates() * (trans_model.NumPhones() + 1));
  for (int32 g = 1; g <= trans_model.NumGraphLabels(); g++) {
    int32 p = trans_model.GraphLabelToPhone(g),
        h = trans_model.GraphLabelToHistoryState(g);
    KALDI_ASSERT(trans_model.GetGraphLabel(h, p) == g);

    KALDI_ASSERT(trans_model.GraphLabelToNextHistoryState(g) ==
                 trans_model.GetNextHistoryState(h, p));
    KALDI_ASSERT(trans_model.GetLmProb(h, p) ==
                 trans_model.GraphLabelToLmProb(g));
    KALDI_ASSERT(trans_model.GetOutputIndex(h, p) ==
                 trans_model.GraphLabelToOutputIndex(g));
  }
  
  delete dep;
}



}  // namespace ctc
}  // namespace kaldi

int main() {
  for (int32 i = 0; i < 10; i++)
    kaldi::ctc::CctcTransitionModelTest();
}
