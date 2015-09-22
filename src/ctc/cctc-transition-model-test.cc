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
#include "ctc/cctc-supervision.h"
#include "ctc/cctc-training.h"
#include "ctc/cctc-test-utils.h"
#include "tree/build-tree.h"
#include "tree/build-tree-utils.h"

// This test program tests things declared in ctc-supervision.h and cctc-graph.h
// and cctc-training.h, as well as cctc-transition-model.h.

namespace kaldi {
namespace ctc {



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


void CctcTransitionModelTest() {
  int32 order = RandInt(1, 4);
  int32 vocab_size;
  std::vector<std::vector<int32> > data, validation_data;

  GenerateLanguageModelingData(&vocab_size, &data, &validation_data);
  
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
  for (int32 p = 1; p <= vocab_size; p++)
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
