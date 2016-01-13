// hmm/hmm-utils-test.cc

// Copyright 2009-2011  Microsoft Corporation
//                2015  Johns Hopkins University (author: Daniel Povey)

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

#include "hmm/hmm-utils.h"
#include "hmm/tree-accu.h"
#include "hmm/hmm-test-utils.h"

namespace kaldi {


void TestConvertPhnxToProns() {

  int32 word_start_sym = 1, word_end_sym = 2;
  { // empty test.
    std::vector<int32> phnx;
    std::vector<int32> words;
    std::vector<std::vector<int32> > ans, ans_check;
    KALDI_ASSERT(ConvertPhnxToProns(phnx, words, word_start_sym,
                                    word_end_sym, &ans)
                 && ans == ans_check);
  }

  { // test w/ one empty word.
    std::vector<int32> phnx; phnx.push_back(3);
    std::vector<int32> words;
    std::vector<std::vector<int32> > ans;
    std::vector<std::vector<int32> > ans_check(1);
    ans_check[0].push_back(0);
    ans_check[0].push_back(3);
    KALDI_ASSERT(ConvertPhnxToProns(phnx, words, word_start_sym,
                                    word_end_sym, &ans)
                 && ans == ans_check);
  }

  { // test w/ one empty word with two phones.
    std::vector<int32> phnx; phnx.push_back(3); phnx.push_back(4);
    std::vector<int32> words;
    std::vector<std::vector<int32> > ans;
    std::vector<std::vector<int32> > ans_check(1);
    ans_check[0].push_back(0);
    ans_check[0].push_back(3);
    ans_check[0].push_back(4);
    KALDI_ASSERT(ConvertPhnxToProns(phnx, words, word_start_sym,
                                    word_end_sym, &ans)
                 && ans == ans_check);
  }

  { // test w/ zero -> should fail.
    std::vector<int32> phnx; phnx.push_back(3); phnx.push_back(4);
    phnx.push_back(0);
    std::vector<int32> words;
    std::vector<std::vector<int32> > ans;
    KALDI_ASSERT(!ConvertPhnxToProns(phnx, words, word_start_sym,
                                     word_end_sym, &ans));
  }

  { // test w/ unexpected word-end -> should fail.
    std::vector<int32> phnx; phnx.push_back(3); phnx.push_back(4);
    phnx.push_back(word_end_sym);
    std::vector<int32> words;
    std::vector<std::vector<int32> > ans;
    KALDI_ASSERT(!ConvertPhnxToProns(phnx, words, word_start_sym,
                                     word_end_sym, &ans));

  }

  { // test w/ word-start but no word-end -> should fail.
    std::vector<int32> phnx; phnx.push_back(3); phnx.push_back(4);
    phnx.push_back(word_start_sym);
    std::vector<int32> words;
    std::vector<std::vector<int32> > ans;
    KALDI_ASSERT(!ConvertPhnxToProns(phnx, words, word_start_sym,
                                     word_end_sym, &ans));

  }

  { // test w/ one empty word then one real word w/ zero phones.
    std::vector<int32> phnx; phnx.push_back(3); phnx.push_back(4);
    phnx.push_back(word_start_sym); phnx.push_back(word_end_sym);
    std::vector<int32> words; words.push_back(100);
    std::vector<std::vector<int32> > ans;
    std::vector<std::vector<int32> > ans_check(2);
    ans_check[0].push_back(0);
    ans_check[0].push_back(3);
    ans_check[0].push_back(4);
    ans_check[1].push_back(100);
    KALDI_ASSERT(ConvertPhnxToProns(phnx, words, word_start_sym,
                                    word_end_sym, &ans)
                 && ans == ans_check);
  }

  { // test w/ one empty word then one real word w/ one phone..
    std::vector<int32> phnx; phnx.push_back(3); phnx.push_back(4);
    phnx.push_back(word_start_sym); phnx.push_back(5); phnx.push_back(word_end_sym);
    std::vector<int32> words; words.push_back(100);
    std::vector<std::vector<int32> > ans;
    std::vector<std::vector<int32> > ans_check(2);
    ans_check[0].push_back(0);
    ans_check[0].push_back(3);
    ans_check[0].push_back(4);
    ans_check[1].push_back(100);
    ans_check[1].push_back(5);
    KALDI_ASSERT(ConvertPhnxToProns(phnx, words, word_start_sym,
                                    word_end_sym, &ans)
                 && ans == ans_check);
  }

  { // test w/ ONE real word w/ one phone..
    std::vector<int32> phnx;
    phnx.push_back(word_start_sym); phnx.push_back(5); phnx.push_back(word_end_sym);
    std::vector<int32> words; words.push_back(100);
    std::vector<std::vector<int32> > ans;
    std::vector<std::vector<int32> > ans_check(1);
    ans_check[0].push_back(100);
    ans_check[0].push_back(5);
    KALDI_ASSERT(ConvertPhnxToProns(phnx, words, word_start_sym,
                                    word_end_sym, &ans)
                 && ans == ans_check);
  }

  { // test w/ ONE real word w/ one phone, but no
    // words supplied-- should fail.
    std::vector<int32> phnx;
    phnx.push_back(word_start_sym); phnx.push_back(5); phnx.push_back(word_end_sym);
    std::vector<int32> words;
    std::vector<std::vector<int32> > ans;
    KALDI_ASSERT(!ConvertPhnxToProns(phnx, words, word_start_sym,
                                    word_end_sym, &ans));
  }

  { // test w/ ONE real word w/ one phone, but two
    // words supplied-- should fail.
    std::vector<int32> phnx;
    phnx.push_back(word_start_sym); phnx.push_back(5); phnx.push_back(word_end_sym);
    std::vector<int32> words(2, 10);
    std::vector<std::vector<int32> > ans;
    KALDI_ASSERT(!ConvertPhnxToProns(phnx, words, word_start_sym,
                                    word_end_sym, &ans));
  }

  { // test w/ ONE real word w/ one phone, but word-id
    // is zero-- should fail.
    std::vector<int32> phnx;
    phnx.push_back(word_start_sym); phnx.push_back(5); phnx.push_back(word_end_sym);
    std::vector<int32> words(1, 0);
    std::vector<std::vector<int32> > ans;
    KALDI_ASSERT(!ConvertPhnxToProns(phnx, words, word_start_sym,
                                    word_end_sym, &ans));
  }

  { // test w/ ONE real word w/ two phones, then one
    // empty word...
    std::vector<int32> phnx;
    phnx.push_back(word_start_sym); phnx.push_back(5);
    phnx.push_back(7); phnx.push_back(word_end_sym);
    phnx.push_back(10);
    std::vector<int32> words; words.push_back(100);
    std::vector<std::vector<int32> > ans;
    std::vector<std::vector<int32> > ans_check(2);
    ans_check[0].push_back(100);
    ans_check[0].push_back(5);
    ans_check[0].push_back(7);
    ans_check[1].push_back(0);
    ans_check[1].push_back(10);
    KALDI_ASSERT(ConvertPhnxToProns(phnx, words, word_start_sym,
                                    word_end_sym, &ans)
                 && ans == ans_check);
  }
}

void TestAccumulateTreeStatsOptions() {
  AccumulateTreeStatsOptions opts;
  opts.var_floor = RandInt(0, 10);
  opts.ci_phones_str = "3:2:1";
  opts.phone_map_rxfilename = "echo 1 2; echo 2 5 |";
  opts.context_width = RandInt(3, 4);
  opts.central_position = RandInt(0, 2);
  AccumulateTreeStatsInfo info(opts);
  KALDI_ASSERT(info.var_floor == opts.var_floor);
  KALDI_ASSERT(info.ci_phones.size() == 3 && info.ci_phones[2] == 3);
  KALDI_ASSERT(info.phone_map.size() == 3 && info.phone_map[2] == 5);
  KALDI_ASSERT(info.context_width == opts.context_width);
  KALDI_ASSERT(info.central_position == opts.central_position);
}

void TestSplitToPhones() {
  ContextDependency *ctx_dep = NULL;
  TransitionModel *trans_model = GenRandTransitionModel(&ctx_dep);
  std::vector<int32> phone_seq;
  int32 num_phones = RandInt(0, 10);
  const std::vector<int32> &phone_list = trans_model->GetPhones();
  for (int32 i = 0; i < num_phones; i++) {
    int32 rand_phone = phone_list[RandInt(0, phone_list.size() - 1)];
    phone_seq.push_back(rand_phone);
  }
  bool reorder = (RandInt(0, 1) == 0);
  std::vector<int32> alignment;
  GenerateRandomAlignment(*ctx_dep, *trans_model, reorder,
                          phone_seq, &alignment);
  std::vector<std::vector<int32> > split_alignment;
  SplitToPhones(*trans_model, alignment, &split_alignment);
  KALDI_ASSERT(split_alignment.size() == phone_seq.size());
  for (size_t i = 0; i < split_alignment.size(); i++) {
    KALDI_ASSERT(!split_alignment[i].empty());
    for (size_t j = 0; j < split_alignment[i].size(); j++) {
      int32 transition_id = split_alignment[i][j];
      KALDI_ASSERT(trans_model->TransitionIdToPhone(transition_id) ==
                   phone_seq[i]);
    }
  }
  delete trans_model;
  delete ctx_dep;
}

void TestConvertAlignment() {
  bool old_reorder = (RandInt(0, 1) == 1),
      new_reorder = (RandInt(0, 1) == 1),
      new_tree = (RandInt(0, 1) == 1),
      new_topology = (RandInt(0, 1) == 1);
  if (!new_tree)
    new_topology = true;

  int32 subsample_factor = RandInt(1, 3);

  KALDI_LOG << " old-reorder = " << old_reorder
            << ", new-reorder = " << new_reorder
            << ", new-tree = " << new_tree
            << ", subsample-factor = " << subsample_factor;

  std::vector<int32> phones;
  phones.push_back(1);
  for (int32 i = 2; i < 20; i++)
    if (rand() % 2 == 0)
      phones.push_back(i);
  int32 N = 2 + rand() % 2, // context-size N is 2 or 3.
      P = rand() % N;  // Central-phone is random on [0, N)

  std::vector<int32> num_pdf_classes_old,
      num_pdf_classes_new;

  ContextDependencyInterface *ctx_dep_old =
      GenRandContextDependencyLarge(phones, N, P,
                                    true, &num_pdf_classes_old),
      *ctx_dep_new;
  if (new_tree) {
    if (new_topology) {
      ctx_dep_new = GenRandContextDependencyLarge(phones, N, P,
                                                  true, &num_pdf_classes_new);
    } else {
      num_pdf_classes_new = num_pdf_classes_old;
      ctx_dep_new = MonophoneContextDependency(phones, num_pdf_classes_new);
    }
  } else {
    num_pdf_classes_new = num_pdf_classes_old;
    ctx_dep_new = ctx_dep_old->Copy();
  }


  HmmTopology topo_old = GenRandTopology(phones, num_pdf_classes_old),
      topo_new =  (new_topology ?
                   GenRandTopology(phones, num_pdf_classes_new) : topo_old);

  TransitionModel trans_model_old(*ctx_dep_old, topo_old),
      trans_model_new(*ctx_dep_new, topo_new);

  std::vector<int32> phone_sequence;
  int32 phone_sequence_length = RandInt(0, 20);
  for (int32 i = 0; i < phone_sequence_length; i++)
    phone_sequence.push_back(phones[RandInt(0, phones.size() - 1)]);
  std::vector<int32> old_alignment;
  GenerateRandomAlignment(*ctx_dep_old, trans_model_old,
                          old_reorder, phone_sequence,
                          &old_alignment);

  std::vector<int32> new_alignment;

  bool ans = ConvertAlignment(trans_model_old, trans_model_new, *ctx_dep_new,
                              old_alignment, subsample_factor, new_reorder,
                              NULL, &new_alignment);
  if(!ans) {
    KALDI_WARN << "Alignment conversion failed";
    // make sure it failed for a good reason.
    KALDI_ASSERT(new_topology || subsample_factor > 1);
  } else {
    std::vector<std::vector<int32> > old_split, new_split;
    bool b1 = SplitToPhones(trans_model_old, old_alignment, &old_split),
        b2 = SplitToPhones(trans_model_new, new_alignment, &new_split);
    KALDI_ASSERT(b1 && b2);
    KALDI_ASSERT(old_split.size() == new_split.size());
    for (size_t i = 0; i < new_split.size(); i++)
      KALDI_ASSERT(trans_model_old.TransitionIdToPhone(old_split[i].front()) ==
                   trans_model_new.TransitionIdToPhone(new_split[i].front()));
    if (!new_topology && subsample_factor == 1) {
      // we should be able to convert back and it'll be the same.
      std::vector<int32> old_alignment_copy;
      bool ans = ConvertAlignment(trans_model_new, trans_model_old, *ctx_dep_old,
                                  new_alignment, subsample_factor, old_reorder,
                                  NULL, &old_alignment_copy);
      KALDI_ASSERT(ans);
      KALDI_ASSERT(old_alignment_copy == old_alignment);
    }

  }
  delete ctx_dep_old;
  delete ctx_dep_new;
}


}

int main() {
  kaldi::TestConvertPhnxToProns();
#ifndef _MSC_VER
  kaldi::TestAccumulateTreeStatsOptions();
#endif
  for (int32 i = 0; i < 2; i++)
    kaldi::TestSplitToPhones();
  for (int32 i = 0; i < 5; i++)
    kaldi::TestConvertAlignment();
  std::cout << "Test OK.\n";
}

