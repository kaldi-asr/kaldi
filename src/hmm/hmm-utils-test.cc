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

void TestSplitToPhones() {
  ContextDependency *ctx_dep;
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

}

int main() {
  kaldi::TestConvertPhnxToProns();
  for (int32 i = 0; i < 2; i++)
    kaldi::TestSplitToPhones();
  std::cout << "Test OK.\n";
}

