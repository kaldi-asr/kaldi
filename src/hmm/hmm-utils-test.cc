// hmm/hmm-utils-test.cc

// Copyright 2009-2011 Microsoft Corporation

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


}

int main() {
  kaldi::TestConvertPhnxToProns();
  std::cout << "Test OK.\n";
}

