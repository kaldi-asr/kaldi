// hmm/posterior-test.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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

#include "hmm/posterior.h"

namespace kaldi {


void TestVectorToPosteriorEntry() {
  int32 n = 10 + rand () % 50, gselect = 1 + rand() % 5;
  BaseFloat min_post = 0.1 + 0.8 * RandUniform();

  Vector<BaseFloat> loglikes(n);
  loglikes.SetRandn();
  loglikes.Scale(10.0);

  std::vector<std::pair<int32, BaseFloat> > post_entry;

  BaseFloat ans = VectorToPosteriorEntry(loglikes, gselect, min_post, &post_entry);

  KALDI_ASSERT(post_entry.size() <= gselect);

  int32 max_elem;
  BaseFloat max_val = loglikes.Max(&max_elem);
  KALDI_ASSERT(post_entry[0].first == max_elem);

  KALDI_ASSERT(post_entry.back().second >= min_post);
  KALDI_ASSERT(post_entry.back().second <= post_entry.front().second);

  BaseFloat sum = 0.0;
  for (size_t i = 0; i < post_entry.size(); i++)
    sum += post_entry[i].second;
  KALDI_ASSERT(fabs(sum - 1.0) < 0.01);
  KALDI_ASSERT(ans >= max_val);
}

}

int main() {
  // repeat the test ten times
  for (int i = 0; i < 10; i++) {
    kaldi::TestVectorToPosteriorEntry();
  }
  std::cout << "Test OK.\n";
}

