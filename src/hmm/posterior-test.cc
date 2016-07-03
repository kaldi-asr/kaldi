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
#include "base/kaldi-math.h"

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

void TestPosteriorIo() {
  int32 post_size = RandInt(0, 5);
  post_size = post_size * post_size;
  Posterior post(post_size);
  for (int32 i = 0; i < post.size(); i++) {
    int32 s = RandInt(0, 3);
    for (int32 j = 0; j < s; j++)
      post[i].push_back(std::pair<int32,BaseFloat>(
          RandInt(-10, 100), RandUniform()));
  }
  bool binary = (RandInt(0, 1) == 0);
  std::ostringstream os;
  WritePosterior(os, binary, post);
  Posterior post2;
  if (RandInt(0, 1) == 0)
    post2 = post;
  std::istringstream is(os.str());
  ReadPosterior(is, binary, &post2);
  if (binary) {
    KALDI_ASSERT(post == post2);
  } else {
    KALDI_ASSERT(post.size() == post2.size());
    for (int32 i = 0; i < post.size(); i++) {
      KALDI_ASSERT(post[i].size() == post2[i].size());
      for (int32 j = 0; j < post[i].size(); j++) {
        KALDI_ASSERT(post[i][j].first == post2[i][j].first &&
                     fabs(post[i][j].second - post2[i][j].second) < 0.01);
      }
    }
  }
}
}

int main() {
  // repeat the test ten times
  for (int i = 0; i < 10; i++) {
    kaldi::TestVectorToPosteriorEntry();
    kaldi::TestPosteriorIo();
  }
  std::cout << "Test OK.\n";
}

