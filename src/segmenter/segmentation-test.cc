// segmenter/segmentation-test.cc

// Copyright 2015  Vimal Manohar (Johns Hopkins University)

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

#include "segmenter/segmentation.h"

namespace kaldi {
namespace segmenter {

void GenerateRandomSegmentation(int32 max_length, int32 num_classes, 
                                Segmentation *segmentation) {
  Clear();
  int32 s = max_length;
  int32 e = max_length;

  while (s >= 0) {
    int32 chunk_size = rand() % (max_length / 10);
    s = e - chunk_size + 1;
    int32 k = rand() % num_classes;

    if (k != 0) {
      segmentation.Emplace(s, e, k);
    }
    e = s - 1;
  }
  Check();
}


int32 GenerateRandomAlignment(int32 max_length, int32 num_classes, 
                             std::vector<int32> *ali) {
  int32 N = RandInt(1, max_length);
  int32 C = RandInt(1, num_classes);

  ali->clear();

  int32 len = 0;
  while (len < N) {
    int32 c = RandInt(0, C-1);
    int32 n = std::min(RandInt(1, N), N - len);
    ali->insert(ali->begin() + len, n, c);
    len += n;
  }
  KALDI_ASSERT(ali->size() == N && len == N);

  int32 state = -1, num_segments = 0;
  for (std::vector<int32>::const_iterator it = ali->begin(); 
        it != ali->end(); ++it) {
    if (*it != state) num_segments++;
    state = *it;
  }

  return num_segments;
}

void TestConversionToAlignment() {
  std::vector<int32> ali;
  int32 max_length = 1000, num_classes = 3;
  int32 num_segments = GenerateRandomAlignment(max_length, num_classes, &ali);

  Segmentation seg;
  KALDI_ASSERT(num_segments ==  seg.InsertFromAlignment(ali, 0));
  
  std::vector<int32> out_ali;
  {
    seg.ConvertToAlignment(&out_ali);
    KALDI_ASSERT(ali == out_ali);
  }

  {
    seg.ConvertToAlignment(&out_ali, num_classes, max_length * 2);
    std::vector<int32> tmp_ali(out_ali.begin(), out_ali.begin() + ali.size()); 
    KALDI_ASSERT(ali == tmp_ali);
    for (std::vector<int32>::const_iterator it = out_ali.begin() + ali.size(); 
          it != out_ali.end(); ++it) {
      KALDI_ASSERT(*it == num_classes);
    }
  }
  
  seg.Clear();
  KALDI_ASSERT(num_segments ==  seg.InsertFromAlignment(ali, max_length));
  {
    seg.ConvertToAlignment(&out_ali, num_classes, max_length * 2);

    for (std::vector<int32>::const_iterator it = out_ali.begin();
          it != out_ali.begin() + max_length; ++it) {
      KALDI_ASSERT(*it == num_classes);
    }
    std::vector<int32> tmp_ali(out_ali.begin() + max_length, out_ali.begin() + max_length + ali.size()); 
    KALDI_ASSERT(tmp_ali == ali);

    for (std::vector<int32>::const_iterator it = out_ali.begin() + max_length + ali.size();
          it != out_ali.end(); ++it) {
      KALDI_ASSERT(*it == num_classes);
    }
  }
}

void TestRemoveSegments() {
  std::vector<int32> ali;
  int32 max_length = 1000, num_classes = 10;
  int32 num_segments = GenerateRandomAlignment(max_length, num_classes, &ali);

  Segmentation seg;
  KALDI_ASSERT(num_segments ==  seg.InsertFromAlignment(ali, 0));

  for (int32 i = 0; i < num_classes; i++) {
    Segmentation out_seg(seg);
    out_seg.RemoveSegments(i);
    std::vector<int32> out_ali;
    out_seg.ConvertToAlignment(&out_ali, i, ali.size());
    KALDI_ASSERT(ali == out_ali);
  }

  {
    std::vector<int32> classes;
    for (int32 i = 0; i < 3; i++) 
      classes.push_back(RandInt(0, num_classes - 1));
    std::sort(classes.begin(), classes.end());

    Segmentation out_seg1(seg);
    out_seg1.RemoveSegments(classes);
    
    Segmentation out_seg2(seg);
    for (std::vector<int32>::const_iterator it = classes.begin();   
          it != classes.end(); ++it) 
      out_seg2.RemoveSegments(*it);

    std::vector<int32> out_ali1, out_ali2;
    out_seg1.ConvertToAlignment(&out_ali1);
    out_seg2.ConvertToAlignment(&out_ali2);

    KALDI_ASSERT(out_ali1 == out_ali2);
  }
}

void TestIntersectSegments() {
  int32 max_length = 100, num_classes = 3;
  
  std::vector<int32> primary_ali;
  GenerateRandomAlignment(max_length, num_classes, &primary_ali);
  
  std::vector<int32> secondary_ali;
  GenerateRandomAlignment(max_length, num_classes, &secondary_ali);

  Segmentation primary_seg;
  primary_seg.InsertFromAlignment(primary_ali);
  
  Segmentation secondary_seg;
  secondary_seg.InsertFromAlignment(secondary_ali);

  {
    Segmentation out_seg;
    primary_seg.IntersectSegments(secondary_seg, &out_seg, num_classes);

    std::vector<int32> out_ali;
    out_seg.ConvertToAlignment(&out_ali);

    std::vector<int32> oracle_ali(primary_ali.size());

    for (size_t i = 0; i < oracle_ali.size(); i++) {
      int32 p = (i < primary_ali.size()) ? primary_ali[i] : -1;
      int32 s = (i < secondary_ali.size()) ? secondary_ali[i] : -2;

      oracle_ali[i] = (p == s) ? p : num_classes;
    }

    KALDI_ASSERT(oracle_ali == out_ali);
  }
  
  {
    Segmentation out_seg;
    primary_seg.IntersectSegments(secondary_seg, &out_seg);
    
    std::vector<int32> out_ali;
    out_seg.ConvertToAlignment(&out_ali, num_classes);

    std::vector<int32> oracle_ali(out_ali.size());

    for (size_t i = 0; i < oracle_ali.size(); i++) {
      int32 p = (i < primary_ali.size()) ? primary_ali[i] : -1;
      int32 s = (i < secondary_ali.size()) ? secondary_ali[i] : -2;

      oracle_ali[i] = (p == s) ? p : num_classes;
    }

    KALDI_ASSERT(oracle_ali == out_ali);
  }

}

void UnitTestSegmentation() {
  TestConversionToAlignment();
  TestRemoveSegments();
  TestIntersectSegments();
}

} // namespace segmenter
} // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::segmenter;

  for (int32 i = 0; i < 10; i++)
    UnitTestSegmentation();
  return 0;
}
  


