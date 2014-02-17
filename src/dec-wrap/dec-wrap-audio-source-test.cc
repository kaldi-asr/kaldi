// -*- coding: utf-8 -*-
/* Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
 * WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
 * MERCHANTABLITY OR NON-INFRINGEMENT.
 * See the Apache 2 License for the specific language governing permissions and
 * limitations under the License. */
#include "dec-wrap-audio-source.h"
#include "matrix/kaldi-vector.h"

using namespace kaldi;


void test_ReadEmpty(const OnlBuffSourceOptions & opts) {
  int32 dim = 8;
  Vector<BaseFloat> to_fill;
  to_fill.Resize(dim);
  OnlBuffSource s(opts);

  int32 read = s.Read(&to_fill);
  KALDI_ASSERT(read == 0);
  KALDI_ASSERT(to_fill.Dim() == dim);
}

void test_Bits() {
  OnlBuffSourceOptions opts;
  int bits[3] = { 8, 16, 32};
  int32 dim = 100;
  unsigned char data[4] = { 'a', 'h', 'o', 'j' };
  Vector<BaseFloat> to_fill;
  to_fill.Resize(dim);

  for(size_t i = 0; i < 3; ++i) {
    opts.bits_per_sample = bits[i];
    OnlBuffSource s(opts);
    size_t num_samples = (4 * 8) / bits[i];

    // write to less data to read
    OnlBuffSource s1(opts);
    for (size_t k = 0; i < dim -1 ; ++k)
      s1.Write(data, num_samples);
    KALDI_ASSERT(s1.Read(&to_fill) == 0);

    // write as much data as needed
    OnlBuffSource s2(opts);
    for (size_t k = 0; i < dim ; ++k)
      s2.Write(data, num_samples);
    KALDI_ASSERT(s2.Read(&to_fill) == 1);
    KALDI_ASSERT(s2.Read(&to_fill) == 0);

    // write enough data for 1 plus extra
    OnlBuffSource s3(opts);
    for (size_t k = 0; i < (2 * dim + 1); ++k)
      s3.Write(data, num_samples);
    KALDI_ASSERT(s3.Read(&to_fill) == 1);
    KALDI_ASSERT(s3.Read(&to_fill) == 0);
  }
}


int main() {
  OnlBuffSourceOptions opts;
  test_ReadEmpty(opts);
  return 0;
}
