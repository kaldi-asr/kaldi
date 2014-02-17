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
#include "dec-wrap/dec-wrap-feat-input.h"
#include <iostream>

using namespace kaldi;

class DummyFeatInput: public OnlFeatInputItf {
  public:
    DummyFeatInput(MatrixIndexT rows, MatrixIndexT cols) {
      Vector<BaseFloat> v(cols);
      for (MatrixIndexT i = 0; i < cols ; ++i)
        v(i) = 0;
      m_ = Matrix<BaseFloat>();
      m_.Resize(rows, cols);
      for (MatrixIndexT i = 0; i < rows ; ++i)
        m_.Row(i).CopyFromVec(v);
    }

    virtual MatrixIndexT Compute(Matrix<BaseFloat> *output) {
      KALDI_ASSERT(output->NumRows() == m_.NumRows() &&
          output->NumCols() == m_.NumCols());
      output->CopyFromMat(m_);
      m_.Add(1.0);
      return m_.NumRows();
    }

    virtual int32 Dim() const { return m_.NumCols(); }

    virtual void Reset() { m_.Resize(0, 0); }
  private:
    Matrix<BaseFloat> m_;
};


class DummyFeatNoInput: public OnlFeatInputItf {
  public:
    DummyFeatNoInput(MatrixIndexT cols) { dim_ = cols; }
    virtual MatrixIndexT Compute(Matrix<BaseFloat> *output) { output->Resize(0, 0); return 0; }
    virtual int32 Dim() const { return dim_; }
    virtual void Reset() {}
  private:
    int32 dim_;
};

void test_fm_IsValidFrame(OnlFeatureMatrixOptions opts, int32 dim) {
  DummyFeatInput yes_features(opts.batch_size, dim);
  DummyFeatNoInput no_features(dim);
  OnlFeatureMatrix yes_m(opts, &yes_features);
  OnlFeatureMatrix no_m(opts, &no_features);

  int32 i, bigger_batch = 3 * opts.batch_size;
  for (i = 0; i < bigger_batch; ++i) {
    KALDI_ASSERT(yes_m.IsValidFrame(i));
    KALDI_ASSERT(!no_m.IsValidFrame(i));
  }
}

void test_fm_Dim(OnlFeatureMatrixOptions opts, int32 dim) {
  DummyFeatNoInput feat(dim);
  OnlFeatureMatrix m(opts, &feat);
  KALDI_ASSERT(dim == m.Dim());
}

void test_fm_GetFrame(OnlFeatureMatrixOptions opts, int32 dim) {
  DummyFeatInput feat(opts.batch_size, dim);
  OnlFeatureMatrix m(opts, &feat);

  for (int32 b = 0; b < 3; ++b) {
    // features from b batch
    for (int32 i = 0; i < opts.batch_size; ++i) {
      int32 frame = (b * opts.batch_size) + i;
      KALDI_ASSERT(m.IsValidFrame(frame));
      SubVector<BaseFloat> v = m.GetFrame(frame);
      // int32 first_n = dim; // check all the values - they should be equal
      int32 first_n = 1; // check just the first value
      for (int32 j = 0; j < first_n; ++j) {
        if(v(j) != b)
          std::cerr << "DummyFeatInput return vectors with values " << b
              << " for batch of vectors " << b << std::endl 
              << "However the vector " << i << " of batch " << b
              << " has item " << j << " with value " << v(j) << std::endl;
        KALDI_ASSERT(v(j) == b);
      }
    }
  }

}

int main() {
  OnlFeatureMatrixOptions opts;
  opts.batch_size = 1;
  int32 dim = 39;

  test_fm_IsValidFrame(opts, dim);
  test_fm_Dim(opts, dim);
  test_fm_GetFrame(opts, dim);
  return 0;
}
