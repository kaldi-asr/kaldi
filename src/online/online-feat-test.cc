// online/online-feat-test.cc

// Copyright 2013   Daniel Povey

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

#include "online/online-feat-input.h"

namespace kaldi {

// This class is for testing and prototyping purposes, it
// does not really do anything except wrap a matrix of features
// in this class.  Note: it maintains a reference to the input
// matrix, so be careful not to delete it while this object
// Since this is intended for testing purposes, it may occasionally
// "time out" and return fewer than requested
class OnlineMatrixInput : public OnlineFeatInputItf {
 public:
  OnlineMatrixInput(const Matrix<BaseFloat> &feats):
      position_(0), feats_(feats) { }

  virtual int32 Dim() const { return feats_.NumCols(); }
  
  virtual bool Compute(Matrix<BaseFloat> *output) {
    if (feats_.NumRows() == 0) { // empty input.
      output->Resize(0, 0);
      return false;
    }
    
    KALDI_ASSERT(output->NumRows() > 0 &&
                 output->NumCols() == feats_.NumCols());
    
    // Because this is a kind of stress test, we completely ignore
    // the number of frames requested, and return whatever number of
    // frames we please.

    int32 num_frames_left = feats_.NumRows() - position_;
    int32 num_frames_return = std::min((Rand() % 5), num_frames_left);
    if (num_frames_return == 0) {
      output->Resize(0, 0);
    } else {
      output->Resize(num_frames_return, feats_.NumCols());
      output->CopyFromMat(feats_.Range(position_, num_frames_return,
                                       0, feats_.NumCols()));
    }
    position_ += num_frames_return;
    if (position_ == feats_.NumRows()) return false;
    else return true;
  }

 private:
  int32 position_;
  Matrix<BaseFloat> feats_;
};

template<class Real> static void AssertEqual(const Matrix<Real> &A,
                                             const Matrix<Real> &B,
                                             float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
    for (MatrixIndexT j = 0;j < A.NumCols();j++) {
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
    }
}

// This function will crash if the two objects do not
// give the same output.
void GetOutput(OnlineFeatInputItf *a,
               Matrix<BaseFloat> *output) {
  int32 dim = a->Dim();
  OnlineCacheInput cache(a);
  while (true) {
    Matrix<BaseFloat> garbage;
    int32 batch_size = 1 + Rand() % 10;
    garbage.Resize(batch_size, dim); // some random requested amount.
    if (!cache.Compute(&garbage)) // returns false when done.
      break;
  }
  cache.GetCachedData(output);
}

// test the MatrixInput and CacheInput classes.
void TestOnlineMatrixInput() {
  int32 dim = 2 + Rand() % 5; // dimension of features.
  int32 num_frames = 100 + Rand() % 100;

  Matrix<BaseFloat> input_feats(num_frames, dim);
  input_feats.SetRandn();

  OnlineMatrixInput matrix_input(input_feats);
  
  Matrix<BaseFloat> output_feats;
  GetOutput(&matrix_input, &output_feats);
  AssertEqual(input_feats, output_feats);
}

void TestOnlineFeatureMatrix() {
  int32 dim = 2 + Rand() % 5; // dimension of features.
  int32 num_frames = 100 + Rand() % 100;

  Matrix<BaseFloat> input_feats(num_frames, dim);
  input_feats.SetRandn();

  OnlineMatrixInput matrix_input(input_feats);
  OnlineFeatureMatrixOptions opts;
  opts.num_tries = 100; // makes it very unlikely we'll get that many timeouts.
  OnlineFeatureMatrix online_feature_matrix(opts, &matrix_input);

  for (int32 frame = 0; frame < num_frames; frame++) {
    KALDI_ASSERT(online_feature_matrix.IsValidFrame(frame));
    KALDI_ASSERT(online_feature_matrix.GetFrame(frame).ApproxEqual(input_feats.Row(frame)));
  }
  KALDI_ASSERT(!online_feature_matrix.IsValidFrame(num_frames));
}



void TestOnlineLdaInput() {
  int32 dim = 2 + Rand() % 5; // dimension of features.
  int32 num_frames = 100 + Rand() % 100;
  int32 left_context = Rand() % 3, right_context = Rand() % 3;
  bool have_offset = (Rand() % 2 == 0);
  int32 lda_input_dim = (dim * (left_context + 1 + right_context)),
      lda_output_dim = 1 + Rand() % 5; // this can even be more than
       // the input dim, the class doesn't care.
  
  Matrix<BaseFloat> transform(lda_output_dim, lda_input_dim +
                              (have_offset ? 1 : 0));
  transform.SetRandn();
  Matrix<BaseFloat> input_feats(num_frames, dim);
  input_feats.SetRandn();

  OnlineMatrixInput matrix_input(input_feats);
  OnlineLdaInput lda_input(&matrix_input, transform, left_context, right_context);

  Matrix<BaseFloat> output_feats1;
  GetOutput(&lda_input, &output_feats1);
  Matrix<BaseFloat> temp_feats;
  SpliceFrames(input_feats, left_context, right_context, &temp_feats);
  Matrix<BaseFloat> output_feats2(temp_feats.NumRows(), transform.NumRows());
  if (!have_offset) {
    output_feats2.AddMatMat(1.0, temp_feats, kNoTrans, transform, kTrans, 0.0);
  } else {
    SubMatrix<BaseFloat> linear_part(transform, 0, transform.NumRows(),
                                     0, transform.NumCols() - 1);
    output_feats2.AddMatMat(1.0, temp_feats, kNoTrans, linear_part, kTrans, 0.0);
    Vector<BaseFloat> offset(transform.NumRows());
    offset.CopyColFromMat(transform, transform.NumCols() - 1);
    output_feats2.AddVecToRows(1.0, offset);
  }
  KALDI_ASSERT(output_feats1.ApproxEqual(output_feats2));
}


void TestOnlineDeltaInput() {
  int32 dim = 2 + Rand() % 5; // dimension of features.
  int32 num_frames = 100 + Rand() % 100;
  DeltaFeaturesOptions opts;
  opts.order = Rand() % 3;
  opts.window = 1 + Rand() % 3;

  int32 output_dim = dim * (1 + opts.order);
  
  Matrix<BaseFloat> input_feats(num_frames, dim);
  input_feats.SetRandn();
  
  OnlineMatrixInput matrix_input(input_feats);
  OnlineDeltaInput delta_input(opts, &matrix_input);

  Matrix<BaseFloat> output_feats1;
  GetOutput(&delta_input, &output_feats1);

  Matrix<BaseFloat> output_feats2(num_frames, output_dim);
  ComputeDeltas(opts, input_feats, &output_feats2);

  KALDI_ASSERT(output_feats1.ApproxEqual(output_feats2));
}


void TestOnlineCmnInput() { // We're also testing OnlineCacheInput here.
  int32 dim = 2 + Rand() % 5; // dimension of features.
  int32 num_frames = 10 + Rand() % 10;
  
  Matrix<BaseFloat> input_feats(num_frames, dim);
  input_feats.SetRandn();

  OnlineMatrixInput matrix_input(input_feats);
  int32 cmn_window = 10 + Rand() % 20;
  int32 min_window = 1 + Rand() % (cmn_window - 1);
  if (Rand() % 3 == 0) min_window = cmn_window;
  
  OnlineCmnInput cmn_input(&matrix_input, cmn_window,
                           min_window);
  OnlineCacheInput cache_input(&cmn_input);
  
  Matrix<BaseFloat> output_feats1;
  GetOutput(&cache_input, &output_feats1);
  
  Matrix<BaseFloat> output_feats2(input_feats);
  for (int32 i = 0; i < output_feats2.NumRows(); i++) {
    SubVector<BaseFloat> this_row(output_feats2, i);
    if (i == 0 && min_window == 0) this_row.SetZero();
    else if (i < min_window) {
      int32 window_nframes = std::min(min_window, input_feats.NumRows());
      Vector<BaseFloat> this_sum(dim);
      SubMatrix<BaseFloat> this_block(input_feats, 0, window_nframes,
                                      0, dim);
      this_sum.AddRowSumMat(1.0, this_block, 0.0);
      this_row.AddVec(-1.0 / window_nframes, this_sum);
    } else {
      int32 window_nframes = std::min(i, cmn_window);
      Vector<BaseFloat> this_sum(dim);
      SubMatrix<BaseFloat> this_block(input_feats, i - window_nframes, window_nframes,
                                      0, dim);
      this_sum.AddRowSumMat(1.0, this_block, 0.0);
      this_row.AddVec(-1.0 / window_nframes, this_sum);
    }
  }
  KALDI_ASSERT(output_feats1.NumRows() == output_feats2.NumRows());
  for (int32 i = 0; i < output_feats2.NumRows(); i++) {
    if (!output_feats1.Row(i).ApproxEqual(output_feats2.Row(i))) {
      KALDI_ERR << "Rows differ " << i << ", " << input_feats.Row(i) << output_feats1.Row(i)
                << output_feats2.Row(i);
    }
  }
  KALDI_ASSERT(output_feats1.ApproxEqual(output_feats2));
  Matrix<BaseFloat> output_feats3;
  cache_input.GetCachedData(&output_feats3);
  KALDI_ASSERT(output_feats1.ApproxEqual(output_feats3));
}



}  // end namespace kaldi

int main() {
  using namespace kaldi;
  for (int i = 0; i < 40; i++) {
    TestOnlineMatrixInput();
    TestOnlineFeatureMatrix();
    TestOnlineLdaInput();
    TestOnlineDeltaInput();
    TestOnlineCmnInput(); // also tests cache input.
    // I have not tested the delta input yet.
  }
  std::cout << "Test OK.\n";
}
