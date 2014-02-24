// online/online-feat-test.cc

// 2014  IMSL, PKU-HKUST (author: Wei Shi)

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

#include "online2/online-feature.h"

namespace kaldi {


template<class Real> static void AssertEqual(const Matrix<Real> &A,
                                             const Matrix<Real> &B,
                                             float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
	for (MatrixIndexT j = 0;j < A.NumCols();j++) {
	  KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol*std::max(1.0, (double) (std::abs(A(i, j))+std::abs(B(i, j)))));
    }
}

void GetOutput(OnlineFeatureInterface *a,
               Matrix<BaseFloat> *output) {
  int32 dim = a->Dim();
  int32 frame_num = 0;
  OnlineCacheFeature cache(a);

  std::vector<Vector<BaseFloat>* > cached_frames;
  while ( true ) {
    Vector<BaseFloat> garbage(dim);
    cache.GetFrame( frame_num , &garbage );
    cached_frames.push_back(new Vector<BaseFloat>(garbage));
    if ( cache.IsLastFrame(frame_num) )
      break;
    frame_num++;
  }

  KALDI_ASSERT(cached_frames.size() == a->NumFramesReady());

  output->Resize(cached_frames.size(), dim);
  for(int32 i = 0; i < cached_frames.size(); i++){
    output->CopyRowFromVec( *(cached_frames[i]), i );
    delete cached_frames[i];
  }
  cached_frames.clear();
  cache.ClearCache();
}

// test the OnlineMatrixFeature and OnlineCacheFeature classes.
void TestOnlineMatrixCacheFeature() {
  int32 dim = 2 + rand() % 5; // dimension of features.
  int32 num_frames = 100 + rand() % 100;

  Matrix<BaseFloat> input_feats(num_frames, dim);
  input_feats.SetRandn();

  OnlineMatrixFeature matrix_feats(input_feats);

  Matrix<BaseFloat> output_feats;
  GetOutput(&matrix_feats, &output_feats);
  AssertEqual(input_feats, output_feats);
}

void TestOnlineDeltaFeature() {
  int32 dim = 2 + rand() % 5; // dimension of features.
  int32 num_frames = 100 + rand() % 100;
  DeltaFeaturesOptions opts;
  opts.order = rand() % 3;
  opts.window = 1 + rand() % 3;

  int32 output_dim = dim * (1 + opts.order);

  Matrix<BaseFloat> input_feats(num_frames, dim);
  input_feats.SetRandn();

  OnlineMatrixFeature matrix_feats(input_feats);
  OnlineDeltaFeature delta_feats(opts, &matrix_feats);

  Matrix<BaseFloat> output_feats1;
  GetOutput(&delta_feats, &output_feats1);

  Matrix<BaseFloat> output_feats2(num_frames, output_dim);
  ComputeDeltas(opts, input_feats, &output_feats2);

  KALDI_ASSERT(output_feats1.ApproxEqual(output_feats2));
}

void TestOnlineSpliceFrames() {
  int32 dim = 2 + rand() % 5; // dimension of features.
  int32 num_frames = 100 + rand() % 100;
  OnlineSpliceOptions opts;
  opts.left_context  = 1 + rand() % 4;
  opts.right_context = 1 + rand() % 4;

  int32 output_dim = dim * (1 + opts.left_context + opts.right_context);

  Matrix<BaseFloat> input_feats(num_frames, dim);
  input_feats.SetRandn();

  OnlineMatrixFeature matrix_feats(input_feats);
  OnlineSpliceFrames splice_frame(opts, &matrix_feats);

  Matrix<BaseFloat> output_feats1;
  GetOutput(&splice_frame, &output_feats1);

  Matrix<BaseFloat> output_feats2(num_frames, output_dim);
  SpliceFrames(input_feats, opts.left_context, opts.right_context, &output_feats2);

  KALDI_ASSERT(output_feats1.ApproxEqual(output_feats2));
}


}  // end namespace kaldi

int main() {
  using namespace kaldi;
  for (int i = 0; i < 40; i++) {
    TestOnlineMatrixCacheFeature();
    TestOnlineDeltaFeature();
    TestOnlineSpliceFrames();
  }
  std::cout << "Test OK.\n";
}
