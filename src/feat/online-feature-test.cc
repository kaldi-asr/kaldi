// feat/online-feature-test.cc

// Copyright 2014  IMSL, PKU-HKUST (author: Wei Shi)
// Copyright 2014  Yanqing Sun, Junjie Wang,
//                 Daniel Povey, Korbinian Riedhammer

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

#include "feat/online-feature.h"
#include "feat/wave-reader.h"
#include "transform/transform-common.h"

namespace kaldi {


template<class Real> static void AssertEqual(const Matrix<Real> &A,
                                             const Matrix<Real> &B,
                                             float tol = 0.001) {
  KALDI_ASSERT(A.NumRows() == B.NumRows()&&A.NumCols() == B.NumCols());
  for (MatrixIndexT i = 0;i < A.NumRows();i++)
    for (MatrixIndexT j = 0;j < A.NumCols();j++) {
      KALDI_ASSERT(std::abs(A(i, j)-B(i, j)) < tol * std::max(1.0,
        static_cast<double>(std::abs(A(i, j))+std::abs(B(i, j)))));
    }
}

void GetOutput(OnlineFeatureInterface *a,
               Matrix<BaseFloat> *output) {
  int32 dim = a->Dim();
  int32 frame_num = 0;
  OnlineCacheFeature cache(a);

  std::vector<Vector<BaseFloat>* > cached_frames;
  while (true) {
    Vector<BaseFloat> garbage(dim);
    cache.GetFrame(frame_num, &garbage);
    cached_frames.push_back(new Vector<BaseFloat>(garbage));
    if (cache.IsLastFrame(frame_num))
      break;
    frame_num++;
  }

  KALDI_ASSERT(cached_frames.size() == a->NumFramesReady());

  output->Resize(cached_frames.size(), dim);
  for (int32 i = 0; i < cached_frames.size(); i++) {
    output->CopyRowFromVec(*(cached_frames[i]), i);
    delete cached_frames[i];
  }
  cached_frames.clear();
  cache.ClearCache();
}

// Only generate random length for each piece
bool RandomSplit(int32 wav_dim,
                 std::vector<int32> *piece_dim,
                 int32 num_pieces,
                 int32 trials = 5) {
  piece_dim->clear();
  piece_dim->resize(num_pieces, 0);

  int32 dim_mean = wav_dim / (num_pieces * 2);
  int32 cnt = 0;
  while (true) {
    int32 dim_total = 0;
    for (int32 i = 0; i < num_pieces - 1; i++) {
      (*piece_dim)[i] = dim_mean + rand() % dim_mean;
      dim_total += (*piece_dim)[i];
    }
    (*piece_dim)[num_pieces - 1] = wav_dim - dim_total;

    if (dim_total > 0 && dim_total < wav_dim)
      break;
    if (++cnt > trials)
      return false;
  }
  return true;
}

// test the OnlineMatrixFeature and OnlineCacheFeature classes.
void TestOnlineMatrixCacheFeature() {
  int32 dim = 2 + rand() % 5;  // dimension of features.
  int32 num_frames = 100 + rand() % 100;

  Matrix<BaseFloat> input_feats(num_frames, dim);
  input_feats.SetRandn();

  OnlineMatrixFeature matrix_feats(input_feats);

  Matrix<BaseFloat> output_feats;
  GetOutput(&matrix_feats, &output_feats);
  AssertEqual(input_feats, output_feats);
}

void TestOnlineDeltaFeature() {
  int32 dim = 2 + rand() % 5;  // dimension of features.
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
  int32 dim = 2 + rand() % 5;  // dimension of features.
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
  SpliceFrames(input_feats, opts.left_context, opts.right_context,
    &output_feats2);

  KALDI_ASSERT(output_feats1.ApproxEqual(output_feats2));
}

void TestOnlineMfcc() {
  std::ifstream is("../feat/test_data/test.wav", std::ios_base::binary);
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // the parametrization object
  MfccOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.frame_opts.samp_freq = wave.SampFreq();
  op.mel_opts.low_freq = 0.0;
  op.htk_compat = false;
  op.use_energy = false;  // C0 not energy.
  if (RandInt(0, 1) == 0)
    op.frame_opts.snip_edges = false;
  Mfcc mfcc(op);

  // compute mfcc offline
  Matrix<BaseFloat> mfcc_feats;
  mfcc.Compute(waveform, 1.0, &mfcc_feats, NULL);  // vtln not supported

  // compare
  // The test waveform is about 1.44s long, so
  // we try to break it into from 5 pieces to 9(not essential to do so)
  for (int32 num_piece = 5; num_piece < 10; num_piece++) {
    OnlineMfcc online_mfcc(op);
    std::vector<int32> piece_length(num_piece, 0);

    bool ret = RandomSplit(waveform.Dim(), &piece_length, num_piece);
    KALDI_ASSERT(ret);

    int32 offset_start = 0;
    for (int32 i = 0; i < num_piece; i++) {
      Vector<BaseFloat> wave_piece(
        waveform.Range(offset_start, piece_length[i]));
      online_mfcc.AcceptWaveform(wave.SampFreq(), wave_piece);
      offset_start += piece_length[i];
    }
    online_mfcc.InputFinished();

    Matrix<BaseFloat> online_mfcc_feats;
    GetOutput(&online_mfcc, &online_mfcc_feats);

    AssertEqual(mfcc_feats, online_mfcc_feats);
  }
}

void TestOnlinePlp() {
  std::ifstream is("../feat/test_data/test.wav", std::ios_base::binary);
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // the parametrization object
  PlpOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.frame_opts.samp_freq = wave.SampFreq();
  op.mel_opts.low_freq = 0.0;
  op.htk_compat = false;
  op.use_energy = false;  // C0 not energy.
  Plp plp(op);

  // compute plp offline
  Matrix<BaseFloat> plp_feats;
  plp.Compute(waveform, 1.0, &plp_feats, NULL);  // vtln not supported

  // compare
  // The test waveform is about 1.44s long, so
  // we try to break it into from 5 pieces to 9(not essential to do so)
  for (int32 num_piece = 5; num_piece < 10; num_piece++) {
    OnlinePlp online_plp(op);
    std::vector<int32> piece_length(num_piece);
    bool ret = RandomSplit(waveform.Dim(), &piece_length, num_piece);
    KALDI_ASSERT(ret);

    int32 offset_start = 0;
    for (int32 i = 0; i < num_piece; i++) {
      Vector<BaseFloat> wave_piece(
        waveform.Range(offset_start, piece_length[i]));
      online_plp.AcceptWaveform(wave.SampFreq(), wave_piece);
      offset_start += piece_length[i];
    }
    online_plp.InputFinished();

    Matrix<BaseFloat> online_plp_feats;
    GetOutput(&online_plp, &online_plp_feats);

    AssertEqual(plp_feats, online_plp_feats);
  }
}

void TestOnlineTransform() {
  std::ifstream is("../feat/test_data/test.wav", std::ios_base::binary);
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // build online feature interface, take OnlineMfcc as an example
  MfccOptions op;
  op.frame_opts.dither = 0.0;
  op.frame_opts.preemph_coeff = 0.0;
  op.frame_opts.window_type = "hamming";
  op.frame_opts.remove_dc_offset = false;
  op.frame_opts.round_to_power_of_two = true;
  op.frame_opts.samp_freq = wave.SampFreq();
  op.mel_opts.low_freq = 0.0;
  op.htk_compat = false;
  op.use_energy = false;  // C0 not energy.
  OnlineMfcc online_mfcc(op);

  online_mfcc.AcceptWaveform(wave.SampFreq(), waveform);
  online_mfcc.InputFinished();
  Matrix<BaseFloat> mfcc_feats;
  GetOutput(&online_mfcc, &mfcc_feats);

  // Affine transform
  Matrix<BaseFloat> trans(online_mfcc.Dim(), online_mfcc.Dim() + 1);
  trans.SetRandn();
  OnlineTransform online_trans(trans, &online_mfcc);

  Matrix<BaseFloat> trans_feats;
  GetOutput(&online_trans, &trans_feats);

  Matrix<BaseFloat> output_feats(mfcc_feats.NumRows(), mfcc_feats.NumCols());
  for (int32 i = 0; i < mfcc_feats.NumRows(); i++) {
    Vector<BaseFloat> vec_tmp(mfcc_feats.Row(i));
    ApplyAffineTransform(trans, &vec_tmp);
    output_feats.CopyRowFromVec(vec_tmp, i);
  }

  AssertEqual(trans_feats, output_feats);
}

void TestOnlineAppendFeature() {
  std::ifstream is("../feat/test_data/test.wav", std::ios_base::binary);
  WaveData wave;
  wave.Read(is);
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  // the parametrization object for 1st stream mfcc feature
  MfccOptions mfcc_op;
  mfcc_op.frame_opts.dither = 0.0;
  mfcc_op.frame_opts.preemph_coeff = 0.0;
  mfcc_op.frame_opts.window_type = "hamming";
  mfcc_op.frame_opts.remove_dc_offset = false;
  mfcc_op.frame_opts.round_to_power_of_two = true;
  mfcc_op.frame_opts.samp_freq = wave.SampFreq();
  mfcc_op.mel_opts.low_freq = 0.0;
  mfcc_op.htk_compat = false;
  mfcc_op.use_energy = false;  // C0 not energy.
  Mfcc mfcc(mfcc_op);

  // compute mfcc offline
  Matrix<BaseFloat> mfcc_feats;
  mfcc.Compute(waveform, 1.0, &mfcc_feats, NULL);  // vtln not supported

  // the parametrization object for 2nd stream plp feature
  PlpOptions plp_op;
  plp_op.frame_opts.dither = 0.0;
  plp_op.frame_opts.preemph_coeff = 0.0;
  plp_op.frame_opts.window_type = "hamming";
  plp_op.frame_opts.remove_dc_offset = false;
  plp_op.frame_opts.round_to_power_of_two = true;
  plp_op.frame_opts.samp_freq = wave.SampFreq();
  plp_op.mel_opts.low_freq = 0.0;
  plp_op.htk_compat = false;
  plp_op.use_energy = false;  // C0 not energy.
  Plp plp(plp_op);

  // compute plp offline
  Matrix<BaseFloat> plp_feats;
  plp.Compute(waveform, 1.0, &plp_feats, NULL);  // vtln not supported

  // compare
  // The test waveform is about 1.44s long, so
  // we try to break it into from 5 pieces to 9(not essential to do so)
  for (int32 num_piece = 5; num_piece < 10; num_piece++) {
    OnlineMfcc online_mfcc(mfcc_op);
    OnlinePlp online_plp(plp_op);
    OnlineAppendFeature online_mfcc_plp(&online_mfcc, &online_plp);

    std::vector<int32> piece_length(num_piece);
    bool ret = RandomSplit(waveform.Dim(), &piece_length, num_piece);
    KALDI_ASSERT(ret);
    int32 offset_start = 0;
    for (int32 i = 0; i < num_piece; i++) {
      Vector<BaseFloat> wave_piece(
        waveform.Range(offset_start, piece_length[i]));
      online_mfcc.AcceptWaveform(wave.SampFreq(), wave_piece);
      online_plp.AcceptWaveform(wave.SampFreq(), wave_piece);
      offset_start += piece_length[i];
    }
    online_mfcc.InputFinished();
    online_plp.InputFinished();

    Matrix<BaseFloat> online_mfcc_plp_feats;
    GetOutput(&online_mfcc_plp, &online_mfcc_plp_feats);

    // compare mfcc_feats & plp_features with online_mfcc_plp_feats
    KALDI_ASSERT(mfcc_feats.NumRows() == online_mfcc_plp_feats.NumRows()
      && plp_feats.NumRows() == online_mfcc_plp_feats.NumRows()
      && mfcc_feats.NumCols() + plp_feats.NumCols()
         == online_mfcc_plp_feats.NumCols());
    for (MatrixIndexT i = 0; i < online_mfcc_plp_feats.NumRows(); i++) {
      for (MatrixIndexT j = 0; j < mfcc_feats.NumCols(); j++) {
        KALDI_ASSERT(std::abs(mfcc_feats(i, j) - online_mfcc_plp_feats(i, j))
          < 0.0001*std::max(1.0, static_cast<double>(std::abs(mfcc_feats(i, j))
                                    + std::abs(online_mfcc_plp_feats(i, j)))));
      }
      for (MatrixIndexT k = 0; k < plp_feats.NumCols(); k++) {
        KALDI_ASSERT(
          std::abs(plp_feats(i, k) -
            online_mfcc_plp_feats(i, mfcc_feats.NumCols() + k))
          < 0.0001*std::max(1.0, static_cast<double>(std::abs(plp_feats(i, k))
            +std::abs(online_mfcc_plp_feats(i, mfcc_feats.NumCols() + k)))));
      }
    }
  }
}

}  // end namespace kaldi

int main() {
  using namespace kaldi;
  for (int i = 0; i < 10; i++) {
    TestOnlineMatrixCacheFeature();
    TestOnlineDeltaFeature();
    TestOnlineSpliceFrames();
    TestOnlineMfcc();
    TestOnlinePlp();
    TestOnlineTransform();
    TestOnlineAppendFeature();
  }
  std::cout << "Test OK.\n";
}
