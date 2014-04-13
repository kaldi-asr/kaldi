// feat/pitch-functions-test.cc

// Copyright    2013  Pegah Ghahremani
//              2014  IMSL, PKU-HKUST (author: Wei Shi)
//              2014  Yanqing Sun, Junjie Wang,
//                    Daniel Povey, Korbinian Riedhammer

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


#include <iostream>
#include "feat/pitch-functions.cc"
#include "feat/feature-plp.h"
#include "base/kaldi-math.h"
#include "feat/wave-reader.h"
#include "util/timer.h"
#include "sys/stat.h"
#include "sys/types.h"
using namespace kaldi;

std::string ConvertIntToString(const int &number) {
  std::stringstream ss;  // create a stringstream
  ss << number;  // add number to the stream
  return ss.str();  // return a string with the contents of the stream
}
bool DirExist(const std::string &dirname) {
  struct stat st;
  if (stat(dirname.c_str(), &st) != 0) {
    KALDI_LOG << " directory " << dirname << " does not exist!";
    return false;
  }
  return true;
}

static void UnitTestSimple() {
  KALDI_LOG << "=== UnitTestSimple() ===\n";
  Vector<BaseFloat> v(1000);
  Matrix<BaseFloat> m;
  // init with noise
  for (int32 i = 0; i < v.Dim(); i++) {
    v(i) = (abs(i * 433024253) % 65535) - (65535 / 2);
  }
  KALDI_LOG << "<<<=== Just make sure it runs... Nothing is compared\n";
  // the parametrization object
  PitchExtractionOptions op;
  // trying to have same opts as baseline.
  // compute pitch.
  ComputeKaldiPitch(op, v, &m);
  KALDI_LOG << "Test passed :)\n";
}


// Make sure that doing a calculation on the whole waveform gives
// the same results as doing on the waveform broken into pieces.
static void UnitTestPieces() {
  KALDI_LOG << "=== UnitTestPieces() ===\n";
  for (int32 n = 0; n < 10; n++) {
    // the parametrization object
    PitchExtractionOptions op;
    PostProcessPitchOptions op2;
    op2.delta_pitch_noise_stddev = 0.0;  // to avoid mismatch of delta_log_pitch
                                         // brought by rand noise.
    op.nccf_ballast_online = true;  // this is necessary for the computation
    // to be identical regardless how many pieces we break the signal into.

    int32 size = 10000 + rand() % 50000;

    Vector<BaseFloat> v(size);
    // init with noise plus a sine-wave whose frequency is changing randomly.

    double cur_freq = 200.0, normalized_time = 0.0;

    for (int32 i = 0; i < size; i++) {
      v(i) = RandGauss() + cos(normalized_time * M_2PI);
      cur_freq += RandGauss();  // let the frequency wander a little.
      if (cur_freq < 100.0) cur_freq = 100.0;
      if (cur_freq > 300.0) cur_freq = 300.0;
      normalized_time += cur_freq / op.samp_freq;
    }

    Matrix<BaseFloat> m1, m1p;

    // trying to have same opts as baseline.
    // compute pitch.
    ComputeKaldiPitch(op, v, &m1);
    PostProcessPitch(op2, m1, &m1p);

    Matrix<BaseFloat> m2, m2p;

    { // compute it online with multiple pieces.
      OnlinePitchFeature pitch_extractor(op);
      OnlinePostProcessPitch postprocess_pitch(op2, &pitch_extractor);
      int32 start_samp = 0;
      while (start_samp < v.Dim()) {
        int32 num_samp = rand() % (v.Dim() + 1 - start_samp);
        SubVector<BaseFloat> v_part(v, start_samp, num_samp);
        pitch_extractor.AcceptWaveform(op.samp_freq, v_part);
        start_samp += num_samp;
      }
      pitch_extractor.InputFinished();
      int32 num_frames = pitch_extractor.NumFramesReady();
      m2.Resize(num_frames, 2);
      m2p.Resize(num_frames, postprocess_pitch.Dim());
      for (int32 frame = 0; frame < num_frames; frame++) {
        SubVector<BaseFloat> row(m2, frame);
        pitch_extractor.GetFrame(frame, &row);
        SubVector<BaseFloat> rowp(m2p, frame);
        postprocess_pitch.GetFrame(frame, &rowp);
      }
    }
    AssertEqual(m1, m2);
    AssertEqual(m1p, m2p);
    KALDI_LOG << "Test passed :)\n";
  }
}

// if we didn't #include pitch-functions.cc, we'd do this:
// extern bool pitch_use_naive_search; // was declared in pitch-functions.cc

// Make sure that doing a calculation on the whole waveform gives
// the same results as doing on the waveform broken into pieces.
static void UnitTestSearch() {
  KALDI_LOG << "=== UnitTestSearch() ===\n";
  for (int32 n = 0; n < 3; n++) {
    // the parametrization object
    PitchExtractionOptions op;
    op.nccf_ballast_online = true;  // this is necessary for the computation
    // to be identical regardless how many pieces we break the signal into.

    int32 size = 10000 + rand() % 10000;

    Vector<BaseFloat> v(size);
    // init with noise plus a sine-wave whose frequency is changing randomly.

    double cur_freq = 200.0, normalized_time = 0.0;

    for (int32 i = 0; i < size; i++) {
      v(i) = RandGauss() + cos(normalized_time * M_2PI);
      cur_freq += RandGauss();  // let the frequency wander a little.
      if (cur_freq < 100.0) cur_freq = 100.0;
      if (cur_freq > 300.0) cur_freq = 300.0;
      normalized_time += cur_freq / op.samp_freq;
    }

    Matrix<BaseFloat> m1;
    ComputeKaldiPitch(op, v, &m1);

    pitch_use_naive_search = true;

    Matrix<BaseFloat> m2;
    ComputeKaldiPitch(op, v, &m2);

    pitch_use_naive_search = false;

    AssertEqual(m1, m2, 1.0e-08);  // should be identical.
  }
  KALDI_LOG << "Test passed :)\n";
}


// Compare pitch using Kaldi pitch tracker on KEELE corpora
static void UnitTestKeele() {
  KALDI_LOG << "=== UnitTestKeele() ===\n";
  for (int32 i = 1; i < 11; i++) {
    std::string wavefile;
    std::string num;
    if (i < 6) {
      num = "f" + ConvertIntToString(i) + "nw0000";
      wavefile = "keele/16kHz/"+num+".wav";
    } else {
      num = "m" + ConvertIntToString(i-5) + "nw0000";
      wavefile = "keele/16kHz/"+num+".wav";
    }
    KALDI_LOG << "--- " << wavefile << " ---\n";
    std::ifstream is(wavefile.c_str());
    WaveData wave;
    wave.Read(is);
    KALDI_ASSERT(wave.Data().NumRows() == 1);
    SubVector<BaseFloat> waveform(wave.Data(), 0);
    // use pitch code with default configuration..
    PitchExtractionOptions op;
    op.nccf_ballast = 0.1;
    // compute pitch.
    Matrix<BaseFloat> m;
    ComputeKaldiPitch(op, waveform, &m);
    std::string outfile = "keele/"+num+"-kaldi.txt";
    std::ofstream os(outfile.c_str());
    m.Write(os, false);
  }
}
/* change freq_weight to investigate the results */
static void UnitTestPenaltyFactor() {
  KALDI_LOG << "=== UnitTestPenaltyFactor() ===\n";
  for (int32 k = 1; k < 5; k++) {
    for (int32 i = 1; i < 4; i++) {
      std::string wavefile;
      std::string num;
      if (i < 6) {
        num = "f"+ConvertIntToString(i)+"nw0000";
        wavefile = "keele/16kHz/"+num+".wav";
      } else {
        num = "m"+ConvertIntToString(i-5)+"nw0000";
        wavefile = "keele/16kHz/"+num+".wav";
      }
      KALDI_LOG << "--- " << wavefile << " ---\n";
      std::ifstream is(wavefile.c_str());
      WaveData wave;
      wave.Read(is);
      KALDI_ASSERT(wave.Data().NumRows() == 1);
      SubVector<BaseFloat> waveform(wave.Data(), 0);
      // use pitch code with default configuration..
      PitchExtractionOptions op;
      op.penalty_factor = k * 0.05;
      op.nccf_ballast = 0.1;
      // compute pitch.
      Matrix<BaseFloat> m;
      ComputeKaldiPitch(op, waveform, &m);
      std::string penaltyfactor = ConvertIntToString(k);
      std::string outfile = "keele/"+num+"-kaldi-penalty-"+penaltyfactor+".txt";
      std::ofstream os(outfile.c_str());
      m.Write(os, false);
    }
  }
}
static void UnitTestKeeleNccfBallast() {
  KALDI_LOG << "=== UnitTestKeeleNccfBallast() ===\n";
  for (int32 k = 1; k < 10; k++) {
    for (int32 i = 1; i < 2; i++) {
      std::string wavefile;
      std::string num;
      if (i < 6) {
        num = "f"+ConvertIntToString(i)+"nw0000";
        wavefile = "keele/16kHz/"+num+".wav";
      } else {
        num = "m"+ConvertIntToString(i-5)+"nw0000";
        wavefile = "keele/16kHz/"+num+".wav";
      }
      KALDI_LOG << "--- " << wavefile << " ---\n";
      std::ifstream is(wavefile.c_str());
      WaveData wave;
      wave.Read(is);
      KALDI_ASSERT(wave.Data().NumRows() == 1);
      SubVector<BaseFloat> waveform(wave.Data(), 0);
      // use pitch code with default configuration..
      PitchExtractionOptions op;
      op.nccf_ballast = 0.05 * k;
      KALDI_LOG << " nccf_ballast " << op.nccf_ballast << std::endl;
      // compute pitch.
      Matrix<BaseFloat> m;
      ComputeKaldiPitch(op, waveform, &m);
      std::string nccfballast = ConvertIntToString(op.nccf_ballast);
      std::string outfile = "keele/"+num
        +"-kaldi-nccf-ballast-"+nccfballast+".txt";
      std::ofstream os(outfile.c_str());
      m.Write(os, false);
    }
  }
}
static void UnitTestWeightedMovingWindowNormalize() {
  KALDI_LOG << "=== UnitTestWeightedMovingWindowNormalize1() ===\n";
  // compare the results of WeightedMovingWindowNormalize and Sliding CMN
  // with uniform weights.
  for (int32 i = 0; i < 1000; i++) {
    int32 num_frames = 1 + (rand()%10 * 10);
    int32 normalization_win_size = 5 + rand() % 50;
    Matrix<BaseFloat> feat(num_frames, 2),
                      output_feat(num_frames, 2);
    feat.SetRandn();
    for (int32 j = 0; j < num_frames; j++)
      feat(j, 0) = 1;

    Vector<BaseFloat> pov(num_frames),
                      log_pitch(num_frames),
                      mean_subtracted_log_pitch(num_frames);
    pov.CopyColFromMat(feat, 0);
    log_pitch.CopyColFromMat(feat, 1);
    WeightedMovingWindowNormalize(normalization_win_size, pov, log_pitch ,
                                  &mean_subtracted_log_pitch);
    output_feat.CopyColFromVec(mean_subtracted_log_pitch, 1);

    // SlidingWindow
    SlidingWindowCmnOptions opts;
    opts.cmn_window = normalization_win_size;
    opts.center = true;
    opts.min_window = 1 + rand() % 100;
    if (opts.min_window > opts.cmn_window)
      opts.min_window = opts.cmn_window;
    Matrix<BaseFloat> output_feat2(num_frames, 2);
    SlidingWindowCmn(opts, feat, &output_feat2);
    for (int32 j = 0; j < num_frames; j++)
      output_feat(j, 0) = 0.0;
    if (!output_feat.ApproxEqual(output_feat2, 0.001)) {
      Matrix<BaseFloat> output_all(num_frames, 2);
      Vector<BaseFloat> pitch(num_frames),
                        pitch2(num_frames);
      pitch.CopyColFromMat(output_feat, 1);
      pitch2.CopyColFromMat(output_feat2, 1);
      output_all.CopyColFromVec(pitch, 0);
      output_all.CopyColFromVec(pitch2, 1);
      KALDI_ERR << "Feafures differ:\n" << output_all;
    }
  }
  // Weighted Moving Window Normalization with non-uniform weights
  /*
  int32 num_frames = 1 + (rand()%10 * 20);
  int32 normalization_win_size = 5 + rand() % 50;
  Matrix<BaseFloat> feat(num_frames, 2),
    output_feat(num_frames, 2);
  for (int32 j = 0; j < num_frames; j++) {
    int32 r = rand() % 2;
    feat(j, 0) = RandUniform() / (1 + 1000.0 * r);
    feat(j, 1) = feat(j, 1) * feat(j, 0);
  }
  ProcessPovFeatures(&feat, 2, true);
  WeightedMovingWindowNormalize(normalization_win_size, feat, &output_feat);
  */
}

static void UnitTestPitchExtractionSpeed() {
  KALDI_LOG << "=== UnitTestPitchExtractionSpeed() ===\n";
  // use pitch code with default configuration..
  PitchExtractionOptions op;
  op.nccf_ballast = 0.1;
  op.lowpass_cutoff = 1000;
  for (int32 i = 1; i < 2; i++) {
    std::string wavefile;
    std::string num;
    if (i < 6) {
      num = "f"+ConvertIntToString(i)+"nw0000";
      wavefile = "keele/16kHz/"+num+".wav";
    } else {
      num = "m"+ConvertIntToString(i-5)+"nw0000";
      wavefile = "keele/16kHz/"+num+".wav";
    }
    KALDI_LOG << "--- " << wavefile << " ---\n";
    std::ifstream is(wavefile.c_str());
    WaveData wave;
    wave.Read(is);
    KALDI_ASSERT(wave.Data().NumRows() == 1);
    SubVector<BaseFloat> waveform(wave.Data(), 0);
    // compute pitch.
    int test_num = 10;
    Matrix<BaseFloat> m;
    Timer timer;
    for (int32 t = 0; t < test_num; t++)
      ComputeKaldiPitch(op, waveform, &m);
    double tot_time = timer.Elapsed(),
        speech_time = test_num * waveform.Dim() / wave.SampFreq();
    KALDI_LOG << " Pitch extraction time per second of speech is "
              << (tot_time / speech_time) << " seconds " << std::endl;
  }
}
static void UnitTestPitchExtractorCompareKeele() {
  KALDI_LOG << "=== UnitTestPitchExtractorCompareKeele() ===\n";
  // use pitch code with default configuration..
  PitchExtractionOptions op;
  op.nccf_ballast = 0.1;
  for (int32 i = 1; i < 11; i++) {
    std::string wavefile;
    std::string num;
    if (i < 6) {
      num = "f"+ConvertIntToString(i)+"nw0000";
      wavefile = "keele/16kHz/"+num+".wav";
    } else {
      num = "m"+ConvertIntToString(i-5)+"nw0000";
      wavefile = "keele/16kHz/"+num+".wav";
    }
    KALDI_LOG << "--- " << wavefile << " ---\n";
    std::ifstream is(wavefile.c_str());
    WaveData wave;
    wave.Read(is);
    KALDI_ASSERT(wave.Data().NumRows() == 1);
    SubVector<BaseFloat>  waveform(wave.Data(), 0);
    // compute pitch.
    Matrix<BaseFloat> m;
    ComputeKaldiPitch(op, waveform, &m);
    std::string outfile = "keele/"+num+"-speedup-kaldi1.txt";
    std::ofstream os(outfile.c_str());
    m.Write(os, false);
  }
}
void UnitTestDiffSampleRate() {
  // you need to use sox to change sampling rate
  // e.g. sox -r 10k input.wav output.wav
  // put them in keele/(samp_rate in kHz)+"kHz" e.g. keele/10kHz
  int sample_rate = 16000;
  PitchExtractionOptions op;
  op.samp_freq = static_cast<double>(sample_rate);
  op.lowpass_cutoff = 1000;
  op.max_f0 = 400;
  std::string samp_rate = ConvertIntToString(sample_rate/1000);
  for (int32 i = 1; i < 11; i++) {
    std::string wavefile;
    std::string num;
    if (i < 6) {
      num = "f"+ConvertIntToString(i)+"nw0000";
      wavefile = "keele/"+samp_rate+"kHz/"+num+".wav";
    } else {
      num = "m"+ConvertIntToString(i-5)+"nw0000";
      wavefile = "keele/"+samp_rate+"kHz/"+num+".wav";
    }
    KALDI_LOG << "--- " << wavefile << " ---\n";
    std::ifstream is(wavefile.c_str());
    WaveData wave;
    wave.Read(is);
    KALDI_ASSERT(wave.Data().NumRows() == 1);
    SubVector<BaseFloat> waveform(wave.Data(), 0);
    Matrix<BaseFloat> m;
    ComputeKaldiPitch(op, waveform, &m);
    std::string outfile = "keele/"+num+"-kaldi-samp-freq-"+samp_rate+"kHz.txt";
    std::ofstream os(outfile.c_str());
    m.Write(os, false);
  }
}
void UnitTestPostProcess() {
  for (int32 i = 1; i < 11; i++) {
    std::string wavefile;
    std::string num;
    if (i < 6) {
      num = "f"+ConvertIntToString(i)+"nw0000";
      wavefile = "keele/16kHz/"+num+".wav";
    } else {
      num = "m"+ConvertIntToString(i-5)+"nw0000";
      wavefile = "keele/16kHz/"+num+".wav";
    }
    KALDI_LOG << "--- " << wavefile << " ---\n";
    std::ifstream is(wavefile.c_str());
    WaveData wave;
    wave.Read(is);
    KALDI_ASSERT(wave.Data().NumRows() == 1);
    SubVector<BaseFloat> waveform(wave.Data(), 0);
    PitchExtractionOptions op;
    op.lowpass_cutoff = 1000;
    op.nccf_ballast = 0.1;
    op.max_f0 = 400;
    Matrix<BaseFloat> m, m2;
    ComputeKaldiPitch(op, waveform, &m);
    PostProcessPitchOptions postprop_op;
    // postprop_op.pov_nonlinearity = 2;
    PostProcessPitch(postprop_op, m, &m2);
    std::string outfile = "keele/"+num+"-processed-kaldi.txt";
    std::ofstream os(outfile.c_str());
    m2.Write(os, false);
  }
}
void UnitTestDeltaPitch() {
  KALDI_LOG << "=== UnitTestDeltaPitch() ===\n";
  for (int32 i = 0; i < 1; i++) {
    int32 num_frames = 1 + (rand()%10 * 1000);
    Vector<BaseFloat> feat(num_frames),
      output_feat(num_frames), output_feat2(num_frames);
    for (int32 j = 0; j < num_frames; j++)
      feat(j) = 0.2 * j;
    for (int32 j = 2; j < num_frames - 2; j++)
      output_feat2(j) = 1.0 / 10.0  *
        (-2.0 * feat(j - 2) - feat(j - 1) + feat(j + 1) + 2.0 * feat(j + 2));
    PostProcessPitchOptions op;
    op.delta_pitch_noise_stddev = 0;
    ExtractDeltaPitch(op, feat, &output_feat);
    if (!output_feat.ApproxEqual(output_feat2, 0.05)) {
      KALDI_ERR << "output feat " << output_feat << " vs. "
        << " ouput feat2 " << output_feat2;
    }
  }
}

static void UnitTestFeatNoKeele() {
  UnitTestSimple();
  UnitTestPieces();
  UnitTestSearch();
  UnitTestDeltaPitch();
  UnitTestWeightedMovingWindowNormalize();
}
static void UnitTestFeatWithKeele() {
  UnitTestKeele();
  UnitTestPenaltyFactor();
  UnitTestKeeleNccfBallast();
  UnitTestPitchExtractionSpeed();
  UnitTestPitchExtractorCompareKeele();
  UnitTestDiffSampleRate();
  UnitTestPostProcess();
}

int main() {
  SetVerboseLevel(3);
  try {
    UnitTestFeatNoKeele();
    if (DirExist("keele/16kHz")) {
      UnitTestFeatWithKeele();
    } else {
      KALDI_LOG << "Not running tests that require the Keele database, "
        << "please ask g.meyer@somewhere.edu for the database if you need it.\n"
        << " you need to put keele wave file in keele/16kHz directory";
    }
    KALDI_LOG << "Tests succeeded.\n";
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return 1;
  }
}


