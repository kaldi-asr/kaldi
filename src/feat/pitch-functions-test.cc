// feat/pitch-functions-test.cc

// Copyright    2013  Pegah Ghahremani
//              2014  IMSL, PKU-HKUST (author: Wei Shi)
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
#include "matrix/kaldi-matrix-inl.h"
#include "feat/wave-reader.h"
#include "sys/timeb.h"
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
  KALDI_LOG << "=== UnitTestSimple() ===";
  Vector<BaseFloat> v(1000);
  Vector<BaseFloat> out;
  Matrix<BaseFloat> m;
  // init with noise
  for (int32 i = 0; i < v.Dim(); i++) {
    v(i) = (abs(i * 433024253) % 65535) - (65535 / 2);
  }
  KALDI_LOG << "<<<=== Just make sure it runs... Nothing is compared";
  // the parametrization object
  PitchExtractionOptions op;
  // trying to have same opts as baseline.
  // compute pitch.
  Compute(op, v, &m);
  KALDI_LOG << "Test passed :)";
}
// Compare pitch using Kaldi pitch tracker on KEELE corpora
static void UnitTestKeele() {
  KALDI_LOG << "=== UnitTestKeele() ===";
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
    KALDI_LOG << "--- " << wavefile << " ---";
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
    Compute(op, waveform, &m);
    std::string outfile = "keele/"+num+"-kaldi.txt";
    std::ofstream os(outfile.c_str());
    m.Write(os, false);
  }
}
/* change freq_weight to investigate the results */
static void UnitTestPenaltyFactor() {
  KALDI_LOG << "=== UnitTestPenaltyFactor() ===";
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
      KALDI_LOG << "--- " << wavefile << " ---";
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
      Compute(op, waveform, &m);
      std::string penaltyfactor = ConvertIntToString(k);
      std::string outfile = "keele/"+num+"-kaldi-penalty-"+penaltyfactor+".txt";
      std::ofstream os(outfile.c_str());
      m.Write(os, false);
    }
  }
}
static void UnitTestKeeleNccfBallast() {
  KALDI_LOG << "=== UnitTestKeeleNccfBallast() ===";
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
      KALDI_LOG << "--- " << wavefile << " ---";
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
      Compute(op, waveform, &m);
      std::string nccfballast = ConvertIntToString(op.nccf_ballast);
      std::string outfile = "keele/"+num
        +"-kaldi-nccf-ballast-"+nccfballast+".txt";
      std::ofstream os(outfile.c_str());
      m.Write(os, false);
    }
  }
}
static void UnitTestWeightedMwn() {
  KALDI_LOG << "=== UnitTestWeightedMwn1() ===";
  // compare the results of WeightedMwn1 and Sliding CMN with uniform weights.
  for (int32 i = 0; i < 1000; i++) {
    int32 num_frames = 1 + (Rand()%10 * 10);
    int32 normalization_win_size = 5 + Rand() % 50;
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
    WeightedMwn(normalization_win_size, pov, log_pitch ,
                &mean_subtracted_log_pitch);
    output_feat.CopyColFromVec(mean_subtracted_log_pitch, 1);

    // SlidingWindow
    SlidingWindowCmnOptions opts;
    opts.cmn_window = normalization_win_size;
    opts.center = true;
    opts.min_window = 1 + Rand() % 100;
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
  int32 num_frames = 1 + (Rand()%10 * 20);
  int32 normalization_win_size = 5 + Rand() % 50;
  Matrix<BaseFloat> feat(num_frames, 2),
    output_feat(num_frames, 2);
  for (int32 j = 0; j < num_frames; j++) {
    int32 r = Rand() % 2;
    feat(j, 0) = RandUniform() / (1 + 1000.0 * r);
    feat(j, 1) = feat(j, 1) * feat(j, 0);
  }
  ProcessPovFeatures(&feat, 2, true);
  WeightedMwn(normalization_win_size, feat, &output_feat);
  */
}
static void UnitTestTakeLogOfPitch() {
  for (int32 i = 0; i < 100; i++) {
    int num_frame = 50 + (Rand() % 200 * 200);
    Matrix<BaseFloat> input(num_frame, 2);
    input.SetRandn();
    input.Scale(100);
    Matrix<BaseFloat> output(input);
    for (int j = 0; j < num_frame; j++) {
      if (input(j, 1) < 1) {
        input(j, 1) = 10;
        output(j, 1) = 10;
      }
      output(j, 1) = log(input(j, 2));
    }
    TakeLogOfPitch(&input);
    if (input.ApproxEqual(output, 0.0001)) {
      KALDI_ERR << " Log of Matrix differs " << input << " vs. " << output;
    }
  }
}
static void UnitTestPitchExtractionSpeed() {
  KALDI_LOG << "=== UnitTestPitchExtractionSpeed() ===";
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
    KALDI_LOG << "--- " << wavefile << " ---";
    std::ifstream is(wavefile.c_str());
    WaveData wave;
    wave.Read(is);
    KALDI_ASSERT(wave.Data().NumRows() == 1);
    SubVector<BaseFloat> waveform(wave.Data(), 0);
    // compute pitch.
    int test_num = 10;
    Matrix<BaseFloat> m;
    struct timeb tstruct;
    int tstart = 0, tend = 0;
    double tot_ft = 0;
    // compute time for Pitch Extraction
    ftime(&tstruct);
    tstart = tstruct.time * 1000 + tstruct.millitm;
    for (int32 t = 0; t < test_num; t++)
      Compute(op, waveform, &m);
    ftime(&tstruct);
    tend = tstruct.time * 1000 + tstruct.millitm;
    double tot_real_time = test_num * waveform.Dim() / op.samp_freq;
    tot_ft = (tend - tstart)/tot_real_time;
    KALDI_LOG << " Pitch extraction time per second of speech "
              << tot_ft << " msec " << std::endl;
  }
}
static void UnitTestPitchExtractorCompareKeele() {
  KALDI_LOG << "=== UnitTestPitchExtractorCompareKeele() ===";
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
    KALDI_LOG << "--- " << wavefile << " ---";
    std::ifstream is(wavefile.c_str());
    WaveData wave;
    wave.Read(is);
    KALDI_ASSERT(wave.Data().NumRows() == 1);
    SubVector<BaseFloat>  waveform(wave.Data(), 0);
    // compute pitch.
    Matrix<BaseFloat> m;
    Compute(op, waveform, &m);
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
    KALDI_LOG << "--- " << wavefile << " ---";
    std::ifstream is(wavefile.c_str());
    WaveData wave;
    wave.Read(is);
    KALDI_ASSERT(wave.Data().NumRows() == 1);
    SubVector<BaseFloat> waveform(wave.Data(), 0);
    Matrix<BaseFloat> m;
    Compute(op, waveform, &m);
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
    KALDI_LOG << "--- " << wavefile << " ---";
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
    Compute(op, waveform, &m);
    PostProcessPitchOptions postprop_op;
    postprop_op.pov_nonlinearity = 2;
    PostProcessPitch(postprop_op, m, &m2);
    std::string outfile = "keele/"+num+"-processed-kaldi.txt";
    std::ofstream os(outfile.c_str());
    m2.Write(os, false);
  }
}
void UnitTestDeltaPitch() {
  KALDI_LOG << "=== UnitTestDeltaPitch() ===";
  for (int32 i = 0; i < 1; i++) {
    int32 num_frames = 1 + (Rand()%10 * 1000);
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
    for (int32 j = 0; j < num_frames; j++)
      KALDI_LOG << output_feat(j) << " , " << output_feat2(j) << " ";
  }
}
void UnitTestResample() {
  KALDI_LOG << "=== UnitTestResample() ===";
  // Resample the sine wave
  double sample_freq = 2000;
  double resample_freq = 1000;
  double lowpass_filter_cutoff = 1000;
  int sample_num = 1000;
  int32 lowpass_filter_width = 2;
  Matrix<double> input_wave(1, sample_num);
  for (int32 i = 0; i < sample_num; i++)
    input_wave(0, i) = sin(2*M_PI/sample_freq * i);
  double dt = sample_freq / resample_freq;
  int32 resampled_len = static_cast<int>(sample_num/dt);
  std::vector<double> resampled_t(resampled_len);
  Matrix<double> resampled_wave1(1, resampled_len),
                 resampled_wave2(1, resampled_len),
                 resampled_wave3(1, resampled_len);
  for (int32 i = 0; i < resampled_len; i++)  {
    resampled_t[i] = static_cast<double>(i) / resample_freq;
    resampled_wave2(0, i) = sin(2 * M_PI * resampled_t[i]);
  }
  ArbitraryResample resample(sample_num, sample_freq,
                             lowpass_filter_cutoff, resampled_t,
                             lowpass_filter_width);
  resample.Upsample(input_wave, &resampled_wave1);

  if (!resampled_wave1.ApproxEqual(resampled_wave2, 0.01)) {
    KALDI_ERR << "Resampled wave " << resampled_wave1
              << " vs. " << resampled_wave2;
  }
  // UnitTest of LinearResample, should equals ArbitraryResample
  Vector<double> input_wave_vec(sample_num);
  input_wave_vec.CopyRowFromMat(input_wave, 0);
  Vector<double> resampled_wave_vec(resampled_len);
  LinearResample resample3(sample_freq, resample_freq,
                             lowpass_filter_cutoff,
                             lowpass_filter_width);
  resample3.Upsample(input_wave_vec, &resampled_wave_vec);
  resampled_wave3.CopyRowFromVec(resampled_wave_vec, 0);

  if (!resampled_wave3.ApproxEqual(resampled_wave1, 0.0001)) {
    KALDI_ERR << "Resampled wave " << resampled_wave3
              << " vs. " << resampled_wave1;
  }
}

static void UnitTestFeatNoKeele() {
  UnitTestSimple();
  UnitTestDeltaPitch();
  UnitTestTakeLogOfPitch();
  UnitTestWeightedMwn();
  UnitTestResample();
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
  try {
    UnitTestFeatNoKeele();
    if (DirExist("keele/16kHz")) {
      UnitTestFeatWithKeele();
    } else {
      KALDI_LOG << "Not running tests that require the Keele database, "
        << "please ask g.meyer@somewhere.edu for the database if you need it.\n"
        << " you need to put keele wave file in keele/16kHz directory";
    }
    KALDI_LOG << "Tests succeeded.";
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return 1;
  }
}


