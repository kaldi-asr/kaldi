// feat/pitch-functions-test.cc

// Copyright    2013  Pegah Ghahremani
//              2014  IMSL, PKU-HKUST (author: Wei Shi)
//              2014  Yanqing Sun, Junjie Wang,
//                    Daniel Povey, Korbinian Riedhammer
//                    Xin Lei

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

#include "base/kaldi-math.h"
#include "feat/feature-plp.h"
#include "feat/pitch-functions.h"
#include "feat/wave-reader.h"
#include "sys/stat.h"
#include "sys/types.h"
#include "util/timer.h"


namespace kaldi {

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
  Matrix<BaseFloat> m1, m2;
  // init with noise
  for (int32 i = 0; i < v.Dim(); i++) {
    v(i) = (abs(i * 433024253) % 65535) - (65535 / 2);
  }
  KALDI_LOG << "<<<=== Just make sure it runs... Nothing is compared";
  // trying to compute and process pitch with same opts as baseline.
  PitchExtractionOptions op1;
  ProcessPitchOptions op2;
  ComputeAndProcessKaldiPitch(op1, op2, v, &m1);
  KALDI_LOG << "Test passed :)";
}


// Make sure that doing a calculation on the whole waveform gives
// the same results as doing on the waveform broken into pieces.
static void UnitTestPieces() {
  KALDI_LOG << "=== UnitTestPieces() ===\n";
  for (int32 n = 0; n < 10; n++) {
    // the parametrization object
    PitchExtractionOptions op1;
    ProcessPitchOptions op2;
    op2.delta_pitch_noise_stddev = 0.0;  // to avoid mismatch of delta_log_pitch
                                         // brought by rand noise.
    op1.nccf_ballast_online = true;  // this is necessary for the computation
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
      normalized_time += cur_freq / op1.samp_freq;
    }

    Matrix<BaseFloat> m1, m1p;

    // trying to have same opts as baseline.
    ComputeKaldiPitch(op1, v, &m1);
    ProcessPitch(op2, m1, &m1p);

    Matrix<BaseFloat> m2, m2p;

    { // compute it online with multiple pieces.
      OnlinePitchFeature pitch_extractor(op1);
      OnlineProcessPitch process_pitch(op2, &pitch_extractor);
      int32 start_samp = 0;
      while (start_samp < v.Dim()) {
        int32 num_samp = rand() % (v.Dim() + 1 - start_samp);
        SubVector<BaseFloat> v_part(v, start_samp, num_samp);
        pitch_extractor.AcceptWaveform(op1.samp_freq, v_part);
        start_samp += num_samp;
      }
      pitch_extractor.InputFinished();
      int32 num_frames = pitch_extractor.NumFramesReady();
      m2.Resize(num_frames, 2);
      m2p.Resize(num_frames, process_pitch.Dim());
      for (int32 frame = 0; frame < num_frames; frame++) {
        SubVector<BaseFloat> row(m2, frame);
        pitch_extractor.GetFrame(frame, &row);
        SubVector<BaseFloat> rowp(m2p, frame);
        process_pitch.GetFrame(frame, &rowp);
      }
    }
    AssertEqual(m1, m2);
    if (!ApproxEqual(m1p, m2p)) {
      KALDI_ERR << "Post-processed pitch differs: " << m1p << " vs. " << m2p;
    }
    KALDI_LOG << "Test passed :)\n";
  }
}

// Make sure that the delayed output matches the non-delayed
// version in the online scenario.
static void UnitTestDelay() {
  KALDI_LOG << "=== UnitTestDelay() ===\n";
  for (int32 n = 0; n < 10; n++) {
    // the parametrization object
    PitchExtractionOptions ext_opt;
    ProcessPitchOptions pro_opt1, pro_opt2;
    pro_opt1.delta_pitch_noise_stddev = 0.0;  // to avoid mismatch of delta_log_pitch
                                              // brought by rand noise.
    pro_opt2.delta_pitch_noise_stddev = 0.0;  // to avoid mismatch of delta_log_pitch
                                              // brought by rand noise.
    pro_opt2.delay = rand() % 50;
    ext_opt.nccf_ballast_online = true;  // this is necessary for the computation
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
      normalized_time += cur_freq / ext_opt.samp_freq;
    }

    Matrix<BaseFloat> m1, m2;
    // compute it online with multiple pieces.
    OnlinePitchFeature pitch_extractor(ext_opt);
    OnlineProcessPitch pitch_processor(pro_opt1, &pitch_extractor);
    OnlineProcessPitch pitch_processor_delayed(pro_opt2, &pitch_extractor);
    int32 start_samp = 0;
    while (start_samp < v.Dim()) {
      int32 num_samp = rand() % (v.Dim() + 1 - start_samp);
      SubVector<BaseFloat> v_part(v, start_samp, num_samp);
      pitch_extractor.AcceptWaveform(ext_opt.samp_freq, v_part);
      start_samp += num_samp;
    }
    pitch_extractor.InputFinished();

    int32 num_frames = pitch_processor.NumFramesReady();
    m1.Resize(num_frames, pitch_processor.Dim());
    for (int32 frame = 0; frame < num_frames; frame++) {
      SubVector<BaseFloat> rowp(m1, frame);
      pitch_processor.GetFrame(frame, &rowp);
    }

    int32 num_frames_delayed = pitch_processor_delayed.NumFramesReady();
    m2.Resize(num_frames_delayed, pitch_processor_delayed.Dim());
    for (int32 frame = 0; frame < num_frames_delayed; frame++) {
      SubVector<BaseFloat> rowp(m2, frame);
      pitch_processor_delayed.GetFrame(frame, &rowp);
    }

    KALDI_ASSERT(num_frames_delayed == num_frames + pro_opt2.delay);
    SubMatrix<BaseFloat> m3(m2, pro_opt2.delay, num_frames, 0, m2.NumCols());
    if (!ApproxEqual(m1, m3)) {
      KALDI_ERR << "Post-processed pitch differs: " << m1 << " vs. " << m3;
    }
    KALDI_LOG << "Test passed :)\n";
  }
}

extern bool pitch_use_naive_search; // was declared in pitch-functions.cc

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

static void UnitTestComputeGPE() {
  KALDI_LOG << "=== UnitTestComputeGPE ===\n";
  int32 wrong_pitch = 0, tot_voiced = 0, tot_unvoiced = 0, num_frames = 0;
  BaseFloat tol = 0.1, avg_d_kpitch = 0, real_pitch = 0;
  for (int32 i = 1; i < 11; i++) {
    std::string wavefile;
    std::string num;
    if (i < 6) {
      num = "f" + ConvertIntToString(i) + "nw0000";
    } else {
      num = "m" + ConvertIntToString(i-5) + "nw0000";
    }
    Matrix<BaseFloat> gross_pitch;
    std::string pitchfile = "keele/keele-true-lags/"+num+".txt";
    std::ifstream pitch(pitchfile.c_str());
    gross_pitch.Read(pitch, false);
    Matrix<BaseFloat> kaldi_pitch;
    std::string kfile = "keele/tmp/+"+num+"-kaldi.txt";
    std::ifstream kpitch(kfile.c_str());
    kaldi_pitch.Read(kpitch, false);
    num_frames = std::min(kaldi_pitch.NumRows(),gross_pitch.NumRows());
    for (int32 j = 1; j < num_frames; j++) {
      if (gross_pitch(j,0) > 0.0) {
        tot_voiced++;
        real_pitch = 20000.0/gross_pitch(j,0);
        if (fabs((real_pitch - kaldi_pitch(j,1))/real_pitch) > tol)
          wrong_pitch++;
      } else if (gross_pitch(j,0) == 0.0 && gross_pitch(j-1,0) == 0.0) {
        tot_unvoiced++;
        avg_d_kpitch += fabs(kaldi_pitch(j,1) - kaldi_pitch(j-1,1));
      }
    }
  }
  BaseFloat GPE = 1.0 * wrong_pitch / tot_voiced;
  KALDI_LOG << " Gross Pitch Error with Rel.Error " << tol << " is " << GPE;
  KALDI_LOG << "Average Kaldi delta_pitch for unvoiced regions " << avg_d_kpitch/tot_unvoiced;
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
    op.nccf_ballast = 1;
    op.penalty_factor = 5;
    // compute pitch.
    Matrix<BaseFloat> m;
    ComputeKaldiPitch(op, waveform, &m);
    std::string outfile = "keele/tmp/+"+num+"-kaldi.txt";
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
      ComputeKaldiPitch(op, waveform, &m);
      std::string penaltyfactor = ConvertIntToString(k);
      std::string outfile = "keele/tmp/+"+num+"-kaldi-penalty-"+penaltyfactor+".txt";
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
      ComputeKaldiPitch(op, waveform, &m);
      std::string nccfballast = ConvertIntToString(op.nccf_ballast);
      std::string outfile = "keele/tmp/+"+num
        +"-kaldi-nccf-ballast-"+nccfballast+".txt";
      std::ofstream os(outfile.c_str());
      m.Write(os, false);
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
    ComputeKaldiPitch(op, waveform, &m);
    std::string outfile = "keele/tmp/+"+num+"-speedup-kaldi1.txt";
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
    ComputeKaldiPitch(op, waveform, &m);
    std::string outfile = "keele/tmp/+"+num+"-kaldi-samp-freq-"+samp_rate+"kHz.txt";
    std::ofstream os(outfile.c_str());
    m.Write(os, false);
  }
}
void UnitTestProcess() {
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
    ComputeKaldiPitch(op, waveform, &m);
    ProcessPitchOptions postprop_op;
    // postprop_op.pov_nonlinearity = 2;
    // Use zero noise, or the features won't be identical.
    postprop_op.delta_pitch_noise_stddev = 0.0;
    ProcessPitch(postprop_op, m, &m2);

    std::string outfile = "keele/tmp/+"+num+"-processed-kaldi.txt";
    std::ofstream os(outfile.c_str());
    m2.Write(os, false);
  }
}

static void UnitTestFeatNoKeele() {
  UnitTestSimple();
  UnitTestPieces();
  UnitTestDelay();
  UnitTestSearch();
}

static void UnitTestFeatWithKeele() {
  UnitTestProcess();
  UnitTestKeele();
  UnitTestComputeGPE();
  UnitTestPenaltyFactor();
  UnitTestKeeleNccfBallast();
  UnitTestPitchExtractionSpeed();
  UnitTestPitchExtractorCompareKeele();
  UnitTestDiffSampleRate();
}

}  // namespace kaldi

int main() {
  using namespace kaldi;

  SetVerboseLevel(3);
  try {
    UnitTestFeatNoKeele();
    if (DirExist("keele/16kHz")) {
      UnitTestFeatWithKeele();
    } else {
      KALDI_LOG
          << "Not running tests that require the Keele database, "
          << "please ask g.meyer@liverpool.ac.uk for the database if you need it.\n"
          << "Once you have the keele/ subdirectory, containing *.{pel,pet,pev,raw,wav}, do this:\n"
          << "cd keele; mkdir -p 16kHz; mkdir -p tmp; for x in *.wav; do \n"
          << "sox $x -r 16000 16kHz/$x; done  \n"
          << "mkdir -p keele-true-lags; for f in *.pev; do \n"
          << "out_f=keele-true-lags/$(echo $f | sed s:pev:txt:); ( echo ' ['; len=`cat $f | wc -l`; \n"
          << "head -n $(($len-1)) $f | tail -n $(($len-14)) ; echo -n ']') >$out_f; done \n"
          << "\n"
          << "Note: the GPE reported in paper is computed using pseudo-ground-truth pitch obtained\n"
          << "by voting among the pitch trackers mentioned in the paper.\n";
    }
    KALDI_LOG << "Tests succeeded.";
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return 1;
  }
}
