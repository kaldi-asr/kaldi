// feat/sinusoid-detection-test.cc

// Copyright    2015  Johns Hopkins University (author: Daniel Povey)

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
#include "feat/sinusoid-detection.h"


namespace kaldi {

// this function is used for testing AddSinusoid.
void AddSinusoidSimple(BaseFloat samp_freq,
                       const Sinusoid &sinusoid,
                       VectorBase<BaseFloat> *signal) {
  for (int32 i = 0; i < signal->Dim(); i++)
    (*signal)(i) += sinusoid.amplitude *
        cos(M_2PI * sinusoid.freq / samp_freq * i + sinusoid.phase);
}

void UnitTestAddSinusoid() {
  BaseFloat samp_freq = 560.1;
  int32 length = 511;
  Vector<BaseFloat> orig(length);
  orig.SetRandn();
  Vector<BaseFloat> orig2(orig);
  Sinusoid sinusoid(49.20, 2.111, 1.5);

  AddSinusoid(samp_freq, sinusoid, &orig);
  AddSinusoidSimple(samp_freq, sinusoid, &orig2);
  AssertEqual(orig, orig2);
}
      


void UnitTestQuadraticMaximizeEqualSpaced() {
  for (int32 n = 0; n < 50; n++) {
  
    //  Let the cubic function be y = a x^2 + b x + c, and let
    //   y0,y1,y2 be its values evaluated at x = [0, 1, 2]; we
    // want it evaluated at arbitrary x.
    
    BaseFloat  a = -0.5 + RandUniform(), b = -0.5 + RandUniform(), c = -0.5 + RandUniform();
    BaseFloat y[3];
    for (int32 i = 0; i < 3; i++) {
      BaseFloat x = i;
      y[i] = a * x * x + b * x + c;
    }
    BaseFloat x_max, y_max;
    SinusoidDetector::QuadraticMaximizeEqualSpaced(y[0], y[1], y[2], &x_max, &y_max);

    for (int32 m = 0; m <= 10; m++) {
      BaseFloat x_test = 0.1 * m;
      BaseFloat y_test = a * x_test * x_test + b * x_test + c;
      KALDI_ASSERT(y_test <= y_max + 1.0e-05);
    }
  }
}

void UnitTestQuadraticMaximize() {
  for (int32 n = 0; n < 50; n++) {
  
    //  Let the cubic function be y = a x^2 + b x + c, and let
    //   y0,y1,y2 be its values evaluated at x = [0, 1, 2]; we
    // want it evaluated at arbitrary x.
    
    BaseFloat  a = -0.5 + RandUniform(), b = -0.5 + RandUniform(), c = -0.5 + RandUniform(),
        x = 0.1 + RandUniform() * 0.98;
    BaseFloat y[3];
    for (int32 i = 0; i < 3; i++) {
      BaseFloat this_x;
      if (i == 0) { this_x = 0.0; }
      else if (i == 1) { this_x = x; }
      else { this_x = 1.0; }
      y[i] = a * this_x * this_x + b * this_x + c;
    }
    BaseFloat x_max, y_max;
    SinusoidDetector::QuadraticMaximize(x, y[0], y[1], y[2], &x_max, &y_max);
    
    for (int32 m = 0; m <= 10; m++) {
      BaseFloat x_test = 0.1 * m;
      BaseFloat y_test = a * x_test * x_test + b * x_test + c;
      if (n < 100 && m == 5) {
        KALDI_VLOG(2) << "Checking y_test <= y_max: "
                      << y_test << " <= " << y_max << " [x_max = "
                      << x_max << "]";
        KALDI_ASSERT(y_test <= y_max + 1.0e-05);
      }
    }
  }
}


void UnitTestSinusoidDetector() {
  BaseFloat samp_freq = 4000 + (rand() % 2000);
  int32 num_samp = 128 + rand() % 400;
  SinusoidDetector detector(samp_freq, num_samp);

  for (int32 i = 0; i < 40; i++) {
  
    Vector<BaseFloat> signal(num_samp);

    // Sinusoid ref_sinusoid(1.3, 312.5, M_PI * 0.0);
    // Sinusoid ref_sinusoid(1.3, 324.125, M_PI * 0.5);

    BaseFloat nyquist = samp_freq * 0.5;
    BaseFloat freq = nyquist * RandUniform();
    BaseFloat amplitude = RandUniform();
    BaseFloat phase = M_2PI * RandUniform();

    Sinusoid ref_sinusoid(amplitude, freq, phase);
  
    AddSinusoid(samp_freq, ref_sinusoid, &signal);


    BaseFloat orig_energy = VecVec(signal, signal);
    KALDI_LOG << "Real frequency is " << freq << ", amplitude "
              << amplitude << ", phase " << phase << ", samp-freq "
              << samp_freq;
    KALDI_LOG << "Total energy of signal (with sinusoid) is " << orig_energy;
  
    Sinusoid sinusoid;
    BaseFloat min_energy = 0.0;
    BaseFloat energy = detector.DetectSinusoid(min_energy,
                                               signal, &sinusoid);

    Vector<BaseFloat> new_signal(signal);
    sinusoid.phase += M_PI;  // Reverse the phase.
    AddSinusoid(samp_freq, sinusoid, &new_signal);
    BaseFloat delta_energy = VecVec(signal, signal) -
        VecVec(new_signal, new_signal);
    KALDI_LOG << "Projected delta energy = " << energy
              << " and observed was " << delta_energy;

    BaseFloat remaining_energy = VecVec(new_signal, new_signal);
    if (remaining_energy > 0.01 * orig_energy) {
      KALDI_WARN << "Energy remaining is " << remaining_energy
                 << " vs. original " << orig_energy;
      BaseFloat relative_freq = freq / nyquist;
      BaseFloat inv_num_samp = 1.0 / num_samp;
      // We only tolerate this kind of error for very ridiculous frequency,
      // close to zero or the Nyquist.
      KALDI_ASSERT(relative_freq < inv_num_samp ||
                   relative_freq > 1.0 - inv_num_samp);
    }
  }
}

// as UnitTestSinusoidDetector(), but doing it in noisy signals.
void UnitTestSinusoidDetectorNoisy() {
  BaseFloat samp_freq = 4000 + (rand() % 2000);
  int32 num_samp = 128 + rand() % 400;
  SinusoidDetector detector(samp_freq, num_samp);

  for (int32 i = 0; i < 40; i++) {
  
    Vector<BaseFloat> signal(num_samp);

    signal.SetRandn();

    BaseFloat rand_energy = VecVec(signal, signal);
    
    // Sinusoid ref_sinusoid(1.3, 312.5, M_PI * 0.0);
    // Sinusoid ref_sinusoid(1.3, 324.125, M_PI * 0.5);

    BaseFloat nyquist = samp_freq * 0.5;
    BaseFloat freq = nyquist * RandUniform();
    BaseFloat amplitude = 10.0 * RandUniform();
    BaseFloat phase = M_2PI * RandUniform();

    Sinusoid ref_sinusoid(amplitude, freq, phase);
  
    AddSinusoid(samp_freq, ref_sinusoid, &signal);

    BaseFloat tot_energy = VecVec(signal, signal);

    KALDI_LOG << "Real frequency is " << freq << ", amplitude "
              << amplitude << ", phase " << phase << ", samp-freq "
              << samp_freq;
    KALDI_LOG << "Total energy of signal (with noise + sinusoid) is " << tot_energy;
  
    Sinusoid sinusoid;
    BaseFloat min_energy = 0.0;
    BaseFloat energy = detector.DetectSinusoid(min_energy,
                                               signal, &sinusoid);

    Vector<BaseFloat> new_signal(signal);
    sinusoid.phase += M_PI;  // reverse the phase.
    AddSinusoid(samp_freq, sinusoid, &new_signal);
    BaseFloat delta_energy = VecVec(signal, signal) -
        VecVec(new_signal, new_signal);
    KALDI_LOG << "Projected delta energy = " << energy
              << " and observed was " << delta_energy;

    BaseFloat min_energy_diff = 0.99 * (tot_energy - rand_energy);
    
    if (delta_energy < min_energy_diff) {
      KALDI_WARN << "Energy reduction is " << delta_energy
                 << " vs. expected " << (tot_energy - rand_energy);
      BaseFloat relative_freq = freq / nyquist;
      BaseFloat inv_num_samp = 1.0 / num_samp;
      // We only tolerate this kind of error for very ridiculous frequency,
      // close to zero or the Nyquist.
      KALDI_ASSERT(relative_freq < inv_num_samp ||
                   relative_freq > 1.0 - inv_num_samp);
    }
  }
}


void AddFreqToSignal(BaseFloat base_freq,
                     BaseFloat samp_freq,
                     BaseFloat tolerance,
                     BaseFloat gain,
                     VectorBase<BaseFloat> *signal) {
  BaseFloat error_scale = (2 * RandUniform() - 1) * tolerance;
  BaseFloat freq = base_freq * (1.0 + error_scale);
  KALDI_VLOG(3) << "base-freq = " << base_freq << ", factor = " << error_scale;
  for (int32 i = 0; i < signal->Dim(); i++)
    (*signal)(i) += gain * sin(i * 2.0 * 3.14159 * freq / samp_freq);
}


void GenerateDtmfTestCase(
    BaseFloat sampling_rate,
    Vector<BaseFloat> *signal,
    std::vector<MultiSinusoidDetectorOutput> *ref_output) {
  // the "ref_output" should correlate with the first of each run of frames with the same label.
  
  BaseFloat min_duration_secs = 0.04;  // min duration of dtmf or non-tone segments.
  BaseFloat min_dialtone_duration_secs = 0.1;
  BaseFloat frequency_tolerance = 0.035;
  BaseFloat dialtone_frequency_tolerance = 0.4 * (440.0  - 425.0) / 440.0;

  int32 num_events = 2 * (5 + rand() % 5) + 1; // odd number.
  int32 tot_signal_dim = 0;

  ref_output->resize(num_events);
  std::vector<Vector<BaseFloat> > all_signals(num_events);
  for (int32 i = 0; i < num_events; i++) {
    MultiSinusoidDetectorOutput &this_output = (*ref_output)[i];
    Vector<BaseFloat> &this_signal = all_signals[i];
    BaseFloat duration_secs = min_duration_secs * (1 + rand() % 3);
    int32 num_samp = sampling_rate * duration_secs;
    tot_signal_dim += num_samp;

    this_signal.Resize(num_samp);
    this_signal.SetRandn();
    
    if (i % 2 == 0); // do nothing;
    else if (rand() % 2 == 0 && duration_secs >= min_dialtone_duration_secs) {
      // dialtone.
      BaseFloat freq;
      if (rand() % 3 == 0) { freq = 350; }
      else if (rand() % 2 == 0) { freq = 440; }
      else { freq = 425; }
      BaseFloat gain = 10.0 * (1.0 + rand() % 2);
      AddFreqToSignal(freq, sampling_rate, dialtone_frequency_tolerance,
                      gain, &(this_signal));
      this_output.freq1 = freq;
    } else {
      // dtmf.  use a subset of tones as examples.
      BaseFloat freq1, freq2;
      char c;
      if (rand() % 4 == 0) {
        c = '8'; freq1 = 852; freq2 = 1336;
      } else if (rand() % 3 == 0) {
        c = '0'; freq1 = 941; freq2 = 1336;
      } else if (rand() % 2 == 0) {
        c = '#'; freq1 = 941; freq2 = 1477;
      } else {
        c = '1'; freq1 = 697; freq2 = 1209;
      }
      BaseFloat base_gain = 10.0 * (1.0 +  (rand() % 3)),
          gain_factor = 1.0 + 0.1 * (-2 + rand() % 5),
          gain1 = base_gain, gain2 = gain_factor * base_gain;
      AddFreqToSignal(freq1, sampling_rate, frequency_tolerance, gain1,
                      &(this_signal));
      AddFreqToSignal(freq2, sampling_rate, frequency_tolerance, gain2,
                      &(this_signal));
      this_output.freq1 = freq1;
      this_output.freq2 = freq2;
    }
  }
  signal->Resize(tot_signal_dim);
  int32 signal_offset = 0;
  for (int32 i = 0; i < num_events; i++) {
    int32 this_dim = all_signals[i].Dim();
    signal->Range(signal_offset, this_dim).CopyFromVec(all_signals[i]);
    signal_offset += this_dim;
  }
}


/*

// Just a basic test to check that it produces output.

void UnitTestToneDetection() {
  BaseFloat samp_freq = (rand() % 2) == 0 ? 8000 : 16000;
  ToneDetectionConfig config;
  
  int32 num_frames = 100 + (rand() % 100);
  int32 frame_length = static_cast<int32>(samp_freq * config.frame_length_secs);

  int32 num_samples = frame_length * num_frames + rand() % frame_length;
  Vector<BaseFloat> signal(num_samples);
  signal.SetRandn();

  ToneDetector tone_detector(config, samp_freq);

  int32 signal_offset = 0;

  std::vector<ToneDetectorOutput*> tone_detector_output;
  
  while (signal_offset < num_samples) {
    int32 signal_remaining = num_samples - signal_offset,
        chunk_size = std::min<int32>((rand() % 200) + 100,
                                     signal_remaining);
    SubVector<BaseFloat> signal_part(signal, signal_offset, chunk_size);
    tone_detector.AcceptWaveform(signal_part);
    signal_offset += chunk_size;

    if (signal_offset == num_samples)
      tone_detector.WaveformFinished();
    while (!tone_detector.Done() &&
           (rand() % 2 == 0 || signal_offset == num_samples)) {
      ToneDetectorOutput *output = new ToneDetectorOutput();
      tone_detector.GetNextFrame(output);
      tone_detector_output.push_back(output);
    }
  }
  KALDI_ASSERT(signal_offset == num_samples);  
  
  Vector<BaseFloat> signal2(signal.Dim());
  signal_offset = 0;
  for (int32 i = 0; i < tone_detector_output.size(); i++) {
    ToneDetectorOutput *output = tone_detector_output[i];
    signal2.Range(signal_offset,
                  output->signal.Dim()).CopyFromVec(output->signal);
    signal_offset += output->signal.Dim();
    if (output->frame_type != 'n') {
      KALDI_ERR << "Frame " << i << " badly classified, should be 'n', is: "
                << output->frame_type;
    }
    delete output;
  }
  KALDI_ASSERT(signal_offset == num_samples &&
               signal.ApproxEqual(signal2, 1.0e-10));

}

std::ostringstream & operator << (std::ostringstream &ostr,
             const ToneDetectorOutput &output) {
  ostr << output.frame_type;
  if (output.frame_type == 'd')
    ostr << output.dialtone_freq;
  ostr << ' ';
  return ostr;
}

*/


// This version of the unit-test generates a signal that has tones in it, and
// runs the detection on that signal.
void UnitTestToneDetection2() {
  BaseFloat samp_freq = (rand() % 2) == 0 ? 8000 : 16000;
  Vector<BaseFloat> signal;
  std::vector<MultiSinusoidDetectorOutput> ref_output;
  GenerateDtmfTestCase(samp_freq, &signal, &ref_output);
  
  MultiSinusoidDetectorConfig config;

  int32 num_samples = signal.Dim();
  KALDI_ASSERT(num_samples > 0);

  MultiSinusoidDetector multi_sinusoid_detector(config, samp_freq);
  
  int32 signal_offset = 0;

  std::vector<MultiSinusoidDetectorOutput*> multi_sinusoid_detector_output;

  while (signal_offset < num_samples) {
    int32 signal_remaining = num_samples - signal_offset,
        chunk_size = std::min<int32>((rand() % 200) + 100,
                                     signal_remaining);
    SubVector<BaseFloat> signal_part(signal, signal_offset, chunk_size);
    multi_sinusoid_detector.AcceptWaveform(signal_part);
    signal_offset += chunk_size;

    if (signal_offset == num_samples)
      multi_sinusoid_detector.WaveformFinished();
    while (!multi_sinusoid_detector.Done() &&
           (rand() % 2 == 0 || signal_offset == num_samples)) {
      MultiSinusoidDetectorOutput *output = new MultiSinusoidDetectorOutput();
      multi_sinusoid_detector.GetNextFrame(output);
      multi_sinusoid_detector_output.push_back(output);
    }
  }
  KALDI_ASSERT(signal_offset == num_samples);  
  
  // std::ostringstream str_ref, str_hyp;
  //for (size_t i = 0; i < ref_output.size(); i++)
  //    str_ref << ref_output[i];


  for (size_t i = 0; i < multi_sinusoid_detector_output.size(); i++) {
    MultiSinusoidDetectorOutput *output = multi_sinusoid_detector_output[i];
    KALDI_LOG << "tot-energy = " << output->tot_energy
              << ", freq1 " << output->freq1 << ", energy1 " << output->energy1
              << ", freq2 " << output->freq2 << ", energy2 " << output->energy2;
    delete output;
  }
}



}  // namespace kaldi

int main() {
  using namespace kaldi;

  SetVerboseLevel(4);

  UnitTestToneDetection2();  
  UnitTestAddSinusoid();
  UnitTestQuadraticMaximizeEqualSpaced();
  UnitTestQuadraticMaximize();
  for (int32 i = 0; i < 10; i++) {
    UnitTestSinusoidDetector();
    UnitTestSinusoidDetectorNoisy();
  }

}
