// featbin/wav-reverberate.cc

// Copyright 2015  Tom Ko

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "feat/signal.h"

namespace kaldi {

/*
   This function is to repeatedly concatenate signal1 by itself 
   to match the length of signal2 and add the two signals together.
*/
void AddVectorsOfUnequalLength(const VectorBase<BaseFloat> &signal1,
                                     Vector<BaseFloat> *signal2) {
  for (int32 po = 0; po < signal2->Dim(); po += signal1.Dim()) {
    int32 block_length = signal1.Dim();
    if (signal2->Dim() - po < block_length) block_length = signal2->Dim() - po;
    signal2->Range(po, block_length).AddVec(1.0, signal1.Range(0, block_length));
  }
}

/*
   This function is to add signal1 to signal2 starting at the offset of signal2
   This will not extend the length of signal2.
*/
void AddVectorsWithOffset(const Vector<BaseFloat> &signal1, int32 offset,
                                             Vector<BaseFloat> *signal2) {
  int32 add_length = std::min(signal2->Dim() - offset, signal1.Dim());
  if (add_length > 0)
    signal2->Range(offset, add_length).AddVec(1.0, signal1.Range(0, add_length));
}


BaseFloat MaxAbsolute(const Vector<BaseFloat> &vector) {
  return std::max(std::abs(vector.Max()), std::abs(vector.Min()));
}

/* 
   Early reverberation component of the signal is composed of reflections 
   within 0.05 seconds of the direct path signal (assumed to be the peak of 
   the room impulse response). This function returns the energy in 
   this early reverberation component of the signal. 
   The input parameters to this function are the room impulse response, the signal
   and their sampling frequency respectively.
*/
BaseFloat ComputeEarlyReverbEnergy(const Vector<BaseFloat> &rir, const Vector<BaseFloat> &signal,
                                   BaseFloat samp_freq) {
  int32 peak_index = 0;
  rir.Max(&peak_index);
  KALDI_VLOG(1) << "peak index is " << peak_index;

  const float sec_before_peak = 0.001;
  const float sec_after_peak = 0.05;
  int32 early_rir_start_index = peak_index - sec_before_peak * samp_freq;
  int32 early_rir_end_index = peak_index + sec_after_peak * samp_freq;
  if (early_rir_start_index < 0) early_rir_start_index = 0;
  if (early_rir_end_index > rir.Dim()) early_rir_end_index = rir.Dim();

  int32 duration = early_rir_end_index - early_rir_start_index;
  Vector<BaseFloat> early_rir(rir.Range(early_rir_start_index, duration));
  Vector<BaseFloat> early_reverb(signal);
  FFTbasedBlockConvolveSignals(early_rir, &early_reverb);

  // compute the energy
  return VecVec(early_reverb, early_reverb) / early_reverb.Dim();
}

/*
   This is the core function to do reverberation on the given signal.
   The input parameters to this function are the room impulse response,
   the sampling frequency and the signal respectively.
   The length of the signal will be extended to (original signal length +
   rir length - 1) after the reverberation.
*/
float DoReverberation(const Vector<BaseFloat> &rir, BaseFloat samp_freq,
                        Vector<BaseFloat> *signal) {
  float signal_power = ComputeEarlyReverbEnergy(rir, *signal, samp_freq);
  FFTbasedBlockConvolveSignals(rir, signal);
  return signal_power;
}

/*
   The noise will be scaled before the addition
   to match the given signal-to-noise ratio (SNR).
*/
void AddNoise(Vector<BaseFloat> *noise, BaseFloat snr_db,
                BaseFloat time, BaseFloat samp_freq,
                BaseFloat signal_power, Vector<BaseFloat> *signal) {
  float noise_power = VecVec(*noise, *noise) / noise->Dim();
  float scale_factor = sqrt(pow(10, -snr_db / 10) * signal_power / noise_power);
  noise->Scale(scale_factor);
  KALDI_VLOG(1) << "Noise signal is being scaled with " << scale_factor
                << " to generate output with SNR " << snr_db << "db\n";
  int32 offset = time * samp_freq;
  AddVectorsWithOffset(*noise, offset, signal);
}

/*
   This function converts comma-spearted string into float vector.
*/
void ReadCommaSeparatedCommand(const std::string &s,
                                std::vector<BaseFloat> *v) {
  std::vector<std::string> split_string;
  SplitStringToVector(s, ",", true, &split_string);
  for (size_t i = 0; i < split_string.size(); i++) {
    float ret;
    ConvertStringToReal(split_string[i], &ret);
    v->push_back(ret);
  }
}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Corrupts the wave files supplied via input pipe with the specified\n"
        "room-impulse response (rir_matrix) and additive noise distortions\n"
        "(specified by corresponding files).\n"
        "Usage:  wav-reverberate [options...] <wav-in-rxfilename> "
        "<wav-out-wxfilename>\n"
        "e.g.\n"
        "wav-reverberate --duration=20.25 --impulse-response=rir.wav "
        "--additive-signals='noise1.wav,noise2.wav' --snrs='20.0,15.0' "
        "--start-times='0,17.8' input.wav output.wav\n";

    ParseOptions po(usage);
    std::string rir_file;
    std::string additive_signals;
    std::string snrs;
    std::string start_times;
    bool multi_channel_output = false;
    bool shift_output = true;
    int32 input_channel = 0;
    int32 rir_channel = 0;
    int32 noise_channel = 0;
    bool normalize_output = true;
    BaseFloat volume = 0;
    BaseFloat duration = 0;

    po.Register("multi-channel-output", &multi_channel_output,
                "Specifies if the output should be multi-channel or not");
    po.Register("shift-output", &shift_output,
                "If true, the reverberated waveform will be shifted by the "
                "amount of the peak position of the RIR and the length of "
                "the output waveform will be equal to the input waveform. "
                "If false, the length of the output waveform will be "
                "equal to (original input length + rir length - 1). "
                "This value is true by default and "
                "it only affects the output when RIR file is provided.");
    po.Register("input-wave-channel", &input_channel,
                "Specifies the channel to be used from input as only a "
                "single channel will be used to generate reverberated output");
    po.Register("rir-channel", &rir_channel,
                "Specifies the channel of the room impulse response, "
                "it will only be used when multi-channel-output is false");
    po.Register("noise-channel", &noise_channel,
                "Specifies the channel of the noise file, "
                "it will only be used when multi-channel-output is false");
    po.Register("impulse-response", &rir_file,
                "File with the impulse response for reverberating the input wave"
                "It can be either a file in wav format or a piped command. "
                "E.g. --impulse-response='rir.wav' or 'sox rir.wav - |' ");
    po.Register("additive-signals", &additive_signals,
                "A comma separated list of additive signals. "
                "They can be either filenames or piped commands. "
                "E.g. --additive-signals='noise1.wav,noise2.wav' or "
                "'sox noise1.wav - |,sox noise2.wav - |'. "
                "Requires --snrs and --start-times.");
    po.Register("snrs", &snrs,
                "A comma separated list of SNRs(dB). "
                "The additive signals will be scaled according to these SNRs. "
                "E.g. --snrs='20.0,0.0,5.0,10.0' ");
    po.Register("start-times", &start_times,
                "A comma separated list of start times referring to the "
                "input signal. The additive signals will be added to the "
                "input signal starting at the offset. If the start time "
                "exceed the length of the input signal, the addition will "
                "be ignored.");
    po.Register("normalize-output", &normalize_output,
                "If true, then after reverberating and "
                "possibly adding noise, scale so that the signal "
                "energy is the same as the original input signal. "
                "See also the --volume option.");
    po.Register("duration", &duration,
                "If nonzero, it specified the duration (secs) of the output "
                "signal. If the duration t is less than the length of the "
                "input signal, the first t secs of the signal is trimmed, "
                "otherwise, the signal will be repeated to "
                "fulfill the duration specified.");
    po.Register("volume", &volume,
                "If nonzero, a scaling factor on the signal that is applied "
                "after reverberating and possibly adding noise. "
                "If you set this option to a nonzero value, it will be as "
                "if you had also specified --normalize-output=false.");

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    if (multi_channel_output) {
      if (rir_channel != 0 || noise_channel != 0)
        KALDI_WARN << "options for --rir-channel and --noise-channel"
                      "are ignored as --multi-channel-output is true.";
    }

    std::string input_wave_file = po.GetArg(1);
    std::string output_wave_file = po.GetArg(2);

    WaveData input_wave;
    {
      WaveHolder waveholder;
      Input ki(input_wave_file);
      waveholder.Read(ki.Stream());
      input_wave = waveholder.Value();
    }

    const Matrix<BaseFloat> &input_matrix = input_wave.Data();
    BaseFloat samp_freq_input = input_wave.SampFreq();
    int32 num_samp_input = input_matrix.NumCols(),  // #samples in the input
          num_input_channel = input_matrix.NumRows();  // #channels in the input
    KALDI_VLOG(1) << "sampling frequency of input: " << samp_freq_input
                  << " #samples: " << num_samp_input
                  << " #channel: " << num_input_channel;
    KALDI_ASSERT(input_channel < num_input_channel);

    Matrix<BaseFloat> rir_matrix;
    BaseFloat samp_freq_rir = samp_freq_input;
    int32 num_samp_rir = 0,
          num_rir_channel = 0;
    if (!rir_file.empty()) {
      WaveData rir_wave;
      {
        WaveHolder waveholder;
        Input ki(rir_file);
        waveholder.Read(ki.Stream());
        rir_wave = waveholder.Value();
      }
      rir_matrix = rir_wave.Data();
      samp_freq_rir = rir_wave.SampFreq();
      num_samp_rir = rir_matrix.NumCols();
      num_rir_channel = rir_matrix.NumRows();
      KALDI_VLOG(1) << "sampling frequency of rir: " << samp_freq_rir
                    << " #samples: " << num_samp_rir
                    << " #channel: " << num_rir_channel;
      if (!multi_channel_output) {
        KALDI_ASSERT(rir_channel < num_rir_channel);
      }
    }

    std::vector<Matrix<BaseFloat> > additive_signal_matrices;
    if (!additive_signals.empty()) {
      if (snrs.empty() || start_times.empty())
        KALDI_ERR << "--additive-signals option requires "
                     "--snrs and --start-times to be set.";
      std::vector<std::string> split_string;
      SplitStringToVector(additive_signals, ",", true, &split_string);
      for (size_t i = 0; i < split_string.size(); i++) {
        WaveHolder waveholder;
        Input ki(split_string[i]);
        waveholder.Read(ki.Stream());
        WaveData additive_signal_wave = waveholder.Value();
        Matrix<BaseFloat> additive_signal_matrix = additive_signal_wave.Data();
        BaseFloat samp_freq = additive_signal_wave.SampFreq();
        KALDI_ASSERT(samp_freq == samp_freq_input);
        int32 num_samp = additive_signal_matrix.NumCols(),
              num_channel = additive_signal_matrix.NumRows();
        KALDI_VLOG(1) << "sampling frequency of additive signal: " << samp_freq
                      << " #samples: " << num_samp
                      << " #channel: " << num_channel;
        if (multi_channel_output) {
          KALDI_ASSERT(num_rir_channel == num_channel);
        } else {
          KALDI_ASSERT(noise_channel < num_channel);
        }

        additive_signal_matrices.push_back(additive_signal_matrix);
      }
    }

    std::vector<BaseFloat> snr_vector;
    if (!snrs.empty()) {
      ReadCommaSeparatedCommand(snrs, &snr_vector);
    }

    std::vector<BaseFloat> start_time_vector;
    if (!start_times.empty()) {
      ReadCommaSeparatedCommand(start_times, &start_time_vector);
    }

    int32 shift_index = 0;
    int32 num_output_channels = (multi_channel_output ? num_rir_channel : 1);
    int32 num_samp_output = (duration > 0 ? samp_freq_input * duration :
                              (shift_output ? num_samp_input :
                                              num_samp_input + num_samp_rir - 1));
    Matrix<BaseFloat> out_matrix(num_output_channels, num_samp_output);

    for (int32 output_channel = 0; output_channel < num_output_channels; output_channel++) {
      Vector<BaseFloat> input(num_samp_input);
      input.CopyRowFromMat(input_matrix, input_channel);
      float power_before_reverb = VecVec(input, input) / input.Dim();

      int32 this_rir_channel = (multi_channel_output ? output_channel : rir_channel);

      float early_energy = power_before_reverb;
      if (!rir_file.empty()) {
        Vector<BaseFloat> rir;
        rir.Resize(num_samp_rir);
        rir.CopyRowFromMat(rir_matrix, this_rir_channel);
        rir.Scale(1.0 / (1 << 15));
        early_energy = DoReverberation(rir, samp_freq_rir, &input);
        if (shift_output) {
          // find the position of the peak of the impulse response 
          // and shift the output waveform by this amount
          rir.Max(&shift_index);
        }
      }

      if (additive_signal_matrices.size() > 0) {
        Vector<BaseFloat> noise(0);
        int32 this_noise_channel = (multi_channel_output ? output_channel : noise_channel);
        KALDI_ASSERT(additive_signal_matrices.size() == snr_vector.size());
        KALDI_ASSERT(additive_signal_matrices.size() == start_time_vector.size());
        for (int32 i = 0; i < additive_signal_matrices.size(); i++) {
          noise.Resize(additive_signal_matrices[i].NumCols());
          noise.CopyRowFromMat(additive_signal_matrices[i], this_noise_channel);
          AddNoise(&noise, snr_vector[i], start_time_vector[i],
                    samp_freq_input, early_energy, &input);
        }
      }

      float power_after_reverb = VecVec(input, input) / input.Dim();

      if (volume > 0)
        input.Scale(volume);
      else if (normalize_output)
        input.Scale(sqrt(power_before_reverb / power_after_reverb));

      if (num_samp_output <= num_samp_input) {
        // trim the signal from the start
        out_matrix.CopyRowFromVec(input.Range(shift_index, num_samp_output), output_channel);
      } else {
        // repeat the signal to fill up the duration
        Vector<BaseFloat> extended_input(num_samp_output);
        extended_input.SetZero();
        AddVectorsOfUnequalLength(input.Range(shift_index, num_samp_input), &extended_input);
        out_matrix.CopyRowFromVec(extended_input, output_channel);
      }
    }

    WaveData out_wave(samp_freq_input, out_matrix);
    Output ko(output_wave_file, true, false);
    out_wave.Write(ko.Stream());

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

