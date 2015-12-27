// featbin/corrupt-wav.cc

// Copyright 2015  Tom Ko
//           2015  Vimal Manohar

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
                               VectorBase<BaseFloat> *signal2,
                               VectorBase<BaseFloat> *signal1_added) {
  if (signal1_added)
    KALDI_ASSERT(signal2->Dim() == signal1_added->Dim());
  for (int32 po = 0; po < signal2->Dim(); po += signal1.Dim()) {
    int32 block_length = signal1.Dim();
    if (signal2->Dim() - po < block_length) block_length = signal2->Dim() - po;
    signal2->Range(po, block_length).AddVec(1.0, signal1.Range(0, block_length));
    if (signal1_added)
      signal1_added->Range(po, block_length).CopyFromVec(
                                              signal1.Range(0, block_length));
  }
}

inline BaseFloat MaxAbsolute(
    const VectorBase<BaseFloat> &vector) {
  return std::max(std::abs(vector.Max()), std::abs(vector.Min()));
}

inline BaseFloat ComputeEnergy(
    const VectorBase<BaseFloat> &vec) {
  return VecVec(vec, vec) / vec.Dim();
}

inline BaseFloat DbToValue(const BaseFloat &db) {
  return Exp(db * Log(10.0) / 10.0);
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
  return ComputeEnergy(early_reverb);
}

/*
   This is the core function to do reverberation and noise addition
   on the given signal. The noise will be scaled before the addition
   to match the given signal-to-noise ratio (SNR) and it will also concatenate
   itself repeatedly to match the length of the signal.
   The input parameters to this function are the room impulse response,
   the sampling frequency, the SNR(dB), the noise and the signal respectively.
*/
void DoCorruption(BaseFloat samp_freq, const Vector<BaseFloat> &rir, 
                  Vector<BaseFloat> *noise, BaseFloat background_snr_db,
                  const std::vector<Matrix<BaseFloat> > &foreground_noises,
                  int32 channel, BaseFloat foreground_snr_db, 
                  Vector<BaseFloat> *signal, 
                  Vector<BaseFloat> *out_clean = NULL,
                  Vector<BaseFloat> *out_noise = NULL, 
                  BaseFloat min_duration = 0.1, BaseFloat search_fraction = 0.1) {
  BaseFloat input_power = 0;

  if (rir.Dim() > 0) {
    FFTbasedBlockConvolveSignals(rir, signal);
    input_power = ComputeEarlyReverbEnergy(rir, *signal, samp_freq);
  } else {
    input_power = ComputeEnergy(*signal);
  }

  if (out_clean)
    out_clean->CopyFromVec(*signal);

  if (noise->Dim() > 0) {
    BaseFloat noise_power = ComputeEnergy(*noise);
    BaseFloat scale_factor = sqrt(DbToValue(-background_snr_db) 
                                  * input_power / noise_power);
    noise->Scale(scale_factor);
    KALDI_VLOG(1) << "Noise signal is being scaled with " << scale_factor
                  << " to generate output with SNR " << background_snr_db << "db\n";
    AddVectorsOfUnequalLength(*noise, signal, out_noise);
  }

  KALDI_ASSERT(search_fraction <= 1.0);
  if (foreground_noises.size() > 0) {
    int32 t = 0;
    while (t < signal->Dim()) {
      // Start position to add foreground noise must be beyond the current 't'
      // but not more than search_fraction * signal->Dim()
      int32 start_t = t + search_fraction * signal->Dim() * RandUniform();
      int32 max_duration_possible = signal->Dim() - start_t;

      // Check if the max duration possible is less than a minimum duration.
      // This is to avoid adding very short duration of noise, say 1 frame.
      if (max_duration_possible < min_duration * samp_freq) break;
      
      int32 i = RandInt(0, foreground_noises.size() - 1);
      KALDI_ASSERT(channel < foreground_noises[i].NumRows());
      Vector<BaseFloat> foreground_noise(foreground_noises[i].Row(channel));
      if (max_duration_possible < foreground_noise.Dim()) {
        SubVector<BaseFloat> this_foreground_noise(foreground_noise, 
                                                   0, max_duration_possible);
        SubVector<BaseFloat> this_signal(*signal,   
                                         start_t, max_duration_possible);

        BaseFloat noise_power = ComputeEnergy(this_foreground_noise);
        BaseFloat signal_power = ComputeEnergy(this_signal);

        BaseFloat scale_factor = sqrt(DbToValue(-foreground_snr_db) 
                                      * signal_power / noise_power);
        this_signal.AddVec(scale_factor, this_foreground_noise);

        if (out_noise) {
          SubVector<BaseFloat> this_out_noise(*out_noise, 
                                              start_t, max_duration_possible);
          this_out_noise.AddVec(scale_factor, this_foreground_noise);
        }
        
        break;
      } else {
        SubVector<BaseFloat> this_signal(*signal,   
                                         start_t, foreground_noise.Dim());

        BaseFloat noise_power = ComputeEnergy(foreground_noise);
        BaseFloat signal_power = ComputeEnergy(this_signal);

        BaseFloat scale_factor = sqrt(DbToValue(-foreground_snr_db)
                                      * signal_power / noise_power);
        this_signal.AddVec(scale_factor, foreground_noise);
        
        if (out_noise) {
          SubVector<BaseFloat> this_out_noise(*out_noise, 
                                              start_t, foreground_noise.Dim());
          this_out_noise.AddVec(scale_factor, foreground_noise);
        }

        t += foreground_noise.Dim();
      }
    }
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
        "Usage: wav-reverberate [options] <input-wave-file> <output-wave-file>\n"
        " e.g.: wav-reverberate --rir-file=large_roon_rir.wav clean.wav corrupted.wav\n";

    ParseOptions po(usage);
    
    std::string rir_file;
    std::string background_noise_file;
    std::string foreground_noise_files_str;
    std::string out_clean_file;
    std::string out_noise_file;

    BaseFloat background_snr_db = 20;
    BaseFloat foreground_snr_db = 20;
    bool multi_channel_output = false;
    int32 input_channel = 0;
    int32 rir_channel = 0;
    int32 noise_channel = 0;
    bool normalize_output = true;
    BaseFloat volume = 0;
    BaseFloat rms_amplitude = 0.1;
    bool normalize_by_amplitude = false, normalize_by_power = false;
    int32 srand_seed = 0;
    BaseFloat min_duration = 0.1, search_fraction = 0.1;

    po.Register("multi-channel-output", &multi_channel_output,
                "Specifies if the output should be multi-channel or not");
    po.Register("input-wave-channel", &input_channel,
                "Specifies the channel to be used from input as only a "
                "single channel will be used to generate reverberated output");
    po.Register("rir-channel", &rir_channel,
                "Specifies the channel of the room impulse response, "
                "it will only be used when multi-channel-output is false");
    po.Register("noise-channel", &noise_channel,
                "Specifies the channel of the noise file, "
                "it will only be used when multi-channel-output is false");
    po.Register("background-snr-db", &background_snr_db,
                "Desired SNR(dB) of background noise wrt clean signal");
    po.Register("foreground-snr-db", &foreground_snr_db,
                "Desired SNR(db) of foreground noise wrt of background corrupted signal");
    po.Register("normalize-output", &normalize_output,
                "If true, then after reverberating and "
                "possibly adding noise, scale so that the signal "
                "energy is the same as the original input signal.");
    po.Register("volume", &volume,
                "If nonzero, a scaling factor on the signal that is applied "
                "after reverberating and possibly adding noise. "
                "If you set this option to a nonzero value, it will be as"
                "if you had also specified --normalize-output=false. "
                "If you set this option to a negative value, it will be "
                "ignored and instead the --signal-db option would be used.");
    po.Register("rms-amplitude", &rms_amplitude,
                "Desired rms after corruption. This will be used "
                "only if volume is less than 0");
    po.Register("normalize-by-amplitude", &normalize_by_amplitude, 
                "Make the maximum amplitude in the output signal to be 95% of "
                "the amplitude range possible in wave output");
    po.Register("normalize-by-power", &normalize_by_power,
                "Make the amplitude such that the RMS energy of the signal "
                "is rms-amplitude");
    po.Register("output-noise-file", &out_noise_file,
                "Wave file to write the output noise file just before "
                "adding it to the reverberated signal");
    po.Register("output-clean-file", &out_clean_file,
                "Wave file to write the output clean file just before "
                "adding additive noise. It may have reverberation");
    po.Register("rir-file", &rir_file, 
                "File with room impulse response");
    po.Register("background-noise-file", &background_noise_file,
                "File with additive background noise");
    po.Register("foreground-noise-files", &foreground_noise_files_str,
                "Colon separated list of foreground noise files");
    po.Register("min-duration", &min_duration, 
                "If the duration of signal in which we can add foreground "
                "noise is smaller than this min-duration, then the noise "
                "would not be added.");
    po.Register("search-fraction", &search_fraction,
                "The maximum separation between two foreground noise additions "
                "specified as a fraction of the length of the file");
    po.Register("srand", &srand_seed, "Seed for random number generator");

    po.Read(argc, argv);
    
    srand(srand_seed);
    
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

    KALDI_VLOG(1) << "input-wav-file: " << input_wave_file;
    KALDI_VLOG(1) << "output-wav-file: " << output_wave_file;
    KALDI_VLOG(1) << "rir-file: " << (!rir_file.empty() ? rir_file : "None");
    KALDI_VLOG(1) << "background-noise-file: " 
                  << (!background_noise_file.empty() ? background_noise_file : "None");
    KALDI_VLOG(1) << "foreground-noise-files-str: " 
                  << (!foreground_noise_files_str.empty() ? foreground_noise_files_str : "None");

    /**************************************************************************
     * Read input wave 
     **************************************************************************/

    WaveData input_wave;
    {
      Input ki(input_wave_file);
      input_wave.Read(ki.Stream());
    }

    const Matrix<BaseFloat> &input_matrix = input_wave.Data();
    BaseFloat samp_freq_input = input_wave.SampFreq();
    int32 num_samp_input = input_matrix.NumCols(),  // #samples in the input
          num_input_channel = input_matrix.NumRows();  // #channels in the input
    KALDI_VLOG(1) << "sampling frequency of input: " << samp_freq_input
                  << " #samples: " << num_samp_input
                  << " #channel: " << num_input_channel;
    KALDI_ASSERT(input_channel < num_input_channel);
    
    /**************************************************************************
     * Read room impulse response if it exists
     **************************************************************************/

    const Matrix<BaseFloat> *rir_matrix = NULL;
    
    BaseFloat samp_freq_rir = samp_freq_input;
    int32 num_samp_rir = 0,
          num_rir_channel = 1;

    WaveData rir_wave;
    if (!rir_file.empty()) {
      {
        Input ki(rir_file);
        rir_wave.Read(ki.Stream());
      }
      rir_matrix = &rir_wave.Data();

      samp_freq_rir = rir_wave.SampFreq();
      KALDI_ASSERT(samp_freq_input == samp_freq_rir);
      num_samp_rir = rir_matrix->NumCols();
      num_rir_channel = rir_matrix->NumRows();
      KALDI_VLOG(1) << "sampling frequency of rir: " << samp_freq_rir
                    << " #samples: " << num_samp_rir
                    << " #channel: " << num_rir_channel;
      if (!multi_channel_output) {
        KALDI_ASSERT(rir_channel < num_rir_channel);
      }
    } else {
      rir_channel = 0;
      // Cannot create multichannel output without an rir-file
      KALDI_ASSERT(!multi_channel_output);    
    }

    /**************************************************************************
     * Read background noise if it is provided
     **************************************************************************/

    const Matrix<BaseFloat> *background_noise_matrix = NULL;
    WaveData noise_wave;
    if (!background_noise_file.empty()) {
      {
        Input ki(background_noise_file);
        noise_wave.Read(ki.Stream());
      }
      background_noise_matrix = &noise_wave.Data();
      BaseFloat samp_freq_noise = noise_wave.SampFreq();
      KALDI_ASSERT(samp_freq_input == samp_freq_noise);
      int32 num_samp_noise = background_noise_matrix->NumCols(),
            num_noise_channel = background_noise_matrix->NumRows();
      KALDI_VLOG(1) << "sampling frequency of noise: " << samp_freq_noise
                    << " #samples: " << num_samp_noise
                    << " #channel: " << num_noise_channel;
      if (multi_channel_output) {
        KALDI_ASSERT(num_rir_channel == num_noise_channel);
      } else {
        KALDI_ASSERT(noise_channel < num_noise_channel);
      }
    }

    /**************************************************************************
     * Read foreground noises if it is provided
     **************************************************************************/

    std::vector<Matrix<BaseFloat> > foreground_noise_matrices;
    std::vector<std::string> foreground_noise_files;

    if (!foreground_noise_files_str.empty()) {
      SplitStringToVector(foreground_noise_files_str, ":", 
                          true, &foreground_noise_files);

      foreground_noise_matrices.resize(foreground_noise_files.size());
      for (size_t i = 0; i < foreground_noise_files.size(); i++) {
        const std::string &noise_file = foreground_noise_files[i];
        WaveData noise_wave;
        {
          Input ki(noise_file);
          noise_wave.Read(ki.Stream());
        }
        
        Matrix<BaseFloat> &noise_matrix = foreground_noise_matrices[i];
        noise_matrix.Resize(noise_wave.Data().NumRows(), 
                            noise_wave.Data().NumCols());

        noise_matrix.CopyFromMat(noise_wave.Data());

        BaseFloat samp_freq_noise = noise_wave.SampFreq();
        KALDI_ASSERT(samp_freq_input == samp_freq_noise);
        int32 num_samp_noise = noise_matrix.NumCols(),
              num_noise_channel = noise_matrix.NumRows();
        KALDI_VLOG(1) << "sampling frequency of noise: " << samp_freq_noise
                      << " #samples: " << num_samp_noise
                      << " #channel: " << num_noise_channel;
        if (multi_channel_output) {
          KALDI_ASSERT(num_rir_channel == num_noise_channel);
        } else {
          KALDI_ASSERT(noise_channel < num_noise_channel);
        }
      }
    }

    /**************************************************************************
     * Prepare output wave matrix along with the output clean and noise 
     * matrices which need to be written optionally.
     **************************************************************************/

    int32 num_output_channels = (multi_channel_output ? num_rir_channel : 1);
    Matrix<BaseFloat> out_matrix(num_output_channels, num_samp_input);

    Matrix<BaseFloat> out_clean_matrix;
    Matrix<BaseFloat> out_noise_matrix;

    for (int32 output_channel = 0; output_channel < num_output_channels; 
          output_channel++) {
      Vector<BaseFloat> input(num_samp_input);
      input.CopyRowFromMat(input_matrix, input_channel);
      float power_before_corruption = VecVec(input, input) / input.Dim();

      int32 this_rir_channel = (multi_channel_output ? 
                                output_channel : rir_channel);
      Vector<BaseFloat> rir(num_samp_rir);

      if (!rir_file.empty()) {
        // Read a particular channel of room impulse response and convert it
        // to a floating point number
        rir.CopyRowFromMat(*rir_matrix, this_rir_channel);
        rir.Scale(1.0 / (1 << 15));
      }

      Vector<BaseFloat> background_noise;
      
      if (!background_noise_file.empty()) {
        background_noise.Resize(background_noise_matrix->NumCols());
        int32 this_noise_channel = (multi_channel_output ? 
                                    output_channel : noise_channel);
        background_noise.CopyRowFromMat(*background_noise_matrix,   
                                        this_noise_channel);
      }

      Vector<BaseFloat> clean_signal(input.Dim());
      Vector<BaseFloat> noise_signal(input.Dim());
      
      DoCorruption(samp_freq_input, rir, &background_noise, background_snr_db,
                   foreground_noise_matrices, 
                   multi_channel_output ? output_channel : noise_channel,
                   foreground_snr_db, &input, 
                   (!out_clean_file.empty() ? &clean_signal : NULL),
                   (!out_noise_file.empty() ? &noise_signal : NULL),
                   min_duration, search_fraction);

      BaseFloat power_after_corruption = ComputeEnergy(input);

      if (volume > 0) {
        input.Scale(volume);
        if (!out_clean_file.empty())
          clean_signal.Scale(volume);
        if (!background_noise_file.empty()) 
          background_noise.Scale(volume);
      } else if (volume < 0) {
        BaseFloat scale;

        if (normalize_by_amplitude) {
          BaseFloat max = MaxAbsolute(input);

          scale = Exp( Log(rms_amplitude) // signal_db to amplitude
                       - Log(max)                   // actual max amplitude
                       + 15.0 * Log(2.0)            // * 2^15
                       + Log(0.95) );               // Allow only 0.95 of max amplitude possible
        } else if (normalize_by_power) {
          scale = Exp( Log(rms_amplitude) // rms amplitude
                      - 0.5 * Log(power_before_corruption) // clean rms amplitude
                      + 15.0 * Log(2.0));               // * 2^15
        }
        
        input.Scale(scale);

        if (!out_clean_file.empty())
          clean_signal.Scale(scale);
        if (!background_noise_file.empty()) 
          noise_signal.Scale(scale);
      } else if (normalize_output)
        input.Scale(sqrt(power_before_corruption / power_after_corruption));

      out_matrix.CopyRowFromVec(input, output_channel);
      
      if (!out_clean_file.empty()) {
        if (output_channel == 0)
          out_clean_matrix.Resize(out_matrix.NumRows(), out_matrix.NumCols());
        out_clean_matrix.CopyRowFromVec(clean_signal, output_channel);
      }

      if (!out_noise_file.empty()) {
        if (output_channel == 0)
          out_noise_matrix.Resize(out_matrix.NumRows(), out_matrix.NumCols());
        out_noise_matrix.CopyRowFromVec(noise_signal, output_channel);
      }
    }

    WaveData out_wave(samp_freq_input, out_matrix);
    Output ko(output_wave_file, false);
    out_wave.Write(ko.Stream());

    if (!out_clean_file.empty()) {
      WaveData out_clean_wave(samp_freq_input, out_clean_matrix);
      Output ko(out_clean_file, false);
      out_clean_wave.Write(ko.Stream());
    }

    if (!out_noise_file.empty()) {
      WaveData out_noise_wave(samp_freq_input, out_noise_matrix);
      Output ko(out_noise_file, false);
      out_noise_wave.Write(ko.Stream());
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


