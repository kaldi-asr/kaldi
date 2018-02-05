// feat/feature-aperiodic.cc

// Copyright 2013  Arnab Ghoshal
//           2016  CereProc Ltd. (author: Blaise Potard)

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



#include "idlakfeat/feature-aperiodic.h"
#include "feat/feature-window.h"
#include "idlakfeat/banks-computations.h"
#include "idlakfeat/feature-mcep.h"

namespace kaldi {

AperiodicEnergy::AperiodicEnergy(const AperiodicEnergyOptions &opts)
    : opts_(opts), feature_window_function_(opts.frame_opts),
      srfft_(NULL), freq_banks_(NULL) {
  int32 window_size = opts.frame_opts.PaddedWindowSize();

  // If the signal window size is N, then the number of points in the FFT
  // computation is the smallest power of 2 that is greater than or equal to 2N
  padded_window_size_ = RoundUpToNearestPowerOfTwo(window_size);
  if (padded_window_size_ <=
      static_cast<int32>(opts_.frame_opts.samp_freq/opts_.f0_min)) {
    KALDI_ERR << "Padded window size (" << padded_window_size_ << ") too small "
              << " to capture F0 range.";
  }
  srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size_);

  FrameExtractionOptions freq_frame_opts = opts_.frame_opts;
  freq_frame_opts.frame_length_ms =
      (static_cast<BaseFloat>(padded_window_size_)
          * 1000.0) / freq_frame_opts.samp_freq;
  // If using HTS bands, we force the bark scale options on the MelBanks
  if (opts_.use_hts_bands)
    opts_.banks_opts.scale_type = "bark";
  freq_banks_ = new FrequencyBanks(opts_.banks_opts, freq_frame_opts, 1.0);
}

AperiodicEnergy::~AperiodicEnergy() {
  if (srfft_ == NULL) {
    KALDI_WARN << "NULL srfft_ pointer: This should not happen if the class "
               << "was used properly";
    return;
  }
  delete srfft_;
  if (freq_banks_ == NULL) {
    KALDI_WARN << "NULL freq_banks_ pointer: This should not happen if the "
               << "class was used properly";
    return;
  }
  delete freq_banks_;
}

#define bark(x) (1960.0 / (26.81 / ((x) + 0.53) - 1))

/// Average the energy of a power spectrum accross a set of predefined bands,
/// the bands are designed to be somewhat compatible with what is commonly
/// used in the HTS community, e.g. in the "Emime" EU project.
// Note that we do an arithmetic mean for the average computation. Emime
// and straight use a geometric mean, but in my (BP) opinion this is something
// strange to do unless you want to be equivalent to an arithmetic mean
// performed in the log domain...
void AperiodicEnergy::ComputeHtsBands(
    const VectorBase<BaseFloat> &power_spectrum,
    Vector<BaseFloat> *output) {
  output->SetZero();
  BaseFloat sample_frequency = opts_.frame_opts.samp_freq;
  int32 number_bands = output->Dim();

  int32 fftlen = 2 * (power_spectrum.Dim() - 1), band_start, band_end = 0;
  for (int32 band = 0; band < number_bands; band++) {
    // Legacy hard-coded 5 bands from straight
    if (number_bands == 5) {
      // Copy previous value
      band_start = band_end;
      switch (band) {
      case 0:
        band_end = fftlen / 16;
        break;
      case 1:
        band_end = fftlen / 8;
        break;
      case 2:
        band_end = fftlen / 4;
        break;
      case 3:
        band_end = fftlen * 3 / 8;
        break;
      case 4:
        band_end = fftlen / 2 + 1;
        break;
      }
    } else {
      // Use buggy hardcoded Bark-inspired bands from J. Yamagishi
      // in Emime.  Note that we use bands that are less buggy,
      // i.e. their size is strictly increasing.
      // Compared to J. Yamagishi's implementation we fixed other bugs:
      // - if bands higher than Nyqist are requested, their energy is set to 0
      // - last band end always set to Nyqist frequency
      BaseFloat start_frequency, end_frequency;
      // Frequency in Hertz...
      start_frequency = bark(band);
      end_frequency = bark(band+1);
      if (band == 0) start_frequency = 0.0;
      // ... converted to an index in power_spectrum
      band_start = static_cast<int32>(round(start_frequency *
                                            fftlen / sample_frequency));
      // Above nyqist frequency or last band?
      if (end_frequency >= sample_frequency / 2 ||  // over Nyqist
          band >= 25 ||  // highest possible Bark band
          band == number_bands - 1) {  // last requested band
        end_frequency = sample_frequency / 2;
        band_end = fftlen / 2 + 1;
      } else {
        band_end = static_cast<int32>(round(end_frequency *
                                            fftlen / sample_frequency));
      }
    }
    // Empty band; that should never happen, but just in case...
    if (band_end <= band_start) continue;
    float sum = 0.0;
    // NB: in HTS, they normally do geometric mean instead of
    // arithmetic mean; this create risks of underflow (and is
    // probably idiotic anyway) so we do not reproduce that behaviour
    for (int32 i = band_start; i < band_end; i++) {
      sum += power_spectrum(i);
    }
    (*output)(band) = sum / (band_end - band_start);
    // Above Nyqist? We are done.
    if (band_end >= fftlen / 2 + 1) break;
  }
}

/// Compute aperiodic energy coefficients, obtained by estimating the noise /
/// aperiodicity spectrogram by sampling the frequencies that minimise the
/// periodic spectrogram (a.k.a. pitch spectrogram).
/// This function takes a wave file, and synchronised F0 / voicing probability
/// sequences of features.
/// Any suitable F0 extractor may be used, but we recommend using kaldi pitch
/// extractor.
void AperiodicEnergy::Compute(const VectorBase<BaseFloat> &wave,
                              const VectorBase<BaseFloat> &voicing_prob,
                              const VectorBase<BaseFloat> &f0,
                              Matrix<BaseFloat> *output,
                              Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(output != NULL);
  KALDI_ASSERT(srfft_ != NULL &&
              "srfft_ must not be NULL if class is initialized properly.");
  KALDI_ASSERT(freq_banks_ != NULL &&
              "freq_banks_ must not be NULL if class is initialized properly.");
  int32 frames_out = NumFrames(wave.Dim(), opts_.frame_opts),
      dim_out = opts_.banks_opts.num_bins;
  if (frames_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
  if (std::abs(voicing_prob.Dim() - frames_out) > opts_.frame_diff_tolerance) {
    KALDI_ERR << "#frames in probability of voicing vector ("
              << voicing_prob.Dim() << ") doesn't match #frames in data ("
              << frames_out << ").";
  }
  if (std::abs(f0.Dim() - frames_out) > opts_.frame_diff_tolerance) {
    KALDI_ERR << "#frames in F0 vector (" << f0.Dim() << ") doesn't match "
              << "#frames in data (" << frames_out << ").";
  }

  frames_out = std::min(frames_out, f0.Dim());  // will be removed eventually
  output->Resize(frames_out, dim_out);
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);
  Vector<BaseFloat> wave_window;
  Vector<BaseFloat> padded_window(padded_window_size_, kUndefined);
  Vector<BaseFloat> binned_energies(dim_out);
  Vector<BaseFloat> noise_binned_energies(dim_out);
  // Print out band frequency information
  if (!opts_.use_hts_bands || dim_out != 5) {
    Vector<BaseFloat> frqs = freq_banks_->GetCenterFreqs();
    BaseFloat freq_delta;
    // The center frequencies are equally spaced in the mel / bark domain
    freq_delta = freq_banks_->Scale(frqs(1), opts_.banks_opts.scale_type) -
        freq_banks_->Scale(frqs(0), opts_.banks_opts.scale_type);

    for (int i = 0; i < frqs.Dim(); i++) {
      BaseFloat low_freq, high_freq;
      // simulating rectangular window
      low_freq = freq_banks_->InverseScale(
          freq_banks_->Scale(frqs(i), opts_.banks_opts.scale_type) -
          0.5 * freq_delta, opts_.banks_opts.scale_type);
      high_freq = freq_banks_->InverseScale(
          freq_banks_->Scale(frqs(i), opts_.banks_opts.scale_type) +
          0.5 * freq_delta, opts_.banks_opts.scale_type);
      KALDI_LOG << "Band " << i <<
         ": [ " << low_freq << " ; " << frqs(i) << " ; " << high_freq << " ]";
    }
  }

  for (int32 r = 0; r < frames_out; r++) {  // r is frame index
    SubVector<BaseFloat> this_ap_energy(output->Row(r));
    ExtractWindow(0, wave, r, opts_.frame_opts, feature_window_function_,
                  &wave_window, NULL);
    int32 window_size = wave_window.Dim();
    padded_window.SetZero();
    padded_window.Range(padded_window_size_/2 - window_size/2, window_size).
      CopyFromVec(wave_window);
    srfft_->Compute(padded_window.Data(), true);

    Vector<BaseFloat> tmp_spectrum(padded_window);
    ComputePowerSpectrum(&tmp_spectrum);
    SubVector<BaseFloat> power_spectrum(tmp_spectrum, 0,
                                        padded_window_size_/2 + 1);
    KALDI_LOG << "Computing frame " << r << " with F0 " << f0(r);

    // BP: this code allows us to speed things up a bit for unvoiced regions
    if (voicing_prob(r) <= 0.0) {  // unvoiced region
      // While the aperiodic energy will not be used during synthesis of the
      // unvoiced regions, it is still modeled as a separate stream in the HMM
      // and so we set it to the log-filterbank values.
      for (int i = 0; i < dim_out; i++) binned_energies(i) = 0.0;
      this_ap_energy.CopyFromVec(binned_energies);
      continue;
    }
    // Compute energy bands of signal power spectrum; the aperiodic energy
    // will be the ratio of noise over signal.
    // NB: we use ComputeHtsBands only for legacy support
    if (opts_.use_hts_bands && dim_out == 5)
      ComputeHtsBands(power_spectrum, &binned_energies);
    else
      freq_banks_->Compute(power_spectrum, &binned_energies);

    // Identify the area of spectrum where the noise dominates
    std::vector<bool> noise_indices;
    IdentifyNoiseRegions(power_spectrum, f0(r), &noise_indices);

    // Recreate a noise spectrum by interpolating from the noise
    // regions obtained previously.
    // padded_window contains the FFT coefficients
    ObtainNoiseSpectrum(padded_window, noise_indices, &tmp_spectrum);
    ComputePowerSpectrum(&tmp_spectrum);
    SubVector<BaseFloat> noise_spectrum(tmp_spectrum, 0,
                                        padded_window_size_/2 + 1);

    // Compute energy bands of noise power spectrum.
    if (opts_.use_hts_bands && dim_out == 5)
      ComputeHtsBands(noise_spectrum, &noise_binned_energies);
    else
      freq_banks_->Compute(noise_spectrum, &noise_binned_energies);

    // Normalise noise energy by the total energy in the band
    for (int i = 0; i < noise_binned_energies.Dim(); i++) {
      noise_binned_energies(i) /= binned_energies(i);
      // Prevent overflows
      if (noise_binned_energies(i) >= 1.0) noise_binned_energies(i) = 1.0;
      if (noise_binned_energies(i) <= 1e-15) noise_binned_energies(i) = 1e-15;
    }
    noise_binned_energies.ApplyLog();  // take the log
    this_ap_energy.CopyFromVec(noise_binned_energies);
  }
}

// For voiced regions, we reconstruct the harmonic spectrum from the
// cepstral coefficients around the cepstral peak corresponding to F0.
// The frequency samples for which the harmonic spectrum reaches minimum
// values are considered to be "pure noise" and are returned.
void AperiodicEnergy::IdentifyNoiseRegions(
    const VectorBase<BaseFloat> &power_spectrum,
    BaseFloat f0,
    std::vector<bool> *noise_indices) {
  KALDI_ASSERT(noise_indices != NULL);
  KALDI_ASSERT(power_spectrum.Dim() == padded_window_size_/2+1 && "Power "
               "spectrum size expected to be half of padded window plus 1.");
  BaseFloat sampling_freq = opts_.frame_opts.samp_freq;
  int32 f0_index = static_cast<int32>(round(sampling_freq/f0)),
      max_f0_index = static_cast<int32>(round(sampling_freq/opts_.f0_max)),
      cepstrum_peak_index, test_index;
  // Width of F0-induced cepstral peak
  int32 f0_peak_width = static_cast<int32>(round(f0_index *
                                                 opts_.f0_width * 0.5));
  int32 transition_width = static_cast<int32>(round(max_f0_index * 0.1));
  int32 cepstrum_peak_width;
  noise_indices->resize(padded_window_size_/2+1, false);

  Vector<BaseFloat> noise_spectrum(padded_window_size_, kSetZero);
  Vector<BaseFloat> test_cepstrum(padded_window_size_, kSetZero);
  Vector<BaseFloat> abs_cepstrum(padded_window_size_, kSetZero);

  noise_spectrum.Range(0, padded_window_size_/2+1).CopyFromVec(power_spectrum);
  // Hacky: to avoid high frequency noise in cepstrum, we do a
  // low-pass filter on power spectrum, only keeping up to 4kHz
  int32 k4_idx =  static_cast<int32>(round(padded_window_size_
                                           * 4000.0 / sampling_freq));
  for (int32 i = 0; i <= padded_window_size_/2 - k4_idx; i++) {
    if (i < transition_width * 3) {
      // Gradual decrease over transition window
      noise_spectrum(k4_idx + i) *= 0.501 +
        0.499 * cos(i * M_PI/(transition_width * 3));
    } else {
      noise_spectrum(k4_idx + i) *= 0.01;
    }
  }

  // Compute Real Cepstrum
  PowerSpectrumToRealCepstrum(&noise_spectrum);

  // All cepstral coefficients below the one corresponding to maximum F0 are
  // considered to correspond to vocal tract characteristics and ignored from
  // the initial harmonic to noise ratio calculation.
  noise_spectrum.Range(0, max_f0_index - transition_width- 1).SetZero();
  // Cos lifter for smoother spectrum
  for (int32 i = -transition_width; i <= 0; i++) {
      noise_spectrum(max_f0_index + i) *= 0.5 +
        0.5 * cos(i * M_PI / transition_width);
  }
  // Find peak in absolute cepstrum
  for (int32 i = 0; i <=  padded_window_size_/2; i++) {
      abs_cepstrum(i) = fabs(noise_spectrum(i));
  }
  abs_cepstrum.Max(&cepstrum_peak_index);
  cepstrum_peak_width = static_cast<int32>(round(cepstrum_peak_index *
                                                 opts_.f0_width * 0.5));
  // Estimate quality of peak by zeroing peak area and getting max of
  // left-over absolute cepstrum
  test_cepstrum.CopyFromVec(abs_cepstrum);
  //   1. Set to 0 from peak_index-npeak_width to peak_index+npeak_width.
  // (But maybe we should actually set everything above
  // peak_index-npeak_width to 0?)
  test_cepstrum.Range(cepstrum_peak_index - cepstrum_peak_width,
                      cepstrum_peak_width * 2).SetZero();
  //   2. Find vestigial peak
  test_cepstrum.Max(&test_index);
  // Estimate peak quality from ratio of peak amplitudes.
  // Quality between 0.0 (very bad) and 1.0 (perfect). In unvoiced
  // region the peak should be less than 0.1, in well voiced region
  // above 0.5
  BaseFloat cepstrum_peak_quality = (1.0 - test_cepstrum(test_index) /
                                     noise_spectrum(cepstrum_peak_index));

  if (cepstrum_peak_index < f0_index - f0_peak_width ||
      cepstrum_peak_index > f0_index + f0_peak_width) {
    KALDI_VLOG(2)
      << "Actual cepstral peak (index=" << cepstrum_peak_index << "; value = "
      << abs_cepstrum(cepstrum_peak_index) << "; quality = "
      << cepstrum_peak_quality << ") occurs too far from F0 (index="
      << f0_index << "; value = " << noise_spectrum(f0_index) << ").";
  }
  // Slightly hacky: a good quality peak will replace a bad F0 peak
  if (f0 == 0 || (
        (cepstrum_peak_quality >= opts_.quality_threshold) &&
        (f0_index != cepstrum_peak_index))) {
    KALDI_VLOG(2) << "Actual cepstral peak has quality "<< cepstrum_peak_quality
                  << ", moving f0 from "<< f0_index
                  << " to " << cepstrum_peak_index;
    f0_index = cepstrum_peak_index;
    f0_peak_width = cepstrum_peak_width;
  }

  // Generating harmonic spectrum from cepstral coefficients located
  // around cepstral / f0 peak
  Vector<BaseFloat> harmonic_spectrum(padded_window_size_, kSetZero);
  // Note that at this point noise_spectrum contains cepstral coeffs
  // energy
  // In the range of interest, we copy the ceps to harmonic_spectrum
  // Note that to have the "real" noise spectrum, we should zero out these
  // same quefrencies.
  // The harmonic "peaks" are actually present for every multiple of F0,
  // we take only the first 2
  for (int32 k = 1; k <= 4; k++) {
    if (k * f0_index + f0_peak_width >= padded_window_size_/2) break;
    int32 offset = k * f0_index;
    // copy within the peak
    for (int32 i = - f0_peak_width; i <= f0_peak_width; i++) {
      BaseFloat m = (0.5 + 0.5 * cos(i * M_PI / (f0_peak_width + 1) ));
      harmonic_spectrum(offset + i) = m * noise_spectrum(offset + i);
      /*noise_spectrum(offset + i) *= (1 - m);*/
    }
  }
  // Get log spectrum of harmonic cepstrum
  RealCepstrumToMagnitudeSpectrum(&harmonic_spectrum, false);

  // We find the "negative peaks" of the harmonic spectrum, we
  // consider these will always correspond to pure noise spectrum
  // frequencies. We will then interpolate between these samples to
  // build the full "noise spectrogram".
  //
  // This will work well in voiced regions if the F0 has been
  // estimated correctly.  This will work pretty well in noise regions
  // as well, unless somehow the "noise" has strong harmonics :-)
  //
  // Our peak picking is a disgrace, but it works well enough!
#define _is_neg_peak(s, i) ((s(i) < s(i-1)) && (s(i) < s(i+1)) && (s(i) < 0.0))
  //
  // Instead of just taking the min peaks, we could use segments, e.g.
  // the lowest 10% between two peaks.
  for (int32 i = 0; i <= padded_window_size_/2; ++i) {
    // NB: - to behave like STRAIGHT, it seems you should just use
    //       "if ( harmonic_spectrum(i) < 0)"
    //     - in 0, the aperiodic energy should always be 0.0!
    if (i > 0 && i < padded_window_size_/2 &&
        _is_neg_peak(harmonic_spectrum, i)) {
      (*noise_indices)[i] = true;
    } else {
      (*noise_indices)[i] = false;
    }
  }
}

/// Generate the noise spectrum from a set of frequencies where
/// noise should dominate in the signal. We simply interpolate
/// linearly between these frequencies
void AperiodicEnergy::ObtainNoiseSpectrum(
  const VectorBase<BaseFloat> &fft_coeffs,
  const std::vector<bool> &noise_indices,
  Vector<BaseFloat> *noise_spectrum) {
  KALDI_ASSERT(noise_spectrum != NULL);
  KALDI_ASSERT(noise_spectrum->Dim() == padded_window_size_);

  noise_spectrum->SetZero();
  Vector<BaseFloat> prev_estimate(padded_window_size_, kSetZero);
  Vector<BaseFloat> debug_spectrum(padded_window_size_, kSetZero);
  Vector<BaseFloat> window_array(padded_window_size_, kSetZero);
  int32 window_size = opts_.frame_opts.WindowSize();
  int32 last_index = -1;
  window_array.Range(padded_window_size_/2 - window_size/2, window_size).
    CopyFromVec(feature_window_function_.window);

  // Build Noise FFT by copying frequencies marked as noise and
  // interpolating between them.
  for (int32 j = 0; j <= padded_window_size_/2; ++j) {
    if ((j == padded_window_size_/2) || noise_indices[j]) {
      // NB: frequency 0 should never be noise.
      if (j != 0 && j != padded_window_size_/2) {
        (*noise_spectrum)(j*2) = fft_coeffs(j*2);
        (*noise_spectrum)(j*2 + 1) = fft_coeffs(j*2 + 1);
      }
      // Some gaps have to be filled in
      if (last_index != j - 1) {
        // interpolate linearly between last_index and j-1
        if (last_index == -1) last_index = 0;
        BaseFloat init_re, init_im, end_re, end_im;
        // Handles Kaldi exotic FFT structure: 0 and nyqist frequencies real
        // together, all the rest as complex.
        // Interpolate between last_index and j:
        //  - Load start and end values
        if (last_index == 0) {
          // Start from 0
          // NB: for now the value will always be 0
          init_re = (*noise_spectrum)(0);
          init_im = 0.0;
        } else {
          init_re = fft_coeffs(last_index*2);
          init_im = fft_coeffs(last_index*2 + 1);
        }

        if (j == padded_window_size_/2) {
          // End at Nyqist, always pure noise
          end_re = fft_coeffs(1);
          end_im = 0.0;
        } else {
          end_re = fft_coeffs(j*2);
          end_im = fft_coeffs(j*2 + 1);
        }
        // - Interpolate linearly in complex domain
        std::complex<double> init_cplx(init_re, init_im);
        std::complex<double> end_cplx(end_re, end_im);
        double init_abs = std::abs(init_cplx), end_abs = std::abs(end_cplx);
        double init_arg = std::arg(init_cplx), end_arg = std::arg(end_cplx);

        for (int k = last_index + 1; k < j; k++) {
          double m = (static_cast<double>(k) - last_index) / (j - last_index);
          // New complex value from linearly interpolated polar representation
          std::complex<double> z = std::polar(
            init_abs * (1.0 - m) + end_abs * m,
            init_arg * (1.0 - m) + end_arg * m);
          (*noise_spectrum)(k*2) = static_cast<BaseFloat>(z.real());
          (*noise_spectrum)(k*2 + 1) = static_cast<BaseFloat>(z.imag());
        }
      }
      last_index = j;
    }
  }
  //ComputePowerSpectrum(noise_spectrum);
}

}  // namespace kaldi
