// feat/sinusoid-detection.h

// Copyright     2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_FEAT_SINUSOID_DETECTION_H_
#define KALDI_FEAT_SINUSOID_DETECTION_H_


#include "base/kaldi-error.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "feat/resample.h"
#include <deque>

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{


struct Sinusoid {
  // this structure used to represent a sinusoid of type amplitude cos (2 pi
  // freq t + phase), in the SinusoidDetector code.
  BaseFloat amplitude;
  BaseFloat freq;
  BaseFloat phase;
  Sinusoid(BaseFloat a, BaseFloat f, BaseFloat p):
      amplitude(a), freq(f), phase(p) { }
  Sinusoid() {}
};


// This function adds the given sinusoid to the signal, as:
// (*signal)(t) += amplitude * cos(2 pi freq/samp_freq t + phase).
void AddSinusoid(BaseFloat samp_freq,
                 const Sinusoid &sinusoid,
                 VectorBase<BaseFloat> *signal);


class SinusoidDetector {
 public:
  SinusoidDetector(BaseFloat samp_freq,
                    int32 num_samp);
  

  // Detect the dominant sinusoid component in the signal, as long as the
  // energy-reduction of the signal from subtracting that sinuoid would be >=
  // "min_energy_change", and return that energy reduction; or zero if no
  // candidate was found.
  // non-const because the FFT class has a temporary buffer.
  BaseFloat DetectSinusoid(BaseFloat min_energy_change,
                           const VectorBase<BaseFloat> &signal,
                           Sinusoid *sinusoid);
  
  // This function does quadratic interpolation for a function that is known at
  // three equally spaced points [x0 x1 x2] = [0 1 2], and we want the x-value
  // and corresponding y-value at the maximum of the function within the range
  // 0 <= x <= 2.  It's public for testing reasons.
  static void QuadraticMaximizeEqualSpaced(
    BaseFloat y0, BaseFloat y1, BaseFloat y2,
    BaseFloat *x, BaseFloat *y);


  // This function does quadratic interpolation for a function that is known at
  // three points x0, x1 and x2 with x0 = 0, 0 < x1 < 1 and x2 = 1, where we
  // want the x-value and corresponding y-value at the maximum of the function
  // within the range 0 <= x <= 1.  It's public for testing reasons.
  static void QuadraticMaximize(
    BaseFloat x1, BaseFloat y0, BaseFloat y1, BaseFloat y2,
    BaseFloat *x, BaseFloat *y);

  // This function does quadratic interpolation for a function that is known at
  // three points x0, x1 and x2 with x0 = 0, 0 <= x1 <= 1 and x2 = 1, where
  // we want the value at a specific value x.  The corresponding y-value is returned.
  static BaseFloat QuadraticInterpolate(
    BaseFloat x1, BaseFloat y0, BaseFloat y1, BaseFloat y2,
    BaseFloat x);
  

 private:
  BaseFloat samp_freq_;
  int32 num_samples_;
  int32 num_samples_padded_;  // Number of samples, after zero-padding to power of 2.
  SplitRadixRealFft<BaseFloat> fft_;  // Object used to compute FFT of padded_signal_.

  BaseFloat factor1_;  // When we search the range between two FFT bins, we
                       // assume that the maximum energy-reduction within the
                       // range may be greater than the maximum of the
                       // energy-reductions at either side, by at most
                       // "factor1", with factor1 > 1.0.  The analysis is quite
                       // hard so we determine this factor empirically.  Making
                       // this as small as possible helps us avoid searching too
                       // many bins.

  BaseFloat factor2_;  // As factor1, but for searches within a half-fft-bin
                       // range.  Again determined empirically.  After that we
                       // use quadratic interpolation to find the maximum energy.

  // This matrix, of dimension (num_samples_padded_ * 2 + 1) by
  // num_samples_, has in each row, a different frequency of cosine wave.
  Matrix<BaseFloat> cos_;
  // This matrix, of dimension (num_samples_padded_ * 2 + 1) by
  // num_samples_, has in each row, a different frequency of sine wave.
  Matrix<BaseFloat> sin_;

  // M_ is a precomputed matrix of dimension (num_samples_padded_ * 2 + 1) by 3,
  // containing the values x y z of a symmetric matrix [ a b; b c ].  There is
  // one of these matrices for each frequency, sampled at one quarter the
  // spacing of the FFT bins.  There is a long comment next to the definition of
  // ComputeCoefficients that describes this. 
  Matrix<BaseFloat> M_;

  // Minv_ is the coefficients in the same format as M_, but containing the
  // corresponding coefficients of the inverse matrix.  There is a long comment
  // next to the definition of ComputeCoefficients that describes this.
  Matrix<BaseFloat> Minv_;
  

  struct InfoForBin {
    bool valid;
    BaseFloat cos_dot;  // dot product of signal with cosine on left frequency
    BaseFloat sin_dot;  // dot product of signal with sine on left frequency
    BaseFloat energy;  // energy.
    InfoForBin(): valid(false) { }
  };

  // Info after fine optimization within a bin.
  struct OptimizedInfo {
    int32 bin;
    BaseFloat offset;
    BaseFloat energy;
    BaseFloat cos_coeff;
    BaseFloat sin_coeff;
  };
  
  // Compute the coefficients and energies at the original FFT bins (every
  // fourth entry in "info"). 
  void ComputeCoarseInfo(const Vector<BaseFloat> &fft,
                         std::vector<InfoForBin> *info) const;


  // After the coarse-level info is computed using ComputeCoarseInfo, finds a
  // set of intermediate bin indexes to compute, that are the midpoints of
  // coarse-level bins.
  void FindCandidateBins(BaseFloat min_energy,
                         const std::vector<InfoForBin> &info,
                         std::vector<int32> *bins) const;

  void FindCandidateBins2(BaseFloat min_energy,
                          const std::vector<InfoForBin> &info,
                          std::vector<int32> *bins) const;

  
  void ComputeBinInfo(const VectorBase<BaseFloat> &signal,
                      int32 bin, InfoForBin *info) const;

  
  // For each bin b such that we have valid "info" data for bins b, b+1 and b+2,
  // does quadratic interpolation to find the maximum predicted energy in the
  // range [b, b+2].  The location of the maximum predicted energy is output to
  // "bin_out" and "offset_out", and the corresponding predicted energy is
  // returned.
  //
  // Note: if there are two different frequencies with similar maximum energies
  // (e.g. within a factor of probably around 1.2 or so), the fact that
  // OptimizeFrequency only returns one maximum may potentially lead to the
  // smaller maximum being output.  We could have modified this to output
  // multiple different maxima, which could have been more accurate in terms of
  // being guaranteed to output the best maximum, but this probably wouldn't
  // have a measurable impact on our application so we haven't bothered.
  BaseFloat OptimizeFrequency(
      const std::vector<InfoForBin> &info,
      int32 *bin_out,
      BaseFloat *offset_out) const;
  

  // This function does
  // (*cos)(t) = cos(2 pi t freq / samp_freq)
  // (*sin)(t) = sin(2 pi t freq / samp_freq)
  static void CreateCosAndSin(BaseFloat samp_freq,
                              BaseFloat freq,
                              VectorBase<BaseFloat> *cos,
                              VectorBase<BaseFloat> *sin);
  
  // Do fine optimization of the frequency within a bin, given a reasonable
  // approximate position within it based on interpolation (that should be close
  // to the optimum).
  void FineOptimizeFrequency(
      const VectorBase<BaseFloat> &signal,
      int32 bin,
      BaseFloat offset,
      std::vector<InfoForBin> *info,
      OptimizedInfo *opt_info) const;
  
  // Computes the coefficients cos_, sin_, and Minv_.
  void ComputeCoefficients();

  // Calls some self-testing code that prints warnings if
  // some of our assumptions were wrong.
  void SelfTest(const VectorBase<BaseFloat> &signal,
                const std::vector<InfoForBin> &info,
                BaseFloat final_freq,
                BaseFloat final_energy);

};



/**
   This configuration class is for the frame-by-frame detection of
   cases where there are one or two sinusoids that can explain
   a lot of the energy in the signal.
*/
struct MultiSinusoidDetectorConfig {

  // frame length in milliseconds
  BaseFloat frame_length_ms;
  // frame shift in milliseconds
  BaseFloat frame_shift_ms;

  // Proportion of the total energy of the signal that the quieter of
  // the two sinusoids must comprise, in order to be counted, if two
  // sinusoids are detected.
  BaseFloat two_freq_min_energy;

  // Proportion of the total energy of the signal that both sinusoids (if
  // two are detected) must comprise, in order to be output.
  BaseFloat two_freq_min_total_energy;

  // Proportion of the total energy of the signal that a single sinusoid
  // must comprise, in order to be output, if we are considering
  // reporting a single sinusoid.  Note: detection of two sinusoids
  // will take precedence over detection of a single sinusoid.
  BaseFloat one_freq_min_energy;

  // Lower end of frequency range that we consider; frequencies outside
  // this range are not candidates to appear in the detected output.
  BaseFloat min_freq;
  // Upper end of frequency range that we consider, see min_freq.
  BaseFloat max_freq;

  // Frequency to which we subsample the signal before processing it.
  // Must be integer because of how LinearResample code works.
  int32 subsample_freq;

  // Filter cut-off frequency used in sub-sampling.
  BaseFloat subsample_filter_cutoff;

  // the following is not critical and is not exported to the
  // command line.
  int32 subsample_filter_zeros;
  
  MultiSinusoidDetectorConfig():
      frame_length_ms(20), frame_shift_ms(10),
      two_freq_min_energy(0.2), two_freq_min_total_energy(0.6),
      one_freq_min_energy(0.75), min_freq(300.0),
      max_freq(1800.0), subsample_freq(4000),
      subsample_filter_cutoff(1900.0), subsample_filter_zeros(5) {}

  void Register(OptionsItf *po) {
    po->Register("frame-length", &frame_length_ms,
                 "Frame length in milliseconds");
    po->Register("frame-shift", &frame_shift_ms,
                 "Frame shift in milliseconds");
    po->Register("two-freq-min-energy", &two_freq_min_energy,
                 "For detecting two-frequency tones, minimum energy that "
                 "the quieter frequency must have (relative to total "
                 "enegy of frame)");
    po->Register("two-freq-min-total-energy", &two_freq_min_total_energy,
                 "For detecting two-frequency tones, minimum energy that "
                 "the two frequencies together must have (relative to total "
                 "energy of frame)");
    po->Register("one-freq-min-energy", &one_freq_min_energy, "For detecting "
                 "single-frequency tones, minimum energy that the frequency "
                 "must have relative to total energy of frame");
    po->Register("min-freq", &min_freq, "Minimum frequency of sinusoid that "
                 "will be detected");
    po->Register("max-freq", &min_freq, "Maximum frequency of sinusoid that "
                 "will be detected");
    po->Register("subsample-freq", &subsample_freq, "Frequency at which "
                 "we subsample the signal");
    po->Register("subsample-filter-cutoff", &subsample_filter_cutoff, "Filter "
                 "cut-off frequency used in subsampling");
  }
  void Check() const {
    KALDI_ASSERT(frame_length_ms > 0 && frame_length_ms >= frame_shift_ms &&
                 min_freq > 0 && max_freq > min_freq &&
                 subsample_filter_cutoff > max_freq &&
                 subsample_freq/2 > subsample_filter_cutoff &&
                 subsample_filter_zeros > 2 &&
                 subsample_filter_cutoff > 0.25 * subsample_freq &&
                 two_freq_min_total_energy > two_freq_min_energy &&
                 two_freq_min_energy <= 0.5 * two_freq_min_total_energy);
    BaseFloat samples_per_frame_shift =
        frame_shift_ms * 0.001 * subsample_freq;
    // The following assert ensures that the frame-shift is an exact
    // number of samples, so that the locations of the frames
    // don't gradually drift out of sync.
    KALDI_ASSERT(fabs(samples_per_frame_shift -
                      static_cast<int32>(samples_per_frame_shift)) <
                 0.001);
                      
  }             
};

struct MultiSinusoidDetectorOutput {
  BaseFloat tot_energy;  // Total energy per sample of this frame (sum-square of
                         // signal divided by number of samples... this is after
                         // downsampling and mean subtraction.
  BaseFloat freq1;  // Lower frequency detected, or 0 if none detected.
  BaseFloat energy1; // Energy of lower frequency divided by total energy, or 0
                     // if none detected.
  BaseFloat freq2;  // Lower frequency detected, or 0 if zero or one
                    // frequencies detected.
  BaseFloat energy2; // Energy of higher frequency divided by total energy, or 0
                     // if zero or one freqencies detected.
  MultiSinusoidDetectorOutput(): tot_energy(0.0), freq1(0.0),
                                 energy1(0.0), freq2(0.0), energy2(0.0) { }
};


class MultiSinusoidDetector {
 public:

  // Initialize sinusoid detector.  Sampling frequency must be integer.
  MultiSinusoidDetector(const MultiSinusoidDetectorConfig &config,
                        int32 sampling_freq);    

  /// This is how the class acccepts its input.  You can put the waveform in
  /// piece by piece, if it's an online application.
  void AcceptWaveform(const VectorBase<BaseFloat> &waveform);
  
  /// The user calls this to announce to the class that the waveform has ended;
  /// this forces any pending data to be flushed.
  void WaveformFinished();

  /// Resets the state of the class so you can start processing another waveform.
  void Reset(); 
  
  /// This returns true if the class currently has no more data ready to output.
  bool Done() const;

  /// Outputs the next frame of output to "frame", which must be non-NULL.
  /// It is an error to call this if Done() has returned true, or has not been
  /// checked.
  void GetNextFrame(MultiSinusoidDetectorOutput *output);

  BaseFloat FrameShiftSecs() const { return 0.001 * config_.frame_shift_ms; }

  BaseFloat SamplingFrequency() const { return sample_freq_; }
  
 private:
  // Gets the next frame of subsampled signal, and consumes the appropriate
  // amount of stored data.  It is an error to call this if Done() returned
  // true.
  void GetNextFrameOfSignal(Vector<BaseFloat> *frame);

  // returns true and sets freq1, freq1, energy1 and energy2 in "output" if we
  // successfully detected an acceptable two-frequency tone.
  bool DetectedTwoFrequency(BaseFloat signal_energy,
                            const Sinusoid &sinusoid1,
                            BaseFloat energy1,
                            const Sinusoid &sinusoid2,
                            BaseFloat energy2,
                            MultiSinusoidDetectorOutput *output);

  // returns true and sets freq1, freq1, energy1 and energy2 in "output" if we
  // successfully detected an acceptable one-frequency tone.
  bool DetectedOneFrequency(BaseFloat signal_energy,
                            const Sinusoid &sinusoid1,
                            BaseFloat energy1,
                            const Sinusoid &sinusoid2,
                            BaseFloat energy2,
                            MultiSinusoidDetectorOutput *output);
  
  
  // Returns std::min(max_samp, sum-of-samples-in-subsampled_signal_).
  // (the std::min is for efficiency so we don't have to visit the
  //  whole list).
  int32 NumSubsampledSamplesReady(int32 max_samp) const;
  
  MultiSinusoidDetectorConfig config_;
  int32 sample_freq_;
  int32 samples_per_frame_subsampled_;  // (samples per frame at subsampled
                                        // rate).
  int32 samples_shift_subsampled_;  // (samples per frame-shift at subsampled
                                    // rate).

  // True if the user has called WaveformFinished().
  bool waveform_finished_;
  
  // Pieces of the subsampled signal that are awaiting processing.
  // Normally there will be just one element here, but if someone calls
  // AcceptWaveform multiple times before getting output, there could
  // be more elements.  All of these pieces are nonempty.
  std::deque<Vector<BaseFloat>* > subsampled_signal_;

  // stores the number of samples consumed from the first member of
  // subsampled_signal_.  We will always have samples_consumed_ >= 0 and either
  // (subsampled_signal_.empty() && samples_consumed_ == 0) or
  // samples_consumed_ < subsampled_signal_[0]->Dim().
  int32 samples_consumed_;
  
  
  // This object is used to subsample the signal.
  LinearResample resampler_;

  // This object is used to detect sinusoids in the subsampled 
  // frames.
  SinusoidDetector detector_;
};

// Detect sinusoids.  Signal should be sampled at detector->SamplingFrequency().
void DetectSinusoids(const VectorBase<BaseFloat> &signal,
                     MultiSinusoidDetector *detector,
                     Matrix<BaseFloat> *output);





/// @} End of "addtogroup feat"
}  // namespace kaldi
#endif  // KALDI_FEAT_SINUSOID_DETECTION_H_
