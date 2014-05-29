// feat/feature-functions.h

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Microsoft Corporation
//                2014  IMSL, PKU-HKUST (author: Wei Shi)

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


#ifndef KALDI_FEAT_FEATURE_FUNCTIONS_H_
#define KALDI_FEAT_FEATURE_FUNCTIONS_H_

#include <string>
#include <vector>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/mel-computations.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{


struct MelBanksOptions {
  int32 num_bins;  // e.g. 25; number of triangular bins
  BaseFloat low_freq;  // e.g. 20; lower frequency cutoff
  BaseFloat high_freq;  // an upper frequency cutoff; 0 -> no cutoff, negative
  // ->added to the Nyquist frequency to get the cutoff.
  BaseFloat vtln_low;  // vtln lower cutoff of warping function.
  BaseFloat vtln_high;  // vtln upper cutoff of warping function: if negative, added
                        // to the Nyquist frequency to get the cutoff.
  bool debug_mel;
  // htk_mode is a "hidden" config, it does not show up on command line.
  // Enables more exact compatibibility with HTK, for testing purposes.  Affects
  // mel-energy flooring and reproduces a bug in HTK.
  bool htk_mode;
  explicit MelBanksOptions(int num_bins = 25)
      : num_bins(num_bins), low_freq(20), high_freq(0), vtln_low(100),
        vtln_high(-500), debug_mel(false), htk_mode(false) {}

  void Register(OptionsItf *po) {
    po->Register("num-mel-bins", &num_bins,
                 "Number of triangular mel-frequency bins");
    po->Register("low-freq", &low_freq,
                 "Low cutoff frequency for mel bins");
    po->Register("high-freq", &high_freq,
                 "High cutoff frequency for mel bins (if < 0, offset from Nyquist)");
    po->Register("vtln-low", &vtln_low,
                 "Low inflection point in piecewise linear VTLN warping function");
    po->Register("vtln-high", &vtln_high,
                 "High inflection point in piecewise linear VTLN warping function"
                 " (if negative, offset from high-mel-freq");
    po->Register("debug-mel", &debug_mel,
                 "Print out debugging information for mel bin computation");
  }
};


struct FrameExtractionOptions {
  BaseFloat samp_freq;
  BaseFloat frame_shift_ms;  // in milliseconds.
  BaseFloat frame_length_ms;  // in milliseconds.
  BaseFloat dither;  // Amount of dithering, 0.0 means no dither.
  BaseFloat preemph_coeff;  // Preemphasis coefficient.
  bool remove_dc_offset;  // Subtract mean of wave before FFT.
  std::string window_type;  // e.g. Hamming window
  bool round_to_power_of_two;
  bool snip_edges;
  // Maybe "hamming", "rectangular", "povey", "hanning"
  // "povey" is a window I made to be similar to Hamming but to go to zero at the
  // edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85)
  // I just don't think the Hamming window makes sense as a windowing function.
  FrameExtractionOptions():
      samp_freq(16000),
      frame_shift_ms(10.0),
      frame_length_ms(25.0),
      dither(1.0),
      preemph_coeff(0.97),
      remove_dc_offset(true),
      window_type("povey"),
      round_to_power_of_two(true),
      snip_edges(true){ }

  void Register(OptionsItf *po) {
    po->Register("sample-frequency", &samp_freq,
                 "Waveform data sample frequency (must match the waveform file, "
                 "if specified there)");
    po->Register("frame-length", &frame_length_ms, "Frame length in milliseconds");
    po->Register("frame-shift", &frame_shift_ms, "Frame shift in milliseconds");
    po->Register("preemphasis-coefficient", &preemph_coeff,
                 "Coefficient for use in signal preemphasis");
    po->Register("remove-dc-offset", &remove_dc_offset,
                 "Subtract mean from waveform on each frame");
    po->Register("dither", &dither, "Dithering constant (0.0 means no dither)");
    po->Register("window-type", &window_type, "Type of window "
                 "(\"hamming\"|\"hanning\"|\"povey\"|\"rectangular\")");
    po->Register("round-to-power-of-two", &round_to_power_of_two,
                 "If true, round window size to power of two.");
    po->Register("snip-edges", &snip_edges,
                 "If true, end effects will be handled by outputting only frames that "
                 "completely fit in the file, and the number of frames depends on the "
                 "frame-length.  If false, the number of frames depends only on the "
                 "frame-shift, and we reflect the data at the ends.");
  }
  int32 WindowShift() const {
    return static_cast<int32>(samp_freq * 0.001 * frame_shift_ms);
  }
  int32 WindowSize() const {
    return static_cast<int32>(samp_freq * 0.001 * frame_length_ms);
  }
  int32 PaddedWindowSize() const {
    return (round_to_power_of_two ? RoundUpToNearestPowerOfTwo(WindowSize()) :
                                    WindowSize());
  }
};


struct FeatureWindowFunction {
  FeatureWindowFunction() {}
  explicit FeatureWindowFunction(const FrameExtractionOptions &opts);
  Vector<BaseFloat> window;
};

int32 NumFrames(int32 wave_length,
                const FrameExtractionOptions &opts);

void Dither(VectorBase<BaseFloat> *waveform, BaseFloat dither_value);

void Preemphasize(VectorBase<BaseFloat> *waveform, BaseFloat preemph_coeff);


// ExtractWindow extracts a windowed frame of waveform with a power-of-two,
// padded size. If log_energy_pre_window != NULL, outputs the log of the
// sum-of-squared samples before preemphasis and windowing
void ExtractWindow(const VectorBase<BaseFloat> &wave,
                   int32 f,  // with 0 <= f < NumFrames(wave.Dim(), opts)
                   const FrameExtractionOptions &opts,
                   const FeatureWindowFunction &window_function,
                   Vector<BaseFloat> *window,
                   BaseFloat *log_energy_pre_window = NULL);

// ExtractWaveformRemainder is useful if the waveform is coming in segments.
// It extracts the bit of the waveform at the end of this block that you
// would have to append the next bit of waveform to, if you wanted to have
// the same effect as everything being in one big block.
void ExtractWaveformRemainder(const VectorBase<BaseFloat> &wave,
                              const FrameExtractionOptions &opts,
                              Vector<BaseFloat> *wave_remainder);



// ComputePowerSpectrum converts a complex FFT (as produced by the FFT
// functions in matrix/matrix-functions.h), and converts it into
// a power spectrum.  If the complex FFT is a vector of size n (representing
// half the complex FFT of a real signal of size n, as described there),
// this function computes in the first (n/2) + 1 elements of it, the
// energies of the fft bins from zero to the Nyquist frequency.  Contents of the
// remaining (n/2) - 1 elements are undefined at output.
void ComputePowerSpectrum(VectorBase<BaseFloat> *complex_fft);



inline void MaxNormalizeEnergy(Matrix<BaseFloat> *feats) {
  // Just subtract the largest energy value... assume energy is the first
  // column of the mfcc features.  Don't do the flooring of energy (dithering
  // should prevent exact zeros).
  // We didn't put this in the main MFCC computation as we wanted to make sure
  // it is stateless (so we can do it bit by bit for large waveforms).
  // not compatible with the order_as_htk_ option in MfccOptions.
  SubMatrix<BaseFloat> energy(*feats, 0, feats->NumRows(), 0, 1);
  energy.Add(-energy.Max());
}





struct DeltaFeaturesOptions {
  int32 order;
  int32 window;  // e.g. 2; controls window size (window size is 2*window + 1)
  // the behavior at the edges is to replicate the first or last frame.
  // this is not configurable.

  DeltaFeaturesOptions(int32 order = 2, int32 window = 2):
      order(order), window(window) { }
  void Register(OptionsItf *po) {
    po->Register("delta-order", &order, "Order of delta computation");
    po->Register("delta-window", &window,
                 "Parameter controlling window for delta computation (actual window"
                 " size for each delta order is 1 + 2*delta-window-size)");
  }
};

class DeltaFeatures {
 public:
  // This class provides a low-level function to compute delta features.
  // The function takes as input a matrix of features and a frame index
  // that it should compute the deltas on.  It puts its output in an object
  // of type VectorBase, of size (original-feature-dimension) * (opts.order+1).
  // This is not the most efficient way to do the computation, but it's
  // state-free and thus easier to understand

  explicit DeltaFeatures(const DeltaFeaturesOptions &opts);

  void Process(const MatrixBase<BaseFloat> &input_feats,
               int32 frame,
               SubVector<BaseFloat> *output_frame) const;
 private:
  DeltaFeaturesOptions opts_;
  std::vector<Vector<BaseFloat> > scales_;  // a scaling window for each
  // of the orders, including zero: multiply the features for each
  // dimension by this window.
};

struct ShiftedDeltaFeaturesOptions {
  int32 window,           // The time delay and advance
        num_blocks,
        block_shift;      // Distance between consecutive blocks

  ShiftedDeltaFeaturesOptions():
      window(1), num_blocks(7), block_shift(3) { }
  void Register(OptionsItf *po) {
    po->Register("delta-window", &window, "Size of delta advance and delay.");
    po->Register("num-blocks", &num_blocks, "Number of delta blocks in advance"
                 " of each frame to be concatenated");
    po->Register("block-shift", &block_shift, "Distance between each block");
  }
};

class ShiftedDeltaFeatures {
 public:
  // This class provides a low-level function to compute shifted
  // delta cesptra (SDC).
  // The function takes as input a matrix of features and a frame index
  // that it should compute the deltas on.  It puts its output in an object
  // of type VectorBase, of size original-feature-dimension + (1  * num_blocks).

  explicit ShiftedDeltaFeatures(const ShiftedDeltaFeaturesOptions &opts);

  void Process(const MatrixBase<BaseFloat> &input_feats,
               int32 frame,
               SubVector<BaseFloat> *output_frame) const;
 private:
  ShiftedDeltaFeaturesOptions opts_;
  Vector<BaseFloat> scales_;  // a scaling window for each

};

// ComputeDeltas is a convenience function that computes deltas on a feature
// file.  If you want to deal with features coming in bit by bit you would have
// to use the DeltaFeatures class directly, and do the computation frame by
// frame.  Later we will have to come up with a nice mechanism to do this for
// features coming in.
void ComputeDeltas(const DeltaFeaturesOptions &delta_opts,
                   const MatrixBase<BaseFloat> &input_features,
                   Matrix<BaseFloat> *output_features);

// ComputeShiftedDeltas computes deltas from a feature file by applying
// ShiftedDeltaFeatures over the frames. This function is provided for
// convenience, however, ShiftedDeltaFeatures can be used directly.
void ComputeShiftedDeltas(const ShiftedDeltaFeaturesOptions &delta_opts,
                   const MatrixBase<BaseFloat> &input_features,
                   Matrix<BaseFloat> *output_features);

// SpliceFrames will normally be used together with LDA.
// It splices frames together to make a window.  At the
// start and end of an utterance, it duplicates the first
// and last frames.
// Will throw if input features are empty.
// left_context and right_context must be nonnegative.
// these both represent a number of frames (e.g. 4, 4 is
// a good choice).
void SpliceFrames(const MatrixBase<BaseFloat> &input_features,
                  int32 left_context,
                  int32 right_context,
                  Matrix<BaseFloat> *output_features);

// ReverseFrames reverses the frames in time (used for backwards decoding)
void ReverseFrames(const MatrixBase<BaseFloat> &input_features,
                  Matrix<BaseFloat> *output_features);

class MelBanks;

void GetEqualLoudnessVector(const MelBanks &mel_banks,
                            Vector<BaseFloat> *ans);


void InitIdftBases(int32 n_bases, int32 dimension, Matrix<BaseFloat> *mat_out);


// Compute LP coefficients from autocorrelation coefficients.
BaseFloat ComputeLpc(const VectorBase<BaseFloat> &autocorr_in,
                     Vector<BaseFloat> *lpc_out);


struct SlidingWindowCmnOptions {
  int cmn_window;
  int min_window;
  bool normalize_variance;
  bool center;

  SlidingWindowCmnOptions():
      cmn_window(600),
      min_window(100),
      normalize_variance(false),
      center(false) { }

  void Register(OptionsItf *po) {
    po->Register("cmn-window", &cmn_window, "Window in frames for running "
                 "average CMN computation");
    po->Register("min-cmn-window", &min_window, "Minimum CMN window "
                 "used at start of decoding (adds latency only at start). "
                 "Only applicable if center == false, ignored if center==true");
    po->Register("norm-vars", &normalize_variance, "If true, normalize "
                 "variance to one."); // naming this as in apply-cmvn.cc
    po->Register("center", &center, "If true, use a window centered on the "
                 "current frame (to the extent possible, modulo end effects)."
                 "If false, window is to the left.");
  }
  void Check() const;
};


/// Applies sliding-window cepstral mean and/or variance normalization.  See the
/// strings registering the options in the options class for information on how
/// this works and what the options are.  input and output must have the same
/// dimension.
void SlidingWindowCmn(const SlidingWindowCmnOptions &opts,
                      const MatrixBase<BaseFloat> &input,
                      MatrixBase<BaseFloat> *output);


/// @} End of "addtogroup feat"
}  // namespace kaldi



#endif  // KALDI_FEAT_FEATURE_FUNCTIONS_H_
