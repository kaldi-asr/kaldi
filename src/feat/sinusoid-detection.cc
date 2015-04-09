// feat/sinusoid-detection.cc

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


#include "feat/sinusoid-detection.h"
#include "matrix/matrix-functions.h"
#include "feat/resample.h"

namespace kaldi {



// This function adds the given sinusoid to the signal, as:
// (*signal)(t) += amplitude * cos(2 pi freq/samp_freq t + phase).
void AddSinusoid(BaseFloat samp_freq,
                 const Sinusoid &sinusoid,
                 VectorBase<BaseFloat> *signal) {
  // treat "factor" as a complex variable equal to exp(i * 2 pi freq / samp_freq); it's
  // the factor by which we multiply on each frame.
  BaseFloat factor_real = cos(M_2PI * sinusoid.freq / samp_freq),
      factor_im = sin(M_2PI * sinusoid.freq / samp_freq);
  BaseFloat *signal_data = signal->Data();
  int32 dim = signal->Dim(), batch_size = 100;
  // process frames in batches of size "batch_size", after which we recompute
  // the starting point to prevent loss of accuracy due to drift.
  for (int32 b = 0; b * batch_size < dim; b++) {
    int32 t_offset = b * batch_size,
        t_end = std::min(dim, t_offset + batch_size);
    double phase = sinusoid.phase + M_2PI * t_offset * sinusoid.freq / samp_freq;
    // treat x as a complex variable which initially is equal to amplitude * exp(i * phase),
    // but which gets multiplied by "factor" on each frame.
    BaseFloat x_real = sinusoid.amplitude * cos(phase),
        x_im = sinusoid.amplitude * sin(phase);
    for (int32 t = t_offset; t < t_end; t++) {
      signal_data[t] += x_real;
      ComplexMul(factor_real, factor_im, &x_real, &x_im);  // x *= factor.
    }
  }
}


// static
void SinusoidDetector::QuadraticMaximizeEqualSpaced(
    BaseFloat y0, BaseFloat y1, BaseFloat y2,
    BaseFloat *x_max, BaseFloat *y_max) {
  // Let the function be y = a x^2 + b x + c, and
  // suppose we have the values of y(0), y(1) and y(2).
  // We have y0 = c, y1 = a + b + c, and y2 = 4a + 2b + c,
  // so c = y0.
  // Also, y2 - 2 y1 = 2a - c, so
  // a = (y2 - 2 y1 + c) / 2, and
  // b = y1 - a - c.
  BaseFloat c = y0, a = y2 - 2 * y1 + c, b = y1 - a - c;
  if (a >= 0) {
    // The maximum of the function will occur at one of the end points.
    if (y0 > y2) {
      *x_max = 0;
      *y_max = y0;
    } else {
      *x_max = 2;
      *y_max = y2;
    }
  } else {
    // derivative y' = 2a x + b.  y' = 0 at x = -b / 2 a.
    BaseFloat x = -b / (2.0 * a);
    if (x <= 0.0) {
      *x_max = 0;
      *y_max = y0;
    } else if (x >= 2.0) {
      *x_max = 0;
      *y_max = y2;
    } else {
      *x_max = x;
      *y_max = a * x * x + b * x + c;
    }
  }
}

// static
void SinusoidDetector::QuadraticMaximize(
    BaseFloat x1, BaseFloat y0, BaseFloat y1, BaseFloat y2,
    BaseFloat *x_max, BaseFloat *y_max) {
  // Let the function be y = a x^2 + b x + c, and
  // suppose we have the values of y(0), y(x1) and y(1),
  // where 0 < x1 < 1.
  // We have y0 = c, y1 = x1^2 a + x1 b + c, and y2 = a + b + c,
  // so c = y0.
  // Also,  x1.y2 - y1 =  a (x1 - x1^2) + (x1 - 1) c, so
  // a = ( (x1 y2 - y1) - (x1 - 1) c) / (x1 - x1^2), and
  // b = y2 - a - c.
  BaseFloat c = y0, 
      a = (x1 * y2 - y1 - (x1 - 1.0) * c) / (x1 - x1*x1),
      b = y2 - a - c;

  // TODO: remove these lines.
  AssertEqual(y1, a * x1 * x1 + b * x1 + c);
  AssertEqual(y2, a + b + c);

  if (a >= 0) {
    // The maximum of the function will occur at one of the end points.
    if (y0 > y2) {
      *x_max = 0;
      *y_max = y0;
    } else {
      *x_max = 1.0;
      *y_max = y2;
    }
  } else {
    // derivative y' = 2a x + b.  y' = 0 at x = -b / 2 a.
    BaseFloat x = -b / (2.0 * a);
    if (x <= 0.0) {
      *x_max = 0.0;
      *y_max = y0;
    } else if (x >= 1.0) {
      *x_max = 1.0;
      *y_max = y2;
    } else {
      *x_max = x;
      *y_max = a * x * x + b * x + c;
    }
  }
}

//static
BaseFloat SinusoidDetector::QuadraticInterpolate(
    BaseFloat x1, BaseFloat y0, BaseFloat y1, BaseFloat y2,
    BaseFloat x) {
  // Let the function be y = a x^2 + b x + c, and
  // suppose we have the values of y(0), y(x1) and y(1),
  // where 0 < x1 < 1.
  // We have y0 = c, y1 = x1^2 a + x1 b + c, and y2 = a + b + c,
  // so c = y0.
  // Also,  x1.y2 - y1 =  a (x1 - x1^2) + (x1 - 1) c, so
  // a = ( (x1 y2 - y1) - (x1 - 1) c) / (x1 - x1^2), and
  // b = y2 - a - c.
  KALDI_ASSERT(x1 >= 0.0 && x1 <= 1.0);
  if (x1 == 0.0) return y0;
  else if (x1 == 1.0) return y2;
  
  BaseFloat c = y0, 
      a = (x1 * y2 - y1 - (x1 - 1.0) * c) / (x1 - x1*x1),
      b = y2 - a - c;
  return a * x * x + b * x + c;
}

// This function does
// (*cos)(t) = cos(2 pi t freq / samp_freq)
// (*sin)(t) = sin(2 pi t freq / samp_freq)
//static
void SinusoidDetector::CreateCosAndSin(BaseFloat samp_freq,
                                        BaseFloat freq,
                                        VectorBase<BaseFloat> *cos_vec,
                                        VectorBase<BaseFloat> *sin_vec) {
  int32 dim = cos_vec->Dim(), batch_size = 100;
  KALDI_ASSERT(dim == sin_vec->Dim());
  BaseFloat *cos_data = cos_vec->Data(), *sin_data = sin_vec->Data();
  BaseFloat factor_real = cos(M_2PI * freq / samp_freq),
      factor_im = sin(M_2PI * freq / samp_freq);
  
  // process frames in batches of size "batch_size", after which we recompute
  // the starting point to prevent loss of accuracy due to drift.
  for (int32 b = 0; b * batch_size < dim; b++) {
    int32 t_offset = b * batch_size,
        t_end = std::min(dim, t_offset + batch_size);
    double phase = M_2PI * t_offset * freq / samp_freq;
    // treat x as a complex variable which initially is equal to amplitude * exp(i * phase),
    // but which gets multiplied by "factor" on each frame.
    BaseFloat x_real = cos(phase), x_im = sin(phase);
    for (int32 t = t_offset; t < t_end; t++) {
      cos_data[t] = x_real;
      sin_data[t] = x_im;
      ComplexMul(factor_real, factor_im, &x_real, &x_im);  // x *= factor.
    }
  }
}

SinusoidDetector::SinusoidDetector(BaseFloat samp_freq,
                                     int32 num_samp): 
    samp_freq_(samp_freq),
    num_samples_(num_samp),
    num_samples_padded_(RoundUpToNearestPowerOfTwo(num_samp)),
    fft_(num_samples_padded_),
    factor1_(3.1),
    factor2_(1.42) {
  ComputeCoefficients();
}

void SinusoidDetector::SelfTest(
    const VectorBase<BaseFloat> &signal,
    const std::vector<InfoForBin> &info,
    BaseFloat final_freq,
    BaseFloat final_energy) {
  int32 num_bins = num_samples_padded_ * 2 + 1;

  
  {
    BaseFloat cutoff = 0.0;
    for (int32 k = 0; k <= num_bins; k += 4)
      cutoff = std::max(cutoff, info[k].energy);
    BaseFloat energy_upper_bound = factor1_ * cutoff;
    if (final_energy > energy_upper_bound) {
      KALDI_WARN << "Self-testing failed [factor1]: " 
                 << final_energy << " > " << energy_upper_bound
                 << ", num-samples is " << num_samples_
                 << ", freq/nyquist = "
                 << (final_freq / (samp_freq_ * 0.5))
                 << "- would require factor1 >= "
                 << (final_energy / cutoff);
    }
  }
  {
    BaseFloat cutoff = 0.0;
    for (int32 k = 0; k <= num_bins; k += 2)
      if (info[k].valid)
        cutoff = std::max(cutoff, info[k].energy);
    BaseFloat energy_upper_bound = factor2_ * cutoff;
    if (final_energy > energy_upper_bound) {
      KALDI_WARN << "Self-testing failed [factor2]: " 
                 << final_energy << " > " << energy_upper_bound
                 << ", num-samples is " << num_samples_
                 << ", freq/nyquist = "
                 << (final_freq / (samp_freq_ * 0.5))
                 << "- would require factor2 >= "
                 << (final_energy / cutoff);
          
    }
  }
  
}


BaseFloat SinusoidDetector::OptimizeFrequency(
    const std::vector<InfoForBin> &info,
    int32 *bin_out,
    BaseFloat *offset_out) const {
  
  BaseFloat max_energy = 0.0;
  *bin_out = -1;
  int32 max_freq =  num_samples_padded_ * 2;

  // For each bin, we consider the frequency range [bin, bin+1, bin+2],
  // and if we have info for all those bins, do a quadratic interpolation to
  // find the maximum within the range.
  for (int32 bin = 0; bin + 2 <= max_freq; bin++) {
    if (info[bin].valid && info[bin+1].valid && info[bin+2].valid) {
      // First handle the left side of the bin.
      BaseFloat best_x, best_y;
      QuadraticMaximizeEqualSpaced(info[bin].energy, info[bin+1].energy,
                                   info[bin+2].energy, &best_x, &best_y);
      if (best_y > max_energy) {
        max_energy = best_y;
        if (best_x <= 1.0) {
          *bin_out = bin;
          *offset_out = best_x;
        } else {
          *bin_out = bin + 1;
          *offset_out = best_x - 1;
        }
      }
    }
  }
  return max_energy;
}


BaseFloat SinusoidDetector::DetectSinusoid(
    BaseFloat min_energy,
    const VectorBase<BaseFloat> &signal,
    Sinusoid *sinusoid) {
  if (signal(0) == 0.0 && signal.Norm(2.0) == 0.0)
    return 0.0;
  KALDI_ASSERT(signal.Dim() == num_samples_);
  Vector<BaseFloat> fft(num_samples_padded_);
  fft.Range(0, num_samples_).CopyFromVec(signal);
  bool forward = true;
  fft_.Compute(fft.Data(), forward);

  std::vector<InfoForBin> info;
  ComputeCoarseInfo(fft, &info);
  // we now have info for the "coarse" bins.

  // each element b of "bins" will be a multiple of 4: it's possible
  // that the best frequency is in the range [b, b+4]
  std::vector<int32> bins;
  FindCandidateBins(min_energy, info, &bins);

  if (bins.empty())
    return 0.0;  // not enough energy in signal.

  for (size_t i = 0; i < bins.size(); i++) {
    int32 bin = bins[i];
    ComputeBinInfo(signal, bin, &(info[bin]));
  }

  std::vector<int32> bins2;
  FindCandidateBins2(min_energy, info, &bins2);

  for (size_t i = 0; i < bins2.size(); i++) {
    int32 bin = bins2[i];
    ComputeBinInfo(signal, bin, &(info[bin]));
  }

  // compute energy for the predicted-optimum point, which will usually be
  // between bins, with an offset.
  int32 bin;
  BaseFloat offset;
  
  BaseFloat opt_energy = OptimizeFrequency(info,  &bin, &offset);

  if (opt_energy == 0.0)
    return 0.0;

  BaseFloat max_freq = (bin + offset) * samp_freq_ / (num_samples_padded_ * 4);
  
  KALDI_VLOG(4) << "Best frequency based on interpolation is "
                << max_freq << ", best energy is "
                << opt_energy << ", bin is " << bin;

  OptimizedInfo final_info;
  
  FineOptimizeFrequency(signal, bin, offset, &info, &final_info);

  // the following while loop will rarely be accessed.
  while (final_info.offset == 0.0 && bin > 0) {
    bin--;
    FineOptimizeFrequency(signal, bin, 1.0, &info, &final_info);
  }

  // the following while loop will rarely be accessed.  
  while (final_info.offset == 1.0 && bin < num_samples_padded_ * 2) {
    bin++;
    FineOptimizeFrequency(signal, bin, 0.0, &info, &final_info);
  }

  if (bin <= 1 || bin >= num_samples_padded_ * 2 - 2) {
    // If we're in the lowest or next-to-lowest bin, or the highest or
    // next-to-highest allowed bin (note, "bin" here is a range, and it can
    // never have the value num_samples_padded_ * 2), we tend to get more
    // estimation error than usual, so do another round of optimization.
    FineOptimizeFrequency(signal, bin, final_info.offset, &info, &final_info);    
  }
  
  BaseFloat final_freq = (final_info.bin + final_info.offset) * samp_freq_ / (num_samples_padded_ * 4);
  KALDI_VLOG(4) << "Final optimized info is: freq " << final_freq
                << ", cos coeff " << final_info.cos_coeff << ", sin coeff "
                << final_info.sin_coeff << ", energy " << final_info.energy;

  if (GetVerboseLevel() > 1)
    SelfTest(signal, info, final_freq, final_info.energy);

  if (final_info.energy >= min_energy) {
    sinusoid->amplitude = std::sqrt(final_info.cos_coeff * final_info.cos_coeff
                                    + final_info.sin_coeff * final_info.sin_coeff);
    sinusoid->freq = final_freq;
    sinusoid->phase = -std::atan2(final_info.sin_coeff, final_info.cos_coeff);
    KALDI_VLOG(4) << "Phase is " << sinusoid->phase << ", amplitude is "
                  << sinusoid->amplitude << ", freq is " << sinusoid->freq;
    return final_info.energy;
  } else {
    return 0.0;
  }
}


/*
  This function computes, the original FFT bins, the amount of energy in
  the signal that can be explained by a sinusoid at the corresponding frequency.

  Let f be the continuous-valued frequency.

  Define the vector C_f as
    C_f = [ c_0, c_1 ... c_n ]  where  c_k = cos(2 pi k f / samp_freq).   [obviously this notation depends on f].
  and S_f the same thing with sin in place of cos.

  Let the signal, as a vector, be V.
  We want to maximize the (positive) energy-difference:
       ||V||^2  - || V - c C_f - s S_f ||^2 
  where c and s are the coefficients of C_f and S_f.
  This quantity can be expanded as follows, where . means dot product.
   \delta E =    -c^2 C_f.C_f - s^2 S_f.S_f - 2 c s C_f.S_f  + 2 c V.C_f + 2 s V.S_f.
  which can be written as follows, where . means dot-product and ' means transpose:
    \delta E   =   2 [c s] v  -  [c s] M [c s]' 
  where M = [ C_f.C_f, C_f.S_f, C_f.S_f,  S_f.S_f ],
    and v = [V.C_f,  V.S_f].
  If M is invertible (i.e. for nonzero frequencies), this is maximized by
   [c s] = M^-1 v
  giving us the value.
    \delta E = v' M^{-1} v.
  We'll compute the inverse of M in advance, inside ComputeCoefficients(), using
  the formula [a b;c d]^-1 = 1/(ad - bc) [d -b; -c a] For zero frequency and at the
  Nyquist, M has the value [ a 0; 0 0 ], and we have the same type of expression
  limited to the first dim of v, i.e. Minv = [ a^{-1} 0; 0 0 ], a kind of pseudo-inverse.
 */

void SinusoidDetector::ComputeCoarseInfo(
    const Vector<BaseFloat> &fft,
    std::vector<InfoForBin> *info) const {
  info->resize(num_samples_padded_ * 2 + 1);  // 4 times resolution of FFT itself.

  const BaseFloat *fft_data = fft.Data();

  int32 num_bins = num_samples_padded_ / 2 + 1;
  for (int32 k = 0; k < num_bins; k++) {
    BaseFloat real, im;
    if (k == 0) {
      real = fft_data[0];
      im = 0.0;
    } else if (k == num_samples_padded_ / 2) {
      real = fft_data[1];
      im = 0.0;
    } else {
      real = fft_data[k * 2];
      im = fft_data[k * 2 + 1];
    }
    // v1 and v2 are the two components of the vector v in the math above.
    BaseFloat v1 = real, v2 = -im;
    // Minv_'s row indexes correspond to frequencies with 4 times more
    // resolution than the FFT bins.
    const BaseFloat *Minv_data = Minv_.RowData(k * 4);
    // The Matrix M^{-1} is of the form [a b; b d]
    BaseFloat a = Minv_data[0], b = Minv_data[1], d = Minv_data[2];
    // compute \delta E = v' M^{-1} v.
    BaseFloat delta_e = v1 * v1 * a + v2 * v2 * d + 2 * v1 * v2 * b;
    InfoForBin &this_info = (*info)[k * 4];
    this_info.valid = true;
    this_info.cos_dot = real;
    this_info.sin_dot = -im;
    this_info.energy = delta_e;
  }
}


void SinusoidDetector::ComputeCoefficients() {
  int32 num_samp = num_samples_;
  int32 num_freq =  num_samples_padded_ * 2 + 1;
  cos_.Resize(num_freq, num_samp);
  sin_.Resize(num_freq, num_samp);
  
  Vector<BaseFloat> cc(num_freq), cs(num_freq);
  for (int32 k = 0; k < num_freq; k++) {
    BaseFloat freq = k * samp_freq_ / (num_samples_padded_ * 4);
    SubVector<BaseFloat> c(cos_, k), s(sin_, k);
    CreateCosAndSin(samp_freq_, freq, &c, &s);
    cc(k) = VecVec(c, c);
    cs(k) = VecVec(c, s);
  }
  
  M_.Resize(num_freq, 3, kUndefined);  
  Minv_.Resize(num_freq, 3, kUndefined);
  
  for (int32 k = 0; k < num_freq; k++) {
    // Let the matrix M be [ a b; b d ].   [we don't write c because c == b].
    // We want to compute Minv_.
    BaseFloat a = cc(k), b = cs(k), d = num_samples_ - a;
    M_(k, 0) = a;
    M_(k, 1) = b;
    M_(k, 2) = d;
    if (k == 0 || k == num_freq - 1) {
      // this is a special case; it's not really the inverse of M but it will

      // give us the expression we want; it's like an inverse in just one dimension.
      Minv_(k, 0) = 1.0 / a;
      Minv_(k, 1) = 0.0;
      Minv_(k, 2) = 0.0;
    } else {
      BaseFloat inv_det = 1.0 / (a * d - b * b);
      // check for NaN and inf.
      KALDI_ASSERT(inv_det == inv_det && inv_det - inv_det == 0.0);
      // use: [a b;c d]^-1 = 1/(ad - bc) [d -b; -c a], special case where c = b.
      BaseFloat inv_a = d * inv_det, inv_b = -b * inv_det, inv_d = a * inv_det;
      Minv_(k, 0) = inv_a;
      Minv_(k, 1) = inv_b;
      Minv_(k, 2) = inv_d;
    }
  }
}


// Does fine optimization of the frequency within this bin; returns the
// final energy, the optimized frequency, and the cos and sin coefficients.
void SinusoidDetector::FineOptimizeFrequency(
    const VectorBase<BaseFloat> &signal,
    int32 bin,
    BaseFloat bin_offset,
    std::vector<InfoForBin> *info_in,
    OptimizedInfo *opt_info) const {
  std::vector<InfoForBin> &info = *info_in;
  if (!info[bin].valid) ComputeBinInfo(signal, bin, &(info[bin]));
  if (!info[bin+1].valid) ComputeBinInfo(signal, bin+1, &(info[bin+1]));
  
  const BaseFloat epsilon = 0.02, delta = 0.001;

  // If the offset is very close to the edges of the bin, move it
  // closer to the center.  Otherwise we may have problems with the
  // steps below.  The initial offset is only used as a starting point
  // anyway, so this won't affect the final value much.
  if (bin_offset < epsilon)
    bin_offset = epsilon;
  if (bin_offset > 1.0 - epsilon)
    bin_offset = 1.0 - epsilon;
  KALDI_VLOG(4) << "Initial bin offset = " << bin_offset << ", bin = " << bin;

  // create cos and sin waves of the specified frequency.
  BaseFloat freq = (bin + bin_offset) * samp_freq_ / (num_samples_padded_ * 4);
  Vector<BaseFloat> c(num_samples_, kUndefined), s(num_samples_, kUndefined);
  CreateCosAndSin(samp_freq_, freq, &c, &s);

  // these a, b and d values are the elements of the M matrix at this frequency
  // "freq", i.e. the matrix M_f [ a b; b d ].  This will be invertible because
  // we have ensured that the frequency is not too close to zero or the Nyquist.
  BaseFloat a = VecVec(c, c), b = VecVec(c, s), d = num_samples_ - a;
  BaseFloat inv_det = 1.0 / (a * d - b * b);
  BaseFloat inv_a = d * inv_det, inv_b = -b * inv_det, inv_d = a * inv_det;
  

  BaseFloat v1 = VecVec(c, signal), v2 = VecVec(s, signal);
  
  BaseFloat delta_e = v1 * v1 * inv_a + v2 * v2 * inv_d + 2 * v1 * v2 * inv_b;
  
  KALDI_VLOG(4) << "Actual energy-change at frequency " << freq << " is "
                << delta_e;
  // "freq" is frequency somewhere in the middle of the bin.
  
  BaseFloat final_offset, final_energy;
  QuadraticMaximize(bin_offset, info[bin].energy, delta_e, info[bin+1].energy,
                    &final_offset, &final_energy);

  KALDI_VLOG(4) << "After further optimizing, offset was " << final_offset
                << " giving freq "
                << ((bin+final_offset) * samp_freq_ / (num_samples_padded_*4))
                << ", with energy " << final_energy;

  // Use interpolation (using a quadratic function) to get the entries of the M matrix
  // the the final, tuned frequency.  Interpolation on M is better than M^{-1}, as its
  // elements are much better behaved as the frequency varies.
  const BaseFloat *M_left_data = M_.RowData(bin),
      *M_right_data = M_.RowData(bin + 1);

  BaseFloat a_interp = QuadraticInterpolate(bin_offset, M_left_data[0], a, M_right_data[0],
                                            final_offset);
  BaseFloat b_interp = QuadraticInterpolate(bin_offset, M_left_data[1], b, M_right_data[1],
                                            final_offset);
  BaseFloat d_interp = QuadraticInterpolate(bin_offset, M_left_data[2], d, M_right_data[2],
                                            final_offset);

  // Now get the inverse of the M matrix at the final point.
  BaseFloat a_inv_interp, b_inv_interp, d_inv_interp;
  
  if ((bin == 0 && final_offset < delta) ||
      (bin == num_samples_padded_ * 2 && final_offset > 1.0 - delta)) {
    // If we're extremely close to zero or the Nyquist, we'll have trouble
    // inverting M; just invert in the 1st dimension (only have a cos
    // component).
    a_inv_interp = 1.0 / a_interp;
    b_inv_interp = 0.0;
    d_inv_interp = 0.0;
  } else {
    BaseFloat inv_det = 1.0 / (a_interp * d_interp - b_interp * b_interp);
    // check for NaN and inf.
    KALDI_ASSERT(inv_det == inv_det && inv_det - inv_det == 0.0);
    // use: [a b;c d]^-1 = 1/(ad - bc) [d -b; -c a], special case where c = b.
    a_inv_interp = d_interp * inv_det;
    b_inv_interp = -b_interp * inv_det;
    d_inv_interp = a_interp * inv_det;
  }

  BaseFloat v1_interp = QuadraticInterpolate(bin_offset, info[bin].cos_dot, v1,
                                             info[bin+1].cos_dot, final_offset);
  BaseFloat v2_interp = QuadraticInterpolate(bin_offset, info[bin].sin_dot, v2,
                                             info[bin+1].sin_dot, final_offset);
  
  opt_info->bin = bin;
  opt_info->offset = final_offset;
  // Recompute the energy-reduction using the more accurate interpolated values of
  // v1 and v2 (the dot-products of the cos and sin with the signal), and
  // of M.
  opt_info->energy = v1_interp * v1_interp * a_inv_interp +
      v2_interp * v2_interp * d_inv_interp +
      2 * v1_interp * v2_interp * b_inv_interp;
  // Compute the coefficients of the cos and sin in the optimal sinusoid, as
  // M^{-1} v.
  opt_info->cos_coeff = a_inv_interp * v1_interp + b_inv_interp * v2_interp;
  opt_info->sin_coeff = b_inv_interp * v1_interp + d_inv_interp * v2_interp;  
}

void SinusoidDetector::FindCandidateBins(
    BaseFloat min_energy,
    const std::vector<InfoForBin> &info,
    std::vector<int32> *bins) const {

  int32 max_bin = num_samples_padded_ * 2;

  BaseFloat cutoff = min_energy;
  for (int32 k = 0; k <= max_bin; k += 4) {
    KALDI_ASSERT(info[k].valid);
    cutoff = std::max(cutoff, info[k].energy);
  }
  
  for (int32 k = 0; k < max_bin; k += 4) {
    BaseFloat energy_upper_bound =
        factor1_ * std::max(info[k].energy,
                            info[k+4].energy);
    if (energy_upper_bound >= cutoff)
      bins->push_back(k + 2);
  }
}


void SinusoidDetector::FindCandidateBins2(
    BaseFloat min_energy,
    const std::vector<InfoForBin> &info,
    std::vector<int32> *bins2) const {

  int32 max_bin = num_samples_padded_ * 2;
  
  BaseFloat cutoff = min_energy;
  for (int32 k = 0; k <= max_bin; k += 2) {
    if (info[k].valid)
      cutoff = std::max(cutoff, info[k].energy);
  }

  for (int32 k = 0; k < max_bin; k += 2) {  
    if (info[k].valid && info[k+2].valid) {
      BaseFloat energy_upper_bound =
          factor2_ * std::max(info[k].energy,
                              info[k+2].energy);
      if (energy_upper_bound >= cutoff)
        bins2->push_back(k + 1);
    }
  }
}
      

void SinusoidDetector::ComputeBinInfo(
    const VectorBase<BaseFloat> &signal,
    int32 bin,
    InfoForBin *info) const {
  KALDI_ASSERT(!info->valid);  // or wasted time.
  info->valid = true;
  BaseFloat v1 = info->cos_dot = VecVec(cos_.Row(bin), signal);
  BaseFloat v2 = info->sin_dot = VecVec(sin_.Row(bin), signal);
  const BaseFloat *Minv_data = Minv_.RowData(bin);
  BaseFloat a = Minv_data[0], b = Minv_data[1], d = Minv_data[2];
  // compute \delta E = v' M^{-1} v.
  BaseFloat delta_e = v1 * v1 * a + v2 * v2 * d + 2 * v1 * v2 * b;
  info->energy = delta_e;
}


MultiSinusoidDetector::MultiSinusoidDetector(
    const MultiSinusoidDetectorConfig &config,
    int32 sampling_freq):
    config_(config),
    sample_freq_(sampling_freq),
    samples_per_frame_subsampled_(0.001 * config.frame_length_ms *
                                  static_cast<BaseFloat>(config.subsample_freq)),
    samples_shift_subsampled_(0.001 * config.frame_shift_ms *
                              static_cast<BaseFloat>(config.subsample_freq)),
    waveform_finished_(false),
    samples_consumed_(0),
    resampler_(sampling_freq, config.subsample_freq,
               config.subsample_filter_cutoff, config.subsample_filter_zeros),
    detector_(config.subsample_freq, samples_per_frame_subsampled_) {
  config.Check();
}


void MultiSinusoidDetector::Reset() {
  waveform_finished_ = false;
  samples_consumed_ = 0;
  while(!subsampled_signal_.empty()) {
    delete subsampled_signal_.front();
    subsampled_signal_.pop_front();
  }
  resampler_.Reset();
}

void MultiSinusoidDetector::WaveformFinished() {
  KALDI_ASSERT(!waveform_finished_ &&
               "WaveformFinished() called twice.");

  Vector<BaseFloat> empty_waveform;
  subsampled_signal_.push_back(new Vector<BaseFloat>());
  bool flush = true;
  resampler_.Resample(empty_waveform, flush,
                      subsampled_signal_.back());
  waveform_finished_ = true;
  if (subsampled_signal_.back()->Dim() == 0) {
    delete subsampled_signal_.back();
    subsampled_signal_.pop_back();
  }
}

void MultiSinusoidDetector::AcceptWaveform(
    const VectorBase<BaseFloat> &waveform) {


  subsampled_signal_.push_back(new Vector<BaseFloat>());
  bool flush = false;
  resampler_.Resample(waveform, flush,
                      subsampled_signal_.back());
  if (subsampled_signal_.back()->Dim() == 0) {
    delete subsampled_signal_.back();
    subsampled_signal_.pop_back();
  }
}

int32 MultiSinusoidDetector::NumSubsampledSamplesReady(int32 max_samp) const {
  KALDI_ASSERT(samples_consumed_ >= 0 &&
               ((subsampled_signal_.empty() && samples_consumed_ == 0) ||
                (!subsampled_signal_.empty () && samples_consumed_ <
                 subsampled_signal_[0]->Dim())));
      
  int32 ans = -samples_consumed_;
  for (size_t i = 0; i < subsampled_signal_.size(); i++) {
    ans += subsampled_signal_[i]->Dim();
    if (ans > max_samp) break;
  }
  KALDI_ASSERT(ans >= 0);
  return std::min(ans, max_samp);
}

bool MultiSinusoidDetector::Done() const {
  int32 samp_ready = NumSubsampledSamplesReady(samples_per_frame_subsampled_);
  if ((samp_ready >= samples_per_frame_subsampled_ && !waveform_finished_) ||
      (samp_ready > 0 && waveform_finished_))
    return false;
  else
    return true;
}

void MultiSinusoidDetector::GetNextFrameOfSignal(Vector<BaseFloat> *frame) {
  frame->Resize(samples_per_frame_subsampled_, kUndefined);

  int32 sample_offset = 0,
      samples_needed = samples_per_frame_subsampled_;
  while (samples_needed > 0 &&
         !subsampled_signal_.empty()) {
    Vector<BaseFloat> *src = subsampled_signal_.front();
    int32 num_samples_avail = src->Dim() - samples_consumed_;
    KALDI_ASSERT(num_samples_avail > 0);
    int32 chunk_size = std::min(num_samples_avail,
                                 samples_needed);
    frame->Range(sample_offset, chunk_size).CopyFromVec(
        src->Range(samples_consumed_, chunk_size));
    sample_offset += chunk_size;
    samples_needed -= chunk_size;
    samples_consumed_ += chunk_size;
    if (samples_consumed_ == src->Dim()) {
      samples_consumed_ = 0;
      delete src;
      subsampled_signal_.pop_front();
    }
  }
  if (samples_needed > 0) {
    KALDI_ASSERT(waveform_finished_ && sample_offset > 0);  // or code error.
    frame->Range(sample_offset, samples_needed).SetZero();
  }
}


void MultiSinusoidDetector::GetNextFrame(MultiSinusoidDetectorOutput *output) {
  Vector<BaseFloat> frame;
  GetNextFrameOfSignal(&frame);
  // Mean subtraction
  frame.Add(-1.0 * frame.Sum() / frame.Dim());
  *output = MultiSinusoidDetectorOutput();  // reset to default.

  BaseFloat signal_energy = VecVec(frame, frame);
  output->tot_energy = signal_energy / frame.Dim();
  if (signal_energy == 0.0) return;

  // min_energy1 is the lowest energy we might care about.
  BaseFloat min_energy1 = signal_energy * 
      std::min<BaseFloat>(config_.two_freq_min_total_energy * 0.5,
                          config_.one_freq_min_energy);

  Sinusoid sinusoid1;
  BaseFloat energy1 = detector_.DetectSinusoid(min_energy1,
                                                      frame,
                                                      &sinusoid1);

  if (energy1 == 0.0) return;  // Nothing detected.

  // we only care about the 2nd sinusoid if
  // energy1 + energy2 >= signal_energy * two_freq_min_total_energy,
  // and energy2 >= signal_energy * config.two_freq_min_energy.

  BaseFloat min_energy2 =
      std::max(signal_energy * config_.two_freq_min_energy,
               signal_energy * config_.two_freq_min_total_energy
               - energy1);

  BaseFloat energy2;
  Sinusoid sinusoid2;

  // If there is enough energy left in the signal that we could
  // possibly detect a sinusoid of energy at least min_energy2...
  if (min_energy2 <= signal_energy - energy1) {
    sinusoid1.phase += M_PI;  // reverse the phase.
    AddSinusoid(config_.subsample_freq, sinusoid1, &frame);


    energy2 = detector_.DetectSinusoid(min_energy2,
                                              frame,
                                              &sinusoid2);

    if (energy2 > energy1) {
      // The following is just for our information, so we are aware
      // when the sinusoid detection gives us the non-optimal sinusoid
      // first.
      BaseFloat factor = energy2 / energy1;
      KALDI_VLOG(2) << "Second sinusoid greater than first by a factor of "
                    << factor << ".  (This means sinusoid detection is not "
                    << " working ideally).";
    }
    
    if (DetectedTwoFrequency(signal_energy,
                             sinusoid1, energy1,
                             sinusoid2, energy2,
                             output))
      return;
  } else {
    energy2 = 0.0;
  }
  // We don't need the return status of the following; we just return anyway.
  DetectedOneFrequency(signal_energy,
                       sinusoid1, energy1,
                       sinusoid2, energy2,
                       output);
}

// acceptable two-frequency tone.
bool MultiSinusoidDetector::DetectedTwoFrequency(
    BaseFloat signal_energy,
    const Sinusoid &sinusoid1,
    BaseFloat energy1,
    const Sinusoid &sinusoid2,
    BaseFloat energy2,
    MultiSinusoidDetectorOutput *output) {

  if (energy1 + energy2 >= signal_energy *
      config_.two_freq_min_total_energy &&
      std::min(energy1, energy2) >= signal_energy *
      config_.two_freq_min_energy &&
      std::min(sinusoid1.freq, sinusoid2.freq) >= config_.min_freq &&
      std::max(sinusoid1.freq, sinusoid2.freq) <= config_.max_freq) {
    output->freq1 = sinusoid1.freq;
    output->energy1 = energy1 / signal_energy;
    output->freq2 = sinusoid2.freq;
    output->energy2 = energy2 / signal_energy;
    if (output->freq1 > output->freq2) {
      std::swap(output->freq1, output->freq2);
      std::swap(output->energy1, output->energy2);
    }
    return true;
  } else {
    return false;
  }
}


// acceptable two-frequency tone.
bool MultiSinusoidDetector::DetectedOneFrequency(
    BaseFloat signal_energy,
    const Sinusoid &sinusoid1,
    BaseFloat energy1,
    const Sinusoid &sinusoid2,
    BaseFloat energy2,
    MultiSinusoidDetectorOutput *output) {
  // If sinusoid detection were performing exactly to spec, we could assume
  // energy1 >= energy2, but we don't assume this as it's not guaranteed.
  if (energy1 > energy2 && energy1 > signal_energy *
      config_.one_freq_min_energy &&
      sinusoid1.freq >= config_.min_freq &&
      sinusoid1.freq <= config_.max_freq) {
    output->freq1 = sinusoid1.freq;
    output->energy1 = energy1 / signal_energy;
    output->freq2 = 0.0;
    output->energy2 = 0.0;
    return true;
  } else if (energy2 > energy1 && energy2 > signal_energy *
             config_.one_freq_min_energy &&
             sinusoid2.freq >= config_.min_freq &&
             sinusoid2.freq <= config_.max_freq) {
    output->freq1 = sinusoid2.freq;
    output->energy1 = energy2 / signal_energy;
    output->freq2 = 0.0;
    output->energy2 = 0.0;
    return true;
  } else {
    return false;
  }
}


void DetectSinusoids(const VectorBase<BaseFloat> &signal,
                     MultiSinusoidDetector *detector,
                     Matrix<BaseFloat> *output) {
  std::vector<MultiSinusoidDetectorOutput> output_vec;
  detector->AcceptWaveform(signal);
  detector->WaveformFinished();

  int32 safety_margin = 10, approx_num_frames = safety_margin + 
      (signal.Dim() / (detector->SamplingFrequency() *
                       detector->FrameShiftSecs()));
  output_vec.reserve(approx_num_frames);
  while (!detector->Done()) {
    output_vec.resize(output_vec.size() + 1);
    detector->GetNextFrame(&(output_vec.back()));
  }  
  detector->Reset();
  if (output_vec.empty()) {
    output->Resize(0, 0);
  } else {
    output->Resize(output_vec.size(), 5, kUndefined);
    for (int32 i = 0; i < output->NumRows(); i++) {
      BaseFloat *row_data = output->RowData(i);
      MultiSinusoidDetectorOutput &this_output = output_vec[i];
      row_data[0] = this_output.tot_energy;
      row_data[1] = this_output.freq1;
      row_data[2] = this_output.energy1;
      row_data[3] = this_output.freq2;
      row_data[4] = this_output.energy2;
    }
  }
}


}  // namespace kaldi

