// feat/feature-mcep.cc

// Copyright 2013  Arnab Ghoshal
//           2016  CereProc Ltd. (author: Blaise Potard)

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


#include "feat/feature-functions.h"
#include "matrix/matrix-functions.h"


namespace kaldi {

template<class Real>
void ComputePowerSpectrum(VectorBase<Real> *complex_fft) {
  int32 dim = complex_fft->Dim();
  // Letting it be non-power-of-two for now.
  // KALDI_ASSERT(dim > 0 && (dim & (dim-1) == 0));
  // We have in complex_fft the first half of complex spectrum
  // it's stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
  int32 half_dim = dim/2;
  // Handle special cases first
  Real first_energy = (*complex_fft)(0) * (*complex_fft)(0),
      last_energy = (*complex_fft)(1) * (*complex_fft)(1);
  for (int32 i = 1; i < half_dim; i++) {
    Real real = (*complex_fft)(i*2), im = (*complex_fft)(i*2 + 1);
    (*complex_fft)(i) = real*real + im*im;
  }
  (*complex_fft)(0) = first_energy;
  (*complex_fft)(half_dim) = last_energy;  // Will actually never be used, and
  // anyway if the signal has been bandlimited sensibly this should be zero.
}

template void ComputePowerSpectrum(VectorBase<float> *complex_fft);
template void ComputePowerSpectrum(VectorBase<double> *complex_fft);

template<class Real>
void PowerSpectrumToRealCepstrum(VectorBase<Real> *power_spectrum) {
  int32 dim = power_spectrum->Dim();
  int32 half_dim = dim/2;
  power_spectrum->Range(0, half_dim + 1).ApplyLog();
  power_spectrum->Range(0, half_dim + 1).Scale(0.5);  // square root
  // Now reconstruct the last N/2 - 1 elements of the symmetric spectrum that
  // correspond to the negative frequencies.
  Real nyquist_power = (*power_spectrum)(half_dim);
  for (int32 i = 1; i < half_dim; i++) {
    (*power_spectrum)(dim - 2*i) = (*power_spectrum)(half_dim - i);
    (*power_spectrum)(dim - 2*i + 1) = 0;
  }
  (*power_spectrum)(1) = nyquist_power;
  RealFft(power_spectrum, false);  // Doing IFFT.
  power_spectrum->Scale(1.0 / dim);
}

template void PowerSpectrumToRealCepstrum(VectorBase<float> *power_spectrum);
template void PowerSpectrumToRealCepstrum(VectorBase<double> *power_spectrum);


template<class Real>
void RealCepstrumToMagnitudeSpectrum(VectorBase<Real> *real_cepstrum,
                                     bool apply_exp) {
  int32 dim = real_cepstrum->Dim();
  int32 half_dim = dim/2;
  // Now reconstruct the last N/2 - 1 elements of the symmetric cepstrum that
  // correspond to the negative quefrencies.
  for (int32 i = 1; i < half_dim; i++)
    (*real_cepstrum)(dim-i) = (*real_cepstrum)(i);
  RealFft(real_cepstrum, true);
  Real last_spectrum = (*real_cepstrum)(1);
  for (int32 i = 1; i < half_dim; i++) {
    Real real = (*real_cepstrum)(i*2),
        im = (*real_cepstrum)(i*2 + 1);
    (*real_cepstrum)(i) = real;
    if (std::abs(im) > 1e-4) {
      (*real_cepstrum)(i) = 0.0;
      KALDI_WARN <<
        "FFT of real cepstrum not expected to have imaginary value.";
    }
  }
  (*real_cepstrum)(half_dim) = last_spectrum;
  if (apply_exp)
    real_cepstrum->Range(0, half_dim+1).ApplyExp();
}

template void RealCepstrumToMagnitudeSpectrum(VectorBase<float> *real_cepstrum,
                                              bool apply_exp);
template void RealCepstrumToMagnitudeSpectrum(VectorBase<double> *real_cepstrum,
                                              bool apply_exp);






}  // namespace kaldi
