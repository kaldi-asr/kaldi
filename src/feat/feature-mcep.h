// feat/feature-mcep.h

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


#ifndef KALDI_FEAT_FEATURE_MCEP_H_
#define KALDI_FEAT_FEATURE_MCEP_H_

#include <string>
#include <vector>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{


// ComputePowerSpectrum converts a complex FFT (as produced by the FFT
// functions in matrix/matrix-functions.h), and converts it into
// a power spectrum.  If the complex FFT is a vector of size n (representing
// half the complex FFT of a real signal of size n, as described there),
// this function computes in the first (n/2) + 1 elements of it, the
// energies of the fft bins from zero to the Nyquist frequency.  Contents of the
// remaining (n/2) - 1 elements are undefined at output.
template<class Real> void ComputePowerSpectrum(VectorBase<Real> *complex_fft);


/// PowerSpecToRealCeps takes the output of ComputePowerSpectrum
/// (which is a squared magnitude spectrum) and produces the real
/// cepstrum coefficients.  If the power spectrum corresponds to an
/// N-point FFT, then only the first N/2 + 1 elements of it are
/// calculated by ComputePowerSpectrum. This function recreates the
/// symmetric log-magnitude spectrum by filling in the remaining N/2 - 1
/// values and computes an IFFT. The IFFT computation happens
/// "in-place" and the first N/2 + 1 elements of the vector are the
/// real part of the cepstral coefficients (the imaginary parts should
/// be 0 anyway, since the power spectrum is symmetric/even). Contents
/// of the last last N/2 - 1 elements are undefined at output, as they
/// are not needed (as real cepstrum is even).
template<class Real>
void PowerSpectrumToRealCepstrum(VectorBase<Real> *power_spectrum);

/// RealCepsToMagnitudeSpec takes the output of PowerSpecToRealCeps
/// and computes an N-point FFT. PowerSpecToRealCeps computes only the
/// first N/2 + 1 elements (corresponding to positive
/// quefrencies). This function recreates the symmetric real cepstrum
/// by filling in the remaining N/2 - 1 values and computes an FFT.
/// The FFT computation happens "in-place" and the first N/2 + 1
/// elements of the vector are the real part of the log magnitude
/// spectrum (the imaginary parts should be close to 0). Contents of
/// the last last N/2 - 1 elements are undefined at output, as they
/// are not needed.
template<class Real>
void RealCepstrumToMagnitudeSpectrum(VectorBase<Real> *real_cepstrum,
                                     bool apply_exp);




/// @} End of "addtogroup feat"
}  // namespace kaldi



#endif  // KALDI_FEAT_FEATURE_MCEP_H_
