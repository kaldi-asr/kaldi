#ifndef KALDI_FVECTOR_PERTURB_H_
#define KALDI_FVECTOR_PERTURB_H_

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

#include "base/kaldi-error.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"

#include "feat/resample.h"
#include "matrix/matrix-functions.h"

namespace kaldi {

// options class for distorting signals in egs
struct FvectorPerturbOptions {
  BaseFloat sample_frequency;
  BaseFloat expected_chunk_length;
  BaseFloat max_speed_perturb_rate;
  BaseFloat max_volume_variance;
  BaseFloat max_snr;
  BaseFloat min_snr;
  bool volume_perturbation;
  bool speed_perturbation;
  bool time_shift;
  BaseFloat add_noise;

  FvectorPerturbOptions(): sample_frequency(16000),
                           expected_chunk_length(100),
                           max_speed_perturb_rate(0.1),
                           max_volume_variance(0.03),
                           max_snr(20),
                           min_snr(0),
                           volume_perturbation(true),
                           speed_perturbation(true),
                           time_shift(true),
                           add_noise(0.85) { }

  void Register(OptionsItf *opts) {
    opts->Register("sample-frequency", &sample_frequency, "The sample frequency "
                   "of the wav signal.");
    opts->Register("expected-chunk-length", &expected_chunk_length, "It show the "
                   "length of chunk you expected. e.g. 100ms. That means the length "
                   "of output will correspond to 100ms. At the same time, it will "
                   "affect the speed_perturb_rate, the speed_perturb_rate factor will "
                   "in the range of min{expected-chunk-length/original-length, "
                   "max-speed-perturb-rate}.");
    opts->Register("max-speed-perturb-rate", &max_speed_perturb_rate,
                   "Max speed perturbation applied on matrix. It will work together "
                   "with expected_chunk_length. E.g. 0.1 means we will generate "
                   "speed_factor randomly from range (1-a, 1+a), where a="
                   "min{original_length/expected_length-1, 0.1}.");
    opts->Register("max-volume-variance", &max_volume_variance, "The variation in "
                   "volume will vary form 1-max-volume-variance to 1+max-volume-variance "
                   "randomly.");
    opts->Register("max-snr",&max_snr,"Specify a upperbound Signal to Noise Ratio. We will scale the noise according "
                   "to the original signal and SNR. Normally, it's a non-zero number between -30 and 30.");
    opts->Register("min-snr",&min_snr,"Specify a lowerbound Signal to Noise Ratio. We will scale the noise according "
                   "to the original signal and SNR. Normally, it's a non-zero number between -30 and 30.");
    opts->Register("volume-perturbation", &volume_perturbation, "If true, we will "
                   "conduct variations in volume.");
    opts->Register("speed-perturbation", &speed_perturbation, "If true, we will "
                   "conduct variations in speed.");
    opts->Register("time-shift", &time_shift, "If true, we will "
                   "conduct time shift. That means randomly select the start point from "
                   "range [0, input.NumCols() - expected_chunk_length], and then "
                   "get the successive 'expected_chunk_length' data. Otherwise, we get "
                   "the data from the head.");
    opts->Register("add-noise", &add_noise, "Add additive noise to source chunk with "
                   "probability. E.g. 0.85 means we add noise with 85 percent probability, "
                   "and remain with 15 percent probability.");
  }
};

/* This class is used to do (0-4) kinds of perturbation operation to fvector.
 * According to the FvectorPerturbOptions, we choose do or not.
 * The input always is a Matrix which contains four lines(S1, S2, N1, N2)[S1=S2]
 * Then we will call different perturbation methods. (For details, see the comments
 * of FvectorPerturbOption.)
 * For the details about the four kinds of perturbation operation, please see
 * the document in fvector-perturb.cc.
 */
class FvectorPerturb {
 public:
  FvectorPerturb(FvectorPerturbOptions opts) { opts_ = opts; }
  void ApplyPerturbation(const MatrixBase<BaseFloat>& input_chunk,
                         Matrix<BaseFloat>* perturbed_chunk);

  // Randomly Generate 2 scale number and scale each line respectively
  void VolumePerturbation(MatrixBase<BaseFloat>* chunk);

  // Use ArbitraryResample. For each line, randomly generate a speed factor. 
  // Then do time axis stretch. As speed factor is different, so we deal with
  // each vector separately. The dim of output_vector is bigger than
  // expected_chunk_length(ms)
  void SpeedPerturbation(VectorBase<BaseFloat>& input_vector,
                         BaseFloat samp_freq,
                         BaseFloat speed_factor,
                         VectorBase<BaseFloat>* output_vector);

  // Randomly choose a expect_chunk_length(ms) vector.
  void TimeShift(VectorBase<BaseFloat>& input_vector,
                 VectorBase<BaseFloat>* output_vector);

  // The input is a matrix contains four consecutive rows.
  // It is (S1, S2, N1, N2). Each line is original_chunk_length(ms)(e.g. 960 dims = 120ms)
  // add N1 to S1, add N2 to S2 with random snr. with probability (probability_threshold).
  // After that, only the first two lines is meaningful, which represents two 
  // perturbed signals from the same source wavform signal.
  // After use this function, maybe you need to resize the output.
  // (Notice: Resize() belongs to Matrix<> rather than MatrixBase<>)
  void AddNoise(BaseFloat probability_threshold,
                MatrixBase<BaseFloat>* chunk);

 private:
  FvectorPerturbOptions opts_;
};


/* This class is used to do (0-4) kinds of perturbation operation to fvector.
 * According to the FvectorPerturbOptions, we choose do or not. 
 * It is block version code that means it will process a matrix each time.
 * Different from class FvectorPerturb, the class will process its private members
 * (perturbed1, perturbed2, noise1 and noise2) to conducte perturbation operations.
 * We will call different perturbation methods. (For details, see the comments
 * of FvectorPerturbOption.)
 * For the details about the four kinds of perturbation operation, please see
 * the document in fvector-perturb.cc.
 */
class FvectorPerturbBlock {
 public:
  FvectorPerturbBlock(FvectorPerturbOptions opts, 
                      const MatrixBase<BaseFloat> &source,
                      const MatrixBase<BaseFloat> &noise1,
                      const MatrixBase<BaseFloat> &noise2) : opts_(opts),
    perturbed1(source), perturbed2(source), noise1(noise1), noise2(noise2) {}

  // The interface to apply different perturbation opertaions. Firstly, the
  // function will conduct different perturbation operations. And then
  // it will compose the final matrices--perturbed1, perturbed2 together.
  void ApplyPerturbationBlock(Matrix<BaseFloat>* perturbed_chunk);

  // The input is two matrices. One is source matrix(e.g. perturbed1), another
  // is noise matrix(e.g. noise1). Each line is original_chunk_length(ms)(e.g. 960 dims = 120ms)
  // add noise row-by-row with random snr. with probability (probability_threshold).
  // After that, the source signal is perturbed by noise signal.
  void AddNoiseBlock(BaseFloat probability_threshold,
                     MatrixBase<BaseFloat>& source,
                     MatrixBase<BaseFloat>& noise);

  // Randomly Generate a scale number and scale the whole matrix
  void VolumePerturbationBlock(MatrixBase<BaseFloat>& block);

  // Use ArbitraryResample. Generate a speed factor randomly for the whole matrix.
  // Then do time axis stretch.  The dim of output_vector is bigger than
  // expected_chunk_length(ms)
  void SpeedPerturbationBlock(Matrix<BaseFloat>& block);

  // Randomly choose a expect_chunk_length(ms) vector.
  void TimeShiftBlock(Matrix<BaseFloat>& block);

 private:
  FvectorPerturbOptions opts_;
  Matrix<BaseFloat> perturbed1, perturbed2, noise1, noise2;
  
};

} // end of namespace kaldi
#endif // KALDI_FVECTOR_PERTURB_H_
