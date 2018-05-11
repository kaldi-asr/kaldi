#include "fvector/fvector-perturb.h"

namespace kaldi {

void FvectorPerturb::ApplyPerturbation(const MatrixBase<BaseFloat>& input_chunk,
                                       Matrix<BaseFloat>* perturbed_chunk) {
  // The original_dim_matrix is a matrix whose dimension is same with input_chunk.
  // Assume the sample_frequency=8kHz, the original length is 120ms.
  // It will be a (4, 960) matrix.
  Matrix<BaseFloat> original_dim_matrix(input_chunk);
  // Firstly, we add additive noise with probability.
  AddNoise(opts_.add_noise, &original_dim_matrix);
  // we do Resize() here, because Resize() belongs to Matrix<> rather than MatrixBase<>
  original_dim_matrix.Resize(2, original_dim_matrix.NumCols(), kCopyData);
  KALDI_ASSERT(original_dim_matrix.NumRows() == 2);
  // After AddNoise(), the shape of original_dim_matrix is (2, original_dim).
  if (opts_.volume_perturbation) {
    VolumePerturbation(&original_dim_matrix);
  }
  // The expected_dim_matrix is a matrix (input_chunk.NumRows(), expected-chunk-length
  // * sample_frequency / 1000). E.g. it is a (4, 800) matrix.
  Matrix<BaseFloat> expected_dim_matrix(original_dim_matrix.NumRows(),
      opts_.expected_chunk_length * opts_.sample_frequency / 1000);
  if (opts_.speed_perturbation) {
    //1. generate speed perturb factor randomly(Noice: the expected_length is
    //always smaller than original_length) for each line.
    //(1) a=min{original_length/expected_length -1, max-speed-perturb-rate}
    //(2) the range of factor is (1-a, 1+a)
    BaseFloat boundary = std::min(static_cast<BaseFloat>((original_dim_matrix.NumCols() * 1.0 / opts_.sample_frequency)
          * 1000 / opts_.expected_chunk_length - 1), opts_.max_speed_perturb_rate);
    for (MatrixIndexT i = 0; i < original_dim_matrix.NumRows(); ++i) {
      //caculate the speed factor
      BaseFloat factor =static_cast<BaseFloat> (RandInt(
          (int)((1-boundary)*100),(int)((1+boundary)*100)) * 1.0 / 100.0);
      
      Vector<BaseFloat> speed_input_vector(original_dim_matrix.Row(i));
      
      MatrixIndexT speed_output_dim = static_cast<MatrixIndexT>(ceil(original_dim_matrix.NumCols() / factor));
      KALDI_ASSERT(speed_output_dim >= opts_.expected_chunk_length * opts_.sample_frequency / 1000);
      Vector<BaseFloat> speed_output_vector(speed_output_dim);

      SpeedPerturbation(speed_input_vector, opts_.sample_frequency, factor, &speed_output_vector);
      
      Vector<BaseFloat> time_shifted_vector(expected_dim_matrix.NumCols());
      if (opts_.time_shift) {
        TimeShift(speed_output_vector, &time_shifted_vector);
      } else {
        time_shifted_vector.CopyFromVec(speed_output_vector.Range(0, expected_dim_matrix.NumCols()));
      }
      expected_dim_matrix.CopyRowFromVec(time_shifted_vector, i); 
    }
  } else { //no speed_perturbation
    if (opts_.time_shift) {
      for (MatrixIndexT i = 0; i < original_dim_matrix.NumRows(); ++i) {
        Vector<BaseFloat> input_vector(original_dim_matrix.Row(i));
        Vector<BaseFloat> time_shifted_vector(expected_dim_matrix.NumCols());
        TimeShift(input_vector, &time_shifted_vector);
        expected_dim_matrix.CopyRowFromVec(time_shifted_vector, i);
      }  
    } else {
      expected_dim_matrix.CopyFromMat(original_dim_matrix.Range(0, expected_dim_matrix.NumRows(),
                                                                0, expected_dim_matrix.NumCols()));
    }
  }
  // Now we operate the "expected_dim_matrix"  
  perturbed_chunk->Resize(2, expected_dim_matrix.NumCols());
  MatrixIndexT indices[2] = {0, 1};
  perturbed_chunk->CopyRows(expected_dim_matrix, indices);
}

void FvectorPerturb::VolumePerturbation(MatrixBase<BaseFloat>* chunk) {
  //1. Randomly generate 2 number from (1-max-volume-variance, 1+max-volume-variance)
  std::vector<BaseFloat> volume_factors;
  for (MatrixIndexT i = 0; i < chunk->NumRows(); ++i) {
    BaseFloat factor = static_cast<BaseFloat>(
        RandInt((int)((1-opts_.max_volume_variance)*100),
                (int)((1+opts_.max_volume_variance)*100)) / 100.0);
    volume_factors.push_back(factor);
  }
  //2. scale each line respectively.
  for (MatrixIndexT i = 0; i < chunk->NumRows(); ++i) {
    chunk->Row(i).Scale(volume_factors[i]);
  }
}

// we stretch the signal from the beginning to end.
// y(t) = x(s*t) for t = 0,...,n. If s>0, the output will be shorter than
// input. It represents speeding up. Vice versa.
// Use ArbitraryResample deal with each line.
//
// In ArbitraryResample, according to num_zeros and filter_cutoff, it generates
// the "filter_with". And then each output_sample(t) corresponds to few input_samples
// from (t-filter_with) to (t+filter_with), which is stored in "first_index_".
// And "weights_" will be adjust by a Hanning window in function FilterFunc.
// In brief, you can think each output sample is the weighted sum of few input_samples.
void FvectorPerturb::SpeedPerturbation(VectorBase<BaseFloat>& input_vector,
                                       BaseFloat samp_freq,
                                       BaseFloat speed_factor,
                                       VectorBase<BaseFloat>* output_vector) {
  if (speed_factor == 1.0) {
    output_vector->CopyFromVec(input_vector);
  } else {
    Vector<BaseFloat> in_vec(input_vector),
                      out_vec(output_vector->Dim());
    int32 input_dim = in_vec.Dim(),
          output_dim = out_vec.Dim();
    Vector<BaseFloat> samp_points_secs(output_dim);
    int32 num_zeros = 4; // Number of zeros of the sinc function that the window extends out to.
    // lowpass frequency that's lower than 95% of the Nyquist.
    BaseFloat filter_cutoff_hz = samp_freq * 0.475; 
    for (int32 i = 0; i < output_dim; i++) {
      samp_points_secs(i) = static_cast<BaseFloat>(speed_factor * i / samp_freq);
    }
    ArbitraryResample time_resample(input_dim, samp_freq,
                                    filter_cutoff_hz, 
                                    samp_points_secs,
                                    num_zeros);
    time_resample.Resample(in_vec, &out_vec);
    output_vector->CopyFromVec(out_vec);
  }
}

void FvectorPerturb::TimeShift(VectorBase<BaseFloat>& input_vector,
                               VectorBase<BaseFloat>* output_vector) {
  //1. generate start point randomly whose range is
  // [0, row.NumCols()- expected_chunk_length * sample_frequency)
  int32 start_point = static_cast<int32>(RandInt(0, input_vector.Dim() - output_vector->Dim()));
  //2. get the successive expected_chunk_length * sample_frequency data.
  output_vector->CopyFromVec(input_vector.Range(start_point, output_vector->Dim()));
}

void FvectorPerturb::AddNoise(BaseFloat probability_threshold, 
                              MatrixBase<BaseFloat>* chunk) {
  //1. generate 2 SNR from (min-snr, max-snr)
  //2. add N1(line3) to S1(line1) with snr1 with probability
  //   add N2(line4) to S2(line2) with snr2 with probability
  for (MatrixIndexT i = 0; i < 2; i++) {
    BaseFloat  probability = static_cast<BaseFloat>(RandInt(0, 100) / 100.0);
    if (probability <= probability_threshold) {
      Vector<BaseFloat> source(chunk->Row(i));
      Vector<BaseFloat> noise(chunk->Row(i+2));
      BaseFloat source_power = VecVec(source, source) / source.Dim();
      BaseFloat noise_power = VecVec(noise, noise) / noise.Dim();
      int32 snr = RandInt(opts_.min_snr, opts_.max_snr);
      BaseFloat scale_factor = sqrt(pow(10, -snr/10) * source_power / noise_power);
      //BaseFloat source_energy = VecVec(source, source);
      //BaseFloat noise_energy = VecVec(noise, noise);
      // The smaller the value, the greater the snr
      //BaseFloat scale_factor = sqrt(source_energy/ noise_energy / (pow(10, snr/20)));
      chunk->Row(i).AddVec(scale_factor, noise);
    }
  }
}


// The following functions belong to Class FvectorPerturbBlock

void FvectorPerturbBlock::ApplyPerturbationBlock(Matrix<BaseFloat>* perturbed_chunk) {
  // 1.Add additive noise with probability
  AddNoiseBlock(opts_.add_noise, perturbed1, noise1);
  AddNoiseBlock(opts_.add_noise, perturbed2, noise2);
  
  // 2.After AddNoise(), conduct volume perturbation.
  if (opts_.volume_perturbation) {
    VolumePerturbationBlock(perturbed1);
    VolumePerturbationBlock(perturbed2);
  }

  // 3. Conduct the speed perturbation and time shift together. At last the
  // NumCols of perturbed matrix equals expected_chunk_length(e.g. 100ms)
  if (opts_.speed_perturbation) {
    SpeedPerturbationBlock(perturbed1);
    SpeedPerturbationBlock(perturbed2);
    if (opts_.time_shift) {
      TimeShiftBlock(perturbed1);
      TimeShiftBlock(perturbed2);
    } else {
      int32 output_cols = static_cast<int32>(opts_.expected_chunk_length * opts_.sample_frequency);
      KALDI_ASSERT(perturbed1.NumRows() == perturbed2.NumRows());
      int32 output_rows = perturbed1.NumRows();
      perturbed1.Resize(output_rows, output_cols, kCopyData);
      perturbed2.Resize(output_rows, output_cols, kCopyData);
    }
  } else {
    if (opts_.time_shift) {
      TimeShiftBlock(perturbed1);
      TimeShiftBlock(perturbed2);
    } else {
      int32 output_cols = static_cast<int32>(opts_.expected_chunk_length * opts_.sample_frequency);
      KALDI_ASSERT(perturbed1.NumRows() == perturbed2.NumRows());
      int32 output_rows = perturbed1.NumRows();
      perturbed1.Resize(output_rows, output_cols, kCopyData);
      perturbed2.Resize(output_rows, output_cols, kCopyData);
    }
  }

  // 4. At last, compose two different perturbed matrices into one matrix.
  // Each two consecutive lines come from the same original source signal.
  KALDI_ASSERT(perturbed1.NumRows() == perturbed2.NumRows());
  int32 output_rows = perturbed1.NumRows() *2;
  KALDI_ASSERT(perturbed1.NumCols() == perturbed2.NumCols());
  int32 output_cols = perturbed1.NumCols();
  perturbed_chunk->Resize(output_rows, output_cols);
  for (MatrixIndexT i = 0; i < output_rows/2 ; i++) {
    perturbed_chunk->Row(2*i).CopyFromVec(perturbed1.Row(i));
    perturbed_chunk->Row(2*i+1).CopyFromVec(perturbed2.Row(i));
  }
}

// For each row of source, we add additive noise to it with random snr with
// probability. 
void FvectorPerturbBlock::AddNoiseBlock(BaseFloat probability_threshold,
                                        MatrixBase<BaseFloat>& source,
                                        MatrixBase<BaseFloat>& noise) {
  KALDI_ASSERT(source.NumRows() == noise.NumRows());
  for (MatrixIndexT i = 0; i < source.NumRows(); i++) {
    BaseFloat  probability = static_cast<BaseFloat>(RandInt(0, 10000) / 100.0);
    if (probability <= probability_threshold) {
      Vector<BaseFloat> source_signal(source.Row(i));
      Vector<BaseFloat> noise_signal(noise.Row(i));
      BaseFloat source_energy = VecVec(source_signal, source_signal);
      BaseFloat noise_energy = VecVec(noise_signal, noise_signal);
      // The smaller the value, the greater the snr
      int32 snr = RandInt(opts_.min_snr, opts_.max_snr);
      BaseFloat scale_factor = sqrt(source_energy/ noise_energy / (pow(10, snr/20)));
      source.Row(i).AddVec(scale_factor, noise_signal);
    }
  }
}

// For the whole block, we use a uniform scale factor.
void FvectorPerturbBlock::VolumePerturbationBlock(MatrixBase<BaseFloat>& block) {
  BaseFloat factor = static_cast<BaseFloat>(
        RandInt((int)((1-opts_.max_volume_variance)*100),
                (int)((1+opts_.max_volume_variance)*100)) / 100.0);
  block.Scale(factor);
}

// It is similar with FvectorPerturb::SpeedPerturbation(). Use ArbitraryResample.
// For the whole block, we use a uniform random speed factor.
void FvectorPerturbBlock::SpeedPerturbationBlock(Matrix<BaseFloat>& block) {
  //1. generate speed perturb factor randomly(Noice: the expected_length is
  //always smaller than original_length) for each line.
  //(1) a=min{original_length/expected_length -1, max-speed-perturb-rate}
  //(2) the range of factor is (1-a, 1+a)
  BaseFloat boundary = std::min((block.NumCols() / opts_.sample_frequency) / opts_.expected_chunk_length - 1,
                                  opts_.max_speed_perturb_rate);
  //caculate the speed factor
  BaseFloat speed_factor =static_cast<BaseFloat> (RandInt(
          (int)((1-boundary)*100),(int)((1+boundary)*100)) * 1.0 / 100.0);
  MatrixIndexT output_dim = static_cast<MatrixIndexT>(ceil(block.NumCols() / speed_factor));
  KALDI_ASSERT(output_dim >= opts_.expected_chunk_length * opts_.sample_frequency / 1000);

  if (speed_factor == 1.0) {
    // return the original block
  } else {
    int32 input_dim = block.NumCols();
    Vector<BaseFloat> samp_points_secs(output_dim);
    int32 num_zeros = 4; // Number of zeros of the sinc function that the window extends out to.
    // lowpass frequency that's lower than 95% of the Nyquist.
    BaseFloat filter_cutoff_hz = opts_.sample_frequency * 0.475; 
    for (int32 i = 0; i < output_dim; i++) {
      samp_points_secs(i) = static_cast<BaseFloat>(speed_factor * i / opts_.sample_frequency);
    }
    ArbitraryResample time_resample(input_dim, opts_.sample_frequency,
                                    filter_cutoff_hz, 
                                    samp_points_secs,
                                    num_zeros);
    Matrix<BaseFloat> tmp_block(block.NumRows(), output_dim);
    time_resample.Resample(block, &tmp_block);
    block.Resize(tmp_block.NumRows(), tmp_block.NumCols());
    block.CopyFromMat(tmp_block);    
  }
}

// Choose a uniform start_point randomly, whose range is
// [0, row.NumCols()- expected_chunk_length * sample_frequency)]
// get the successive expected_chunk_length * sample_frequency data.
void FvectorPerturbBlock::TimeShiftBlock(Matrix<BaseFloat>& block) {
  int32 output_cols = static_cast<int32>(opts_.expected_chunk_length * opts_.sample_frequency);
  int32 output_rows = block.NumRows();
  int32 start_point = static_cast<int32>(RandInt(0, block.NumCols() - output_cols));
  Matrix<BaseFloat> tmp_block(output_rows, output_cols);
  tmp_block.CopyFromMat(block.Range(0, output_rows, start_point, output_cols));
  block.Resize(output_rows, output_cols);
  block.CopyFromMat(tmp_block);
}

} // end of namespace kaldi
