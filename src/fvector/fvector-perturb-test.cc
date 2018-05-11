// fvector/fvector-perturb-test.cc

#include <iostream>
#include <math.h>
#include "fvector/fvector-perturb.h"
#include "feat/wave-reader.h"

using namespace kaldi;

static void UnitTestSpeedPerturb() {
  std::cout << "=== UnitTestSpeedPerturb ===" << std::endl;
  Vector<BaseFloat> input, output;
  BaseFloat sample_freq;
  {
    std::ifstream is("./test_data/test.wav", std::ios_base::binary);
    WaveData wave;
    wave.Read(is);
    const Matrix<BaseFloat> data(wave.Data());
    KALDI_ASSERT(data.NumRows() == 1);
    input.Resize(data.NumCols());
    input.CopyFromVec(data.Row(0));
    sample_freq = wave.SampFreq();
  }
  BaseFloat speed_factor = 1.2;
  FvectorPerturbOptions opts;
  FvectorPerturb perturb(opts);
  output.Resize(static_cast<MatrixIndexT>(ceil(input.Dim()/speed_factor)));
  perturb.SpeedPerturbation(input, sample_freq, speed_factor, &output);
  {
    std::ofstream os("./test_data/test_speedperturbed.wav.txt", std::ios::out);
    output.Write(os, false);
  }
  std::cout << "With fvector class, The dim of input is: " << input.Dim() << std::endl;
  std::cout << "With fvector class, The dim of output is: " << output.Dim() << std::endl;
  // Write the perturbed data into wav format
  {
    Matrix<BaseFloat> output_matrix(1, output.Dim());
    output_matrix.CopyRowFromVec(output, 0);
    WaveData perturbed_wave(sample_freq, output_matrix);
    std::ofstream os("./test_data/test_speedperturbed.wav", std::ios::out);
    perturbed_wave.Write(os);
  }
  // print the wav data which is dealed by sox. 
  // Command: sox -t wav test.wav -t wav test_speed12.wav speed 1.2
  Vector<BaseFloat> sox_input;
  BaseFloat sox_sample_freq;
  {
    std::ifstream is("./test_data/test_speed12.wav", std::ios_base::binary);
    WaveData wave;
    wave.Read(is);
    const Matrix<BaseFloat> data(wave.Data());
    KALDI_ASSERT(data.NumRows() == 1);
    sox_input.Resize(data.NumCols());
    sox_input.CopyFromVec(data.Row(0));
    sox_sample_freq = wave.SampFreq();
    std::ofstream os("./test_data/test_sox.wav.txt", std::ios::out);
    sox_input.Write(os, false);
  }
  KALDI_ASSERT(sample_freq == sox_sample_freq);
  if (output.ApproxEqual(sox_input, 0.01)) { 
    std::cout << "Equal" << std::endl;
  } else {
    std::cout << "Not Equal" << std::endl;
    BaseFloat prod_output = VecVec(output, output);
    BaseFloat prod_sox = VecVec(sox_input, sox_input);
    BaseFloat cross_prod = VecVec(output, sox_input);
    std::cout << "The cosin distance is: " 
              << cross_prod/(sqrt(prod_output)*sqrt(prod_sox))
              << std::endl;
  }
  std::cout << "=== UnitTestSpeedPerturb finish ===" << std::endl;
}

static void UnitTestFvectorPerturb() {
  UnitTestSpeedPerturb();
}

int main() {
  try{
    UnitTestFvectorPerturb();
    std::cout << "Tests succeeded." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
}
