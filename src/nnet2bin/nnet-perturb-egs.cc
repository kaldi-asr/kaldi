// nnet2bin/nnet-perturb-egs.cc

// Copyright 2012-2014  Johns Hopkins University (author:  Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet2/nnet-example-functions.h"

namespace kaldi {
namespace nnet2 {

void PerturbTrainingExample(const TpMatrix<BaseFloat> &cholesky,
                            BaseFloat noise_factor,
                            NnetExample *eg) {
  Matrix<BaseFloat> input_frames(eg->input_frames);
  int32 dim = input_frames.NumRows() * input_frames.NumCols();
  if (dim != cholesky.NumRows()) {
    KALDI_ERR << "Dimension mismatch: egs have total dim " << dim
              << " vs. cholesky factor " << cholesky.NumRows();
  }
  Vector<BaseFloat> vec(dim, kUndefined);
  vec.CopyRowsFromMat(input_frames);
  Vector<BaseFloat> noise(dim);
  noise.SetRandn();  // Gaussian noise with unit variance and zero mean
  vec.AddTpVec(noise_factor, cholesky, kNoTrans, noise, 1.0);
  input_frames.CopyRowsFromVec(vec);
  eg->input_frames.CopyFromMat(input_frames);
}

}
}



int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples, perturbing them by adding a specified amount (--noise-factor)\n"
        "times the within-class covariance of the examples. the Cholesky factor of\n"
        "the examples (obtained from the --write-cholesky option of\n"
        "nnet-get-feature-transform) must be supplied.\n"
        "\n"
        "Usage:  nnet-perturb-egs [options] <cholesky> <egs-rspecifier> <egs-wspecifier>\n"
        "\n"
        "nnet-perturb-egs --noise-factor=0.2 exp/nnet5/cholesky.tpmat ark:- ark:-\n";
    
        
    BaseFloat noise_factor = 0.1;
    int32 srand_seed = 0;
    
    ParseOptions po(usage);
    po.Register("noise-factor", &noise_factor, "Factor to multiply noise generated "
                "from within-class variance by before adding to egs");
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    
    po.Read(argc, argv);

    srand(srand_seed);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string cholesky_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    TpMatrix<BaseFloat> cholesky;
    ReadKaldiObject(cholesky_rxfilename, &cholesky);
    
    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    
    
    int64 num_done = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_done++) {
      std::string key = example_reader.Key();
      NnetExample eg = example_reader.Value();
      PerturbTrainingExample(cholesky, noise_factor, &eg);
      example_writer.Write(key, eg);
    }
    
    KALDI_LOG << "Perturbed " << num_done << " neural-network training examples "
              << "with noise factor " << noise_factor;
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


