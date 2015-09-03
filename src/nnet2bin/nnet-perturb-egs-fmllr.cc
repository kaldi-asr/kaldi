// nnet2bin/nnet-perturb-egs-fmllr.cc

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
#include "transform/transform-common.h"
#include "nnet2/nnet-example-functions.h"

namespace kaldi {
namespace nnet2 {

void TransformTrainingExample(const Matrix<BaseFloat> &fmllr_mat,
                              BaseFloat noise_factor,
                              NnetExample *eg) {

  Matrix<BaseFloat> input_frames(eg->input_frames);
  Matrix<BaseFloat> transformed_frames(input_frames);

  for (int32 t = 0; t < transformed_frames.NumRows(); t++) {
    SubVector<BaseFloat> row(transformed_frames, t);
    ApplyAffineTransform(fmllr_mat, &row);
  }
  input_frames.Scale(1.0 - noise_factor);
  input_frames.AddMat(noise_factor, transformed_frames);
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
        "Copy examples, perturbing them by multiplying by a randomly chosen fMLLR\n"
        "transform from a fixed set.  The option --noise-factor interpolates the\n"
        "un-transformed feature (times 1.0 - noise-factor) with the fMLLR feature\n"
        "(times noise-factor)\n"
        "\n"
        "Usage:  nnet-perturb-egs-fmllr [options] <fmllr-rspecifier> <egs-rspecifier> <egs-wspecifier>\n"
        "\n"
        "nnet-perturb-egs-fmllr --noise-factor=0.2 'ark:cat exp/tri4_ali/trans.*|' ark:- ark:-\n";
    
        
    BaseFloat noise_factor = 0.1;
    int32 srand_seed = 0;
    
    ParseOptions po(usage);
    po.Register("noise-factor", &noise_factor, "Factor to interpolate fMLLR-projected "
                "data with raw data (1.0 would be pure fMLLR)");
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    
    po.Read(argc, argv);

    srand(srand_seed);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string fmllr_rspecifier = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    std::vector<Matrix<BaseFloat>* > fmllr_transforms;
    
    SequentialBaseFloatMatrixReader transform_reader(fmllr_rspecifier);
    for (; !transform_reader.Done(); transform_reader.Next())
      fmllr_transforms.push_back(new Matrix<BaseFloat>(transform_reader.Value()));

    if (fmllr_transforms.empty()) {
      KALDI_ERR << "Read no fMLLR transforms";
    }
    KALDI_LOG << "Read " << fmllr_transforms.size() << " transforms.";
    
    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    
    
    int64 num_done = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_done++) {
      std::string key = example_reader.Key();
      NnetExample eg = example_reader.Value();
      int32 n = RandInt(0, fmllr_transforms.size() - 1);
      const Matrix<BaseFloat> &fmllr_mat = *(fmllr_transforms[n]);
      TransformTrainingExample(fmllr_mat, noise_factor, &eg);
      example_writer.Write(key, eg);
    }

    while (!fmllr_transforms.empty()) {
      delete fmllr_transforms.back();
      fmllr_transforms.pop_back();
    }
    
    KALDI_LOG << "Perturbed " << num_done << " neural-network training examples "
              << "using fMLLR, with noise factor " << noise_factor;
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


