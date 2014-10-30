// online2bin/ivector-randomize.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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
#include "matrix/kaldi-matrix.h"
#include "transform/transform-common.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy matrices of online-estimated iVectors, but randomize them;\n"
        "this is intended primarily for training the online nnet2 setup\n"
        "with iVectors.  For each input matrix, each row with index t is,\n"
        "with probability given by the option --randomize-prob, replaced\n"
        "with the contents an input row chosen randomly from the interval [t, T]\n"
        "where T is the index of the last row of the matrix.\n"
        "\n"
        "Usage: ivector-randomize [options] <ivector-rspecifier> <ivector-wspecifier>\n"
        " e.g.: ivector-randomize ark:- ark:-\n"
        "See also: ivector-extract-online, ivector-extract-online2, subsample-feats\n";

    int32 srand_seed = 0;
    BaseFloat randomize_prob = 0.5;
    
    ParseOptions po(usage);

    po.Register("srand", &srand_seed, "Seed for random number generator");
    po.Register("randomize-prob", &randomize_prob, "For each row, replace it with a "
                "random row with this probability.");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }


    std::string ivector_rspecifier = po.GetArg(1),
        ivector_wspecifier = po.GetArg(2);

    int num_done = 0;
    SequentialBaseFloatMatrixReader reader(ivector_rspecifier);
    BaseFloatMatrixWriter writer(ivector_wspecifier);
    
    for (; !reader.Done(); reader.Next(), num_done++) {
      std::string utt = reader.Key();
      const Matrix<BaseFloat> &ivectors_in = reader.Value();
      int32 T = ivectors_in.NumRows(), dim = ivectors_in.NumCols();
      Matrix<BaseFloat> ivectors_out(T, dim, kUndefined);
      for (int32 t = 0; t < T; t++) {
        int32 t_src;
        if (WithProb(randomize_prob)) t_src = RandInt(t, T-1);
        else t_src = t;
        ivectors_out.Row(t).CopyFromVec(ivectors_in.Row(t_src));
      }
      writer.Write(utt, ivectors_out);
      num_done++;
    }
    KALDI_LOG << "Randomized " << num_done << " iVectors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


