// bin/vector-extract-dims.cc

// Copyright 2015   Vimal Manohar (Johns Hopkins University)

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
#include "matrix/kaldi-vector.h"
#include "transform/transform-common.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Extract only some dimensions of a vector\n"
        "\n"
        "Usage: vector-extract-dims [options] <vector-in-rspecifier> <mask-rspecifier> <vector-out-wspecifier>\n"
        " e.g.: copy-vector ark:2.vec ark:2.mask ark,t:-\n"
        "see also: extract-rows, select-voiced-frames\n";
    
    bool select_unmasked_dims = false;

    ParseOptions po(usage);
    po.Register("select-unmasked-dims", &select_unmasked_dims,
                "Reverses the operation of this file and selects "
                "dimensions that have 0 in the mask");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string vector_rspecifier = po.GetArg(1),   
      mask_rspecifier = po.GetArg(2),
      vector_wspecifier = po.GetArg(3);
    
    SequentialBaseFloatVectorReader vector_reader(vector_rspecifier);
    RandomAccessBaseFloatVectorReader mask_reader(mask_rspecifier);
    BaseFloatVectorWriter vector_writer(vector_wspecifier);

    int32 num_done = 0, num_err = 0;
    long long num_dims = 0, num_select = 0;
    
    for (;!vector_reader.Done(); vector_reader.Next()) {
      std::string utt = vector_reader.Key();
      const Vector<BaseFloat> &vec = vector_reader.Value();
      if (vec.Dim() == 0) {
        KALDI_WARN << "Empty feature matrix for utterance " << utt;
        num_err++;
        continue;
      }
      if (!mask_reader.HasKey(utt)) {
        KALDI_WARN << "No VAD input found for utterance " << utt;
        num_err++;
        continue;
      }
      const Vector<BaseFloat> &mask = mask_reader.Value(utt);

      if (vec.Dim() != mask.Dim()) {
        KALDI_WARN << "Mismatch in number for dimensions " << vec.Dim() 
                   << " for vector and mask " << mask.Dim() 
                   << ", for utterance " << utt;
        num_err++;
        continue;
      }
      
      int32 dim = 0;
      for (int32 i = 0; i < mask.Dim(); i++)
        if (!select_unmasked_dims) {
          if (mask(i) != 0.0)
            dim++;
        } else {
          if (mask(i) == 0.0)
            dim++;
        }

      if (dim == 0) {
        KALDI_WARN << "No dimensions were selected for utterance "
                   << utt;
        num_err++;
        continue;
      }

      Vector<BaseFloat> masked_vec(dim);
      
      int32 index = 0;
      for (int32 i = 0; i < vec.Dim(); i++) {
        if (!select_unmasked_dims) {
          if (mask(i) != 0.0) {
            KALDI_ASSERT(mask(i) == 1.0); // should be zero or one.
            masked_vec(index) = vec(i);
            index++;
          }
        } else {
          if (mask(i) == 0.0) {
            masked_vec(index) = vec(i);
            index++;
          }
        }
      }
      KALDI_ASSERT(index == dim);
      vector_writer.Write(utt, masked_vec);
      num_done++;
      num_select += dim;
      num_dims += vec.Dim();
    }

    KALDI_LOG << "Done selecting " << num_select << " unmasked dimensions "
              << "out of " << num_dims << " dims total dimensions ; processed "
              << num_done << " utterances, "
              << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

