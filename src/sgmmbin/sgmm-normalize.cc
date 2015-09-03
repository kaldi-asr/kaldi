// sgmmbin/sgmm-normalize.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Renormalize SGMM so that within certain subsets of UBM Gaussians (typically \n"
        "corresponding to gender), probabilities sum to one; write it out, including\n"
        "normalizers."
        "Note: gaussians-rspecifier will normally be \"ark:foo\" where foo looks like\n"
        "  m  0 1 2 3 4 5\n"
        "  f  6 7 8 9 10\n"
        "Usage: sgmm-normalize [options] <model-in> <gaussians-rspecifier> <model-out>\n";

    bool binary_write = true;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_in_filename = po.GetArg(1),
        gaussians_rspecifier = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    AmSgmm am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    std::vector<std::vector<int32> > norm_sets;
    SequentialInt32VectorReader vec_reader(gaussians_rspecifier);
    for (;!vec_reader.Done(); vec_reader.Next()) 
      norm_sets.push_back(vec_reader.Value());

    am_sgmm.ComputeNormalizersNormalized(norm_sets);
    
    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_sgmm.Write(ko.Stream(), binary_write, kSgmmWriteAll);
    }
    
    
    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


