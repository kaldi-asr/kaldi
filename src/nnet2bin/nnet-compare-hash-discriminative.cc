// nnet2bin/nnet-compare-hash-discriminative.cc

// Copyright 2012-2013  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet2/nnet-randomize.h"
#include "nnet2/nnet-example-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Compares two archives of discriminative training examples and checks\n"
        "that they behave the same way for purposes of discriminative training.\n"
        "This program was created as a way of testing nnet-get-egs-discriminative\n"
        "The model is only needed for its transition-model.\n"
        "\n"
        "Usage:  nnet-compare-hash-discriminative [options] <model-rxfilename> "
        "<egs-rspecifier1> <egs-rspecifier2>\n"
        "\n"
        "Note: options --drop-frames and --criterion should be matched with the\n"
        "command line of nnet-get-egs-discriminative used to get the examples\n"
        "nnet-compare-hash-discriminative --drop-frames=true --criterion=mmi ark:1.degs ark:2.degs\n";
    
    std::string criterion = "smbr";
    bool drop_frames = false;
    BaseFloat threshold = 0.002;
    BaseFloat acoustic_scale = 1.0, lm_scale = 1.0;
    ParseOptions po(usage);

    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale,
                "Scaling factor for \"graph costs\" (including LM costs)");
    po.Register("criterion", &criterion, "Training criterion, 'mmi'|'mpfe'|'smbr'");
    po.Register("drop-frames", &drop_frames, "If true, for MMI training, drop "
                "frames where num and den do not intersect.");
    po.Register("threshold", &threshold, "Threshold for equality testing "
                "(relative)");
    
    po.Read(argc, argv);

    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        examples_rspecifier1 = po.GetArg(2),
        examples_rspecifier2 = po.GetArg(3);

    int64 num_done1 = 0, num_done2 = 0;


    TransitionModel tmodel;
    ReadKaldiObject(model_rxfilename, &tmodel);
    
    Matrix<double> hash1, hash2;

    // some additional diagnostics:
    double num_weight1 = 0.0, den_weight1 = 0.0, tot_t1 = 0.0;
    double num_weight2 = 0.0, den_weight2 = 0.0, tot_t2 = 0.0;
    
    SequentialDiscriminativeNnetExampleReader
        example_reader1(examples_rspecifier1),
        example_reader2(examples_rspecifier2);

    KALDI_LOG << "Computing first hash function";
    for (; !example_reader1.Done(); example_reader1.Next(), num_done1++) {
      DiscriminativeNnetExample eg = example_reader1.Value();
      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale),
                        &(eg.den_lat));
      UpdateHash(tmodel, eg, criterion, drop_frames, &hash1,
                 &num_weight1, &den_weight1, &tot_t1);
    }
    KALDI_LOG << "Processed " << num_done1 << " examples.";

    KALDI_LOG << "Computing second hash function";
    for (; !example_reader2.Done(); example_reader2.Next(), num_done2++) {
      DiscriminativeNnetExample eg = example_reader2.Value();
      fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale),
                        &(eg.den_lat));
      UpdateHash(tmodel, eg, criterion, drop_frames, &hash2,
                 &num_weight2, &den_weight2, &tot_t2);
    }
    KALDI_LOG << "Processed " << num_done2 << " examples.";
    
    double prod1 = TraceMatMat(hash1, hash1, kTrans),
        prod2 = TraceMatMat(hash2, hash2, kTrans),
        cross_prod = TraceMatMat(hash1, hash2, kTrans);

    KALDI_LOG << "Products are as follows (should be the same): prod1 = "
              << prod1 << ", prod2 = " << prod2 << ", cross_prod = "
              << cross_prod;

    KALDI_LOG << "Num-weight1 = " << num_weight1 << ", den-weight1 = "
              << den_weight1 << ", tot_t1 = " << tot_t1;
    KALDI_LOG << "Num-weight2 = " << num_weight2 << ", den-weight2 = "
              << den_weight2 << ", tot_t2 = " << tot_t2;
        
    KALDI_ASSERT(ApproxEqual(prod1, prod2, threshold) &&
                 ApproxEqual(prod2, cross_prod, threshold));
    KALDI_ASSERT(prod1 > 0.0);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


