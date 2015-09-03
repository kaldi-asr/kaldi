// nnet2bin/nnet-logprob2.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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
#include "nnet2/nnet-update-parallel.h"
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-compute.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Do the forward computation for a neural net acoustic model, and output\n"
        "matrix of logprobs.  This version of the program outputs to two tables,\n"
        "one table of probabilities without prior division and one table of\n"
        "log-probs with prior division.  It is intended for use in discriminative\n"
        "training.\n"
        "\n"
        "Usage: nnet-logprob2 [options] <model-in> <features-rspecifier> "
        "<probs-wspecifier-not-divided> <logprobs-wspecifier-divided>\n"
        "\n"
        "e.g.: nnet-logprob2 1.nnet \"$feats\" ark:- \"ark:|logprob-to-post ark:- 1.post\" ark:- \\"
        "        | latgen-faster-mapped [args]\n";
    
    bool pad_input = true; // This is not currently configurable.
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        feats_rspecifier = po.GetArg(2),
        prob_wspecifier_nodiv = po.GetArg(3),
        logprob_wspecifier_divided = po.GetArg(4);
        
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    int64 num_done = 0, num_err = 0;
    
    CuVector<BaseFloat> inv_priors(am_nnet.Priors());
    KALDI_ASSERT(inv_priors.Dim() == am_nnet.NumPdfs() &&
                 "Priors in neural network not set up.");
    inv_priors.ApplyPow(-1.0);
    
    SequentialBaseFloatCuMatrixReader feature_reader(feats_rspecifier);
    BaseFloatCuMatrixWriter prob_writer_nodiv(prob_wspecifier_nodiv);
    BaseFloatCuMatrixWriter logprob_writer_divided(logprob_wspecifier_divided);

    
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const CuMatrix<BaseFloat> &feats = feature_reader.Value();
      
      CuMatrix<BaseFloat> log_probs(feats.NumRows(), am_nnet.NumPdfs());
      NnetComputation(am_nnet.GetNnet(), feats, pad_input, &log_probs);
      // at this point "log_probs" contains actual probabilities, not logs.

      // at this point they are probabilities, not log-probs, without prior division.
      prob_writer_nodiv.Write(key, log_probs);
          

      {
        log_probs.MulColsVec(inv_priors); // scales each column by the corresponding element
        // of inv_priors.
        for (int32 i = 0; i < log_probs.NumRows(); i++) {
          CuSubVector<BaseFloat> frame(log_probs, i);
          BaseFloat p = frame.Sum();
          if (!(p > 0.0)) {
            KALDI_WARN << "Bad sum of probabilities " << p;
          } else {
            frame.Scale(1.0 / p); // re-normalize to sum to one.
          }
        }
      }
      log_probs.ApplyFloor(1.0e-20); // To avoid log of zero which leads to NaN.
      log_probs.ApplyLog();
      
      logprob_writer_divided.Write(key, log_probs);
      num_done++;
    }
    
    KALDI_LOG << "Finished computing neural net log-probs, processed "
              << num_done << " utterances, " << num_err << " with errors.";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


