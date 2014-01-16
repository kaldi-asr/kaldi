// nnet2bin/nnet-logprob.cc

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
#include "nnet2/nnet-randomize.h"
#include "nnet2/nnet-update-parallel.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Do the forward computation for a neural net acoustic model (and division by\n"
        "the prior, if --divide-by-priors=true), and output as an archive the matrix\n"
        "of log probabilities for each utterance, e.g. for input to latgen-faster-mapped\n"
        "(note: you can also directly use nnet-latgen-faster.\n"
        "\n"
        "Usage: nnet-logprob [options] <model-in> <features-rspecifier> <logprobs-wspecifier>\n"
        "\n"
        "e.g.: nnet-logprob 1.nnet \"$feats\" ark:- | latgen-faster-mapped ... \n";
    
    std::string spk_vecs_rspecifier, utt2spk_rspecifier;
    bool pad_input = true; // This is not currently configurable.
    bool divide_by_priors = true;
    
    ParseOptions po(usage);
    
    po.Register("spk-vecs", &spk_vecs_rspecifier, "Rspecifier for a vector that "
                "describes each speaker; only needed if the neural net was "
                "trained this way.");
    po.Register("utt2spk", &utt2spk_rspecifier, "Rspecifier for map from "
                "utterance to speaker; only relevant in conjunction with the "
                "--spk-vecs option.");
    po.Register("divide-by-priors", &divide_by_priors, "If true, before getting "
                "the log-probs, divide by the priors stored with the model");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string nnet_rxfilename = po.GetArg(1),
        feats_rspecifier = po.GetArg(2),
        logprob_wspecifier = po.GetArg(3);
    
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
    // note: spk_vecs_rspecifier and utt2spk_rspecifier may be empty.
    RandomAccessBaseFloatVectorReaderMapped vecs_reader(spk_vecs_rspecifier,
                                                        utt2spk_rspecifier); 
    BaseFloatCuMatrixWriter logprob_writer(logprob_wspecifier);
    
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const CuMatrix<BaseFloat> &feats = feature_reader.Value();
      CuVector<BaseFloat> spk_vec;
      if (!spk_vecs_rspecifier.empty()) {
        if (!vecs_reader.HasKey(key)) {
          KALDI_ERR << "No speaker vector available for key " << key;
          num_err++;
          continue;
        }
        spk_vec = vecs_reader.Value(key);
      }
      
      CuMatrix<BaseFloat> log_probs(feats.NumRows(), am_nnet.NumPdfs());
      NnetComputation(am_nnet.GetNnet(), feats, spk_vec, pad_input, &log_probs);
      // at this point "log_probs" contains actual probabilities, not logs.

      if (divide_by_priors) {
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
      logprob_writer.Write(key, log_probs);
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


