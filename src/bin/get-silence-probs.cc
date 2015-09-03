// bin/get-silence-probs.cc

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"



int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "This program takes two archives of Vector<BaseFloat>, representing\n"
        "per-frame log-likelihoods for silence and non-silence models respectively.\n"
        "It outputs per-frame silence probabilities in the same format.\n"
        "To get non-silence probs instead, use --write-nonsil-probs "
        "Usage:  get-silence-probs [options] <silence-loglikes-rspecifier> "
        " <nonsilence-loglikes-rspecifier> <silence-probs-wspecifier>\n"
        "e.g.: get-silence-probs --silence-prior=0.9 --quantize=0.25 ark:sil.likes "
        "ark:nonsil.likes ark:sil.probs\n";


    ParseOptions po(usage);

    BaseFloat sil_prior = 0.5;
    BaseFloat quantize = 0.0;
    bool write_nonsil_probs = false;
    po.Register("sil-prior", &sil_prior,
                "Prior probability of silence, must be strictly between 0 and 1.");
    po.Register("quantize", &quantize,
                "If nonzero, quantize probs to this level (to improve "
                "compressibility).");
    po.Register("write-nonsil-probs", &write_nonsil_probs,
                "If true, write non-silence probs instead of silence probs");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(sil_prior > 0.0 && sil_prior < 1.0);
    KALDI_ASSERT(quantize >= 0.0 && quantize <= 1.0);


    double sil_log_bias = Log(sil_prior / (1.0 - sil_prior));
    
    std::string silence_likes_rspecifier = po.GetArg(1),
        nonsilence_likes_rspecifier = po.GetArg(2),
        silence_probs_wspecifier = po.GetArg(3);

    SequentialBaseFloatVectorReader silence_likes_reader(silence_likes_rspecifier);
    RandomAccessBaseFloatVectorReader nonsilence_likes_reader(nonsilence_likes_rspecifier);
    BaseFloatVectorWriter silence_probs_writer(silence_probs_wspecifier);

    int num_done = 0, num_err = 0;
    double tot_frames = 0.0, tot_sil_prob = 0.0;
    
    for (; !silence_likes_reader.Done(); silence_likes_reader.Next()) {
      std::string key = silence_likes_reader.Key();
      if (!nonsilence_likes_reader.HasKey(key)) {
        KALDI_WARN << "No non-silence likes available for utterance " << key;
        num_err++;
        continue;
      }
      const Vector<BaseFloat> &sil_likes = silence_likes_reader.Value();
      const Vector<BaseFloat> &nonsil_likes = nonsilence_likes_reader.Value(key);
      if (sil_likes.Dim() != nonsil_likes.Dim()) {
        KALDI_WARN << "Dimension mismatch between sil and non-sil likes";
        num_err++;
        continue;
      }
      int32 num_frames = sil_likes.Dim();
      Vector<BaseFloat> sil_probs(num_frames);
      for (int32 f = 0; f < num_frames; f++) {
        // We're basically just applying Bayes' rule here to get the
        // posterior prob of silence.
        BaseFloat sil_loglike = sil_likes(f), nonsil_loglike = nonsil_likes(f);
        sil_loglike -= nonsil_loglike; nonsil_loglike = 0; // improve floating-point range.
        sil_loglike += sil_log_bias; // relates to prior.  Zero if prior==0.5.
        if (sil_loglike > 10) {
          sil_probs(f) = 1.0; // because the exp below might fail.
        } else {
          BaseFloat e_sil_loglike = Exp(sil_loglike);
          BaseFloat sil_prob = e_sil_loglike / (1.0 + e_sil_loglike);
          if ( !(sil_prob >= 0.0 && sil_prob  <= 1.0)) {
            KALDI_WARN << "Bad silence prob (NaNs found?), setting to 0.5";
            sil_prob = 0.5;
          }
          sil_probs(f) = sil_prob;
        }
        if (quantize != 0.0) {
          int64 i = static_cast<int64>(0.5 + (sil_probs(f) / quantize));
          sil_probs(f) = quantize * i;
        }
      }
      tot_frames += num_frames;
      tot_sil_prob += sil_probs.Sum();
      if (write_nonsil_probs) { // sil_prob <-- 1.0 - sil_prob
        sil_probs.Scale(-1.0);
        sil_probs.Add(1.0);
      }
      silence_probs_writer.Write(key, sil_probs);
      num_done++;
    }
    KALDI_LOG << "Done " << num_done << " utterances, " << num_err << " with errors.";
    KALDI_LOG << "Average silence prob is " << (tot_sil_prob/tot_frames)
              << " over " << tot_frames << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


