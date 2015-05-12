// bin/loglike-to-pred.cc

// Copyright 2015  Vimal Manohar (Johns Hopkins University)

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
#include "hmm/posterior.h"

/* For each frame, predict its class given the log-likelihoods under
 * different class models based on maximum-likelihood decoding */

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert a set of vectors of log-likelihoods (e.g. from gmm-global-get-frame-likes) to class predictions using ML decoding\n"
        "Usage:  loglikes-to-pred [options] <loglikes-vector-rspecifier1> [ <loglikes-vector-rspecifier2> ... <loglikes-vector-rspecifierN> ] <prediction-wspecifier>\n"
        "e.g.:\n"
        " loglikes-to-pred ark:silence_likes.ark ark:speech_likes.ark ark:vad.ark\n";
    
    std::string weights_wspecifier;

    ParseOptions po(usage);
    po.Register("weights", &weights_wspecifier, "Write posterior probability of each class.");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string loglikes_vec_rspecifier1 = po.GetArg(1);
    std::string prediction_wspecifier = po.GetArg(po.NumArgs());

    int32 num_done = 0;
    SequentialBaseFloatVectorReader loglikes_reader1(loglikes_vec_rspecifier1);
    std::vector<RandomAccessBaseFloatVectorReader*> loglikes_readers(po.NumArgs()-2,
        static_cast<RandomAccessBaseFloatVectorReader*>(NULL));
    BaseFloatVectorWriter prediction_writer(prediction_wspecifier);
    BaseFloatVectorWriter weights_writer(weights_wspecifier);

    for (int32 i = 0; i < po.NumArgs()-2; i++) 
      loglikes_readers[i] = new RandomAccessBaseFloatVectorReader(po.GetArg(i+2));

    std::vector<BaseFloat> class_loglikes(po.NumArgs()-1);
    std::vector<int32> class_counts(po.NumArgs()-1);

    for (; !loglikes_reader1.Done(); loglikes_reader1.Next()) {
      const Vector<BaseFloat> &loglikes1 = loglikes_reader1.Value();
      std::string key = loglikes_reader1.Key();
      std::vector<Vector<BaseFloat>*> loglikes(po.NumArgs()-2,
          static_cast<Vector<BaseFloat>*>(NULL));
      for (int32 i = 0; i < po.NumArgs()-2; i++) {
        if (!loglikes_readers[i]->HasKey(key)) {
          KALDI_ERR << "Key " << key << " not found in "
                     << po.GetArg(i+2);
        }
        loglikes[i] = new Vector<BaseFloat>(loglikes_readers[i]->Value(key));
      }
      
      Vector<BaseFloat> prediction(loglikes1.Dim());
      Vector<BaseFloat> weights;
      
      if (weights_wspecifier != "") 
        weights.Resize(loglikes1.Dim());

      for (int32 j = 0; j < loglikes1.Dim(); j++) {
        BaseFloat max_like = loglikes1(j); 
        Vector<BaseFloat> this_log_likes(po.NumArgs()-1);
        this_log_likes(0) = max_like;
        for (int32 i = 0; i < po.NumArgs()-2; i++) {
          if (loglikes[i] == NULL) continue;
          this_log_likes(i+1) = (*loglikes[i])(j);
          KALDI_VLOG(1) << loglikes1(j) << " " << (*loglikes[i])(j);
          if ((*(loglikes[i]))(j) > max_like) {
            prediction(j) = i+1;
            max_like = (*(loglikes[i]))(j);
          }
        }
        if (weights_wspecifier != "") {
          weights(j) = Exp(max_like - this_log_likes.LogSumExp());
          KALDI_ASSERT(weights(j) <= 1.0);
        }
        class_loglikes[prediction(j)] += max_like;
        class_counts[prediction(j)]++;
      }
    
      for (int32 i = 0; i < po.NumArgs()-2; i++) {
        delete loglikes[i];
      }

      prediction_writer.Write(key, prediction);
      if (weights_wspecifier != "") 
        weights_writer.Write(key, weights);

      num_done++;
    }
    
    KALDI_LOG << "Average log-likelihood of frames of class " << 0 
              << " is " << class_loglikes[0] / class_counts[0]
              << " over " << class_counts[0] << " frames.";

    for (int32 i = 0; i < po.NumArgs()-2; i++) {
      delete loglikes_readers[i];
      KALDI_LOG << "Average log-likelihood of frames of class " << i+1 
                << " is " << class_loglikes[i+1] / class_counts[i+1]
                << " over " << class_counts[i+1] << " frames.";
    }

    KALDI_LOG << "Converted " << num_done << " sets of log-likes vectors to predictions.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

