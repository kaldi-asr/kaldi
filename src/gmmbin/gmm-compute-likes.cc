// gmmbin/gmm-compute-likes.cc

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
#include "fstext/fstext-lib.h"
#include "base/timer.h"



int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Compute log-likelihoods from GMM-based model\n"
        "(outputs matrices of log-likelihoods indexed by (frame, pdf)\n"
        "Usage: gmm-compute-likes [options] model-in features-rspecifier likes-wspecifier\n";
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        loglikes_wspecifier = po.GetArg(3);

    AmDiagGmm am_gmm;
    {
      bool binary;
      TransitionModel trans_model;  // not needed.
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    BaseFloatMatrixWriter loglikes_writer(loglikes_wspecifier);
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    int32 num_done = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &features (feature_reader.Value());
      Matrix<BaseFloat> loglikes(features.NumRows(), am_gmm.NumPdfs());
      for (int32 i = 0; i < features.NumRows(); i++) {
        for (int32 j = 0; j < am_gmm.NumPdfs(); j++) {
          SubVector<BaseFloat> feat_row(features, i);
          loglikes(i, j) = am_gmm.LogLikelihood(j, feat_row);
        }
      }
      loglikes_writer.Write(key, loglikes);
      num_done++;
    }

    KALDI_LOG << "gmm-compute-likes: computed likelihoods for " << num_done
              << " utterances.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


