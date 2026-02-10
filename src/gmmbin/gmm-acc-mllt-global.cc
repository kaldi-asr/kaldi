// gmmbin/gmm-acc-mllt-global.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Guoguo Chen
//                2014  Daniel Povey

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
#include "transform/mllt.h"
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Accumulate MLLT (global STC) statistics: this version is for where there is\n"
        "one global GMM (e.g. a UBM)\n"
        "Usage:  gmm-acc-mllt-global [options] <gmm-in> <feature-rspecifier> <stats-out>\n"
        "e.g.: \n"
         " gmm-acc-mllt-global 1.dubm scp:feats.scp 1.macc\n";

    ParseOptions po(usage);
    bool binary = true;
    BaseFloat rand_prune = 0.25;
    std::string gselect_rspecifier;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("rand-prune", &rand_prune, "Randomized pruning parameter to speed up "
                "accumulation (larger -> more pruning.  May exceed one).");
    po.Register("gselect", &gselect_rspecifier, "Rspecifier for Gaussian selection "
                "information");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string gmm_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        accs_wxfilename = po.GetArg(3);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    DiagGmm gmm;
    ReadKaldiObject(gmm_filename, &gmm);

    MlltAccs mllt_accs(gmm.Dim(), rand_prune);

    double tot_like = 0.0;
    double tot_t = 0.0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    
    int32 num_done = 0, num_err = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      const Matrix<BaseFloat> &mat = feature_reader.Value();      
      
      num_done++;
      BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;
      
      if (gselect_rspecifier == "") {
        for (int32 i = 0; i < mat.NumRows(); i++) {
          tot_like_this_file += mllt_accs.AccumulateFromGmm(gmm, mat.Row(i), 1.0);
          tot_weight += 1.0;
        }
      } else {
        if (!gselect_reader.HasKey(utt)) {
          KALDI_WARN << "No gselect information for utterance " << utt;
          num_err++;
          continue;
        }
        const std::vector<std::vector<int32> > &gselect= gselect_reader.Value(utt);
        if (static_cast<int32>(gselect.size()) != mat.NumRows()) {
          KALDI_WARN << "Gselect information has wrong size for utterance "
                     << utt << ", " << gselect.size() << " vs. "
                     << mat.NumRows();
          num_err++;
          continue;
        }
        
        for (int32 i = 0; i < mat.NumRows(); i++) {
          tot_like_this_file += mllt_accs.AccumulateFromGmmPreselect(
              gmm, gselect[i], mat.Row(i), 1.0);
          tot_weight += 1.0;
        }
      }
      KALDI_LOG << "Average like for this file is "
                << (tot_like_this_file/tot_weight) << " over "
                << tot_weight << " frames.";
      tot_like += tot_like_this_file;
      tot_t += tot_weight;
      if (num_done % 10 == 0)
        KALDI_LOG << "Avg like per frame so far is "
                  << (tot_like/tot_t);
    }
  
    KALDI_LOG << "Done " << num_done << " files. ";
    
    KALDI_LOG << "Overall avg like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";

    WriteKaldiObject(mllt_accs, accs_wxfilename, binary);
    KALDI_LOG << "Written accs.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


