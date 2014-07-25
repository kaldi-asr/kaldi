// featbin/apply-cmvn.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University

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
#include "transform/cmvn.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Apply cepstral mean and (optionally) variance normalization\n"
        "Per-utterance by default, or per-speaker if utt2spk option provided\n"
        "Usage: apply-cmvn [options] (cmvn-stats-rspecifier|cmvn-stats-rxfilename) feats-rspecifier feats-wspecifier\n";

    ParseOptions po(usage);
    std::string utt2spk_rspecifier;
    bool norm_vars = false;
    bool norm_means = true;
    std::string skip_dims_str;
    
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("norm-vars", &norm_vars, "If true, normalize variances.");
    po.Register("norm-means", &norm_means, "You can set this to false to turn off mean "
                "normalization.  Note, the same can be achieved by using 'fake' CMVN stats; "
                "see the --fake option to compute_cmvn_stats.sh");
    po.Register("skip-dims", &skip_dims_str, "Dimensions for which to skip "
                "normalization: colon-separated list of integers, e.g. 13:14:15)");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    if (norm_vars && !norm_means)
      KALDI_ERR << "You cannot normalize the variance but not the mean.";

    std::vector<int32> skip_dims;  // optionally use "fake"
                                   // (zero-mean/unit-variance) stats for some
                                   // dims to disable normalization.
    if (!SplitStringToIntegers(skip_dims_str, ":", false, &skip_dims)) {
      KALDI_ERR << "Bad --skip-dims option (should be colon-separated list of "
                <<  "integers)";
    }
    
    
    kaldi::int32 num_done = 0, num_err = 0;
    
    std::string cmvn_rspecifier_or_rxfilename = po.GetArg(1);
    std::string feat_rspecifier = po.GetArg(2);
    std::string feat_wspecifier = po.GetArg(3);
    
    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);
    
    if (ClassifyRspecifier(cmvn_rspecifier_or_rxfilename, NULL, NULL)
        != kNoRspecifier) { // reading from a Table: per-speaker or per-utt CMN/CVN.
      std::string cmvn_rspecifier = cmvn_rspecifier_or_rxfilename;

      RandomAccessDoubleMatrixReaderMapped cmvn_reader(cmvn_rspecifier,
                                                       utt2spk_rspecifier);
      
      for (; !feat_reader.Done(); feat_reader.Next()) {
        std::string utt = feat_reader.Key();
        Matrix<BaseFloat> feat(feat_reader.Value());
        if (norm_means) {
          if (!cmvn_reader.HasKey(utt)) {
            KALDI_WARN << "No normalization statistics available for key "
                       << utt << ", producing no output for this utterance";
            num_err++;
            continue;
          }
          Matrix<double> cmvn_stats = cmvn_reader.Value(utt);
          if (!skip_dims.empty())
            FakeStatsForSomeDims(skip_dims, &cmvn_stats);
          
          ApplyCmvn(cmvn_stats, norm_vars, &feat);
        
          feat_writer.Write(utt, feat);
        } else {
          feat_writer.Write(utt, feat);
        }
        num_done++;
      }
    } else {
      if (utt2spk_rspecifier != "")
        KALDI_ERR << "--utt2spk option not compatible with rxfilename as input "
                  << "(did you forget ark:?)";
      std::string cmvn_rxfilename = cmvn_rspecifier_or_rxfilename;
      bool binary;
      Input ki(cmvn_rxfilename, &binary);
      Matrix<double> cmvn_stats;
      cmvn_stats.Read(ki.Stream(), binary);
      if (!skip_dims.empty())
        FakeStatsForSomeDims(skip_dims, &cmvn_stats);
      
      for (;!feat_reader.Done(); feat_reader.Next()) {
        std::string utt = feat_reader.Key();
        Matrix<BaseFloat> feat(feat_reader.Value());
        if (norm_means)
          ApplyCmvn(cmvn_stats, norm_vars, &feat);
        feat_writer.Write(utt, feat);
        num_done++;
      }
    }
    if (norm_vars) 
      KALDI_LOG << "Applied cepstral mean and variance normalization to "
                << num_done << " utterances, errors on " << num_err;
    else
      KALDI_LOG << "Applied cepstral mean normalization to "
                << num_done << " utterances, errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


