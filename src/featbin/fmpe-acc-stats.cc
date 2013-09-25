// featbin/fmpe-acc-stats.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
#include "transform/fmpe.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using kaldi::int32;
  try {
    const char *usage =
        "Compute statistics for fMPE training\n"
        "Usage:  fmpe-acc-stats [options...] <fmpe-object> "
        "<feat-rspecifier> <feat-diff-rspecifier> <gselect-rspecifier> <stats-out>\n"
        "Note: gmm-fmpe-acc-stats avoids computing the features an extra time\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "If true, output stats in binary mode.");
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string fmpe_rxfilename = po.GetArg(1),
        feat_rspecifier = po.GetArg(2),
        feat_diff_rspecifier = po.GetArg(3),
        gselect_rspecifier = po.GetArg(4),
        stats_wxfilename = po.GetArg(5);
    
    Fmpe fmpe;
    ReadKaldiObject(fmpe_rxfilename, &fmpe);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    RandomAccessBaseFloatMatrixReader diff_reader(feat_diff_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);

    // fmpe stats...
    FmpeStats fmpe_stats(fmpe);

    int32 num_done = 0, num_err = 0;
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> feat_in(feat_reader.Value());
      if (!gselect_reader.HasKey(key)) {
        KALDI_WARN << "No gselect information for key " << key;
        num_err++;
        continue;
      }
      const std::vector<std::vector<int32> > &gselect =
          gselect_reader.Value(key);
      if (static_cast<int32>(gselect.size()) != feat_in.NumRows()) {
        KALDI_WARN << "gselect information has wrong size";
        num_err++;
        continue;
      }
      if (!diff_reader.HasKey(key)) {
        KALDI_WARN << "No gradient information for key " << key;
        num_err++;
        continue;
      }
      const Matrix<BaseFloat> &feat_deriv = diff_reader.Value(key);

      if (feat_deriv.NumCols() == feat_in.NumCols()) { // Only direct derivative.
        fmpe.AccStats(feat_in, gselect, feat_deriv, NULL, &fmpe_stats);
      } else if (feat_deriv.NumCols() == feat_in.NumCols() * 2) { // +indirect.
        SubMatrix<BaseFloat> direct_deriv(feat_deriv, 0, feat_deriv.NumRows(),
                                          0, feat_in.NumCols()),
            indirect_deriv(feat_deriv, 0, feat_deriv.NumRows(),
                           feat_in.NumCols(), feat_in.NumCols());
        fmpe.AccStats(feat_in, gselect, direct_deriv, &indirect_deriv, &fmpe_stats);
      } else {
        KALDI_ERR << "Mismatch in dimension of feature derivative.";
      }
      num_done++;
    }

    KALDI_LOG << " Done " << num_done << " utterances, " << num_err
              << " had errors.";

    WriteKaldiObject(fmpe_stats, stats_wxfilename, binary);
    
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
