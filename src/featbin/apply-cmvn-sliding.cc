// featbin/apply-cmvn-sliding.cc

// Copyright  2013     Johns Hopkins University (Author: Daniel Povey)

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
#include "feat/feature-functions.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Apply sliding-window cepstral mean (and optionally variance)\n"
        "normalization per utterance.  If center == true, window is centered\n"
        "on frame being normalized; otherwise it precedes it in time.\n"
        "Useful for speaker-id and for offline training of a system intended\n"
        "for use with online feature normalization, as in src/online/.\n"
        "\n"
        "Usage: apply-cmvn-sliding [options] <feats-rspecifier> <feats-wspecifier>\n";
    
    ParseOptions po(usage);
    SlidingWindowCmnOptions opts;
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0, num_err = 0;
    
    std::string feat_rspecifier = po.GetArg(1);
    std::string feat_wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);
    
    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      Matrix<BaseFloat> feat(feat_reader.Value());
      if (feat.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for utterance " << utt;
        num_err++;
        continue;
      }
      Matrix<BaseFloat> cmvn_feat(feat.NumRows(),
                                  feat.NumCols(), kUndefined);

      SlidingWindowCmn(opts, feat, &cmvn_feat);
      
      feat_writer.Write(utt, cmvn_feat);
      num_done++;
    }

    KALDI_LOG << "Applied sliding-window cepstral mean "
              << (opts.normalize_variance ? "and variance " : "")
              << "normalization to " << num_done << " utterances, "
              << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


