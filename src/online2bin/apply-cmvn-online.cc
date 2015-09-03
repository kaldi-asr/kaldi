// online2bin/apply-cmvn-online.cc

// Copyright      2014  Johns Hopkins University (author: Daniel Povey)

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

#include <string>
#include <vector>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/online-feature.h"

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Apply online cepstral mean (and possibly variance) computation online,\n"
        "using the same code as used for online decoding in the 'new' setup in\n"
        "online2/ and online2bin/.  If the --spk2utt option is used, it uses\n"
        "prior utterances from the same speaker to back off to at the utterance\n"
        "beginning.  See also apply-cmvn-sliding.\n"
        "\n"
        "Usage: apply-cmvn-online [options] <global-cmvn-stats> <feature-rspecifier> "
        "<feature-wspecifier>\n"
        "e.g. apply-cmvn-online 'matrix-sum scp:data/train/cmvn.scp -|' data/train/split8/1/feats.scp ark:-\n"
        "or: apply-cmvn-online --spk2utt=ark:data/train/split8/1/spk2utt 'matrix-sum scp:data/train/cmvn.scp -|' "
        " data/train/split8/1/feats.scp ark:-\n";
    
    ParseOptions po(usage);

    OnlineCmvnOptions cmvn_opts;
    
    std::string spk2utt_rspecifier;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
                "utterance-list map");
    cmvn_opts.Register(&po);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string global_stats_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);

    // global_cmvn_stats helps us initialize to online CMVN to
    // reasonable values at the beginning of the utterance.
    Matrix<double> global_cmvn_stats;
    ReadKaldiObject(global_stats_rxfilename, &global_cmvn_stats);

    
    
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);
    int32 num_done = 0, num_err = 0;
    int64 tot_t = 0;

    if (spk2utt_rspecifier != "") {
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        OnlineCmvnState cmvn_state(global_cmvn_stats);
        const std::vector<std::string> &uttlist = spk2utt_reader.Value();
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "No features for utterance " << utt;
            num_err++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          
          Matrix<BaseFloat> normalized_feats(feats.NumRows(), feats.NumCols(),
                                             kUndefined);

          OnlineMatrixFeature online_matrix(feats);
          OnlineCmvn online_cmvn(cmvn_opts,
                                 cmvn_state,
                                 &online_matrix);

          for (int32 t = 0; t < feats.NumRows(); t++) {
            SubVector<BaseFloat> row(normalized_feats, t);
            online_cmvn.GetFrame(t, &row);
          }
          online_cmvn.GetState(feats.NumRows() - 1, &cmvn_state);
          
          num_done++;
          tot_t += feats.NumRows();
          feature_writer.Write(utt, normalized_feats);
        }
      }
    } else {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        const Matrix<BaseFloat> &feats = feature_reader.Value();
        OnlineCmvnState cmvn_state(global_cmvn_stats);

        Matrix<BaseFloat> normalized_feats(feats.NumRows(), feats.NumCols(),
                                           kUndefined);
        OnlineMatrixFeature online_matrix(feats);
        OnlineCmvn online_cmvn(cmvn_opts,
                               cmvn_state,
                               &online_matrix);

        for (int32 t = 0; t < feats.NumRows(); t++) {
          SubVector<BaseFloat> row(normalized_feats, t);
          online_cmvn.GetFrame(t, &row);
        }
        num_done++;
        tot_t += feats.NumRows();
        feature_writer.Write(utt, normalized_feats);
        
        num_done++;
      }
    }
    
    KALDI_LOG << "Applied online CMVN to " << num_done << " files, or "
              << tot_t << " frames.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

