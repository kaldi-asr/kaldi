// featbin/compute-cmvn-stats-balanced.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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
#include "transform/balanced-cmvn.h"

namespace kaldi {

bool AccCmvnStatsWrapper(std::string utt,
                        const MatrixBase<BaseFloat> &feats,
                        RandomAccessBaseFloatVectorReader *weights_reader,
                        BalancedCmvn *stats) {
  if (!weights_reader->HasKey(utt)) {
    KALDI_WARN << "No weights available for utterance " << utt;
    return false;
  }
  const Vector<BaseFloat> &weights = weights_reader->Value(utt);
  if (weights.Dim() != feats.NumRows()) {
    KALDI_WARN << "Weights for utterance " << utt << " have wrong dimension "
               << weights.Dim() << " vs. " << feats.NumRows();
    return false;
  }
  stats->AccStats(feats, weights);
  return true;
}
                  

} // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Compute cepstral mean and variance normalization statistics\n"
        " Per-utterance by default, or per-speaker if spk2utt option provided.\n"
        " This version of the program uses speech/silence probabilities in\n"
        " <weights-wspecifier> to refine the normalization (see code for details)\n"
        " Note: the cmvn stats we write are not \"real\" statistics, but \"faked\"\n"
        "ones that will give us the normalization we want.\n"
        "\n"
        "Usage: compute-cmvn-stats-balanced [options] <sil-global-cmvn-stats> <nonsil-global-cmvn-stats> "
        "<feats-rspecifier> <nonsilence-weight-rspecifier> <cmvn-stats-wspecifier>\n";
    
    ParseOptions po(usage);
    std::string spk2utt_rspecifier;
    bool binary = true;
    BalancedCmvnConfig config;
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to utterance-list map");
    po.Register("binary", &binary, "write in binary mode (applies only to global CMVN/CVN)");
    config.Register(&po);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0, num_err = 0;

    std::string sil_cmvn_stats_rxfilename = po.GetArg(1),
        nonsil_cmvn_stats_rxfilename = po.GetArg(2),
        feats_rspecifier = po.GetArg(3),
        weights_rspecifier = po.GetArg(4),
        cmvn_stats_wspecifier = po.GetArg(5);

    Matrix<double> sil_cmvn_stats, nonsil_cmvn_stats;
    ReadKaldiObject(sil_cmvn_stats_rxfilename, &sil_cmvn_stats);
    ReadKaldiObject(nonsil_cmvn_stats_rxfilename, &nonsil_cmvn_stats);

    RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);
    DoubleMatrixWriter writer(cmvn_stats_wspecifier);
    
    if (spk2utt_rspecifier != "") {
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feat_reader(feats_rspecifier);
      
      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        std::string spk = spk2utt_reader.Key();
        const std::vector<std::string> &uttlist = spk2utt_reader.Value();
        BalancedCmvn stats(config, sil_cmvn_stats, nonsil_cmvn_stats);
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feat_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            num_err++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feat_reader.Value(utt);
          if (!AccCmvnStatsWrapper(utt, feats, &weights_reader, &stats)) {
            num_err++;
          } else {
            num_done++;
          }
        }
        if (stats.TotCount() == 0) {
          KALDI_WARN << "No stats accumulated for speaker " << spk;
        } else {
          writer.Write(spk, stats.GetStats());
        }
      }
    } else {  // per-utterance normalization
      SequentialBaseFloatMatrixReader feat_reader(feats_rspecifier);
        
      for (; !feat_reader.Done(); feat_reader.Next()) {
        std::string utt = feat_reader.Key();
        BalancedCmvn stats(config, sil_cmvn_stats, nonsil_cmvn_stats);
        const Matrix<BaseFloat> &feats = feat_reader.Value();
        
        if (!AccCmvnStatsWrapper(utt, feats, &weights_reader, &stats)) {
          num_err++;
          continue;
        }
        writer.Write(feat_reader.Key(), stats.GetStats());
        num_done++;
      }
    }
    KALDI_LOG << "Done accumulating CMVN stats for " << num_done
              << " utterances; " << num_err << " had errors.";
   return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


