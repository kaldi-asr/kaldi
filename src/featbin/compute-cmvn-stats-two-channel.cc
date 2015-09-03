// featbin/compute-cmvn-stats-two-channel.cc

// Copyright          2013  Johns Hopkins University (author: Daniel Povey)

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

namespace kaldi {


/*
  This function gets the utterances that are the first field of the
  contents of the file reco2file_and_channel_rxfilename, and sorts
  them into pairs corresponding to A/B sides, or singletons in case
  we get one without the other.
 */
void GetUtterancePairs(const std::string &reco2file_and_channel_rxfilename,
                       std::vector<std::vector<std::string> > *utt_pairs) {
  Input ki(reco2file_and_channel_rxfilename);
  std::string line;
  std::map<std::string, std::vector<std::string> > call_to_uttlist;
  while (std::getline(ki.Stream(), line)) {
    std::vector<std::string> split_line;
    SplitStringToVector(line, " \t\r", true, &split_line);
    if (split_line.size() != 3) {
      KALDI_ERR << "Expecting 3 fields per line of reco2file_and_channel file "
                << PrintableRxfilename(reco2file_and_channel_rxfilename)
                << ", got: " << line;
    }
    // lines like: sw02001-A sw02001 A
    std::string utt = split_line[0],
        call = split_line[1];
    call_to_uttlist[call].push_back(utt);
  }
  for (std::map<std::string, std::vector<std::string> >::const_iterator
         iter = call_to_uttlist.begin(); iter != call_to_uttlist.end(); ++iter) {
    const std::vector<std::string> &uttlist = iter->second;
    if (uttlist.size() == 2) {
      utt_pairs->push_back(uttlist);
    } else {
      KALDI_WARN << "Call " << iter->first << " has " << uttlist.size()
                 << " utterances, expected two; treating them singly.";
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::vector<std::string> singleton_list;
        singleton_list.push_back(uttlist[i]);
        utt_pairs->push_back(singleton_list);
      }
    }
  }
}

void AccCmvnStatsForPair(const std::string &utt1, const std::string &utt2,
                         const MatrixBase<BaseFloat> &feats1,
                         const MatrixBase<BaseFloat> &feats2,
                         BaseFloat quieter_channel_weight,
                         MatrixBase<double> *cmvn_stats1,
                         MatrixBase<double> *cmvn_stats2) {
  KALDI_ASSERT(feats1.NumCols() == feats2.NumCols()); // same dim.
  if (feats1.NumRows() != feats2.NumRows()) {
    KALDI_WARN << "Number of frames differ between " << utt1 << " and " << utt2
               << ": " << feats1.NumRows() << " vs. " << feats2.NumRows()
               << ", treating them separately.";
    AccCmvnStats(feats1, NULL, cmvn_stats1);
    AccCmvnStats(feats2, NULL, cmvn_stats2);
    return;
  }

  for (int32 i = 0; i < feats1.NumRows(); i++) {
    if (feats1(i, 0) > feats2(i, 0)) {
      AccCmvnStats(feats1.Row(i), 1.0, cmvn_stats1);
      AccCmvnStats(feats2.Row(i), quieter_channel_weight, cmvn_stats2);
    }
    else {
      AccCmvnStats(feats2.Row(i), 1.0, cmvn_stats2);
      AccCmvnStats(feats1.Row(i), quieter_channel_weight, cmvn_stats1);
    }
  }
}


}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Compute cepstral mean and variance normalization statistics\n"
        "Specialized for two-sided telephone data where we only accumulate\n"
        "the louder of the two channels at each frame (and add it to that\n"
        "side's stats).  Reads a 'reco2file_and_channel' file, normally like\n"
        "sw02001-A sw02001 A\n"
        "sw02001-B sw02001 B\n"
        "sw02005-A sw02005 A\n"
        "sw02005-B sw02005 B\n"
        "interpreted as <utterance-id> <call-id> <side> and for each <call-id>\n"
        "that has two sides, does the 'only-the-louder' computation, else doesn\n"
        "per-utterance stats in the normal way.\n"
        "Note: loudness is judged by the first feature component, either energy or c0;\n"
        "only applicable to MFCCs or PLPs (this code could be modified to handle filterbanks).\n"
        "\n"
        "Usage: compute-cmvn-stats-two-channel  [options] <reco2file-and-channel> <feats-rspecifier> <stats-wspecifier>\n"
        "e.g.: compute-cmvn-stats-two-channel data/train_unseg/reco2file_and_channel scp:data/train_unseg/feats.scp ark,t:-\n";
        
    
    ParseOptions po(usage);
    BaseFloat quieter_channel_weight = 0.01;

    po.Register("quieter-channel-weight", &quieter_channel_weight,
                "For the quieter channel, apply this weight to the stats, so "
                "that we still get stats if one channel always dominates.");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0, num_err = 0;

    std::string reco2file_and_channel_rxfilename = po.GetArg(1),
        feats_rspecifier = po.GetArg(2),
        stats_wspecifier = po.GetArg(3);


    std::vector<std::vector<std::string> > utt_pairs;
    GetUtterancePairs(reco2file_and_channel_rxfilename, &utt_pairs);
    
    RandomAccessBaseFloatMatrixReader feat_reader(feats_rspecifier);
    DoubleMatrixWriter writer(stats_wspecifier);

    for (size_t i = 0; i < utt_pairs.size(); i++) {
      std::vector<std::string> this_pair(utt_pairs[i]);

      KALDI_ASSERT(this_pair.size() == 2 || this_pair.size() == 1);
      if (this_pair.size() == 2) {
        std::string utt1 = this_pair[0], utt2 = this_pair[1];
        if (!feat_reader.HasKey(utt1)) {
          KALDI_WARN << "No feature data for utterance " << utt1;
          num_err++;          
          this_pair[0] = utt2;
          this_pair.pop_back();
          // and fall through to the singleton code below.
        } else if (!feat_reader.HasKey(utt2)) {
          KALDI_WARN << "No feature data for utterance " << utt2;
          num_err++;
          this_pair.pop_back();
          // and fall through to the singleton code below.
        } else {
          Matrix<BaseFloat> feats1 = feat_reader.Value(utt1),
              feats2 = feat_reader.Value(utt2);
          int32 dim = feats1.NumCols();
          Matrix<double> cmvn_stats1(2, dim + 1), cmvn_stats2(2, dim + 1);
          AccCmvnStatsForPair(utt1, utt2, feats1, feats2, quieter_channel_weight,
                              &cmvn_stats1, &cmvn_stats2);
          writer.Write(utt1, cmvn_stats1);
          writer.Write(utt2, cmvn_stats2);
          num_done += 2;
          continue; // continue so we don't go to the singleton-processing code
                    // below.
        }
      }
      // process singletons.
      std::string utt = this_pair[0];
      if (!feat_reader.HasKey(utt)) {
        KALDI_WARN << "No feature data for utterance " << utt;
        num_err++;
        continue;
      }
      const Matrix<BaseFloat> &feats = feat_reader.Value(utt);
      Matrix<double> cmvn_stats(2, feats.NumCols() + 1);
      AccCmvnStats(feats, NULL, &cmvn_stats);
      writer.Write(utt, cmvn_stats);
      num_done++;
    }
    KALDI_LOG << "Done accumulating CMVN stats for " << num_done
              << " utterances; " << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


