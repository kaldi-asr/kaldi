// ivectorbin/agglomerative-cluster.cc

// Copyright 2016-2018  David Snyder
//           2017-2018  Matthew Maciejewski

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
#include "util/stl-utils.h"
#include "ivector/agglomerative-clustering.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "Cluster utterances by similarity score, used in diarization.\n"
      "Takes a table of score matrices indexed by recording, with the\n"
      "rows/columns corresponding to the utterances of that recording in\n"
      "sorted order and a reco2utt file that contains the mapping from\n"
      "recordings to utterances, and outputs a list of labels in the form\n"
      "<utt> <label>.  Clustering is done using agglomerative hierarchical\n"
      "clustering with a score threshold as stop criterion.  By default, the\n"
      "program reads in similarity scores, but with --read-costs=true\n"
      "the scores are interpreted as costs (i.e. a smaller value indicates\n"
      "utterance similarity).\n"
      "Usage: agglomerative-cluster [options] <scores-rspecifier> "
      "<reco2utt-rspecifier> <labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster ark:scores.ark ark:reco2utt \n"
      "   ark,t:labels.txt\n";

    ParseOptions po(usage);
    std::string reco2num_spk_rspecifier;
    BaseFloat threshold = 0.0;
    bool read_costs = false;

    po.Register("reco2num-spk-rspecifier", &reco2num_spk_rspecifier,
      "If supplied, clustering creates exactly this many clusters for each"
      " recording and the option --threshold is ignored.");
    po.Register("threshold", &threshold, "Merge clusters if their distance"
      " is less than this threshold.");
    po.Register("read-costs", &read_costs, "If true, the first"
      " argument is interpreted as a matrix of costs rather than a"
      " similarity matrix.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rspecifier = po.GetArg(1),
      reco2utt_rspecifier = po.GetArg(2),
      label_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
    RandomAccessTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessInt32Reader reco2num_spk_reader(reco2num_spk_rspecifier);
    Int32Writer label_writer(label_wspecifier);

    if (!read_costs)
      threshold = -threshold;
    for (; !scores_reader.Done(); scores_reader.Next()) {
      std::string reco = scores_reader.Key();
      Matrix<BaseFloat> costs = scores_reader.Value();
      // By default, the scores give the similarity between pairs of
      // utterances.  We need to multiply the scores by -1 to reinterpet
      // them as costs (unless --read-costs=true) as the agglomerative
      // clustering code requires.
      if (!read_costs)
        costs.Scale(-1);
      std::vector<std::string> uttlist = reco2utt_reader.Value(reco);
      std::vector<int32> spk_ids;
      if (reco2num_spk_rspecifier.size()) {
        int32 num_speakers = reco2num_spk_reader.Value(reco);
        AgglomerativeCluster(costs,
          std::numeric_limits<BaseFloat>::max(), num_speakers, &spk_ids);
      } else {
        AgglomerativeCluster(costs, threshold, 1, &spk_ids);
      }
      for (int32 i = 0; i < spk_ids.size(); i++)
        label_writer.Write(uttlist[i], spk_ids[i]);
    }
    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
