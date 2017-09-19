// ivectorbin/agglomerative-cluster.cc

// Copyright 2016  David Snyder

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
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"
#include "ivector/group-clusterable.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Cluster matrices of scores per utterance. Used in diarization\n"
      "TODO better documentation\n"
      "Usage: agglomerative-cluster [options] <scores-rspecifier> "
      "<spk2utt-rspecifier> <labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster ark:scores.ark ark:spk2utt \n"
      "   ark,t:labels.txt\n";

    ParseOptions po(usage);
    std::string utt2num_rspecifier;
    BaseFloat threshold = 0.5;

    po.Register("utt2num-rspecifier", &utt2num_rspecifier,
      "If supplied, clustering creates exactly this many clusters for each"
      "utterance and the option --threshold is ignored.");
    po.Register("threshold", &threshold, "Merging clusters if their distance"
      "is less than this threshold.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rspecifier = po.GetArg(1),
      spk2utt_rspecifier = po.GetArg(2),
      label_wspecifier = po.GetArg(3);

    // TODO  Maybe should make the PLDA scoring binary output segmentation so that this can read it
    // directly. If not, at least make sure the utt2seg in that binary is NOT sorted. Might sort it in a different
    // order than here.
    SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
    RandomAccessTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessInt32Reader utt2num_reader(utt2num_rspecifier);
    Int32Writer label_writer(label_wspecifier);

    for (; !scores_reader.Done(); scores_reader.Next()) {
      std::string utt = scores_reader.Key();
      const Matrix<BaseFloat> &scores = scores_reader.Value();
      std::vector<std::string> seglist = spk2utt_reader.Value(utt);
      std::sort(seglist.begin(), seglist.end());

      std::vector<Clusterable*> clusterables;
      std::vector<int32> spk_ids;

      for (int32 i = 0; i < scores.NumRows(); i++) {
        std::set<int32> points;
        points.insert(i);
        clusterables.push_back(new GroupClusterable(points, &scores));
      }
      if (utt2num_rspecifier.size()) {
        int32 num_speakers = utt2num_reader.Value(utt);
        ClusterBottomUp(clusterables, std::numeric_limits<BaseFloat>::max(),
          num_speakers, NULL, &spk_ids);
      } else {
        ClusterBottomUp(clusterables, threshold, 1, NULL, &spk_ids);
      }
      for (int32 i = 0; i < spk_ids.size(); i++)
        label_writer.Write(seglist[i], spk_ids[i]);
      DeletePointers(&clusterables);
    }
    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
