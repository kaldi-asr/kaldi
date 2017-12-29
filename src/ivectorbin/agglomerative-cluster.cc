// ivectorbin/agglomerative-cluster.cc

// Copyright 2016  David Snyder
//           2017  Matthew Maciejewski

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
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Cluster utterances by score, used in diarization.\n"
      "Takes a table of score matrices indexed by recording,\n"
      "with the rows/columns corresponding to the utterances\n"
      "of that recording in sorted order and a spk2utt file that\n"
      "contains the mapping from recordings to utterances, and \n"
      "outputs a list of labels in the form <utt1> <label>.\n"
      "Clustering is done using Agglomerative Hierarchical\n"
      "Clustering with a score threshold as stop criterion.\n"
      "Usage: agglomerative-cluster [options] <scores-rspecifier> "
      "<spk2utt-rspecifier> <labels-wspecifier>\n"
      "e.g.: \n"
      " agglomerative-cluster ark:scores.ark ark:spk2utt \n"
      "   ark,t:labels.txt\n";

    ParseOptions po(usage);
    std::string spk2num_rspecifier;
    BaseFloat threshold = 0.5;
    BaseFloat max_dist = 1.0;

    po.Register("spk2num-rspecifier", &spk2num_rspecifier,
      "If supplied, clustering creates exactly this many clusters for each"
      "recording and the option --threshold is ignored.");
    po.Register("threshold", &threshold, "Merging clusters if their distance"
      "is less than this threshold.");
    po.Register("max-dist", &max_dist, "Values missing from scores file"
      "are assigned this value.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rspecifier = po.GetArg(1),
      spk2utt_rspecifier = po.GetArg(2),
      label_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
    RandomAccessTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessInt32Reader spk2num_reader(spk2num_rspecifier);
    Int32Writer label_writer(label_wspecifier);

    for (; !scores_reader.Done(); scores_reader.Next()) {
      std::string spk = scores_reader.Key();
      const Matrix<BaseFloat> &scores = scores_reader.Value();
      std::vector<std::string> uttlist = spk2utt_reader.Value(spk);
      std::vector<int32> spk_ids;
      if (spk2num_rspecifier.size()) {
        int32 num_speakers = spk2num_reader.Value(spk);
        AgglomerativeCluster(scores,
          std::numeric_limits<BaseFloat>::max(), num_speakers, &spk_ids);
      } else {
        AgglomerativeCluster(scores, threshold, 1, &spk_ids);
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
