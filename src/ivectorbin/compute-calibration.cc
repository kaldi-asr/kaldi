// ivectorbin/compute-calibration.cc

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
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Computes a calibration threshold from scores (e.g., PLDA scores)."
      "Generally, the scores are the result of a comparison between two"
      "iVectors.  This is typically used to find the stopping criteria for"
      "agglomerative clustering."
      "Usage: compute-calibration [options] <scores-rspecifier> "
      "<calibration-wxfilename>\n"
      "e.g.: \n"
      " compute-calibration ark:scores.ark threshold.txt\n";

    ParseOptions po(usage);
    bool read_matrices = true;
    po.Register("read-matrices", &read_matrices, "If true, read scores as"
      "matrices, probably output from ivector-plda-scoring-dense");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rspecifier = po.GetArg(1),
      calibration_wxfilename = po.GetArg(2);
    ClusterKMeansOptions opts;
    BaseFloat mean = 0.0;
    int32 num_done = 0,
      num_err = 0;
    Output output(calibration_wxfilename, false);
    if (read_matrices) {
      SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
      for (; !scores_reader.Done(); scores_reader.Next()) {
        std::string utt = scores_reader.Key();
        const Matrix<BaseFloat> scores = scores_reader.Value();
        if (scores.NumRows() <= 2 && scores.NumCols() <= 2) {
          KALDI_WARN << "Too few scores in " << utt << " to cluster";
          num_err++;
          continue;
        }
        std::vector<Clusterable*> this_clusterables;
        std::vector<Clusterable*> this_clusters;
        for (int32 i = 0; i < scores.NumRows(); i++) {
          for (int32 j = 0; j < scores.NumCols(); j++) {
            this_clusterables.push_back(new ScalarClusterable(scores(i,j)));
          }
        }
        ClusterKMeans(this_clusterables, 2, &this_clusters, NULL, opts);
        DeletePointers(&this_clusterables);
        BaseFloat this_mean1 = dynamic_cast<ScalarClusterable*>(
          this_clusters[0])->Mean(),
          this_mean2 = dynamic_cast<ScalarClusterable*>(
          this_clusters[1])->Mean();
        mean += this_mean1 + this_mean2;
        num_done++;
      }
      mean = mean / (2*num_done);
    } else {
      std::vector<Clusterable*> clusterables;
      std::vector<Clusterable*> clusters;
      SequentialBaseFloatReader scores_reader(scores_rspecifier);
      for (; !scores_reader.Done(); scores_reader.Next()) {
        std::string utt = scores_reader.Key();
        const BaseFloat score = scores_reader.Value();
        clusterables.push_back(new ScalarClusterable(score));
        num_done++;
      }
      ClusterKMeans(clusterables, 2, &clusters, NULL, opts);
      DeletePointers(&clusterables);
      BaseFloat mean1 = dynamic_cast<ScalarClusterable*>(clusters[0])->Mean(),
        mean2 = dynamic_cast<ScalarClusterable*>(clusters[1])->Mean();
      mean = (mean1 + mean2) / 2;
    }
    output.Stream() << mean;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
