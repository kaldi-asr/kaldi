// ivectorbin/compute-calibration.cc

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
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
      "Computes a calibration threshold from scores (e.g., PLDA scores).\n"
      "Generally, the scores are the result of a comparison between two\n"
      "iVectors.  This is typically used to find the stopping criteria for\n"
      "agglomerative clustering.\n"
      "Usage: compute-calibration [options] <scores-rspecifier|scores-rxfilename>\n"
      "<calibration-wxfilename|calibration-wspecifier>\n"
      "e.g.: \n"
      " compute-calibration ark:scores.ark threshold.txt\n";

    ParseOptions po(usage);
    std::string spk2utt_rspecifier;
    bool read_matrices = true;

    po.Register("spk2utt-rspecifier", &spk2utt_rspecifier, "If supplied,\n"
      "computes a threshold per spk and writes a table accordingly");
    po.Register("read-matrices", &read_matrices, "If true, reads scores\n"
      "as matrices, probably output from ivector-plda-scoring-dense");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string scores_rspecifier = po.GetArg(1),
        calibration_fn = po.GetArg(2);
    bool out_is_wspecifier =
        (ClassifyWspecifier(calibration_fn, NULL, NULL, NULL) != kNoWspecifier);
    ClusterKMeansOptions opts;
    BaseFloat mean = 0.0;
    int32 num_done = 0,
        num_err = 0;
    if (spk2utt_rspecifier.size()) {
      if (!out_is_wspecifier)
        KALDI_ERR << "If spk2utt is supplied, output must be a wspecifier";
      BaseFloatWriter output(calibration_fn);
      if (read_matrices) {
        SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
        for (; !scores_reader.Done(); scores_reader.Next()) {
          std::string spk = scores_reader.Key();
          const Matrix<BaseFloat> scores = scores_reader.Value();
          if (scores.NumRows() <= 2 && scores.NumCols() <= 2) {
            KALDI_WARN << "Too few scores in " << spk << " to cluster";
            num_err++;
            continue;
          }
          std::vector<Clusterable*> clusterables;
          std::vector<Clusterable*> clusters;
          for (int32 i = 0; i < scores.NumRows(); i++) {
            for (int32 j = 0; j < scores.NumCols(); j++) {
              clusterables.push_back(new ScalarClusterable(scores(i,j)));
            }
          }
          ClusterKMeans(clusterables, 2, &clusters, NULL, opts);
          DeletePointers(&clusterables);
          BaseFloat mean1 = dynamic_cast<ScalarClusterable*>(clusters[0])->Mean(),
                    mean2 = dynamic_cast<ScalarClusterable*>(clusters[1])->Mean();
          mean = (mean1 + mean2) / 2;
          output.Write(spk, mean);
          num_done++;
        }
      } else {
        std::vector<std::string> spk_list;
        std::vector<std::vector<Clusterable*>*> clusterables_list;
        std::unordered_map<std::string, std::vector<Clusterable*>*> utt2clust;
        SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
        for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
          spk_list.push_back(spk2utt_reader.Key());
          std::vector<std::string> uttlist = spk2utt_reader.Value();
          std::vector<Clusterable*>* clusterables = new std::vector<Clusterable*>;
          clusterables_list.push_back(clusterables);
          for (std::vector<std::string>::iterator it = uttlist.begin();
               it != uttlist.end(); ++it)
            utt2clust[*it] = clusterables;
        }

        SequentialBaseFloatReader scores_reader(scores_rspecifier);
        for (; !scores_reader.Done(); scores_reader.Next()) {
          std::string utt = scores_reader.Key();
          const BaseFloat score = scores_reader.Value();
          utt2clust[utt]->push_back(new ScalarClusterable(score));
        }

        std::vector<Clusterable*> clusters;
        for (int32 i = 0; i < spk_list.size(); i++) {
          ClusterKMeans(*(clusterables_list[i]), 2, &clusters, NULL, opts);
          DeletePointers(clusterables_list[i]);
          BaseFloat mean1 = dynamic_cast<ScalarClusterable*>(clusters[0])->Mean(),
                    mean2 = dynamic_cast<ScalarClusterable*>(clusters[1])->Mean();
          mean = (mean1 + mean2) / 2;
          output.Write(spk_list[i], mean);
          clusters.clear();
          num_done++;
        }
        DeletePointers(&clusterables_list);
      }
    } else {
      Output output(calibration_fn, false);
      if (read_matrices) {
        std::vector<Clusterable*> clusterables;
        std::vector<Clusterable*> clusters;
        SequentialBaseFloatMatrixReader scores_reader(scores_rspecifier);
        for (; !scores_reader.Done(); scores_reader.Next()) {
          std::string spk = scores_reader.Key();
          const Matrix<BaseFloat> scores = scores_reader.Value();
          for (int32 i = 0; i < scores.NumRows(); i++) {
            for (int32 j = 0; j < scores.NumCols(); j++) {
              clusterables.push_back(new ScalarClusterable(scores(i,j)));
              num_done++;
            }
          }
        }
        ClusterKMeans(clusterables, 2, &clusters, NULL, opts);
        DeletePointers(&clusterables);
        BaseFloat mean1 = dynamic_cast<ScalarClusterable*>(clusters[0])->Mean(),
                  mean2 = dynamic_cast<ScalarClusterable*>(clusters[1])->Mean();
        mean = (mean1 + mean2) / 2;
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
    }
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
