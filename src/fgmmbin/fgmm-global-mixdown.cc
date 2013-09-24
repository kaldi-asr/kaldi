// fgmmbin/fgmm-global-mixdown.cc

// Copyright 2012  Johns Hopkins Universithy (author: Daniel Povey)

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

#include <queue>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/full-gmm.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Merge Gaussians in a full-covariance GMM to get a smaller number;\n"
        "this program supports a --gselect option which is used to select\n"
        "\"good\" pairs of Gaussians to consider merging (pairs that most often\n"
        "co-occur in the gselect information are considered).  If no gselect\n"
        "info supplied, we consider all pairs (very slow for big models).\n"
        "Usage:  fgmm-global-mixdown [options] <model-in> <model-out>\n"
        "e.g.: fgmm-global-mixdown --gselect=gselect.1 --mixdown-target=120 1.ubm 2.ubm\n"
        "Note: --mixdown-target option is required.\n";

    bool binary_write = true;
    std::string gselect_rspecifier;
    int32 mixdown_target = -1, num_pairs = 20000;
    BaseFloat power = 0.75; // Power used in choosing pairs; between 0.5 and 1 make sense.
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("gselect", &gselect_rspecifier, "Gaussian-selection info, used "
                "to select most promising pairs");
    po.Register("num-pairs", &num_pairs, "Number of pairs of Gaussians to try merging "
                "(only relevant if you use --gselect option");
    po.Register("mixdown-target", &mixdown_target,
                "Number of Gaussians we want in mixed-down GMM.");
    po.Register("power", &power,
                "Power used in choosing pairs from gselect (should be between 0.5 and 1)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    KALDI_ASSERT(mixdown_target >= 0 && "--mixdown-target option is required and must be >0.");
    
    FullGmm fgmm;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      fgmm.Read(ki.Stream(), binary_read);
    }
    std::vector<std::pair<int32, int32> > pairs;
    if (gselect_rspecifier == "") { // use all pairs.
      for (int32 i = 0; i < fgmm.NumGauss(); i++)
        for (int32 j = 0; j < i; j++) pairs.push_back(std::make_pair(i, j));
    } else {
      unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > counts; // co-occurrence map:
      // if i <= j, then maps from (i,j) -> #co-occurrences in gselect info.
      SequentialInt32VectorVectorReader gselect_reader(gselect_rspecifier);
      for (; !gselect_reader.Done(); gselect_reader.Next()) {
        const std::vector<std::vector<int32> > &gselect = gselect_reader.Value();
        for (int32 i = 0; i < gselect.size(); i++) {
          for (int32 j = 0; j < gselect[i].size(); j++) {
            for (int32 k = 0; k < gselect[i].size(); k++) {
              int32 idx1 = gselect[i][j], idx2 = gselect[i][k];
              if (idx1 <= idx2) {
                std::pair<int32, int32> pr(idx1, idx2);
                if (counts.count(pr) == 0) counts[pr] = 1;
                else counts[pr]++;
              }
            }
          }
        }
      }
      // take greatest according to count(i,j) / pow(count(i,i)*count(j,j), pow)
      typedef std::pair<BaseFloat, std::pair<int32,int32> > QueueElem;
      std::priority_queue<QueueElem> queue;
      for (unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> >::iterator iter = counts.begin();
             iter != counts.end(); ++iter) {
        int32 idx1 = iter->first.first, idx2 = iter->first.second,
            count = iter->second;
        if (idx1 != idx2) {
          BaseFloat x = counts[std::make_pair(idx1,idx1)] * counts[std::make_pair(idx2, idx2)];
          BaseFloat f = count / std::pow(x, power);
          queue.push(std::make_pair(f, iter->first));
        }
      }
      while (!queue.empty() && static_cast<int32>(pairs.size()) < num_pairs) {
        KALDI_VLOG(2) << "Pair is " << queue.top().second.first << ", "
                      << queue.top().second.second;
        pairs.push_back(queue.top().second); // the "num_pairs" "best" pairs of
        queue.pop();
        // indices, based on this co-occurrence statistic.
      }
    }
    KALDI_LOG << "Selected " << pairs.size() << " pairs of Gaussians to merge, "
              << "now doing merging.";
    int32 orig_ngauss = fgmm.NumGauss();
    BaseFloat like_change = fgmm.MergePreselect(mixdown_target, pairs);
    int32 cur_ngauss = fgmm.NumGauss();
    KALDI_LOG << "Mixed down GMM from " << orig_ngauss << " to "
              << cur_ngauss << ", likelihood change was " << like_change;
    
    WriteKaldiObject(fgmm, model_out_filename, binary_write);

    KALDI_LOG << "Wrote model to " << model_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


