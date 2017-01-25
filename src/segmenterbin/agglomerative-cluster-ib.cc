// segmenterbin/agglomerative-cluster-ib.cc

// Copyright 2017   Vimal Manohar (Johns Hopkins University)

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
#include "segmenter/information-bottleneck-cluster-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage = 
      "Cluster per-utterance probability distributions of "
      "relevance variables using Information Bottleneck principle.\n"
      "Usage: agglomerative-cluster-ib [options] <relevance-prob-rspecifier> "
      "<reco2utt-rspcifier> <labels-wspecifier>\n"
      " e.g.: agglomerative-cluster-ib ark:avg_post.1.ark "
      "ark,t:data/dev/reco2utt ark,t:labels.txt";

    ParseOptions po(usage);

    InformationBottleneckClustererOptions opts;
    
    std::string reco2num_clusters_rspecifier;
    std::string counts_rspecifier;
    int32 junk_label = -2;
    BaseFloat max_merge_thresh = std::numeric_limits<BaseFloat>::max();
    int32 min_clusters = 1;

    po.Register("reco2num-clusters-rspecifier", &reco2num_clusters_rspecifier,
                "If supplied, clustering creates exactly this many clusters "
                "for the corresponding recording.");
    po.Register("counts-rspecifier", &counts_rspecifier, 
                "The counts for each of the initial segments. If not specified "
                "the count is taken to be 1 for each segment.");
    po.Register("junk-label", &junk_label,
                "Assign this label to utterances that could not be clustered");
    po.Register("max-merge-thresh", &max_merge_thresh,
                "Threshold on cost change from merging clusters; clusters "
                "won't be merged if the cost is more than this.");
    po.Register("min-clusters", &min_clusters,
                "Mininum number of clusters desired; we'll stop merging "
                "after reaching this number.");

    opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string relevance_prob_rspecifier = po.GetArg(1),
      reco2utt_rspecifier = po.GetArg(2),
      label_wspecifier = po.GetArg(3);

    RandomAccessBaseFloatVectorReader relevance_prob_reader(
        relevance_prob_rspecifier);
    SequentialTokenVectorReader reco2utt_reader(reco2utt_rspecifier);
    RandomAccessInt32Reader reco2num_clusters_reader(
        reco2num_clusters_rspecifier);
    Int32Writer label_writer(label_wspecifier);
    RandomAccessBaseFloatReader counts_reader(counts_rspecifier);
     
    int32 count = 1, num_utt_err = 0, num_reco_err = 0, num_done = 0,
          num_reco = 0;

    for (; !reco2utt_reader.Done(); reco2utt_reader.Next()) {
      const std::vector<std::string> &uttlist = reco2utt_reader.Value();
      const std::string &reco = reco2utt_reader.Key();

      std::vector<Clusterable*> points;
      points.reserve(uttlist.size());

      int32 id = 0;
      for (std::vector<std::string>::const_iterator it = uttlist.begin();
           it != uttlist.end(); ++it, id++) {
        if (!relevance_prob_reader.HasKey(*it)) {
          KALDI_WARN << "Could not find relevance probability distribution "
                     << "for utterance " << *it << " in archive " 
                     << relevance_prob_rspecifier;
          num_utt_err++;
          continue;
        }
        
        if (!counts_rspecifier.empty()) {
          if (!counts_reader.HasKey(*it)) {
            KALDI_WARN << "Could not find counts for utterance " << *it;
            num_utt_err++;
            continue;
          }
          count = counts_reader.Value(*it);
        }

        const Vector<BaseFloat>& relevance_prob = 
          relevance_prob_reader.Value(*it);

        points.push_back(
            new InformationBottleneckClusterable(id, count, relevance_prob));
        num_done++;
      }
      
      std::vector<Clusterable*> clusters_out;
      std::vector<int32> assignments_out;
      
      int32 this_num_clusters = min_clusters;

      if (!reco2num_clusters_rspecifier.empty()) {
        if (!reco2num_clusters_reader.HasKey(reco)) {
          KALDI_WARN << "Could not find num-clusters for recording "
                     << reco;
          num_reco_err++;
        } else {
          this_num_clusters = reco2num_clusters_reader.Value(reco);
        }
      }
      
      IBClusterBottomUp(points, opts, max_merge_thresh, this_num_clusters,
                        NULL, &assignments_out);
      
      for (int32 i = 0; i < points.size(); i++) {
        InformationBottleneckClusterable* point 
          = static_cast<InformationBottleneckClusterable*> (points[i]);
        int32 id = point->Counts().begin()->first;
        const std::string &utt = uttlist[id];
        label_writer.Write(utt, assignments_out[i] + 1);
      }

      DeletePointers(&points);
      num_reco++;
    }
    
    KALDI_LOG << "Clustered " << num_done << " segments  from "
              << num_reco << " recordings; failed with " 
              << num_utt_err << " segments and " 
              << num_reco_err << " recordings.";

    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
