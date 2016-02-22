// Copyright 2014  David Snyder

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
#include "tree/cluster-utils.cc"
#include "tree/clusterable-classes.h"
#include "ivector/logistic-regression.h"
#include "ivector/plda.h"

using namespace kaldi;

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage = "Does speaker diarzation using k-means clustering of PLDA transformed i-vectors.\n"
                        "Usage: speaker-diarization <plda> <spk2utt-rspecifier> <per-utt-ivector-rspecifier> <diar-wspecifier>\n"
                        " e.g.: speaker-diarization plda ark,t:data/dev/spk2utt scp:exp/ivectors_dev/ivector.scp ark,t:exp/diarization_dev/diarization.txt\n"
                        "\n";

 
    int32 num_speakers = 2;
    ParseOptions po(usage);

    po.Register("num-speakers", &num_speakers, "Number of speakers to use in the k-means clustering algorithm");

    ClusterKMeansOptions cfg;

    po.Read(argc, argv);

    if (po.NumArgs() != 4 && po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    int32 num_utt_err = 0;
    int32 num_utt_done = 0;

    std::string plda_rxfilename, spk2utt_rspecifier, ivector_rspecifier,
                diar_wspecifier;

    if (po.NumArgs() == 4) {
      plda_rxfilename = po.GetArg(1);
      spk2utt_rspecifier = po.GetArg(2);
      ivector_rspecifier = po.GetArg(3);
      diar_wspecifier = po.GetArg(4);
    } else {
      spk2utt_rspecifier = po.GetArg(1);
      ivector_rspecifier = po.GetArg(2);
      diar_wspecifier = po.GetArg(3);
    }

    Plda plda;
    PldaConfig plda_config;
    int32 dim = 0;

    if (plda_rxfilename != "") {
      ReadKaldiObject(plda_rxfilename, &plda);
      dim = plda.Dim();
    } 

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessBaseFloatVectorReader ivector_reader(ivector_rspecifier);
    Int32Writer diar_writer(diar_wspecifier);
    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();
      std::vector<Clusterable *> ivector_clusters;
      ivector_clusters.reserve(uttlist.size());

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!ivector_reader.HasKey(utt)) {
          KALDI_WARN << "No iVector present in input for utterance " << utt;
          num_utt_err++;
        } else {
          ivector_clusters.resize(ivector_clusters.size() + 1);
          Vector<BaseFloat> ivector = ivector_reader.Value(utt);
          Vector<BaseFloat> *transformed_ivector;

          if (plda_rxfilename != "") {
            transformed_ivector = new Vector<BaseFloat>(dim);
            plda.TransformIvector(plda_config, ivector,
                                  transformed_ivector);
          } else {
            transformed_ivector = &ivector;
          }

          Clusterable *cluster = new VectorClusterable(*transformed_ivector, 1.0);
          ivector_clusters.back() = cluster;
          num_utt_done++;
          
          if (plda_rxfilename != "") {
            delete transformed_ivector;
          }
        }
      }
      std::vector<Clusterable *> ivector_clusters_out;
      std::vector<int32> assignments_out;
      //BaseFloat imprv = ClusterKMeansOnce(ivector_clusters, 2, &ivector_clusters_out, &assignments_out, cfg);
      BaseFloat imprv = ClusterKMeans(ivector_clusters, num_speakers, &ivector_clusters_out, &assignments_out, cfg);
      for (int32 i = 0; i < ivector_clusters.size(); i++) {
        delete ivector_clusters[i];
      }
      for (int32 i = 0; i < ivector_clusters_out.size(); i++) {
        delete ivector_clusters_out[i];
      }

      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        diar_writer.Write(utt, assignments_out[i]);
      }
      KALDI_LOG << "Objf improvement is " << imprv << " for utt " << spk;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
