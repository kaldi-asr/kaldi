// kwsbin/generate-proxy-keywords.cc

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)

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
#include "fstext/fstext-utils.h"

namespace fst {

bool PrintProxyFstPath(const VectorFst<StdArc> &proxy,
                       vector<vector<StdArc::Label> > *path,
                       vector<StdArc::Weight> *weight,
                       StdArc::StateId cur_state,
                       vector<StdArc::Label> cur_path,
                       StdArc::Weight cur_weight) {
  if (proxy.Final(cur_state) != StdArc::Weight::Zero()) {
    // Assume only final state has non-zero weight.
    cur_weight = Times(proxy.Final(cur_state), cur_weight);
    path->push_back(cur_path);
    weight->push_back(cur_weight);
    return true;
  }

  for (ArcIterator<StdFst> aiter(proxy, cur_state);
       !aiter.Done(); aiter.Next()) {
    const StdArc &arc = aiter.Value();
    StdArc::Weight temp_weight = Times(arc.weight, cur_weight);
    cur_path.push_back(arc.ilabel);
    PrintProxyFstPath(proxy, path, weight,
                      arc.nextstate, cur_path, temp_weight);
    cur_path.pop_back();
  }

  return true;
}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    typedef kaldi::int32 int32;
    typedef kaldi::uint64 uint64;
    typedef StdArc::StateId StateId;
    typedef StdArc::Weight Weight;

    const char *usage =
        "Convert the keywords into in-vocabulary words using the given phone\n"
        "level edit distance fst (E.fst). The large lexicon (L2.fst) and\n"
        "inverted small lexicon (L1'.fst) are also expected to be present. We\n"
        "actually use the composed FST L2xE.fst to be more efficient. Ideally\n"
        "we should have used L2xExL1'.fst but this is quite computationally\n"
        "expensive at command level. Keywords.int is in the transcription\n"
        "format. If kwlist-wspecifier is given, the program also prints out\n"
        "the proxy fst in a format where each line is \"kwid weight proxy\".\n"
        "\n"
        "Usage: generate-proxy-keywords [options] <L2xE.fst> <L1'.fst> \\\n"
        "    <keyword-rspecifier> <proxy-wspecifier> [kwlist-wspecifier] \n"
        " e.g.: generate-proxy-keywords L2xE.fst L1'.fst ark:keywords.int \\\n"
        "                           ark:proxy.fsts [ark,t:proxy.kwlist.txt]\n";

    ParseOptions po(usage);

    int32 nBest = 100;
    double cost_threshold = 1;
    po.Register("nBest", &nBest, "n best possible in-vocabulary proxy keywords.");
    po.Register("cost-threshold", &cost_threshold, "Cost threshold.");

    po.Read(argc, argv);

    if (po.NumArgs() < 4 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string L2xE_filename = po.GetArg(1),
        L1_filename = po.GetArg(2),
        keyword_rspecifier = po.GetArg(3),
        proxy_wspecifier = po.GetArg(4),
        kwlist_wspecifier = (po.NumArgs() == 5) ? po.GetArg(5) : "";

    VectorFst<StdArc> *L2xE = ReadFstKaldi(L2xE_filename);
    VectorFst<StdArc> *L1 = ReadFstKaldi(L1_filename);
    SequentialInt32VectorReader keyword_reader(keyword_rspecifier);
    TableWriter<VectorFstHolder> proxy_writer(proxy_wspecifier);
    TableWriter<BasicVectorHolder<double> > kwlist_writer(kwlist_wspecifier);

    // Start processing the keywords
    int32 n_done = 0;
    for (; !keyword_reader.Done(); keyword_reader.Next()) {
      std::string key = keyword_reader.Key();
      std::vector<int32> keyword = keyword_reader.Value();
      keyword_reader.FreeCurrent();

      KALDI_LOG << "Processing " << key;

      VectorFst<StdArc> proxy;
      VectorFst<StdArc> tmp;
      MakeLinearAcceptor(keyword, &proxy);

      KALDI_VLOG(1) << "Compose(KW, L2xE)";
      ArcSort(&proxy, OLabelCompare<StdArc>());
      Compose(proxy, *L2xE, &tmp);
      KALDI_VLOG(1) << "Compose(KWxL2xE, L1')";
      ArcSort(&tmp, OLabelCompare<StdArc>());
      Compose(tmp, *L1, &proxy);
      KALDI_VLOG(1) << "Project";
      Project(&proxy, PROJECT_OUTPUT);
      KALDI_VLOG(1) << "Prune";
      Prune(&proxy, cost_threshold);
      if (nBest > 0) {
        KALDI_VLOG(1) << "Shortest Path";
        ShortestPath(proxy, &tmp, nBest, true, true);
      } else {
        tmp = proxy;
      }
      KALDI_VLOG(1) << "Remove epsilon";
      RmEpsilon(&tmp);
      KALDI_VLOG(1) << "Determinize";
      Determinize(tmp, &proxy);
      ArcSort(&proxy, fst::OLabelCompare<StdArc>());

      // Write the proxy FST.
      proxy_writer.Write(key, proxy);

      // Print the proxy FST with each line looks like "kwid weight proxy"
      if (po.NumArgs() == 5) {
        if (proxy.Properties(kAcyclic, true) == 0) {
          KALDI_WARN << "Proxy FST has cycles, skip printing paths for " << key;
        } else {
          vector<vector<StdArc::Label> > path;
          vector<StdArc::Weight> weight;
          PrintProxyFstPath(proxy, &path, &weight, proxy.Start(),
                            vector<StdArc::Label>(), StdArc::Weight::One());
          KALDI_ASSERT(path.size() == weight.size());
          for (int32 i = 0; i < path.size(); i++) {
            vector<double> kwlist;
            kwlist.push_back(static_cast<double>(weight[i].Value()));
            for (int32 j = 0; j < path[i].size(); j++) {
              kwlist.push_back(static_cast<double>(path[i][j]));
            }
            kwlist_writer.Write(key, kwlist);
          }
        }
      }

      n_done++;
    }

    delete L1;
    delete L2xE;
    KALDI_LOG << "Done " << n_done << " keywords";
    return (n_done != 0 ? 0 : 1);    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
