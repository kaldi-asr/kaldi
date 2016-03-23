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
#include "fstext/kaldi-fst-io.h"
#include "fstext/fstext-utils.h"
#include "fstext/prune-special.h"

namespace fst {

bool PrintProxyFstPath(const VectorFst<StdArc> &proxy,
                       vector<vector<StdArc::Label> > *path,
                       vector<StdArc::Weight> *weight,
                       StdArc::StateId cur_state,
                       vector<StdArc::Label> cur_path,
                       StdArc::Weight cur_weight) {
  if (proxy.Final(cur_state) != StdArc::Weight::Zero()) {
    // Assumes only final state has non-zero weight.
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

    int32 max_states = 100000;
    int32 phone_nbest = 50;
    int32 proxy_nbest = 100;
    double phone_beam = 5;
    double proxy_beam = 5;
    po.Register("phone-nbest", &phone_nbest, "Prune KxL2xE transducer to only "
                "contain top n phone sequences, -1 means all sequences.");
    po.Register("proxy-nbest", &proxy_nbest, "Prune KxL2xExL1' transducer to "
                "only contain top n proxy keywords, -1 means all proxies.");
    po.Register("phone-beam", &phone_beam, "Prune KxL2xE transducer to the "
                "given beam, -1 means no prune.");
    po.Register("proxy-beam", &proxy_beam, "Prune KxL2xExL1' transducer to the "
                "given beam, -1 means no prune.");
    po.Register("max-states", &max_states, "Prune kxL2xExL1' transducer to the "
                "given number of states, 0 means no prune.");

    po.Read(argc, argv);

    // Checks input options.
    if (phone_nbest != -1 && phone_nbest <= 0) {
      KALDI_ERR << "--phone-nbest must either be -1 or positive.";
      exit(1);
    }
    if (proxy_nbest != -1 && proxy_nbest <= 0) {
      KALDI_ERR << "--proxy-nbest must either be -1 or positive.";
      exit(1);
    }
    if (phone_beam != -1 && phone_beam < 0) {
      KALDI_ERR << "--phone-beam must either be -1 or non-negative.";
      exit(1);
    }
    if (proxy_beam != -1 && proxy_beam <=0) {
      KALDI_ERR << "--proxy-beam must either be -1 or non-negative.";
      exit(1);
    }

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

    // Processing the keywords.
    int32 n_done = 0;
    for (; !keyword_reader.Done(); keyword_reader.Next()) {
      std::string key = keyword_reader.Key();
      std::vector<int32> keyword = keyword_reader.Value();
      keyword_reader.FreeCurrent();

      KALDI_LOG << "Processing " << key;

      VectorFst<StdArc> proxy;
      VectorFst<StdArc> tmp_proxy;
      MakeLinearAcceptor(keyword, &proxy);

      // Composing K and L2xE. We assume L2xE is ilabel sorted.
      KALDI_VLOG(1) << "Compose(K, L2xE)";
      ArcSort(&proxy, OLabelCompare<StdArc>());
      Compose(proxy, *L2xE, &tmp_proxy);

      // Processing KxL2xE.
      KALDI_VLOG(1) << "Project(KxL2xE, PROJECT_OUTPUT)";
      Project(&tmp_proxy, PROJECT_OUTPUT);
      if (phone_beam >= 0) {
        KALDI_VLOG(1) << "Prune(KxL2xE, " << phone_beam << ")";
        Prune(&tmp_proxy, phone_beam);
      }
      if (phone_nbest > 0) {
        KALDI_VLOG(1) << "ShortestPath(KxL2xE, " << phone_nbest << ")";
        RmEpsilon(&tmp_proxy);
        ShortestPath(tmp_proxy, &proxy, phone_nbest, true, true);
        tmp_proxy.DeleteStates();   // Not needed for now.
        KALDI_VLOG(1) << "Determinize(KxL2xE)";
        Determinize(proxy, &tmp_proxy);
        proxy.DeleteStates();       // Not needed for now.
      }
      KALDI_VLOG(1) << "ArcSort(KxL2xE, OLabel)";
      proxy = tmp_proxy;
      tmp_proxy.DeleteStates();     // Not needed for now.
      ArcSort(&proxy, OLabelCompare<StdArc>());


      // Processing KxL2xExL1'.
      RmEpsilon(&proxy);
      ArcSort(&proxy, OLabelCompare<StdArc>());
      if (proxy_beam >= 0) {
        // We only use the delayed FST when pruning is requested, because we do
        // the optimization in pruning.
        // Composing KxL2xE and L1'. We assume L1' is ilabel sorted.
        KALDI_VLOG(1) << "Compose(KxL2xE, L1')";
        ComposeFst<StdArc> lazy_compose(proxy, *L1);
        proxy.DeleteStates();

        KALDI_VLOG(1) << "Project(KxL2xExL1', PROJECT_OUTPUT)";
        ProjectFst<StdArc> lazy_project(lazy_compose, PROJECT_OUTPUT);

        // This will likely be the most time consuming part, we use a special
        // pruning algorithm where we don't expand the full FST.
        KALDI_VLOG(1) << "Prune(KxL2xExL1', " << proxy_beam << ")";
        PruneSpecial(lazy_project, &tmp_proxy, proxy_beam, max_states);
      } else {
        // If no pruning is requested, we do the normal composition.
        KALDI_VLOG(1) << "Compose(KxL2xE, L1')";
        Compose(proxy, *L1, &tmp_proxy);
        proxy.DeleteStates();

        KALDI_VLOG(1) << "Project(KxL2xExL1', PROJECT_OUTPUT)";
        Project(&tmp_proxy, PROJECT_OUTPUT);
      }
      if (proxy_nbest > 0) {
        KALDI_VLOG(1) << "ShortestPath(KxL2xExL1', " << proxy_nbest << ")";
        proxy = tmp_proxy;
        tmp_proxy.DeleteStates(); // Not needed for now.
        RmEpsilon(&proxy);
        ShortestPath(proxy, &tmp_proxy, proxy_nbest, true, true);
        proxy.DeleteStates();     // Not needed for now.
      }
      KALDI_VLOG(1) << "RmEpsilon(KxL2xExL1')";
      RmEpsilon(&tmp_proxy);
      KALDI_VLOG(1) << "Determinize(KxL2xExL1')";
      Determinize(tmp_proxy, &proxy);
      tmp_proxy.DeleteStates();
      KALDI_VLOG(1) << "ArcSort(KxL2xExL1', OLabel)";
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
