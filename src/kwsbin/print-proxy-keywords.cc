// kwsbin/print-proxy-keywords.cc
//
// Copyright 2014-2016  Johns Hopkins University (Author: Guoguo Chen,
//                                                        Yenda Trmal)
//
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
#include "fstext/kaldi-fst-io.h"

namespace fst {

bool PrintProxyFstPath(const VectorFst<StdArc> &proxy,
                       vector<vector<StdArc::Label> > *path,
                       vector<StdArc::Weight> *cost,
                       StdArc::StateId cur_state,
                       vector<StdArc::Label> cur_path,
                       StdArc::Weight cur_cost) {

  if (proxy.Final(cur_state) != StdArc::Weight::Zero()) {
    cur_cost = Times(proxy.Final(cur_state), cur_cost);
    path->push_back(cur_path);
    cost->push_back(cur_cost);
    // even final state can have outgoing args, so no return here
  }

  for (ArcIterator<StdFst> aiter(proxy, cur_state);
       !aiter.Done(); aiter.Next()) {
    const StdArc &arc = aiter.Value();
    StdArc::Weight temp_cost = Times(arc.weight, cur_cost);
    cur_path.push_back(arc.ilabel);
    PrintProxyFstPath(proxy, path, cost,
                      arc.nextstate, cur_path, temp_cost);
    cur_path.pop_back();
  }

  return true;
}
}   // namespace fst

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    typedef kaldi::int32 int32;
    typedef kaldi::uint64 uint64;
    typedef StdArc::StateId StateId;
    typedef StdArc::Weight Weight;

    const char *usage =
        "Reads in the proxy keywords FSTs and print them to a file where each\n"
        "line is \"kwid w1 w2 .. 2n\"\n"
        "\n"
        "Usage: \n"
        " print-proxy-keywords [options] <proxy-rspecifier> "
        " <kwlist-wspecifier> [<cost-wspecifier>]]\n"
        "e.g.:\n"
        " print-proxy-keywords ark:proxy.fsts ark,t:kwlist.txt"
        " ark,t:costs.txt\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string proxy_rspecifier = po.GetArg(1),
                kwlist_wspecifier = po.GetArg(2),
                cost_wspecifier = po.GetOptArg(3);


    SequentialTableReader<VectorFstHolder> proxy_reader(proxy_rspecifier);
    TableWriter<BasicVectorHolder<int32> > kwlist_writer(kwlist_wspecifier);
    TableWriter<BasicVectorHolder<double> > cost_writer(cost_wspecifier);

    // Start processing the keywords
    int32 n_done = 0;
    for (; !proxy_reader.Done(); proxy_reader.Next()) {
      std::string key = proxy_reader.Key();
      VectorFst<StdArc> proxy = proxy_reader.Value();
      proxy_reader.FreeCurrent();

      if (proxy.Properties(kAcyclic, true) == 0) {
        KALDI_WARN << "Proxy FST has cycles, skip printing paths for " << key;
        continue;
      }

      vector<vector<StdArc::Label> > paths;
      vector<StdArc::Weight> costs;
      PrintProxyFstPath(proxy, &paths, &costs, proxy.Start(),
                        vector<StdArc::Label>(), StdArc::Weight::One());
      KALDI_ASSERT(paths.size() == costs.size());
      for (int32 i = 0; i < paths.size(); i++) {
        vector<int32> kwlist;
        vector<double> cost;
        cost.push_back(costs[i].Value());
        for (int32 j = 0; j < paths[i].size(); j++) {
          kwlist.push_back(paths[i][j]);
        }
        kwlist_writer.Write(key, kwlist);
        if (cost_wspecifier != "")
          cost_writer.Write(key, cost);
      }
      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " keywords";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


