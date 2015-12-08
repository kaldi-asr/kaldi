// kwsbin/print-proxy-keywords.cc
//
// Copyright 2014  Johns Hopkins University (Author: Guoguo Chen)
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
        "Reads in the proxy keywords FSTs and print them to a file where each\n"
        "line is \"kwid weight proxies\"\n"
        "\n"
        "Usage: print-proxy-keywords [options] <proxy-rspecifier> \\\n"
        "                                      <kwlist-wspecifier>\n"
        " e.g.: print-proxy-keywords ark:proxy.fsts ark,t:kwlist.txt [ark,t:weights.txt]\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if ((po.NumArgs() < 2) && (po.NumArgs() > 3)) {
      po.PrintUsage();
      exit(1);
    }

    std::string proxy_rspecifier = po.GetArg(1),
                kwlist_wspecifier = po.GetArg(2),
                weight_wspecifier = po.GetOptArg(3);


    SequentialTableReader<VectorFstHolder> proxy_reader(proxy_rspecifier);
    TableWriter<BasicVectorHolder<int32> > kwlist_writer(kwlist_wspecifier);
    TableWriter<BasicVectorHolder<double> > weight_writer(weight_wspecifier);

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
      vector<StdArc::Weight> weights;
      PrintProxyFstPath(proxy, &paths, &weights, proxy.Start(),
                        vector<StdArc::Label>(), StdArc::Weight::One());
      KALDI_ASSERT(paths.size() == weights.size());
      for (int32 i = 0; i < paths.size(); i++) {
        vector<int32> kwlist;
        vector<double> weight;
        weight.push_back(weights[i].Value());
        for (int32 j = 0; j < paths[i].size(); j++) {
          kwlist.push_back(paths[i][j]);
        }
        kwlist_writer.Write(key, kwlist);
        weight_writer.Write(key, weight);
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


