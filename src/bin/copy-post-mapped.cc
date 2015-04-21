// bin/copy-post-mapped.cc

// Copyright 2015   Vimal Manohar (Johns Hopkins University)

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
#include "hmm/posterior.h"
#include "util/kaldi-io.h"

namespace kaldi {
  void MapPosterior(std::vector<int32> id_map, 
                   Posterior *post) {
    for (size_t i = 0; i < post->size(); i++) {
      for (size_t j = 0; j < (*post)[i].size(); j++) {
        (*post)[i][j].first = id_map[(*post)[i][j].first];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;  

    const char *usage =
        "Copy archives of posteriors mapping to new ids\n"
        "(Also see copy-post, rand-prune-post and sum-post)\n"
        "\n"
        "Usage: copy-post-mapped <post-rspecifier> <post-wspecifier>\n";

    BaseFloat scale = 1.0;
    std::string id_map_rxfilename = "";

    ParseOptions po(usage);
    po.Register("scale", &scale, "Scale for posteriors");
    po.Register("id-map", &id_map_rxfilename,
                "File name containing old->new id mapping (each line is: "
                "old-integer-id new-integer-id)");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
      
    std::string post_rspecifier = po.GetArg(1),
        post_wspecifier = po.GetArg(2);

    kaldi::SequentialPosteriorReader posterior_reader(post_rspecifier);
    kaldi::PosteriorWriter posterior_writer(post_wspecifier); 
    
    std::vector<int32> id_map;
    if (id_map_rxfilename != "") {  // read id map.
      std::vector<std::vector<int32> > vec;
      if (!ReadIntegerVectorVectorSimple(id_map_rxfilename, &vec))
        KALDI_ERR << "Could not read map from " << id_map_rxfilename;
      for (size_t i = 0; i < vec.size(); i++) {
        if (vec[i].size() != 2 || vec[i][0]<0 || vec[i][1]<=0 ||
            (vec[i][0]<static_cast<int32>(id_map.size()) &&
             id_map[vec[i][0]] != -1))
          KALDI_ERR << "Error reading id map from " 
                    << id_map_rxfilename 
                    << " (bad line " << i << ")";
        if (vec[i][0] >= static_cast<int32>(id_map.size()))
          id_map.resize(vec[i][0]+1, -1);
        KALDI_ASSERT(id_map[vec[i][0]] == -1);
        id_map[vec[i][0]] = vec[i][1];
      }
      if (id_map.empty()) {
        KALDI_ERR << "Read empty id map from "  
                  << id_map_rxfilename;
      }
    }

    int32 num_done = 0;
   
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      std::string key = posterior_reader.Key();

      kaldi::Posterior posterior = posterior_reader.Value();
      if (scale != 1.0)
        ScalePosterior(scale, &posterior);
      if (id_map_rxfilename != "")
        MapPosterior(id_map, &posterior);
      posterior_writer.Write(key, posterior);

      num_done++;
    }
    KALDI_LOG << "Done copying " << num_done << " posteriors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


