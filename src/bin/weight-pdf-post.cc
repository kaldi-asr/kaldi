// bin/weight-pdf-post.cc

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/posterior.h"

namespace kaldi {

void WeightPdfPost(const ConstIntegerSet<int32> &pdf_set,
                BaseFloat pdf_scale,
                Posterior *post) {
  for (size_t i = 0; i < post->size(); i++) {
    std::vector<std::pair<int32, BaseFloat> > this_post;
    this_post.reserve((*post)[i].size());
    for (size_t j = 0; j < (*post)[i].size(); j++) {
      int32 pdf_id = (*post)[i][j].first;
      BaseFloat weight = (*post)[i][j].second;
      if (pdf_set.count(pdf_id) != 0) {  // is a silence.
        if (pdf_scale != 0.0)
          this_post.push_back(std::make_pair(pdf_id, weight*pdf_scale));
      } else {
        this_post.push_back(std::make_pair(pdf_id, weight));
      }
    }
    (*post)[i].swap(this_post);
  }
}

void WeightPdfPostDistributed(const ConstIntegerSet<int32> &pdf_set,
                                  BaseFloat pdf_scale,
                                  Posterior *post) {
  for (size_t i = 0; i < post->size(); i++) {
    std::vector<std::pair<int32, BaseFloat> > this_post;
    this_post.reserve((*post)[i].size());
    BaseFloat sil_weight = 0.0, nonsil_weight = 0.0;   
    for (size_t j = 0; j < (*post)[i].size(); j++) {
      int32 pdf_id = (*post)[i][j].first;
      BaseFloat weight = (*post)[i][j].second;
      if (pdf_set.count(pdf_id) != 0) sil_weight += weight;
      else nonsil_weight += weight;
    }
    KALDI_ASSERT(sil_weight >= 0.0 && nonsil_weight >= 0.0); // This "distributed"
    // weighting approach doesn't make sense if we have negative weights.
    if (sil_weight + nonsil_weight == 0.0) continue;
    BaseFloat frame_scale = (sil_weight * pdf_scale + nonsil_weight) /
                            (sil_weight + nonsil_weight);
    if (frame_scale != 0.0) {
      for (size_t j = 0; j < (*post)[i].size(); j++) {
        int32 pdf_id = (*post)[i][j].first;
        BaseFloat weight = (*post)[i][j].second;    
        this_post.push_back(std::make_pair(pdf_id, weight * frame_scale));
      }
    }
    (*post)[i].swap(this_post);    
  }
}

}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Apply weight to specific pdfs or tids in posts\n"
        "Usage:  weight-pdf-post [options] <pdf-weight> <pdf-list-phones> "
        "<posteriors-rspecifier> <posteriors-wspecifier>\n"
        "e.g.:\n"
        " weight-pdf-post 0.00001 0:2 ark:1.post ark:nosil.post\n";

    ParseOptions po(usage);

    bool distribute = false;

    po.Register("distribute", &distribute, "If true, rather than weighting the "
                "individual posteriors, apply the weighting to the whole frame: "
                "i.e. on time t, scale all posterior entries by "
                "p(sil)*silence-weight + p(non-sil)*1.0");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string pdf_weight_str = po.GetArg(1),
        pdfs_str = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        posteriors_wspecifier = po.GetArg(4);

    BaseFloat pdf_weight = 0.0;
    if (!ConvertStringToReal(pdf_weight_str, &pdf_weight))
      KALDI_ERR << "Invalid pdf-weight parameter: expected float, got \""
                 << pdf_weight << '"';
    std::vector<int32> pdfs;
    if (!SplitStringToIntegers(pdfs_str, ":", false, &pdfs))
      KALDI_ERR << "Invalid pdf string string " << pdfs_str;
    if (pdfs.empty())
      KALDI_WARN <<"No pdf specified, this will have no effect";
    ConstIntegerSet<int32> pdf_set(pdfs); // faster lookup.

    int32 num_posteriors = 0;
    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);

    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      num_posteriors++;
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      Posterior post = posterior_reader.Value();
      // Posterior is vector<vector<pair<int32, BaseFloat> > >
      if (distribute)
        WeightPdfPostDistributed(pdf_set,
                                     pdf_weight, &post);
      else
        WeightPdfPost(pdf_set,
                          pdf_weight, &post);
      
      posterior_writer.Write(posterior_reader.Key(), post);
    }
    KALDI_LOG << "Done " << num_posteriors << " posteriors.";
    return (num_posteriors != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


