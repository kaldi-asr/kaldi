// bin/post-to-pdf-post.cc

// Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey)

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

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "This program turns per-frame posteriors, which have transition-ids as\n"
        "the integers, into pdf-level posteriors\n"
        "\n"
        "Usage:  post-to-pdf-post [options] <model-file> <posteriors-rspecifier> <posteriors-wspecifier>\n"
        "e.g.: post-to-pdf-post 1.mdl ark:- ark:-\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        posteriors_rspecifier = po.GetArg(2),
        posteriors_wspecifier = po.GetArg(3);

    TransitionModel trans_model;    
    {
      bool binary_in;
      Input ki(model_rxfilename, &binary_in);
      trans_model.Read(ki.Stream(), binary_in);
    }

    int32 num_done = 0;
    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    PosteriorWriter posterior_writer(posteriors_wspecifier);
    
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      const kaldi::Posterior &posterior = posterior_reader.Value();
      kaldi::Posterior pdf_posterior;
      ConvertPosteriorToPdfs(trans_model, posterior, &pdf_posterior);
      posterior_writer.Write(posterior_reader.Key(), pdf_posterior);
      num_done++;
    }
    KALDI_LOG << "Done converting posteriors to pdf-level posteriors for "
              << num_done << " utterances.";

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


