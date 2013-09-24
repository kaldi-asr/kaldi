// gmmbin/gmm-boost-silence.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
#include "hmm/transition-model.h"
#include "gmm/am-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Modify GMM-based model to boost (by a certain factor) all\n"
        "probabilities associated with the specified phones (could be\n"
        "all silence phones, or just the ones used for optional silence).\n"
        "Note: this is done by modifying the GMM weights.  If the silence\n"
        "model shares a GMM with other models, then it will modify the GMM\n"
        "weights for all models that may correspond to silence.\n"
        "\n"
        "Usage:  gmm-boost-silence [options] <silence-phones-list> <model-in> <model-out>\n"
        "e.g.: gmm-boost-silence --boost=1.5 1:2:3 1.mdl 1_boostsil.mdl\n";
    
    bool binary_write = true;
    BaseFloat boost = 1.5;
        
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("boost", &boost, "Factor by which to boost silence probs");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string
        silence_phones_string = po.GetArg(1),
        model_rxfilename = po.GetArg(2),
        model_wxfilename = po.GetArg(3);
    
    std::vector<int32> silence_phones;
    if (silence_phones_string != "") {
      SplitStringToIntegers(silence_phones_string, ":", false, &silence_phones);
      std::sort(silence_phones.begin(), silence_phones.end());
      KALDI_ASSERT(IsSortedAndUniq(silence_phones) && "Silence phones non-unique.");
    } else {
      KALDI_WARN << "gmm-boost-silence: no silence phones specified, doing nothing.";
    }
    
    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    { // Do the modification to the am_gmm object.
      std::vector<int32> pdfs;
      bool ans = GetPdfsForPhones(trans_model, silence_phones, &pdfs);
      if (!ans) {
        KALDI_WARN << "The pdfs for the silence phones may be shared by other phones "
                   << "(note: this probably does not matter.)";
      }
      for (size_t i = 0; i < pdfs.size(); i++) {
        int32 pdf = pdfs[i];
        DiagGmm &gmm = am_gmm.GetPdf(pdf);
        Vector<BaseFloat> weights(gmm.weights());
        weights.Scale(boost);
        gmm.SetWeights(weights);
        gmm.ComputeGconsts();
      }
      KALDI_LOG << "Boosted weights for " << pdfs.size()
                << " pdfs, by factor of " << boost;
    }
    
    {
      Output ko(model_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_gmm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Wrote model to " << model_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


