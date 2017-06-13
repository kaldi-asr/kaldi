// nnet2bin/nnet2-boost-silence.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
       "This program can be used to change the acoustic probabilities associated \n"
       "with a certain set of phones by a given factor, for nnet2 models. \n" 
       "Can be useful to control the amount of silence, noise, and so on. \n"
       "It is implemented by dividing the corresponding priors by that\n"
       "factor (since we divide by the prior when we evaluate likelihoods).\n"
       "\n"
       "Usage: nnet2-boost-silence [options] <silence-phones-list> <model-in> <model-out>\n"
       "e.g.:   nnet2-boost-silence --boost=0.2 1:2:3 final.mdl final_boost.mdl\n"
       "See also: gmm-boost-silence\n";
   
    bool binary = true; 
    BaseFloat boost = 0.1;
 
    ParseOptions po(usage);
    po.Register("binary", &binary, "Read/Write in binary mode");
    po.Register("boost", &boost, "Factor by which to boost silence priors");   
 
    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string silence_phones_string = po.GetArg(1);
    std::string nnet_rxfilename = po.GetArg(2);
    std::string nnet_wxfilename = po.GetArg(3);
    
    std::vector<int32> silence_phones;
    if (silence_phones_string != "") {
      SplitStringToIntegers(silence_phones_string, ":", false, &silence_phones);
      std::sort(silence_phones.begin(), silence_phones.end());
      KALDI_ASSERT(IsSortedAndUniq(silence_phones) && "Silence phones non-unique.");
    } else {
      KALDI_WARN << "nnet-boost-silence: no silence phones specified, doing nothing.";
    }

    TransitionModel trans_model;
    AmNnet am_nnet;
    { 
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }    
    
    std::vector<int32> pdfs;
    bool ans = GetPdfsForPhones(trans_model, silence_phones, &pdfs);
    if (!ans) {
      KALDI_WARN << "The pdfs for the silence phones may be shared by other phones "
                 << "(note: this probably does not matter.)";
    }

    int32 num_pdfs = trans_model.NumPdfs();
    Vector<BaseFloat> priors(num_pdfs);
    priors.CopyFromVec(am_nnet.Priors());
    
    for(int32 i=0; i<pdfs.size(); i++){
        priors(pdfs[i]) /= boost;
    }
    am_nnet.SetPriors(priors); 

    {
      Output ko(nnet_wxfilename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_nnet.Write(ko.Stream(), binary);
    }

  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


