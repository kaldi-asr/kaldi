// nnet3bin/nnet3-average.cc

// Copyright 2015  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet3/nnet-utils.h"


namespace kaldi {

void GetWeights(const std::string &weights_str,
                int32 num_inputs,
                std::vector<BaseFloat> *weights) {
  KALDI_ASSERT(num_inputs >= 1);
  if (!weights_str.empty()) {
    SplitStringToFloats(weights_str, ":", true, weights);
    if (weights->size() != num_inputs) {
      KALDI_ERR << "--weights option must be a colon-separated list "
                << "with " << num_inputs << " elements, got: "
                << weights_str;
    }
  } else {
    for (int32 i = 0; i < num_inputs; i++)
      weights->push_back(1.0 / num_inputs);
  }
  // normalize the weights to sum to one.
  float weight_sum = 0.0;
  for (int32 i = 0; i < num_inputs; i++)
    weight_sum += (*weights)[i];
  for (int32 i = 0; i < num_inputs; i++)
    (*weights)[i] = (*weights)[i] / weight_sum;
  if (fabs(weight_sum - 1.0) > 0.01) {
    KALDI_WARN << "Normalizing weights to sum to one, sum was " << weight_sum;
  }
}

}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program averages the parameters over a number of 'raw' nnet3 neural nets.\n"
        "\n"
        "Usage:  nnet3-average [options] <model1> <model2> ... <modelN> <model-out>\n"
        "\n"
        "e.g.:\n"
        " nnet3-average 1.1.nnet 1.2.nnet 1.3.nnet 2.nnet\n";

    bool binary_write = true;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    string weights_str;
    po.Register("weights", &weights_str, "Colon-separated list of weights, one "
                "for each input model.  These will be normalized to sum to one.");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        first_nnet_rxfilename = po.GetArg(1),
        nnet_wxfilename = po.GetArg(po.NumArgs());

    Nnet nnet;
    ReadKaldiObject(first_nnet_rxfilename, &nnet);
    
    int32 num_inputs = po.NumArgs() - 1;

    std::vector<BaseFloat> model_weights;
    GetWeights(weights_str, num_inputs, &model_weights);
    
    ScaleNnet(model_weights[0], &nnet);
              
    for (int32 i = 2; i <= num_inputs; i++) {
      Nnet src_nnet;
      ReadKaldiObject(po.GetArg(i), &src_nnet);
      AddNnet(src_nnet, model_weights[i - 1], &nnet);
    }
    

    WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
    
    KALDI_LOG << "Averaged parameters of " << num_inputs
              << " neural nets, and wrote to " << nnet_wxfilename;
    return 0; // it will throw an exception if there are any problems.
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

