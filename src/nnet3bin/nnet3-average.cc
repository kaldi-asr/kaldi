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

// This job is run in a spawned thread; it reads a subset of models with
// specified weights.  Sets *success to 1 for success and 0 for failure.  (We
// don't use bool because of the weird implementation of std::vector<bool>).
void ReadModels(std::vector<std::pair<std::string, BaseFloat> > models_and_weights,
                nnet3::Nnet *output_nnet,
                int32 *success) {
  using namespace nnet3;
  try {
    int32 n = models_and_weights.size();
    ReadKaldiObject(models_and_weights[0].first, output_nnet);
    ScaleNnet(models_and_weights[0].second, output_nnet);
    for (int32 i = 1; i < n; i++) {
      Nnet nnet;
      ReadKaldiObject(models_and_weights[i].first, &nnet);
      AddNnet(nnet, models_and_weights[i].second, output_nnet);
    }
    *success = 1;
  } catch (...) {
    *success = 0;
  }
}

}  // namespace kaldi


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
    int32 num_threads = -1;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    std::string weights_str;
    po.Register("weights", &weights_str, "Colon-separated list of weights, one "
                "for each input model.  These will be normalized to sum to one.");
    po.Register("num-threads", &num_threads, "Number of threads to read the "
                "models (will be set automatically if not set.");

    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        first_nnet_rxfilename = po.GetArg(1),
        nnet_wxfilename = po.GetArg(po.NumArgs());

    int32 num_inputs = po.NumArgs() - 1;

    if (num_threads <= 0) {
      // Default logic for selecting the number of threads.
      if (num_inputs > 10) num_threads = 3;
      else if (num_inputs > 5) num_threads = 2;
      else num_threads = 1;
    }

    if (num_threads > 1 && num_threads * 2 > num_inputs) {
      num_threads = num_inputs / 2;
    }

    std::vector<BaseFloat> model_weights;
    GetWeights(weights_str, num_inputs, &model_weights);

    std::vector<Nnet> nnets(num_threads);
    std::vector<int32> return_statuses(num_threads);

    std::vector<std::thread*> threads(num_threads);

    for (int32 thread_id = 0; thread_id < num_threads; thread_id++) {
      std::vector<std::pair<std::string, BaseFloat> > this_models_and_weights;
      for (int32 j = 1 + thread_id; j < po.NumArgs(); j += num_threads) {
        this_models_and_weights.push_back(std::pair<std::string, BaseFloat>(
            po.GetArg(j), model_weights[j - 1]));
      }
      threads[thread_id] = new std::thread(ReadModels, this_models_and_weights,
                                           &(nnets[thread_id]),
                                           &(return_statuses[thread_id]));
    }

    bool success = true;
    for (int32 thread_id = 0; thread_id < num_threads; thread_id++) {
      threads[thread_id]->join();
      delete threads[thread_id];
      if (!return_statuses[thread_id])
        success = false;
      if (success && thread_id > 0)
        AddNnet(nnets[thread_id], 1.0, &(nnets[0]));
    }

    if (!success) {
      KALDI_ERR << "Error detected in a model-reading thread.";
    }

    WriteKaldiObject(nnets[0], nnet_wxfilename, binary_write);

    KALDI_LOG << "Averaged parameters of " << num_inputs
              << " neural nets, and wrote to " << nnet_wxfilename;
    return 0; // it will throw an exception if there are any problems.
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
