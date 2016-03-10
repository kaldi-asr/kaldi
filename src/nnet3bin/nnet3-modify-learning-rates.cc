// nnet3bin/nnet3-modify-learning-rates.cc

// Copyright 2013  Guoguo Chen
//           2015  Vimal Manohar

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program modifies the learning rates so as to equalize the\n"
        "relative changes in parameters for each layer, while keeping their\n"
        "geometric mean the same (or changing it to a value specified using\n"
        "the --average-learning-rate option).\n"
        "\n"
        "Usage: nnet3-modify-learning-rates [options] <prev-model> \\\n"
        "                                  <cur-model> <modified-cur-model>\n"
        "e.g.: nnet-modify-learning-rates --average-learning-rate=0.0002 \\\n"
        "                                 5.mdl 6.mdl 6.mdl\n";

    bool binary_write = true;
    bool retroactive = false;
    BaseFloat average_learning_rate = 0.0;
    BaseFloat first_layer_factor = 1.0;
    BaseFloat last_layer_factor = 1.0;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("average-learning-rate", &average_learning_rate,
                "If supplied, change learning rate geometric mean to the given "
                "value.");
    po.Register("first-layer-factor", &first_layer_factor, "Factor that "
                "reduces the target relative learning rate for first layer.");
    po.Register("last-layer-factor", &last_layer_factor, "Factor that "
                "reduces the target relative learning rate for last layer.");
    po.Register("retroactive", &retroactive, "If true, scale the parameter "
                "differences as well.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    KALDI_ASSERT(average_learning_rate >= 0);

    std::string prev_nnet_rxfilename = po.GetArg(1),
        cur_nnet_rxfilename = po.GetArg(2),
        modified_cur_nnet_rxfilename = po.GetOptArg(3);

    TransitionModel trans_model;
    Nnet prev_nnet, cur_nnet;
    {
      bool binary_read;
      Input ki(prev_nnet_rxfilename, &binary_read);
      prev_nnet.Read(ki.Stream(), binary_read);
    }
    {
      bool binary_read;
      Input ki(cur_nnet_rxfilename, &binary_read);
      cur_nnet.Read(ki.Stream(), binary_read);
    }

    int32 ret = 0;

    // Get info about magnitude of parameter change.
    Nnet diff_nnet(prev_nnet);
    AddNnet(cur_nnet, -1.0, &diff_nnet);
    int32 num_updatable = NumUpdatableComponents(diff_nnet);
    Vector<BaseFloat> dot_prod(num_updatable);
    ComponentDotProducts(diff_nnet, diff_nnet, &dot_prod);
    dot_prod.ApplyPow(0.5); // take sqrt to get l2 norm of diff
    KALDI_LOG << "Parameter differences per layer are "
      << PrintVectorPerUpdatableComponent(prev_nnet, dot_prod);

    Vector<BaseFloat> baseline_prod(num_updatable);
    ComponentDotProducts(prev_nnet, prev_nnet, &baseline_prod);
    baseline_prod.ApplyPow(0.5);
    dot_prod.DivElements(baseline_prod);
    KALDI_LOG << "Relative parameter differences per layer are "
      << PrintVectorPerUpdatableComponent(prev_nnet, dot_prod);

    // If relative parameter difference for a certain is zero, set it to the
    // mean of the rest values.
    int32 num_zero = 0;
    for (int32 i = 0; i < num_updatable; i++) {
      if (dot_prod(i) == 0.0) {
        num_zero++;
      }
    }
    
    if (num_zero > 0) {
      BaseFloat average_diff = dot_prod.Sum()
        / static_cast<BaseFloat>(num_updatable - num_zero);
      for (int32 i = 0; i < num_updatable; i++) {
        if (dot_prod(i) == 0.0) {
          dot_prod(i) = average_diff;
        }
      }
      KALDI_LOG << "Zeros detected in the relative parameter difference "
        << "vector, updating the vector to " << dot_prod ;
    }

    // Gets learning rates for previous neural net.
    Vector<BaseFloat> prev_nnet_learning_rates(num_updatable),
                      cur_nnet_learning_rates(num_updatable);
    GetLearningRates(prev_nnet, &prev_nnet_learning_rates);
    GetLearningRates(cur_nnet, &cur_nnet_learning_rates);
    KALDI_LOG << "Learning rates for previous model per layer are "
              << prev_nnet_learning_rates;
    KALDI_LOG << "Learning rates for current model per layer are "
              << cur_nnet_learning_rates;
    
    // Gets target geometric mean.
    BaseFloat target_geometric_mean = 0.0; 
    if (average_learning_rate == 0.0) {
      target_geometric_mean = Exp(cur_nnet_learning_rates.SumLog()
                                  / static_cast<BaseFloat>(num_updatable));
    } else {
      target_geometric_mean = average_learning_rate;
    }
    KALDI_ASSERT(target_geometric_mean > 0.0);

    // Works out the new learning rates.  We start from the previous model;
    // this ensures that if this program is run twice, we get consistent
    // results even if it's overwritten the current model.
    Vector<BaseFloat> nnet_learning_rates(prev_nnet_learning_rates);
    nnet_learning_rates.DivElements(dot_prod);
    KALDI_ASSERT(last_layer_factor > 0.0);
    nnet_learning_rates(num_updatable - 1) *= last_layer_factor;
    KALDI_ASSERT(first_layer_factor > 0.0);
    nnet_learning_rates(0) *= first_layer_factor;
    BaseFloat cur_geometric_mean = Exp(nnet_learning_rates.SumLog()
                                 / static_cast<BaseFloat>(num_updatable));
    nnet_learning_rates.Scale(target_geometric_mean / cur_geometric_mean);
    KALDI_LOG << "New learning rates for current model per layer are "
              << nnet_learning_rates;

    // Changes the parameter differences if --retroactivate is set to true.
    if (retroactive) {
      Vector<BaseFloat> scale_factors(nnet_learning_rates);
      scale_factors.DivElements(prev_nnet_learning_rates);
      AddNnet(prev_nnet, -1.0, &cur_nnet);
      ScaleNnetComponents(scale_factors, &cur_nnet);
      AddNnet(prev_nnet, 1.0, &cur_nnet);
      KALDI_LOG << "Scale parameter difference retroactively. Scaling factors "
                << "are " << scale_factors;
    }

    // Sets learning rates and writes updated model.
    SetLearningRates(nnet_learning_rates, &cur_nnet);

    Output ko(modified_cur_nnet_rxfilename, binary_write);
    cur_nnet.Write(ko.Stream(), binary_write);

    return ret;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

