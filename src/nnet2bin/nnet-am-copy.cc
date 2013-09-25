// nnet2bin/nnet-am-copy.cc

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
#include "hmm/transition-model.h"
#include "nnet2/am-nnet.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy a (cpu-based) neural net and its associated transition model,\n"
        "possibly changing the binary mode\n"
        "Also supports multiplying all the learning rates by a factor\n"
        "(the --learning-rate-factor option) and setting them all to a given\n"
        "value (the --learning-rate options)\n"
        "\n"
        "Usage:  nnet-am-copy [options] <nnet-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet-am-copy --binary=false 1.mdl text.mdl\n";

    int32 truncate = -1;
    bool binary_write = true;
    bool remove_dropout = false;
    BaseFloat dropout_scale = -1.0;
    bool remove_preconditioning = false;
    bool collapse = false;
    bool match_updatableness = true;
    BaseFloat learning_rate_factor = 1.0, learning_rate = -1;
    std::string learning_rates = "";
    std::string scales = "";
    std::string stats_from;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("learning-rate-factor", &learning_rate_factor,
                "Before copying, multiply all the learning rates in the "
                "model by this factor.");
    po.Register("learning-rate", &learning_rate,
                "If supplied, all the learning rates of \"updatable\" layers"
                "are set to this value.");
    po.Register("learning-rates", &learning_rates,
                "If supplied (a colon-separated list of learning rates), sets "
                "the learning rates of \"updatable\" layers to these values.");
    po.Register("scales", &scales,
                "A colon-separated list of scaling factors, one for each updatable "
                "layer: a mechanism to scale the parameters.");
    po.Register("truncate", &truncate, "If set, will truncate the neural net "
                "to this many components by removing the last components.");
    po.Register("remove-dropout", &remove_dropout, "Set this to true to remove "
                "any dropout components.");
    po.Register("dropout-scale", &dropout_scale, "If set, set the dropout scale in any "
                "dropout components to this value.  Note: in traditional dropout, this "
                "is always zero; you can set it to any value between zero and one.");
    po.Register("remove-preconditioning", &remove_preconditioning, "Set this to true to replace "
                "components of type AffineComponentPreconditioned with AffineComponent.");
    po.Register("stats-from", &stats_from, "Before copying neural net, copy the "
                "statistics in any layer of type NonlinearComponent, from this "
                "neural network: provide the extended filename.");
    po.Register("collapse", &collapse, "If true, collapse sequences of AffineComponents "
                "and FixedAffineComponents to compactify model");
    po.Register("match-updatableness", &match_updatableness, "Only relevant if "
                "collapse=true; set this to false to collapse mixed types.");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        nnet_wxfilename = po.GetArg(2);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    if (learning_rate_factor != 1.0)
      am_nnet.GetNnet().ScaleLearningRates(learning_rate_factor);

    if (learning_rate >= 0)
      am_nnet.GetNnet().SetLearningRates(learning_rate);

    if (learning_rates != "") {
      std::vector<BaseFloat> learning_rates_vec;
      if (!SplitStringToFloats(learning_rates, ":", false, &learning_rates_vec)
          || static_cast<int32>(learning_rates_vec.size()) !=
             am_nnet.GetNnet().NumUpdatableComponents()) {
        KALDI_ERR << "Expected --learning-rates option to be a "
                  << "colon-separated string with "
                  << am_nnet.GetNnet().NumUpdatableComponents()
                  << " elements, instead got \"" << learning_rates << '"';
      }
      SubVector<BaseFloat> learning_rates_vector(&(learning_rates_vec[0]),
                                                 learning_rates_vec.size());
      am_nnet.GetNnet().SetLearningRates(learning_rates_vector);
    }

    if (scales != "") {
      std::vector<BaseFloat> scales_vec;
      if (!SplitStringToFloats(scales, ":", false, &scales_vec)
          || static_cast<int32>(scales_vec.size()) !=
             am_nnet.GetNnet().NumUpdatableComponents()) {
        KALDI_ERR << "Expected --scales option to be a "
                  << "colon-separated string with "
                  << am_nnet.GetNnet().NumUpdatableComponents()
                  << " elements, instead got \"" << scales << '"';
      }
      SubVector<BaseFloat> scales_vector(&(scales_vec[0]),
                                         scales_vec.size());
      am_nnet.GetNnet().ScaleComponents(scales_vector);
    }

    if (truncate >= 0) {
      am_nnet.GetNnet().Resize(truncate);
      if (am_nnet.GetNnet().OutputDim() != am_nnet.Priors().Dim()) {
        Vector<BaseFloat> empty_priors;
        am_nnet.SetPriors(empty_priors); // so dims don't disagree.
      }
    }

    if (remove_dropout) am_nnet.GetNnet().RemoveDropout();

    if (dropout_scale != -1.0) am_nnet.GetNnet().SetDropoutScale(dropout_scale);

    if (remove_preconditioning) am_nnet.GetNnet().RemovePreconditioning();

    if (collapse) am_nnet.GetNnet().Collapse(match_updatableness);
    
    if (stats_from != "") {
      // Copy the stats associated with the layers descending from
      // NonlinearComponent.
      bool binary;
      Input ki(stats_from, &binary);
      TransitionModel trans_model;
      trans_model.Read(ki.Stream(), binary);
      AmNnet am_nnet_stats;
      am_nnet_stats.Read(ki.Stream(), binary);
      am_nnet.GetNnet().CopyStatsFrom(am_nnet_stats.GetNnet());
    }
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Copied neural net from " << nnet_rxfilename
              << " to " << nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
