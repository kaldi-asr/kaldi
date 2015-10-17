// nnet2bin/nnet-am-average.cc

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

#include <algorithm>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet2/combine-nnet-a.h"
#include "nnet2/am-nnet.h"

namespace kaldi {

void GetWeights(const std::string &weights_str,
                int32 num_inputs,
                vector<BaseFloat> *weights) {
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



std::vector<bool> GetSkipLayers(const std::string &skip_layers_str,
                                const int32 first_layer_idx,
                                const int32 last_layer_idx) {

  std::vector<bool> skip_layers(last_layer_idx, false);

  if (skip_layers_str.empty()) {
    return skip_layers;
  }

  std::vector<int> layer_indices;
  bool ret = SplitStringToIntegers(skip_layers_str, ":", true, &layer_indices);
  if (!ret) {
    KALDI_ERR << "Cannot parse the skip layers specifier. It should be"
              << "colon-separated list of integers";
  }

  int min_elem = std::numeric_limits<int>().max(),
      max_elem = std::numeric_limits<int>().min();

  std::vector<int>::iterator it;
  for ( it = layer_indices.begin(); it != layer_indices.end(); ++it ) {
    if ( *it < 0 )
      *it = last_layer_idx + *it;  // convert the negative indices to
                                       // correct indices -- -1 would be the
                                       // last one, -2 the one before the last
                                       // and so on.
    if (*it > max_elem)
      max_elem = *it;

    if (*it < min_elem)
      min_elem = *it;
  }

  if (max_elem >= last_layer_idx) {
    KALDI_ERR << "--skip-layers option has to be a colon-separated list"
              << "of indices which are supposed to be skipped.\n"
              << "Maximum expected index: " << last_layer_idx
              << " got: " << max_elem ;
  }
  if (min_elem < first_layer_idx) {
    KALDI_ERR << "--skip-layers option has to be a colon-separated list"
              << "of indices which are supposed to be skipped.\n"
              << "Minimum expected index: " << first_layer_idx
              << " got: " << min_elem ;
  }

  for ( it = layer_indices.begin(); it != layer_indices.end(); ++it ) {
    skip_layers[*it] = true;
  }
  return skip_layers;
}

}
int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program averages (or sums, if --sum=true) the parameters over a\n"
        "number of neural nets.  If you supply the option --skip-last-layer=true,\n"
        "the parameters of the last updatable layer are copied from <model1> instead\n"
        "of being averaged (useful in multi-language scenarios).\n"
        "The --weights option can be used to weight each model differently.\n"
        "\n"
        "Usage:  nnet-am-average [options] <model1> <model2> ... <modelN> <model-out>\n"
        "\n"
        "e.g.:\n"
        " nnet-am-average 1.1.nnet 1.2.nnet 1.3.nnet 2.nnet\n";

    bool binary_write = true;
    bool sum = false;

    ParseOptions po(usage);
    po.Register("sum", &sum, "If true, sums instead of averages.");
    po.Register("binary", &binary_write, "Write output in binary mode");
    string weights_str;
    bool skip_last_layer = false;
    string skip_layers_str;
    po.Register("weights", &weights_str, "Colon-separated list of weights, one "
                "for each input model.  These will be normalized to sum to one.");
    po.Register("skip-last-layer", &skip_last_layer, "If true, averaging of "
                "the last updatable layer is skipped (result comes from model1)");
    po.Register("skip-layers", &skip_layers_str, "Colon-separated list of "
                "indices of the layers that should be skipped during averaging."
                "Be careful: this parameter uses an absolute indexing of "
                "layers, i.e. iterates over all components, not over updatable "
                "ones only.");

    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        nnet1_rxfilename = po.GetArg(1),
        nnet_wxfilename = po.GetArg(po.NumArgs());

    TransitionModel trans_model1;
    AmNnet am_nnet1;
    {
      bool binary_read;
      Input ki(nnet1_rxfilename, &binary_read);
      trans_model1.Read(ki.Stream(), binary_read);
      am_nnet1.Read(ki.Stream(), binary_read);
    }

    int32 num_inputs = po.NumArgs() - 1;

    vector<BaseFloat> model_weights;
    GetWeights(weights_str, num_inputs, &model_weights);

    int32 c_begin = 0,
        c_end = (skip_last_layer ?
                 am_nnet1.GetNnet().LastUpdatableComponent() :
                 am_nnet1.GetNnet().NumComponents());
    KALDI_ASSERT(c_end != -1 && "Network has no updatable components.");

    int32 last_layer_idx = am_nnet1.GetNnet().NumComponents();
    vector<bool> skip_layers = GetSkipLayers(skip_layers_str,
                                             0,
                                             last_layer_idx);

    // scale the components - except the last layer, if skip_last_layer == true.
    for (int32 c = c_begin; c < c_end; c++) {
      if (skip_layers[c]) {
        KALDI_VLOG(2) << "Not averaging layer " << c << " (as requested)";
        continue;
      }
      bool updated = false;
      UpdatableComponent *uc =
        dynamic_cast<UpdatableComponent*>(&(am_nnet1.GetNnet().GetComponent(c)));
      if (uc != NULL)  {
        KALDI_VLOG(2) << "Averaging layer " << c << " (UpdatableComponent)";
        uc->Scale(model_weights[0]);
        updated = true;
      }
      NonlinearComponent *nc =
        dynamic_cast<NonlinearComponent*>(&(am_nnet1.GetNnet().GetComponent(c)));
      if (nc != NULL) {
        KALDI_VLOG(2) << "Averaging layer " << c << " (NonlinearComponent)";
        nc->Scale(model_weights[0]);
        updated = true;
      }
      if (! updated) {
        KALDI_VLOG(2) << "Not averaging layer " << c
          << " (unscalable component)";
      }
    }

    for (int32 i = 2; i <= num_inputs; i++) {
      bool binary_read;
      Input ki(po.GetArg(i), &binary_read);
      TransitionModel trans_model;
      trans_model.Read(ki.Stream(), binary_read);
      AmNnet am_nnet;
      am_nnet.Read(ki.Stream(), binary_read);

      for (int32 c = c_begin; c < c_end; c++) {
        if (skip_layers[c]) continue;

        UpdatableComponent *uc_average =
          dynamic_cast<UpdatableComponent*>(&(am_nnet1.GetNnet().GetComponent(c)));
        const UpdatableComponent *uc_this =
          dynamic_cast<const UpdatableComponent*>(&(am_nnet.GetNnet().GetComponent(c)));
        if (uc_average != NULL) {
          KALDI_ASSERT(uc_this != NULL &&
                       "Networks must have the same structure.");
          uc_average->Add(model_weights[i-1], *uc_this);
        }

        NonlinearComponent *nc_average =
          dynamic_cast<NonlinearComponent*>(&(am_nnet1.GetNnet().GetComponent(c)));
        const NonlinearComponent *nc_this =
          dynamic_cast<const NonlinearComponent*>(&(am_nnet.GetNnet().GetComponent(c)));
        if (nc_average != NULL) {
          KALDI_ASSERT(nc_this != NULL &&
                       "Networks must have the same structure.");
          nc_average->Add(model_weights[i-1], *nc_this);
        }
      }
    }

    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model1.Write(ko.Stream(), binary_write);
      am_nnet1.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Averaged parameters of " << num_inputs
              << " neural nets, and wrote to " << nnet_wxfilename;
    return 0; // it will throw an exception if there are any problems.
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

