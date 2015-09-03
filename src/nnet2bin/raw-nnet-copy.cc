// nnet2bin/raw-nnet-copy.cc

// Copyright 2014 Johns Hopkins University (author:  Daniel Povey)

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

#include <typeinfo>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet2/am-nnet.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy a raw neural net (this version works on raw nnet2 neural nets,\n"
        "without the transition model.  Supports the 'truncate' option.\n"
        "\n"
        "Usage:  raw-nnet-copy [options] <raw-nnet-in> <raw-nnet-out>\n"
        "e.g.:\n"
        " raw-nnet-copy --binary=false 1.mdl text.mdl\n"
        "See also: nnet-to-raw-nnet, nnet-am-copy\n";
    
    int32 truncate = -1;
    bool binary_write = true;
    std::string learning_rate_scales_str = " ";
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("truncate", &truncate, "If set, will truncate the neural net "
                "to this many components by removing the last components.");
    po.Register("learning-rate-scales", &learning_rate_scales_str,
                "Colon-separated list of scaling factors for learning rates, "
                "applied after the --learning-rate and --learning-rates options."
                "Used to scale learning rates for particular layer types.  E.g."
                "--learning-rate-scales=AffineComponent=0.5");

    po.Read(argc, argv);
    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string raw_nnet_rxfilename = po.GetArg(1),
        raw_nnet_wxfilename = po.GetArg(2);
    
    Nnet nnet;
    ReadKaldiObject(raw_nnet_rxfilename, &nnet);
    
    if (truncate >= 0)
      nnet.Resize(truncate);

    if (learning_rate_scales_str != " ")  {
      // parse the learning_rate_scales provided as an option
      std::map<std::string, BaseFloat> learning_rate_scales;
      std::vector<std::string> learning_rate_scale_vec;
      SplitStringToVector(learning_rate_scales_str, ":", true,
                          &learning_rate_scale_vec);
      for (int32 index = 0; index < learning_rate_scale_vec.size();
          index++) {
        std::vector<std::string> parts;
        BaseFloat scale_factor;
        SplitStringToVector(learning_rate_scale_vec[index],
                            "=", false,  &parts);
        if (!ConvertStringToReal(parts[1], &scale_factor)) {
          KALDI_ERR << "Unknown format for --learning-rate-scales option. "
              << "Expected format is "
              << "--learning-rate-scales=AffineComponent=0.1:AffineComponentPreconditioned=0.5 "
              << "instead got "
              << learning_rate_scales_str;
        }
        learning_rate_scales.insert(std::pair<std::string, BaseFloat>(
                parts[0], scale_factor));
      }
      // use the learning_rate_scales to scale the component learning rates
      nnet.ScaleLearningRates(learning_rate_scales);
    }

    WriteKaldiObject(nnet, raw_nnet_wxfilename, binary_write);

    KALDI_LOG << "Copied raw neural net from " << raw_nnet_rxfilename
              << " to " << raw_nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
