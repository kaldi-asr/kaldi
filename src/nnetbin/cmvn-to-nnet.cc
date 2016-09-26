// nnetbin/cmvn-to-nnet.cc

// Copyright 2012-2016  Brno University of Technology

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
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-various.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    const char *usage =
      "Convert cmvn-stats into <AddShift> and <Rescale> components.\n"
      "Usage:  cmvn-to-nnet [options] <transf-in> <nnet-out>\n"
      "e.g.:\n"
      " cmvn-to-nnet --binary=false transf.mat nnet.mdl\n";


    bool binary_write = false;
    float std_dev = 1.0;
    float var_floor = 1e-10;
    float learn_rate_coef = 0.0;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("std-dev", &std_dev, "Standard deviation of the output.");
    po.Register("var-floor", &var_floor,
        "Floor the variance, so the factors in <Rescale> are bounded.");
    po.Register("learn-rate-coef", &learn_rate_coef,
        "Initialize learning-rate coefficient to a value.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string cmvn_stats_rxfilename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // read the matrix,
    Matrix<double> cmvn_stats;
    {
      bool binary_read;
      Input ki(cmvn_stats_rxfilename, &binary_read);
      cmvn_stats.Read(ki.Stream(), binary_read);
    }
    KALDI_ASSERT(cmvn_stats.NumRows() == 2);
    KALDI_ASSERT(cmvn_stats.NumCols() > 1);

    int32 num_dims = cmvn_stats.NumCols() - 1;
    double frame_count = cmvn_stats(0, cmvn_stats.NumCols() - 1);

    // buffers for shift and scale
    Vector<BaseFloat> shift(num_dims);
    Vector<BaseFloat> scale(num_dims);

    // compute the shift and scale per each dimension
    for (int32 d = 0; d < num_dims; d++) {
      BaseFloat mean = cmvn_stats(0, d) / frame_count;
      BaseFloat var = cmvn_stats(1, d) / frame_count - mean * mean;
      if (var <= var_floor) {
        KALDI_WARN << "Very small variance " << var
                   << " flooring to " << var_floor;
        var = var_floor;
      }
      shift(d) = -mean;
      scale(d) = std_dev / sqrt(var);
    }

    // create empty nnet,
    Nnet nnet;

    // append shift component to nnet,
    {
      AddShift shift_component(shift.Dim(), shift.Dim());
      shift_component.SetParams(shift);
      shift_component.SetLearnRateCoef(learn_rate_coef);
      nnet.AppendComponent(shift_component);
    }

    // append scale component to nnet,
    {
      Rescale scale_component(scale.Dim(), scale.Dim());
      scale_component.SetParams(scale);
      scale_component.SetLearnRateCoef(learn_rate_coef);
      nnet.AppendComponent(scale_component);
    }

    // write the nnet,
    {
      Output ko(model_out_filename, binary_write);
      nnet.Write(ko.Stream(), binary_write);
      KALDI_LOG << "Written cmvn in 'nnet1' model to: " << model_out_filename;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
