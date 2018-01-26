// nnetbin/minmax-to-nnet.cc

// Copyright 2015  IDIAP Research Institute
// Author: B. Potard

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
        "Convert minmax-stats into <AddShift> and <Rescale> components.\n"
        "Usage:  minmax-to-nnet [options] <transf-in> <nnet-out>\n"
        "e.g.:\n"
        " minmax-to-nnet --binary=false transf.mat nnet.mdl\n";


    bool binary_write = false;
    float learn_rate_coef = 0.0;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("learn-rate-coef", &learn_rate_coef, "Initialize learning-rate coefficient to a value.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string transform_rxfilename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // read the matrix
    Matrix<double> minmax_stats;
    {
      bool binary_read;
      Input ki(transform_rxfilename, &binary_read);
      minmax_stats.Read(ki.Stream(), binary_read);
    }
    KALDI_ASSERT(minmax_stats.NumRows() == 2);
    KALDI_ASSERT(minmax_stats.NumCols() > 1);

    int32 num_dims = minmax_stats.NumCols() - 1;
    double count = minmax_stats(0, minmax_stats.NumCols()-1);
   
    // buffers for shift and scale 
    Vector<BaseFloat> shift(num_dims);
    Vector<BaseFloat> scale(num_dims);
    
    // Make this configurable?
    double ca = 0.01;
    double da = 0.99;

    // compute the shift and scale per each dimension
    for(int32 d = 0; d < num_dims; d++) {
        BaseFloat min = minmax_stats(0, d);
        BaseFloat max = minmax_stats(1, d);
        if (min + 1e-7 >= max) max = min + 1e-7;
        shift(d) = (max * ca - min * da) / (max - min);
        scale(d) = (da - ca) / (max - min);
        std::cerr << d << ": " <<  shift(d) << "," << scale(d) << '\n';
    }

    // we will put the shift and scale to the nnet
    Nnet nnet;


    // create the scale component
    {
      Rescale scale_component = Rescale(scale.Dim(), scale.Dim());

      // set the weights
      scale_component.SetParams(scale);

      // set the learn-rate coef
      scale_component.SetLearnRateCoef(learn_rate_coef);

      // append layer to the nnet
      nnet.AppendComponent(scale_component);
    }
    // create the shift component
    {
      AddShift shift_component = AddShift(shift.Dim(), shift.Dim());

      // set the weights
      shift_component.SetParams(shift);

      // set the learn-rate coef
      shift_component.SetLearnRateCoef(learn_rate_coef);

      // append layer to the nnet
      nnet.AppendComponent(shift_component);
    }
      
    // write the nnet
    {
      Output ko(model_out_filename, binary_write);
      nnet.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


