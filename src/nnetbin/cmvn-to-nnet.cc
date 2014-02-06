// nnetbin/cmvn-to-nnet.cc

// Copyright 2012  Brno University of Technology

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
        "Convert transformation matrix to <biasedlinearity>\n"
        "Usage:  cmvn-to-nnet [options] <transf-in> <nnet-out>\n"
        "e.g.:\n"
        " cmvn-to-nnet --binary=false transf.mat nnet.mdl\n";


    bool binary_write = false;
    bool tied_normalzation = false;
    float var_floor = 1e-10;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("tied-normalization", &tied_normalzation, "The normalization is tied accross all the input dimensions");
    po.Register("var-floor", &var_floor, "Floor the variance, so the factors in <Rescale> are bounded.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string transform_rxfilename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    //read the matrix
    Matrix<double> cmvn_stats;
    {
      bool binary_read;
      Input ki(transform_rxfilename, &binary_read);
      cmvn_stats.Read(ki.Stream(), binary_read);
    }
    KALDI_ASSERT(cmvn_stats.NumRows() == 2);
    KALDI_ASSERT(cmvn_stats.NumCols() > 1);

    //get the count
    double count = cmvn_stats(0,cmvn_stats.NumCols()-1);
   
    //buffers for shift and scale 
    Vector<BaseFloat> shift(cmvn_stats.NumCols()-1);
    Vector<BaseFloat> scale(cmvn_stats.NumCols()-1);
    
    //compute the shift and scale per each dimension
    for(int32 d=0; d<cmvn_stats.NumCols()-1; d++) {
      BaseFloat mean = cmvn_stats(0,d)/count;
      BaseFloat var = cmvn_stats(1,d)/count - mean*mean;
      if (var <= var_floor) {
        KALDI_WARN << "Very small variance " << var << " flooring to " << var_floor;
        var = var_floor;
      }
      shift(d) = -mean;
      scale(d) = 1.0 / sqrt(var);
    }

    if(tied_normalzation) {
      //just average the variances
      BaseFloat sum_var = 0.0;
      for(int32 i=0; i<scale.Dim(); i++) {
        sum_var += 1.0 / (scale(i)*scale(i));
      }
      BaseFloat mean_var = sum_var / scale.Dim();
      BaseFloat tied_scale = 1.0 / sqrt(mean_var);
      scale.Set(tied_scale);
    }

    //we will put the shift and scale to the nnet
    Nnet nnet;

    //create the shift component
    {
      AddShift* shift_component = new AddShift(shift.Dim(), shift.Dim());
      //the pointer will be given to the nnet, so we don't need to call delete
      
      //convert Vector to CuVector
      CuVector<BaseFloat> cu_shift(shift);

      //set the weights
      shift_component->SetShiftVec(cu_shift);

      //append layer to the nnet
      nnet.AppendComponent(shift_component);
    }

    //create the scale component
    {
      Rescale* scale_component = new Rescale(scale.Dim(), scale.Dim());
      //the pointer will be given to the nnet, so we don't need to call delete
      
      //convert Vector to CuVector
      CuVector<BaseFloat> cu_scale(scale);

      //set the weights
      scale_component->SetScaleVec(cu_scale);

      //append layer to the nnet
      nnet.AppendComponent(scale_component);
    }
      
    //write the nnet
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


