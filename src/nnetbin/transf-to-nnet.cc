// nnetbin/transf-to-nnet.cc

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
#include "nnet/nnet-affine-transform.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    const char *usage =
        "Convert transformation matrix to <affine-transform>\n"
        "Usage:  transf-to-nnet [options] <transf-in> <nnet-out>\n"
        "e.g.:\n"
        " transf-to-nnet --binary=false transf.mat nnet.mdl\n";


    bool binary_write = false;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string transform_rxfilename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    //read the matrix
    Matrix<BaseFloat> transform;
    {
      bool binary_read;
      Input ki(transform_rxfilename, &binary_read);
      transform.Read(ki.Stream(), binary_read);
    }
    
    //we will put the transform to the nnet
    Nnet nnet;
    //create affine transform layer
    AffineTransform* layer = new AffineTransform(transform.NumCols(),transform.NumRows());
    //the pointer will be given to the nnet, so we don't need to call delete

    //convert Matrix to CuMatrix
    CuMatrix<BaseFloat> cu_transform(transform);

    //set the weights
    layer->SetLinearity(cu_transform);

    //append layer to the nnet
    nnet.AppendComponent(layer);
    
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


