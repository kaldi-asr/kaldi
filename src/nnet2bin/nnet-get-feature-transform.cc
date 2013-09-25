// nnet2bin/nnet-get-feature-transform.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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
#include "nnet2/get-feature-transform.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Get feature-projection transform using stats obtained with acc-lda.\n"
        "Usage:  nnet-get-feature-transform [options] <matrix-out> <lda-acc-1> <lda-acc-2> ...\n";

    bool binary = true;
    FeatureTransformEstimateOptions opts;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write accumulators in binary mode.");
    opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    FeatureTransformEstimate fte;
    std::string projection_wxfilename = po.GetArg(1);

    for (int32 i = 2; i <= po.NumArgs(); i++) {
      bool binary_in, add = true;
      Input ki(po.GetArg(i), &binary_in);
      fte.Read(ki.Stream(), binary_in, add);
    }

    Matrix<BaseFloat> mat;
    fte.Estimate(opts, &mat);
    WriteKaldiObject(mat, projection_wxfilename, binary);
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


