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
        "See comments in the code of nnet2/get-feature-transform.h for more\n"
        "information.\n"
        "\n"
        "Usage:  nnet-get-feature-transform [options] <matrix-out> <lda-acc-1> <lda-acc-2> ...\n";

    bool binary = true;
    FeatureTransformEstimateOptions opts;
    std::string write_cholesky;
    std::string write_within_covar;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write outputs in binary mode.");
    po.Register("write-cholesky", &write_cholesky, "If supplied, write to this "
                "wxfilename the Cholesky factor of the within-class covariance. "
                "Can be used for perturbing features.  E.g. "
                "--write-cholesky=exp/nnet5/cholesky.tpmat");
    po.Register("write-within-covar", &write_within_covar, "If supplied, write "
                "to this wxfilename the within-class covariance (as a symmetric "
                "matrix). E.g. --write-within-covar=exp/nnet5/within_covar.mat");
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
    TpMatrix<BaseFloat> cholesky;
    fte.Estimate(opts, &mat,
                 (write_cholesky != "" || write_within_covar != "" ?
                  &cholesky : NULL));
    WriteKaldiObject(mat, projection_wxfilename, binary);
    if (write_cholesky != "") {
      WriteKaldiObject(cholesky, write_cholesky, binary);
    }
    if (write_within_covar != "") {
      SpMatrix<BaseFloat> within_var(cholesky.NumRows());
      within_var.AddTp2(1.0, cholesky, kNoTrans, 0.0);
      WriteKaldiObject(within_var, write_within_covar, binary);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


