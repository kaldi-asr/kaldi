// fgmmbin/fgmm-global-init-from-accs.cc

// Copyright 2015-2017 David Snyder
//                2015 Johns Hopkins University (Author: Daniel Povey)
//                2015 Johns Hopkins University (Author: Daniel Garcia-Romero)

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
#include "gmm/full-gmm.h"
#include "gmm/mle-full-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef int32 int32;
    MleFullGmmOptions gmm_opts;

    const char *usage =
        "Initialize a full-covariance GMM from the accumulated stats.\n"
        "This binary is similar to fgmm-global-est, but does not use "
        "a preexisting model.  See also fgmm-global-est.\n"
        "Usage:  fgmm-global-init-from-accs [options] <stats-in> "
        "<number-of-components> <model-out>\n";

    bool binary_write = true;
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    gmm_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_filename = po.GetArg(1),
        model_out_filename = po.GetArg(3);
    int32 num_components = atoi(po.GetArg(2).c_str());

    AccumFullGmm gmm_accs;
    {
      bool binary;
      Input ki(stats_filename, &binary);
      gmm_accs.Read(ki.Stream(), binary, true /* add accs. */);
    }

    int32 num_gauss = gmm_accs.NumGauss(), dim = gmm_accs.Dim(),
          tot_floored = 0, gauss_floored = 0, tot_low_occ = 0;

    FullGmm fgmm(num_components, dim);

    Vector<BaseFloat> weights(num_gauss);
    Matrix<BaseFloat> means(num_gauss, dim);
    std::vector<SpMatrix<BaseFloat> > invcovars;

    for (int32 i = 0; i < num_components; i++) {
      BaseFloat occ = gmm_accs.occupancy()(i);
      weights(i) = occ;
      Vector<BaseFloat> mean(dim, kSetZero);
      SpMatrix<BaseFloat> covar(dim, kSetZero);

      // If the occupancy for a Gaussian is very low, set it to a small value.
      if (occ < 1e-10) {
        weights(i) = 1e-10;
        mean.SetRandn();
        Vector<BaseFloat> diag(mean.Dim());
        diag.Set(1.0);
        covar.AddDiagVec(1.0, diag);
        tot_low_occ++;
      // This is the typical case.
      } else {
        mean.CopyRowFromMat(gmm_accs.mean_accumulator(), i);
        mean.Scale(1.0 / occ);
        covar.CopyFromSp(gmm_accs.covariance_accumulator()[i]);
        covar.Scale(1.0 / occ);
        covar.AddVec2(-1.0, mean);  // subtract squared means.
      }
      means.CopyRowFromVec(mean, i);

      // Floor variance Eigenvalues.
      BaseFloat floor = std::max(
          static_cast<BaseFloat>(gmm_opts.variance_floor),
          static_cast<BaseFloat>(covar.MaxAbsEig() / gmm_opts.max_condition));
      int32 floored = covar.ApplyFloor(floor);
      if (floored) {
        tot_floored += floored;
        gauss_floored++;
      }
      covar.InvertDouble();
      invcovars.push_back(covar);
    }
    weights.Scale(1.0 / weights.Sum());
    fgmm.SetWeights(weights);
    fgmm.SetInvCovarsAndMeans(invcovars, means);
    int32 num_bad = fgmm.ComputeGconsts();
    KALDI_LOG << "FullGmm has " << num_bad << " bad GConsts";

    if (tot_floored > 0) {
      KALDI_WARN << tot_floored << " variances floored in " << gauss_floored
                 << " Gaussians.";
    }
    if (tot_low_occ > 0) {
      KALDI_WARN << tot_low_occ << " out of " << num_gauss
                 << " Gaussians had very low occupancy.";
    }

    WriteKaldiObject(fgmm, model_out_filename, binary_write);

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
