// gmmbin/fmpe-init-gmms.cc

// Copyright 2009-2011   Yanmin Qian

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
#include "util/kaldi-io.h"
#include "gmm/diag-gmm.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/fmpe-am-diag-gmm.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    typedef kaldi::BaseFloat BaseFloat;

    const char *usage =
        "Cluster the Gaussians in a diagonal-GMM acoustic model\n"
        "to two single diag-covariance GMMs used in fmpe: one is the gmm model\n"
        "for compute gaussian posteriors and one is the gaussian\n"
        "cluster centers which is used to speed up gaussian calculations"
        "Usage: fmpe-init-gmms [options] <model-file> <state-occs> <gmm-out> <gmm-cluster-centers-out> <gaussian-cluster-center-map-out>\n";

    bool binary_write = false;
    int32 gmm_num_comps = 2048;
    int32 gmm_num_cluster_centers = 128;
    BaseFloat cluster_varfloor = 0.01;
    kaldi::FmpeConfig fmpe_opts;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("gmm-num-comps", &gmm_num_comps, "Number of the Gaussian"
        " components in the gmm model to calculate the gaussian posteriors.");
    po.Register("gmm-num-cluster-centers", &gmm_num_cluster_centers, "Number"
        " of the Gaussian cluster centers for fast posteriors evaluation.");
    po.Register("cluster-varfloor", &cluster_varfloor,
      "Variance floor used in bottom-up state clustering.");

    fmpe_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        occs_in_filename = po.GetArg(2),
        gmm_out_filename = po.GetArg(3),
        gmm_cluster_centers_out_filename = po.GetArg(4),
        gauss_cluster_center_map_out_filename = po.GetArg(5);

    kaldi::AmDiagGmm am_gmm;
    kaldi::TransitionModel trans_model;
    {
      bool binary_read;
      kaldi::Input is(model_in_filename, &binary_read);
      trans_model.Read(is.Stream(), binary_read);
      am_gmm.Read(is.Stream(), binary_read);
    }

    kaldi::Vector<BaseFloat> state_occs;
    state_occs.Resize(am_gmm.NumPdfs());
    {
      bool binary_read;
      kaldi::Input is(occs_in_filename, &binary_read);
      state_occs.Read(is.Stream(), binary_read);
    }

    kaldi::DiagGmm gmm;
    kaldi::DiagGmm gmm_cluster_centers;
    std::vector<int32> gaussian_cluster_center_map;
    ObtainUbmAndSomeClusterCenters(
                     am_gmm,
                     state_occs,
                     fmpe_opts,
                     &gmm,
                     &gmm_cluster_centers,
                     &gaussian_cluster_center_map);

    // Write out the gmms model
    {
      kaldi::Output os(gmm_out_filename, binary_write);
      gmm.Write(os.Stream(), binary_write);
      gmm_cluster_centers.Write(os.Stream(), binary_write);
      kaldi::WriteIntegerVector(os.Stream(), binary_write, gaussian_cluster_center_map);
    }

    KALDI_LOG << "Written GMMs to " << gmm_out_filename;
  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


