// gmmbin/gmm-init-model-flat.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "gmm/mle-am-diag-gmm.h"
#include "tree/build-tree-utils.h"
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

namespace kaldi {

void GetFeatureMeanAndVariance(const std::string &feat_rspecifier,
                               Vector<BaseFloat> *inv_var_out,                               
                               Vector<BaseFloat> *mean_out) {
  double count = 0.0;
  Vector<double> x_stats, x2_stats;

  SequentialDoubleMatrixReader feat_reader(feat_rspecifier);
  for (; !feat_reader.Done(); feat_reader.Next()) {
    const Matrix<double> &mat = feat_reader.Value();
    if (x_stats.Dim() == 0) {
      int32 dim = mat.NumCols();
      x_stats.Resize(dim);
      x2_stats.Resize(dim);
    }
    for (int32 i = 0; i < mat.NumRows(); i++) {
      count += 1.0;
      x_stats.AddVec(1.0, mat.Row(i));
      x2_stats.AddVec2(1.0, mat.Row(i));
    }
  }
  if (count == 0) { KALDI_ERR << "No features were read!"; }
  x_stats.Scale(1.0/count);
  x2_stats.Scale(1.0/count);
  x2_stats.AddVec2(-1.0, x_stats);
  if (x2_stats.Min() <= 0.0)
    KALDI_ERR << "Variance is zero or negative!";
  x2_stats.InvertElements();
  int32 dim = x_stats.Dim();
  inv_var_out->Resize(dim);
  mean_out->Resize(dim);
  inv_var_out->CopyFromVec(x2_stats);
  mean_out->CopyFromVec(x_stats);
}


}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize GMM, with Gaussians initialized to mean and variance\n"
        "of some provided example data (or to 0,1 if not provided: in that\n"
        "case, provide --dim option)\n"
        "Usage:  gmm-init-model-flat [options] <tree-in> <topo-file> <model-out> [<features-rspecifier>]\n"
        "e.g.: \n"
        "  gmm-init-model-flat tree topo 1.mdl ark:feats.scp\n";
    
    bool binary = true;
    int32 dim = 40;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("dim", &dim, "Dimension of model (this matters only if not providing features).");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        tree_filename = po.GetArg(1),
        topo_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3),
        feats_rspecifier = po.GetOptArg(4);

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);

    HmmTopology topo; 
    ReadKaldiObject(topo_filename, &topo);

    Vector<BaseFloat> global_inverse_var, global_mean;
    if (po.NumArgs() == 4) {
      GetFeatureMeanAndVariance(feats_rspecifier,
                                &global_inverse_var,
                                &global_mean);
      dim = global_mean.Dim();
    } else {
      global_inverse_var.Resize(dim);
      global_inverse_var.Set(1.0);
      global_mean.Resize(dim); // leave it at zero.
    }

    int32 num_pdfs = ctx_dep.NumPdfs();

    AmDiagGmm am_gmm;
    DiagGmm gmm;
    gmm.Resize(1, dim);
    {  // Initialize the gmm.
      Matrix<BaseFloat> inv_var(1, dim);
      inv_var.Row(0).CopyFromVec(global_inverse_var);
      Matrix<BaseFloat> mu(1, dim);
      mu.Row(0).CopyFromVec(global_mean);
      Vector<BaseFloat> weights(1);
      weights.Set(1.0);
      gmm.SetInvVarsAndMeans(inv_var, mu);
      gmm.SetWeights(weights);
      gmm.ComputeGconsts();
    }
    for (int i = 0; i < num_pdfs; i++)
      am_gmm.AddPdf(gmm);
    
    TransitionModel trans_model(ctx_dep, topo);

    {
      Output ko(model_out_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Wrote model.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
