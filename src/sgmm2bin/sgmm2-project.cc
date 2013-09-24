// sgmm2bin/sgmm2-project.cc

// Copyright 2012    Johns Hopkins University (Author: Daniel Povey)

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
#include "thread/kaldi-thread.h"
#include "hmm/transition-model.h"
#include "sgmm2/am-sgmm2-project.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Compute SGMM model projection that only models a part of a pre-LDA space.\n"
        "Used in predictive SGMMs.  Takes as input an LDA+MLLT transform,\n"
        "and outputs a transform from the pre-LDA+MLLT space to the space that\n"
        "we want to model\n"
        "Usage: sgmm2-project [options] <model-in> <lda-mllt-mat-in> <model-out> <new-projection-out>\n"
        "e.g.: sgmm2-project --start-dim=0 --end-dim=52 final.mdl final.inv_full_mat final_proj1.mdl proj1.mat\n";
    
    std::string write_flags_str = "gsnu";

    bool binary_write = false;
    int32 start_dim = 0;
    int32 end_dim = 0;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("start-dim", &start_dim, "Starting dimension to keep in "
                "pre-LDA-MLLT space.");
    po.Register("end-dim", &end_dim, "Ending dimension to keep in "
                "pre-LDA-MLLT space (equals last retained dimension plus one)");

    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_rxfilename = po.GetArg(1),
        lda_mllt_rxfilename = po.GetArg(2),
        model_wxfilename = po.GetArg(3),
        proj_wxfilename = po.GetArg(4);

    kaldi::SgmmWriteFlagsType write_flags =
        StringToSgmmWriteFlags(write_flags_str);
    
    AmSgmm2 am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }


    Matrix<BaseFloat> lda_mllt_mat;
    ReadKaldiObject(lda_mllt_rxfilename, &lda_mllt_mat);

    // Need the full LDA+MLLT matrix, including the extra rows.
    // See featbin/extend-transform.cc
    KALDI_ASSERT(lda_mllt_mat.NumRows() == lda_mllt_mat.NumCols());

    Matrix<BaseFloat> inv_lda_mllt_mat(lda_mllt_mat);
    inv_lda_mllt_mat.Invert();

    Matrix<BaseFloat> projection;
    Sgmm2Project sgmm_project;
    sgmm_project.ComputeProjection(am_sgmm, inv_lda_mllt_mat, start_dim, end_dim,
                                   &projection);

    Matrix<BaseFloat> total_projection(projection.NumRows(), projection.NumCols());
    total_projection.AddMatMat(1.0, projection, kNoTrans,
                               inv_lda_mllt_mat, kNoTrans, 0.0);
    
    sgmm_project.ApplyProjection(total_projection, &am_sgmm);
    
    am_sgmm.ComputeDerivedVars(); // recompute normalizers, and possibly
    // weights.
    
    {
      Output ko(model_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_sgmm.Write(ko.Stream(), binary_write, write_flags);
    }
    KALDI_LOG << "Wrote model to " << model_wxfilename;

    WriteKaldiObject(projection, proj_wxfilename, binary_write);
    KALDI_LOG << "Wrote projection matrix to " << proj_wxfilename;
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


