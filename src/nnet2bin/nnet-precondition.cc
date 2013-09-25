// nnet2bin/nnet-precondition.cc

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet2/nnet-randomize.h"
#include "nnet2/nnet-lbfgs.h"
#include "nnet2/am-nnet.h"

namespace kaldi {
namespace nnet2 {
void NormalizeNnet(Nnet *nnet) {
  for (int32 c = 0; c < nnet->NumComponents(); c++) {
    AffineComponent *ac =
        dynamic_cast<AffineComponent*>(&(nnet->GetComponent(c)));
    if (ac != NULL) {
      int32 output_dim = ac->OutputDim();
      BaseFloat dot_prod = ac->DotProduct(*ac);
      // Assuming the standard deviation of the elements was 1.0/sqrt(input_dim),
      // so the variance of the elements was 1.0/input_dim, the dot_prod would
      // be (input_dim * output_dim) / input_dim, which equals output_dim.
      // We rescale to make it equal to output_dim;
      if (dot_prod != 0.0) {
        BaseFloat scale = std::sqrt(output_dim / dot_prod);
        ac->Scale(scale);
      }
    }
  }
}

} // namespace nnet2
} // namespace kaldi


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Apply preconditioning to neural-net gradients by multiplying by inverse-Fisher-like\n"
        "quantities obtained by the program nnet-get-preconditioner.  This is useful in batch\n"
        "gradient descent methods.  Note: writing the preconditioner again is supported\n"
        "because the first time a model is preconditioned, certain quantities are pre-computed,\n"
        "and the model written in this way can be used to save computation later on.\n"
        "\n"
        "Usage:  nnet-precondition [options] <preconditioner-in> <model-in> <model-out> [<preconditioner-out>]\n"
        "\n"
        "e.g.:\n"
        "nnet-precondition 1.preconditioner 1.nnet 1pre.nnet\n";
    
    bool binary_write = true;
    BaseFloat scale = 1.0;
    bool normalize = false;
    PreconditionConfig config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("scale", &scale, "Scaling factor to apply to gradients (if not 1.0)");
    po.Register("normalize", &normalize, "If true, normalize each AffineComponent (or descendant) "
                "to have a \"standard\" parameter variance.");
    config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string precon_rxfilename = po.GetArg(1),
        nnet_rxfilename = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3),
        precon_wxfilename = po.GetOptArg(4);
    
    AmNnet am_precon;
    {
      bool binary_read;
      Input ki(precon_rxfilename, &binary_read);
      TransitionModel trans_model;
      trans_model.Read(ki.Stream(), binary_read);
      am_precon.Read(ki.Stream(), binary_read);
    }

    AmNnet am_nnet;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    
    PreconditionNnet(config,
                     &(am_precon.GetNnet()),
                     &(am_nnet.GetNnet()));

    if (scale != 1.0) {
      KALDI_LOG << "Scaling neural net parameters by " << scale;
      Vector<BaseFloat> scales(am_nnet.GetNnet().NumUpdatableComponents());
      scales.Set(scale);
      am_nnet.GetNnet().ScaleComponents(scales);
    }

    if (normalize) {
      NormalizeNnet(&(am_nnet.GetNnet()));
    }
    
    {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Preconditioned neural net, wrote it to "
              << nnet_wxfilename;

    if (precon_wxfilename != "") {
      // we only make it possible write it because when we apply it, it
      // precomputes some stuff and we'd like to make it possible to avoid
      // duplicating that work.
      Output ko(precon_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_precon.Write(ko.Stream(), binary_write);
      KALDI_LOG << "Wrote preconditioner to "
                << precon_wxfilename;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


