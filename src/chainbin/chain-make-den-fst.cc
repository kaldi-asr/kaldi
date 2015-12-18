// chainbin/chain-make-den-fst.cc

// Copyright       2015  Johns Hopkins University (author:  Daniel Povey)

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
#include "chain/chain-den-graph.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;

    const char *usage =
        "Created 'denominator' FST for 'chain' training\n"
        "Outputs in FST format.  <denominator-fst-out> is an epsilon-free acceptor\n"
        "<normalization-fst-out> is a modified version of <denominator-fst> (w.r.t.\n"
        "initial and final probs) that is used in example generation.\n"
        "\n"
        "Usage: chain-make-den-fsth [options] <tree> <transition-model> <phone-lm-fst> "
        "<denominator-fst-out> <normalization-fst-out>\n"
        "e.g.:\n"
        "chain-make-den-fst dir/tree dir/0.trans_mdl dir/phone_lm.fst dir/den.fst dir/normalization.fst\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_rxfilename = po.GetArg(1),
        transition_model_rxfilename = po.GetArg(2),
        phone_lm_rxfilename = po.GetArg(3),
        den_fst_wxfilename = po.GetArg(4),
        normalization_fst_wxfilename = po.GetArg(5);


    ContextDependency ctx_dep;
    TransitionModel trans_model;
    fst::StdVectorFst phone_lm;

    ReadKaldiObject(tree_rxfilename, &ctx_dep);
    ReadKaldiObject(transition_model_rxfilename, &trans_model);
    ReadFstKaldi(phone_lm_rxfilename, &phone_lm);

    fst::StdVectorFst den_fst;
    chain::CreateDenominatorFst(ctx_dep, trans_model, phone_lm,
                                &den_fst);

    fst::StdVectorFst normalization_fst;
    chain::DenominatorGraph den_graph(den_fst, trans_model.NumPdfs());
    den_graph.GetNormalizationFst(den_fst, &normalization_fst);


    WriteFstKaldi(den_fst, den_fst_wxfilename);
    WriteFstKaldi(normalization_fst, normalization_fst_wxfilename);

    KALDI_LOG << "Write denominator FST to " << den_fst_wxfilename
              << " and normalization FST to " << normalization_fst_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

