// gmmbin/gmm-make-regtree.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

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
#include "util/kaldi-io.h"
#include "util/text-utils.h"
#include "gmm/mle-am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "transform/regression-tree.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    typedef kaldi::BaseFloat BaseFloat;

    const char *usage =
        "Build regression class tree.\n"
        "Usage: gmm-make-regtree [options] <model-file> <regtree-out>\n"
        "E.g.: gmm-make-regtree --silphones=1:2:3 --state-occs=1.occs 1.mdl 1.regtree\n"
        " [Note: state-occs come from --write-occs option of gmm-est]\n";

    std::string occs_in_filename;
    std::string sil_phones_str;
    bool binary_write = true;
    int32 max_leaves = 1;
    kaldi::ParseOptions po(usage);
    po.Register("state-occs", &occs_in_filename, "File containing state occupancies (use --write-occs in gmm-est)");
    po.Register("sil-phones", &sil_phones_str, "Colon-separated list of integer ids of silence phones, e.g. 1:2:3; if used, create top-level speech/sil split (only one reg-class for silence).");
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("max-leaves", &max_leaves, "Maximum number of leaves in regression tree.");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        tree_out_filename = po.GetArg(2);

    kaldi::AmDiagGmm am_gmm;
    kaldi::TransitionModel trans_model;
    {
      bool binary_read;
      kaldi::Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    kaldi::Vector<BaseFloat> state_occs;
    if (occs_in_filename != "") {
      bool binary_read;
      kaldi::Input ki(occs_in_filename, &binary_read);
      state_occs.Read(ki.Stream(), binary_read);
    } else {
      KALDI_LOG << "--state-occs option not provided so using constant occupancies.";
      state_occs.Resize(am_gmm.NumPdfs());
      state_occs.Set(1.0);
    }

    std::vector<int32> sil_pdfs;
    if (sil_phones_str != "") {
      std::vector<int32> sil_phones;
      if (!kaldi::SplitStringToIntegers(sil_phones_str, ":", false, &sil_phones))
        KALDI_ERR << "invalid sil-phones option " << sil_phones_str;
      std::sort(sil_phones.begin(), sil_phones.end());
      bool ans = GetPdfsForPhones(trans_model, sil_phones, &sil_pdfs);
      if (!ans)
        KALDI_WARN << "Pdfs associated with silence phones are not only "
            "associated with silence phones: your speech-silence split "
            "may not be meaningful.";
    }

    kaldi::RegressionTree regtree;
    regtree.BuildTree(state_occs, sil_pdfs, am_gmm, max_leaves);
    // Write out the regression tree
    {
      kaldi::Output ko(tree_out_filename, binary_write);
      regtree.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written regression tree to " << tree_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


