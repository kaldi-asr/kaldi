// ctcbin/ctc-init-transition-model.cc

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
#include "tree/context-dep.h"
#include "ctc/language-model.h"
#include "ctc/cctc-transition-model.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize CCTC transition-model object.\n"
        "\n"
        "Usage:  ctc-init-transition-model [options] <tree-in> <phone-seqs-rspecifier> <ctc-transition-model-out>\n"
        "The phone-sequences are used to train a language model.\n"
        "e.g.:\n"
        "gunzip -c input_dir/ali.*.gz | ali-to-phones input_dir/final.mdl ark:- ark:- | \\\n"
        "  cctc-init-transition-model --num-phones=43 dir/left_context_tree ark:- dir/cctc.trans_mdl\n"
        "Note: the --num-phones option is required (it should equal the highest-numbered\n"
        "'real' phone, i.e. not including disambiguation symbols #0, #1, ...)\n";
    
    bool binary_write = true;
    int32 num_phones = -1;
    LanguageModelOptions lm_opts;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("num-phones", &num_phones, "Number of phones in phone-set");
    lm_opts.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_rxfilename = po.GetArg(1),
        phone_seqs_rspecifier = po.GetArg(2),
        ctc_trans_model_wxfilename = po.GetArg(3);

    if (num_phones <= 0)
      KALDI_ERR << "--num-phones option is required (and needs a positive argument)";

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    
    LanguageModel lm;
    {
      LanguageModelEstimator lm_estimator(lm_opts, num_phones);
      
      SequentialInt32VectorReader phones_reader(phone_seqs_rspecifier);
      KALDI_LOG << "Reading phone sequences";
      for (; !phones_reader.Done(); phones_reader.Next()) {
        const std::vector<int32> &phone_seq = phones_reader.Value();
        lm_estimator.AddCounts(phone_seq);
      }
      KALDI_LOG << "Estimating phone-level "
                << lm_opts.ngram_order << "-gram language model";
      lm_estimator.Discount();
      lm_estimator.Output(&lm);
    }

    CctcTransitionModel cctc_trans_model;
    {
      CctcTransitionModelCreator creator(ctx_dep, lm);
      creator.InitCctcTransitionModel(&cctc_trans_model);
    }

    KALDI_LOG << "Writing CCTC transition model to "
              << ctc_trans_model_wxfilename;

    WriteKaldiObject(cctc_trans_model, ctc_trans_model_wxfilename,
                     binary_write);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

