// chainbin/chain-est-phone-lm.cc

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
#include "chain/language-model.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize un-smoothed phone language model for 'chain' training\n"
        "Output in FST format (epsilon-free deterministic acceptor)\n"
        "\n"
        "Usage:  chain-est-phone-lm [options] <phone-seqs-rspecifier> <phone-lm-fst-out>\n"
        "The phone-sequences are used to train a language model.\n"
        "e.g.:\n"
        "gunzip -c input_dir/ali.*.gz | ali-to-phones input_dir/final.mdl ark:- ark:- | \\\n"
        " chain-est-phone-lm --leftmost-context-questions=dir/leftmost_questions.txt ark:- dir/phone_G.fst\n";

    bool binary_write = true;
    LanguageModelOptions lm_opts;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    lm_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string phone_seqs_rspecifier = po.GetArg(1),
        lm_fst_wxfilename = po.GetArg(2);


    LanguageModelEstimator lm_estimator(lm_opts);

    SequentialInt32VectorReader phones_reader(phone_seqs_rspecifier);
    KALDI_LOG << "Reading phone sequences";
    for (; !phones_reader.Done(); phones_reader.Next()) {
      const std::vector<int32> &phone_seq = phones_reader.Value();
      lm_estimator.AddCounts(phone_seq);
    }
    KALDI_LOG << "Estimating phone LM";
    fst::StdVectorFst fst;
    lm_estimator.Estimate(&fst);

    WriteFstKaldi(fst, lm_fst_wxfilename);

    KALDI_LOG << "Estimated phone language model and wrote it to "
              << lm_fst_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

