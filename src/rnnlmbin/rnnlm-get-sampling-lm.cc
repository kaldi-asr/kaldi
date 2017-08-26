// rnnlmbin/rnnlm-get-sampling-lm.cc

// Copyright 2017  Johns Hopkins University (author: Daniel Povey)

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
#include "rnnlm/sampling-lm-estimate.h"
#include "rnnlm/sampling-lm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::rnnlm;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    SamplingLmEstimatorOptions config;
    bool binary = true;

    // Note: we may later make it possible to write in a binary format without the
    // word symbol table being involved.
    const char *usage =
        "Estimate highly-pruned backoff LM for use in importance sampling for\n"
        "RNNLM training.  Reads integerized text.\n"
        "Usage:\n"
        " rnnlm-get-sampling-lm [options] <input-integerized-weighted-text> \\\n"
        "            <sampling-lm-out>\n"
        " (this form writes a non-human-readable format that can be read by\n"
        " rnnlm-get-egs).\n"
        " e.g.:\n"
        "  ... | rnnlm-get-sampling-lm --vocab-size=10002 - sampling.lm\n"
        "The word symbol table is used to write the ARPA file, but is expected\n"
        "to already have been used to convert the words into integer form.\n"
        "Each line of integerized input text should have a corpus weight as\n"
        "the first field, e.g.:\n"
        " 1.0   782 1271 3841 82\n"
        "and lines of input text should not be repeated (just increase the\n"
        "weight).\n"
        "See also: rnnlm-get-egs\n";

    ParseOptions po(usage);
    config.Register(&po);
    po.Register("binary", &binary, "If true, write LM in binary format "
                "(only applies to 2-argument form of the program)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (po.NumArgs() == 3) {
      std::string input_text_rxfilename = po.GetArg(1),
          symbol_table_rxfilename = po.GetArg(2),
          arpa_wxfilename = po.GetArg(3);


      fst::SymbolTable *symtab;
      {
        Input symtab_input(symbol_table_rxfilename);
        symtab = fst::SymbolTable::ReadText(symtab_input.Stream(),
                                            symbol_table_rxfilename);
        if (symtab == NULL)
          KALDI_ERR << "Error reading symbol table.";
      }
      if (config.vocab_size <= 0)
        config.vocab_size = symtab->AvailableKey();

      SamplingLmEstimator estimator(config);

      Input ki(input_text_rxfilename);
      estimator.Process(ki.Stream());
      bool will_write_arpa = true;
      estimator.Estimate(will_write_arpa);

      bool binary = false;
      Output ko(arpa_wxfilename, binary);
      estimator.PrintAsArpa(ko.Stream(), *symtab);
      delete symtab;
    } else {
      std::string input_text_rxfilename = po.GetArg(1),
          lm_wxfilename = po.GetArg(2);

      SamplingLmEstimator estimator(config);
      Input ki(input_text_rxfilename);
      estimator.Process(ki.Stream());
      bool will_write_arpa = false;
      estimator.Estimate(will_write_arpa);
      SamplingLm lm(estimator);
      Output ko(lm_wxfilename, binary);
      lm.Write(ko.Stream(), binary);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
