// nnet3bin/rnnlm-get-egs.cc

// Copyright      2017  Johns Hopkins University (author:  Daniel Povey)

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

#include <sstream>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "rnnlm/sampling-lm.h"
#include "rnnlm/rnnlm-example.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::rnnlm;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program processes lines of text (typically sentences) with weights,\n"
        "in a format like:\n"
        "  1.0 67 5689 21 8940 6723\n"
        "and turns them into examples (class RnnlmExample) for RNNLM training.\n"
        "This involves splitting up the sentences to a maximum length,\n"
        "importance sampling and other procedures.\n"
        "\n"
        "Usage:\n"
        "(1) no sampling:\n"
        " rnnlm-get-egs [options] <sentences-rxfilename> <rnnlm-egs-wspecifier>\n"
        "(2) sampling, ARPA LM read:\n"
        " rnnlm-get-egs [options] <symbol-table> <ARPA-rxfilename> \\\n"
        "                         <sentences-rxfilename>  <rnnlm-egs-wspecifier>\n"
        "(3) sampling, non-ARPA LM read:\n"
        "    rnnlm-get-egs [options] <LM-rxfilename> <sentences-rxfilename>\\\n"
        "                            <rnnlm-egs-wspecifier>\n"
        "E.g.:\n"
        " ... | rnnlm-get-egs --vocab-size=20002 - ark:- | rnnlm-train ...\n"
        "or (with sampling, reading LM as ARPA):\n"
        " ... | rnnlm-get-egs words.txt foo.arpa - ark:- | rnnlm-train ...\n"
        "or (with sampling, reading LM natively):\n"
        " ... | rnnlm-get-egs sampling.lm - ark:- | rnnlm-train ...\n"
        "\n"
        "See also: rnnlm-train\n";

    int32 srand_seed = 0;
    RnnlmEgsConfig egs_config;
    TaskSequencerConfig sequencer_config;  // has --num-threads option; only
                                           // relevant if we are using sampling

    ParseOptions po(usage);

    egs_config.Register(&po);
    sequencer_config.Register(&po);
    po.Register("srand", &srand_seed, "Seed for random number generator; "
                "affects splitting of utterances and arrangement into "
                "minibatches.");

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    srand(srand_seed);

    if (po.NumArgs() == 4) {
      // the language model is provided (as an ARPA file), and we are doing
      // sampling.
      std::string symbol_table_rxfilename = po.GetArg(1),
          arpa_rxfilename = po.GetArg(2),
          sentences_rxfilename = po.GetArg(3),
          egs_wspecifier = po.GetArg(4);

      RnnlmExampleWriter writer(egs_wspecifier);

      fst::SymbolTable *symtab;
      {
        Input symtab_input(symbol_table_rxfilename);
        symtab = fst::SymbolTable::ReadText(symtab_input.Stream(),
                                            symbol_table_rxfilename);
        if (symtab == NULL)
          KALDI_ERR << "Error reading symbol table.";
      }
      ArpaParseOptions arpa_options;
      arpa_options.bos_symbol = egs_config.bos_symbol;
      arpa_options.eos_symbol = egs_config.eos_symbol;
      SamplingLm lm(arpa_options, symtab);
      {
        Input arpa_input(arpa_rxfilename);
        lm.Read(arpa_input.Stream());
      }
      if (egs_config.vocab_size <= 0)
        egs_config.vocab_size = lm.VocabSize();
      RnnlmExampleSampler sampler(egs_config, lm);
      RnnlmExampleCreator creator(egs_config, sequencer_config,
                                  sampler, &writer);
      Input ki(sentences_rxfilename);
      creator.Process(ki.Stream());
      delete symtab;
    } else if (po.NumArgs() == 3) {
      // the language model is provided (in its native format, not as an ARPA
      // file), and we are doing sampling.
      std::string lm_rxfilename = po.GetArg(1),
          sentences_rxfilename = po.GetArg(2),
          egs_wspecifier = po.GetArg(3);

      RnnlmExampleWriter writer(egs_wspecifier);
      SamplingLm lm;
      ReadKaldiObject(lm_rxfilename, &lm);
      if (egs_config.vocab_size <= 0)
        egs_config.vocab_size = lm.VocabSize();
      RnnlmExampleSampler sampler(egs_config, lm);
      RnnlmExampleCreator creator(egs_config, sequencer_config,
                                  sampler, &writer);
      Input ki(sentences_rxfilename);
      creator.Process(ki.Stream());
    } else {
      std::string sentences_rxfilename = po.GetArg(1),
          egs_wspecifier = po.GetArg(2);
      RnnlmExampleWriter writer(egs_wspecifier);
      RnnlmExampleCreator creator(egs_config, &writer);
      Input ki(sentences_rxfilename);
      creator.Process(ki.Stream());
    }
    return 0;  // we'd have died with an exception if there was a problem or if
               // we didn't process any data.
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
