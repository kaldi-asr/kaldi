// bin/arpa2fst.cc
//
// Copyright 2009-2011  Gilles Boulianne.
//
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

#include <string>

#include "lm/arpa-lm-compiler.h"
#include "util/kaldi-io.h"
#include "util/parse-options.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage  =
        "Converts an ARPA format language model into a FST\n"
        "Usage: arpa2fst [opts] (input_arpa|-)  [output_fst|-]\n";
    kaldi::ParseOptions po(usage);

    bool natural_base = true;
    po.Register("natural-base", &natural_base, "Use log-base e (not log-base 10)");
    po.Read(argc, argv);

    if (po.NumArgs() != 1 && po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string arpa_rxfilename = po.GetArg(1),
        fst_wxfilename = po.GetOptArg(2);

    ArpaParseOptions options;
    fst::SymbolTable symbols;
    symbols.AddSymbol("<eps>", 0);
    options.bos_symbol = symbols.AddSymbol("<s>");
    options.eos_symbol = symbols.AddSymbol("</s>");
    options.oov_handling = ArpaParseOptions::kAddToSymbols;
    options.use_log10 = !natural_base;

    kaldi::ArpaLmCompiler lm_compiler(options, 0, &symbols);
    ReadKaldiObject(arpa_rxfilename, &lm_compiler);

    bool write_binary = true, write_header = false;
    kaldi::Output ko(fst_wxfilename, write_binary, write_header);
    fst::FstWriteOptions wopts(kaldi::PrintableWxfilename(fst_wxfilename));
    lm_compiler.Fst().Write(ko.Stream(), wopts);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
