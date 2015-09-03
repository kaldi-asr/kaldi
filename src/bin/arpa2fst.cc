// bin/arpa2fst.cc
//
// Copyright 2009-2011  Gilles Boulianne.
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

/// @addtogroup LanguageModel
/// @{

/**
 * @file arpa2fst.cc
 * @brief Example for converting an ARPA format language model into an FST.
 *
 */

#include <string>
#include "lm/kaldi-lm.h"
#include "util/parse-options.h"

int main(int argc, char *argv[]) {
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
    std::string arpa_filename = po.GetArg(1),
        fst_filename = po.GetOptArg(2);
    
    kaldi::LangModelFst lm;
    // read from standard input and write to standard output
    lm.Read(arpa_filename, kaldi::kArpaLm, NULL, natural_base);
    lm.Write(fst_filename);
    exit(0);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
/// @}

