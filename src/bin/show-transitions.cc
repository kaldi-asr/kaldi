// bin/show-transitions.cc
//
// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (author: Daniel Povey)

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

#include "hmm/transition-model.h"
#include "fst/fstlib.h"
#include "util/common-utils.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Print debugging info from transition model, in human-readable form\n"
        "Usage:  show-transitions <phones-symbol-table> <transition/model-file> [<occs-file>]\n"
        "e.g.: \n"
        " show-transitions phones.txt 1.mdl 1.occs\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string phones_symtab_filename = po.GetArg(1),
        transition_model_filename = po.GetArg(2),
        accumulator_filename = po.GetOptArg(3);


    fst::SymbolTable *syms = fst::SymbolTable::ReadText(phones_symtab_filename);
    if (!syms)
      KALDI_ERR << "Could not read symbol table from file "
                 << phones_symtab_filename;
    std::vector<std::string> names(syms->NumSymbols());
    for (size_t i = 0; i < syms->NumSymbols(); i++)
      names[i] = syms->Find(i);

    TransitionModel trans_model;
    ReadKaldiObject(transition_model_filename, &trans_model);

    Vector<double> occs;
    if (accumulator_filename != "") {
      bool binary_in;
      Input ki(accumulator_filename, &binary_in);
      occs.Read(ki.Stream(), binary_in);
    }

    trans_model.Print(std::cout,
                      names,
                      (accumulator_filename != "" ? &occs : NULL));

    delete syms;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

