// bin/show-alignments.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Display alignments in human-readable form\n"
        "Usage:  show-alignments  [options] <phone-syms> <model> <alignments-rspecifier>\n"
        "e.g.: \n"
        " show-alignments phones.txt 1.mdl ark:1.ali\n";
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string phones_symtab_filename = po.GetArg(1),
        model_filename = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3);

    TransitionModel trans_model;
    ReadKaldiObject(model_filename, &trans_model);

    fst::SymbolTable *phones_symtab = NULL;
    {
      std::ifstream is(phones_symtab_filename.c_str());
      phones_symtab = fst::SymbolTable::ReadText(is, phones_symtab_filename);
      if (!phones_symtab || phones_symtab->NumSymbols() == 0)
        KALDI_ERR << "Error opening symbol table file "<<phones_symtab_filename;
    }


    SequentialInt32VectorReader reader(alignments_rspecifier);

    for (; !reader.Done(); reader.Next()) {
      std::string key = reader.Key();
      const std::vector<int32> &alignment = reader.Value();

      std::vector<std::vector<int32> > split;
      SplitToPhones(trans_model, alignment, &split);

      // split_str is the numerical form of the alignments..
      std::vector<std::string> split_str(split.size());
      std::vector<std::string> split_str_phones(split.size());
      for (size_t i = 0; i < split.size(); i++) {
        std::ostringstream ss;
        ss << "[ ";
        for (size_t j = 0; j < split[i].size(); j++)
          ss << split[i][j] << " ";
        ss << "] ";
        split_str[i] = ss.str();

        int32 tid = split[i][0],
            tstate = trans_model.TransitionIdToTransitionState(tid),
            phone = trans_model.TransitionStateToPhone(tstate);
        split_str_phones[i] =
            phones_symtab->Find(phone) + " ";
        std::string space;
        int len = abs(static_cast<int>(split_str[i].size())-
                      static_cast<int>(split_str_phones[i].size()));
        for (int j = 0; j < len; j++)
          space += " ";
        if (split_str[i].size() < split_str_phones[i].size())
          split_str[i] += space;
        else
          split_str_phones[i] += space;
      }
      std::cout << key << "  ";
      for (size_t i = 0; i < split_str.size(); i++)
        std::cout << split_str[i];
      std::cout << '\n';
      std::cout << key << "  ";
      for (size_t i = 0; i < split_str_phones.size(); i++)
        std::cout << split_str_phones[i];
      std::cout << '\n';
      std::cout << '\n';
    }
    delete phones_symtab;
    phones_symtab = NULL;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


