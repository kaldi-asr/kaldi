// fstbin/fstmakedisambigsyms.cc

// Copyright 2009-2011  Microsoft Corporation

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


// e.g. of test:
// echo "eps 0" > a.txt; fstmakedisambigsyms 5 a.txt

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Prints out the next N symbols after the last symbol defined in symtab.txt\n"
        "\n"
        "Usage:  fstmakedisambigsyms N symtab.txt [out.list] \n";

    // no options.
    // bool binary = false;
    ParseOptions po(usage);
    // po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    size_t N;
    if (!kaldi::ConvertStringToInteger(po.GetArg(1), &N)) {
      std::cerr << "Invalid first argument to fstmakedisambigsyms: expect a number.\n";
      return 1;
    }

    std::string sym_filename = po.GetArg(2);

    SymbolTable *symtab = SymbolTable::ReadText(sym_filename, false);
    if (!symtab) {
      std::cerr << "fstmakedisambigsyms: could not read symbol table from "<<sym_filename << '\n';
      return 1;
    }
    size_t next_avail = symtab->AvailableKey();
    std::vector<int32> syms;
    for (size_t s = next_avail; s < next_avail + N; s++ ) syms.push_back(s);

    std::string out_list_filename;
    out_list_filename = po.GetOptArg(3);
    if (out_list_filename == "-") out_list_filename = "";

    if (!WriteIntegerVectorSimple(out_list_filename, syms))
      std::cerr << "fstpredeterminize: could not write disambig symbols to "<<
          (out_list_filename == "" ? "stdout" : out_list_filename.c_str())
                << '\n';

    delete symtab;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}

