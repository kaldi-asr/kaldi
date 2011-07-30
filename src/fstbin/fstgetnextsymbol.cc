// fstbin/fstgetnextsymbol.cc

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


#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "util/parse-options.h"
#include "fst/fstlib.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Prints out next unused symbol not in supplied symbol table.\n"
        "\n"
        "Usage:  fstgetnextsymbol symtab.txt\n";

    // no options.
    // bool binary = false;
    ParseOptions po(usage);
    // po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string sym_filename = po.GetArg(1);

    SymbolTable *symtab = SymbolTable::ReadText(sym_filename, false);
    if (!symtab) {
      std::cerr << "fstgetnextsymbol: could not read symbol table from "<<sym_filename << '\n';
      return 1;
    }
    size_t ans = symtab->AvailableKey();

    std::cerr << ans << '\n';  // This is the program's output.

    delete symtab;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}

