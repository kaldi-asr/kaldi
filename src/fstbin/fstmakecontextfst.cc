// fstbin/fstmakecontextfst.cc

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
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"
#include "fstext/context-fst.h"

/* for example of testing setup, see fstmakecontextsymbols.cc */

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Constructs a context FST with a specified context-width and context-position.  Outputs\n"
        " the context FST, and a file in Kaldi format that describes what the input labels mean.\n"
        "\n"
        "Usage:  fstmakecontextfst phones_symtab subseq_sym ilabels_output_file [out.fst]\n"
        "E.g.:   fstmakecontextfst phones.txt 42 ilabels.sym > C.fst\n";

    bool binary = true;  // binary output to ilabels_output_file.
    std::string disambig_list_infile;
    std::string disambig_list_outfile;
    int32 N = 3;
    int32 P = 1;

    OptimizeConfig cfg;
    ParseOptions po(usage);
    po.Register("read-disambig-syms", &disambig_list_infile, "List of disambiguation symbols to read");
    po.Register("write-disambig-syms", &disambig_list_outfile, "List of disambiguation symbols to write");
    po.Register("context-size", &N, "Size of phonetic context window");
    po.Register("central-position", &P, "Designated central position in context window");
    po.Register("binary", &binary, "Write ilabels output file in binary Kaldi format");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string phones_symtab_filename = po.GetArg(1);
    int32 subseq_sym;
    if (!ConvertStringToInteger(po.GetArg(2), &subseq_sym))
      KALDI_EXIT << "Invalid subsequential symbol " << po.GetArg(2);
    std::string ilabels_out_filename = po.GetArg(3);
    std::string fst_out_filename = po.GetOptArg(4);


    std::vector<kaldi::int32> phone_syms;
    {
      fst::SymbolTable *phones_symtab = NULL;
      {  // read phone symbol table.
        std::ifstream is(phones_symtab_filename.c_str());
        phones_symtab = fst::SymbolTable::ReadText(is, phones_symtab_filename);
        if (!phones_symtab) KALDI_EXIT << "Could not read phones symbol-table file "<<phones_symtab_filename;
      }
      GetSymbols(*phones_symtab,
                 false,  // don't include eps,
                 &phone_syms);
      delete phones_symtab;
    }

    if ( (disambig_list_outfile != "") && (disambig_list_infile == "") ) {
      std::cerr << "fstmakecontextfst: cannot specify --write-disambig-syms if "
          "not specifying --read-disambig-syms\n";
    }

    std::vector<int32> disambig_in;
    if (disambig_list_infile != "") {
      if (disambig_list_infile == "-") disambig_list_infile = "";
      if (!ReadIntegerVectorSimple(disambig_list_infile, &disambig_in)) {
        std::cerr << "fstcomposecontext: Could not read disambiguation symbols from "
                  << (disambig_list_infile == "" ? "standard input" : disambig_list_infile)
                  << '\n';
        return 1;
      }
    }

    if (std::binary_search(phone_syms.begin(), phone_syms.end(), subseq_sym)
       ||std::binary_search(disambig_in.begin(), disambig_in.end(), subseq_sym))
      KALDI_EXIT << "Invalid subsequential symbol "<<(subseq_sym)<<", already a phone or disambiguation symbol.";


    ContextFst<StdArc, int32> cfst(subseq_sym,
                                   phone_syms,
                                   disambig_in,
                                   N,
                                   P);

    VectorFst<StdArc> vfst(cfst);  // Copy the fst to a VectorFst.

    if (! vfst.Write(fst_out_filename) )
      KALDI_EXIT << "fstmakecontextfst: error writing the output to "<<fst_out_filename;

    const std::vector<std::vector<int32> >  &ilabels = cfst.ILabelInfo();
    WriteILabelInfo(Output(ilabels_out_filename, binary).Stream(),
                    binary, ilabels);

    if (disambig_list_outfile != "") {
      std::vector<int32> disambig_out;
      for (size_t i = 0; i < ilabels.size(); i++)
        if (ilabels[i].size() == 1 && ilabels[i][0] <= 0)
          disambig_out.push_back(static_cast<int32>(i));
      if (!WriteIntegerVectorSimple(disambig_list_outfile, disambig_out)) {
        std::cerr << "fstcomposecontext: Could not write disambiguation symbols to "
                  << (disambig_list_outfile == "" ? "standard input" : disambig_list_outfile)
                  << '\n';
        return 1;
      }
    }
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}

