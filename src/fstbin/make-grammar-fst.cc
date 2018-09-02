// fstbin/make-grammar-fst.cc

// Copyright      2018  Johns Hopkins University (author: Daniel Povey)

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
#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/kaldi-fst-io.h"
#include "decoder/grammar-fst.h"

namespace fst {

// Reads an FST from disk using Kaldi I/O mechanisms, and if it is not of type
// ConstFst, copies it to that stype.
ConstFst<StdArc>* ReadAsConstFst(std::string rxfilename) {
  // the following call will throw if there is an error.
  Fst<StdArc> *fst = ReadFstKaldiGeneric(rxfilename);
  ConstFst<StdArc> *const_fst = dynamic_cast<ConstFst<StdArc>* >(fst);
  if (!const_fst) {
    const_fst = new ConstFst<StdArc>(*fst);
    delete fst;
  }
  return const_fst;
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Construct GrammarFst and write it to disk (or convert it to ConstFst\n"
        "and write that to disk instead).  Mostly intended for demonstration\n"
        "and testing purposes (since it may be more convenient to construct\n"
        "GrammarFst from code).  See kaldi-asr.org/doc/grammar.html\n"
        "Can also be used to prepares FSTs for this use, by calling\n"
        "PrepareForGrammarFst(), which does things like adding final-probs and\n"
        "making small structural tweaks to the FST\n"
        "\n"
        "Usage (1): make-grammar-fst [options] <top-level-fst> <symbol1> <fst1> \\\n"
        "                            [<symbol2> <fst2> ...]] <fst-out>\n"
        "\n"
        "<symbol1>, <symbol2> are the integer ids of the corresponding\n"
        " user-defined nonterminal symbols (e.g. #nonterm:contact_list) in the\n"
        " phones.txt file.\n"
        "e.g.: make-grammar-fst --nonterm-phones-offset=317 HCLG.fst \\\n"
        "            320 HCLG1.fst HCLG_grammar.fst\n"
        "\n"
        "Usage (2): make-grammar-fst <fst-in> <fst-out>\n"
        "  Prepare individual FST for compilation into GrammarFst.\n"
        "  E.g. make-grammar-fst HCLG.fst HCLGmod.fst.  The outputs of this\n"
        "   will then become the arguments <top-level-fst>, <fst1>, ... for usage\n"
        "   pattern (1).\n"
        "\n"
        "The --nonterm-phones-offset option is required for both usage patterns.\n";


    ParseOptions po(usage);


    int32 nonterm_phones_offset = -1;
    bool write_as_grammar = true;

    po.Register("nonterm-phones-offset", &nonterm_phones_offset,
                "Integer id of #nonterm_bos in phones.txt");
    po.Register("write-as-grammar", &write_as_grammar, "If true, "
                "write as GrammarFst object; if false, convert to "
                "ConstFst<StdArc> (readable by standard decoders) "
                "and write that.");

    po.Read(argc, argv);


    if (po.NumArgs() < 2 || po.NumArgs() % 2 != 0) {
      po.PrintUsage();
      exit(1);
    }

    if (nonterm_phones_offset < 0)
      KALDI_ERR << "The --nonterm-phones-offset option must be supplied "
          "and positive.";

    if (po.NumArgs() == 2) {
      // this usage pattern calls PrepareForGrammarFst().
      VectorFst<StdArc> *fst = ReadFstKaldi(po.GetArg(1));
      PrepareForGrammarFst(nonterm_phones_offset, fst);
      // This will write it as VectorFst; to avoid it having to be converted to
      // ConstFst when read again by make-grammar-fst, you may want to pipe
      // through fstconvert --fst_type=const.
      WriteFstKaldi(*fst, po.GetArg(2));
      exit(0);
    }

    std::string top_fst_str = po.GetArg(1),
        fst_out_str = po.GetArg(po.NumArgs());

    ConstFst<StdArc> *top_fst = ReadAsConstFst(top_fst_str);
    std::vector<std::pair<int32, const ConstFst<StdArc>* > > pairs;

    int32 num_pairs = (po.NumArgs() - 2) / 2;
    for (int32 i = 1; i <= num_pairs; i++) {
      int32 nonterminal;
      std::string nonterm_str = po.GetArg(2*i);
      if (!ConvertStringToInteger(nonterm_str, &nonterminal) ||
          nonterminal <= 0)
        KALDI_ERR << "Expected positive integer as nonterminal, got: "
                  << nonterm_str;
      std::string fst_str = po.GetArg(2*i + 1);
      ConstFst<StdArc> *fst = ReadAsConstFst(fst_str);
      pairs.push_back(std::pair<int32, const ConstFst<StdArc>* >(nonterminal, fst));
    }

    GrammarFst *grammar_fst = new GrammarFst(nonterm_phones_offset,
                                             *top_fst,
                                             pairs);

    if (write_as_grammar) {
      bool binary = true;  // GrammarFst does not support non-binary write.
      WriteKaldiObject(*grammar_fst, fst_out_str, binary);
      delete grammar_fst;
    } else {
      VectorFst<StdArc> vfst;
      CopyToVectorFst(grammar_fst, &vfst);
      delete grammar_fst;
      ConstFst<StdArc> cfst(vfst);
      // We don't have a wrapper in kaldi-fst-io.h for writing type
      // ConstFst<StdArc>, so do it manually.
      bool binary = true, write_binary_header = false;  // suppress the ^@B
      Output ko(fst_out_str, binary, write_binary_header);
      FstWriteOptions wopts(kaldi::PrintableWxfilename(fst_out_str));
      cfst.Write(ko.Stream(), wopts);
    }

    delete top_fst;
    for (size_t i = 0; i < pairs.size(); i++)
      delete pairs[i].second;

    KALDI_LOG << "Created grammar FST and wrote it to "
              << fst_out_str;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
