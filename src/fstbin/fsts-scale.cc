// fstbin/fsts-scale.cc

// Copyright 2016  Johns Hopkins University (Authors: Jan "Yenda" Trmal)

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
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

void ScaleFst(fst::VectorFst<fst::StdArc> *fst, double alpha, double beta) {
  typedef typename fst::StdArc::StateId StateId;
  using fst::MutableArcIterator;
  using fst::VectorFst;
  using fst::StdArc;

  StateId num_states = fst->NumStates();
  for (StateId s = 0; s < num_states; s++) {
    for (MutableArcIterator<VectorFst<StdArc> > aiter(fst, s);
         !aiter.Done();
         aiter.Next()) {
      StdArc arc = aiter.Value();
      arc.weight = arc.weight.Value() * alpha + beta;
      aiter.SetValue(arc);
    }
    StdArc::Weight final_weight = fst->Final(s);
    final_weight = final_weight.Value() * alpha + beta;
    fst->SetFinal(s, final_weight);
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    double alpha = 1.0;
    double beta = 0.0;

    const char *usage =
      "Scales the FST scores using new_score = alpha * old_score + beta\n"
      "where alpha and beta can be set as command line parameters. Typically\n"
      "one would set beta!=0 and alpha=0 for logarithmic weights (tropical\n"
      "semiring, for example) and alpha!=0 and beta=0 for probabilistic\n"
      "weightsa\n"
      "\n"
      "Usage: fsts-scale --alpha=1 --beta=0 (fst-rxfilename|fst-rspecifier) "
      " [(out-rxfilename|out-rspecifier)]";

    ParseOptions po(usage);
    po.Register("alpha", &alpha, "The alpha (multiplikative) coefficient");
    po.Register("beta", &beta, "The beta (additive) coefficient");

    po.Read(argc, argv);
    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      KALDI_WARN << po.NumArgs();
      po.PrintUsage();
      exit(1);
    }

    std::string fst_in_str = po.GetArg(1),
        fst_out_str = po.GetArg(2);

    bool is_table_1 =
        (ClassifyRspecifier(fst_in_str, NULL, NULL) != kNoRspecifier),
        is_table_out =
        (ClassifyWspecifier(fst_out_str, NULL, NULL, NULL) != kNoWspecifier);

    if (is_table_out != is_table_1)
      KALDI_ERR << "Incompatible combination of archives and files";

    if (!is_table_1 && !is_table_out) {  // Only dealing with files...
      VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_str);
      ScaleFst(fst, alpha, beta);
      WriteFstKaldi(*fst, fst_out_str);
      return 0;
    } else {
      // is_table_1 && is_table_out
      SequentialTableReader<VectorFstHolder> fst_reader(fst_in_str);
      TableWriter<VectorFstHolder> fst_writer(fst_out_str);

      int32 n_done = 0;
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string key = fst_reader.Key();
        VectorFst<StdArc> fst(fst_reader.Value());

        ScaleFst(&fst, alpha, beta);

        fst_writer.Write(key, fst);
        n_done++;
      }
      KALDI_LOG << "Successfully scaled " << n_done << " FSTs";
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

