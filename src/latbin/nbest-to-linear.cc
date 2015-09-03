// latbin/nbest-to-linear.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Takes as input lattices/n-bests which must be linear (single path);\n"
        "convert from lattice to up to 4 archives containing transcriptions, alignments,\n"
        "and acoustic and LM costs (note: use ark:/dev/null for unwanted outputs)\n"
        "Usage: nbest-to-linear [options] <nbest-rspecifier> <alignments-wspecifier> "
        "[<transcriptions-wspecifier> [<lm-cost-wspecifier> [<ac-cost-wspecifier>]]]\n"
        " e.g.: lattice-to-nbest --n=10 ark:1.lats ark:- | \\\n"
        "   nbest-to-linear ark:1.lats ark,t:1.ali ark,t:1.tra\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        ali_wspecifier = po.GetArg(2),
        trans_wspecifier = po.GetOptArg(3),
        lm_cost_wspecifier = po.GetOptArg(4),
        ac_cost_wspecifier = po.GetOptArg(5);

    SequentialLatticeReader lattice_reader(lats_rspecifier);

    Int32VectorWriter ali_writer(ali_wspecifier);
    Int32VectorWriter trans_writer(trans_wspecifier);
    BaseFloatWriter lm_cost_writer(lm_cost_wspecifier);
    BaseFloatWriter ac_cost_writer(ac_cost_wspecifier);
    
    int32 n_done = 0, n_err = 0;
    
    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      Lattice lat = lattice_reader.Value();

      vector<int32> ilabels;
      vector<int32> olabels;
      LatticeWeight weight;
      
      if (!GetLinearSymbolSequence(lat, &ilabels, &olabels, &weight)) {
        KALDI_WARN << "Lattice/nbest for key " << key << " had wrong format: "
            "note, this program expects input with one path, e.g. from "
            "lattice-to-nbest.";
        n_err++;
      } else {
        if (ali_wspecifier != "") ali_writer.Write(key, ilabels);
        if (trans_wspecifier != "") trans_writer.Write(key, olabels);
        if (lm_cost_wspecifier != "") lm_cost_writer.Write(key, weight.Value1());
        if (ac_cost_wspecifier!= "") ac_cost_writer.Write(key, weight.Value2());
        n_done++;
      }
    }
    KALDI_LOG << "Done " << n_done << " n-best entries, "
              << n_err  << " had errors.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
