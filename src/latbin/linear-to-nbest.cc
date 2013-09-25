// latbin/linear-to-nbest.cc

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

namespace kaldi {
void MakeLatticeFromLinear(const std::vector<int32> &ali,
                           const std::vector<int32> &words,
                           BaseFloat lm_cost,
                           BaseFloat ac_cost,
                           Lattice *lat_out) {
  typedef LatticeArc::StateId StateId;
  typedef LatticeArc::Weight Weight;
  typedef LatticeArc::Label Label;
  lat_out->DeleteStates();
  StateId cur_state = lat_out->AddState(); // will be 0.
  lat_out->SetStart(cur_state);
  for (size_t i = 0; i < ali.size() || i < words.size(); i++) {
    Label ilabel = (i < ali.size()  ? ali[i] : 0);
    Label olabel = (i < words.size()  ? words[i] : 0);
    StateId next_state = lat_out->AddState();
    lat_out->AddArc(cur_state,
                    LatticeArc(ilabel, olabel, Weight::One(), next_state));
    cur_state = next_state;
  }
  lat_out->SetFinal(cur_state, Weight(lm_cost, ac_cost));
}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "This does the opposite of nbest-to-linear.  It takes 4 archives,\n"
        "containing alignments, word-sequences, and acoustic and LM costs,\n"
        "and turns it into a single archive containing FSTs with a linear\n"
        "structure.  The program is called linear-to-nbest because very often\n"
        "the archives concerned will represent N-best lists\n"
        "Usage:  linear-to-nbest [options] <alignments-rspecifier> "
        "<transcriptions-rspecifier> (<lm-cost-rspecifier>|'') (<ac-cost-rspecifier>|'') "
        "<nbest-wspecifier>\n"
        "Note: if the rspecifiers for lm-cost or ac-cost are the empty string,\n"
        "these value will default to zero.\n"
        " e.g.: linear-to-nbest ark:1.ali ark:1.tra ark:1.lmscore ark:1.acscore "
        "ark:1.nbest\n";
    
    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string ali_rspecifier = po.GetArg(1),
        trans_rspecifier = po.GetArg(2),
        lm_cost_rspecifier = po.GetArg(3),
        ac_cost_rspecifier = po.GetArg(4),
        lats_wspecifier = po.GetArg(5); // will probably represent N-best.



    SequentialInt32VectorReader ali_reader(ali_rspecifier);
    RandomAccessInt32VectorReader trans_reader(trans_rspecifier);
    RandomAccessBaseFloatReader lm_cost_reader(lm_cost_rspecifier);
    RandomAccessBaseFloatReader ac_cost_reader(ac_cost_rspecifier);
    
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);
    
    int32 n_done = 0, n_err = 0;
    
    for (; !ali_reader.Done(); ali_reader.Next()) {
      std::string key = ali_reader.Key();
      if (!trans_reader.HasKey(key)) {
        KALDI_ERR << "No transcription for key " << key;
        n_err++;
        continue;
      }
      if (lm_cost_rspecifier != "" && !lm_cost_reader.HasKey(key)) {
        KALDI_ERR << "No LM cost for key " << key;
        n_err++;
        continue;
      }
      if (ac_cost_rspecifier != "" && !ac_cost_reader.HasKey(key)) {
        KALDI_ERR << "No acoustic cost for key " << key;
        n_err++;
        continue;
      }
      const std::vector<int32> &ali = ali_reader.Value();
      const std::vector<int32> &words = trans_reader.Value(key);
      BaseFloat
          ac_cost = (ac_cost_rspecifier == "") ? 0.0 : ac_cost_reader.Value(key),
          lm_cost = (lm_cost_rspecifier == "") ? 0.0 : lm_cost_reader.Value(key);
      Lattice lat;
      MakeLatticeFromLinear(ali, words, lm_cost, ac_cost, &lat);
      CompactLattice clat;
      ConvertLattice(lat, &clat);
      
      compact_lattice_writer.Write(key, clat);
      n_done++;
    }
    KALDI_LOG << "Done " << n_done << " n-best entries ,"
              << n_err  << " had errors.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
