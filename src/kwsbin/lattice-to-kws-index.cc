// kwsbin/lattice-to-kws-index.cc

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
//                 Lucas Ondel

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
#include "fstext/fstext-utils.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/kaldi-kws.h"
#include "lat/kws-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using fst::VectorFst;
    typedef kaldi::int32 int32;
    typedef kaldi::uint64 uint64;

    const char *usage =
        "Create an inverted index of the given lattices. The output index is in the T*T*T\n"
        "semiring. For details for the semiring, please refer to Dogan Can and Muran Saraclar's"
        "lattice indexing paper."
        "\n"
        "Usage: lattice-to-kws-index [options]  utter-symtab-rspecifier lattice-rspecifier index-wspecifier\n"
        " e.g.: lattice-to-kws-index ark:utter.symtab ark:1.lats ark:global.idx\n";

    ParseOptions po(usage);

    int32 max_silence_frames = 50;
    bool strict = true;
    po.Register("max-silence-frames", &max_silence_frames, "Maximum #frames for silence arc.");
    po.Register("strict", &strict, "Will allow 0 lattice if it is set to false.");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string usymtab_rspecifier = po.GetOptArg(1),
        lats_rspecifier = po.GetArg(2),
        index_wspecifier = po.GetOptArg(3);

    // We use RandomAccessInt32Reader to read the utterance symtab table.
    RandomAccessInt32Reader usymtab_reader(usymtab_rspecifier);

    // We read the lattice in as CompactLattice; We need the CompactLattice
    // structure for the rest of the work
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    TableWriter< fst::VectorFstTplHolder<KwsLexicographicArc> > index_writer(index_wspecifier);

    int32 n_done = 0;
    int32 n_fail = 0;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      clat_reader.FreeCurrent();

      //Check if we have the corresponding utterance id
      if (!usymtab_reader.HasKey(key)) {
        KALDI_WARN << "Cannot find utterance id for " << key;
        n_fail++;
        continue;
      }

      // Topologically sort the lattice, if not already sorted.
      uint64 props = clat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        if (fst::TopSort(&clat) == false) {
          KALDI_WARN << "Cycles detected in lattice " << key;
          n_fail++;
          continue;
        }
      } 

      // Get the alignments
      vector<int32> state_times;
      CompactLatticeStateTimes(clat, &state_times);

      // Cluster the arcs in the CompactLattice, write the cluster_id on the
      // output label side.
      // ClusterLattice() corresponds to the second part of the preprocessing in
      // Dogan and Murat's paper -- clustering. Note that we do the first part
      // of preprocessing (the weight pushing step) later when generating the
      // factor transducer.
      bool success = false;
      success = ClusterLattice(&clat, state_times);
      if (!success) {
        KALDI_WARN << "State id's and alignments do not match for lattice " << key;
        n_fail++;
        continue;
      }

      // Generate factor transducer
      // CreateFactorTransducer() corresponds to the "Factor Generation" part of
      // Dogan and Murat's paper. But we also move the weight pushing step to
      // this function as we have to compute the alphas and betas anyway.
      KwsProductFst factor_transducer;
      int32 utterance_id = usymtab_reader.Value(key);
      success = CreateFactorTransducer(clat, state_times, utterance_id, &factor_transducer);
      if (!success) {
        KALDI_WARN << "Cannot generate factor transducer for lattice " << key;
        n_fail++; 
      }

      // Remove long silence arc
      // We add the filtering step in our implementation. This is because gap
      // between two successive words in a query term should be less than 0.5s
      RemoveLongSilences(max_silence_frames, state_times, &factor_transducer);

      // Do factor merging, and return a transducer in T*T*T semiring. This step
      // corresponds to the "Factor Merging" part in Dogan and Murat's paper.
      KwsLexicographicFst index_transducer;
      DoFactorMerging(factor_transducer, &index_transducer);

      // Do factor disambiguation. It corresponds to the "Factor Disambiguation"
      // step in Dogan and Murat's paper.
      DoFactorDisambiguation(&index_transducer);

      // Optimize the above factor transducer. It corresponds to the
      // "Optimization" step in the paper.
      OptimizeFactorTransducer(&index_transducer);

      // Write result
      index_writer.Write(key, index_transducer);  

      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " lattices, failed for " << n_fail;
    if (strict == true)
      return (n_done != 0 ? 0 : 1);
    else
      return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
