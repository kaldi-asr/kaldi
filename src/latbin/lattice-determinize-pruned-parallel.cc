// latbin/lattice-determinize-pruned-parallel.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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
#include "lat/kaldi-lattice.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/lattice-functions.h"
#include "lat/push-lattice.h"
#include "lat/minimize-lattice.h"
#include "util/kaldi-thread.h"

namespace kaldi {

class DeterminizeLatticeTask {
 public:
  // Initializer takes ownership of "lat".
  DeterminizeLatticeTask(
      fst::DeterminizeLatticePrunedOptions &opts,
      std::string key,
      BaseFloat acoustic_scale,
      BaseFloat beam,
      bool minimize,
      Lattice *lat,
      CompactLatticeWriter *clat_writer,
      int32 *num_warn):
      opts_(opts), key_(key), acoustic_scale_(acoustic_scale), beam_(beam),
      minimize_(minimize), lat_(lat), clat_writer_(clat_writer),
      num_warn_(num_warn) { }

  void operator () () {
    Invert(lat_); // to get word labels on the input side.
    // We apply the acoustic scale before determinization and will undo it
    // afterward, since it can affect the result.
    fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale_), lat_);
    if (!TopSort(lat_)) {
      KALDI_WARN << "Could not topologically sort lattice: this probably means it"
          " has bad properties e.g. epsilon cycles.  Your LM or lexicon might "
          "be broken, e.g. LM with epsilon cycles or lexicon with empty words.";
      (*num_warn_)++;
    }
    fst::ArcSort(lat_, fst::ILabelCompare<LatticeArc>());
    if (!DeterminizeLatticePruned(*lat_, beam_, &det_clat_, opts_)) {
      KALDI_WARN << "For key " << key_ << ", determinization did not succeed"
          "(partial output will be pruned tighter than the specified beam.)";
      (*num_warn_)++;
    }
    delete lat_; // This is no longer needed so we can delete it now;
    lat_ = NULL;
    if (minimize_) {
      PushCompactLatticeStrings(&det_clat_);
      PushCompactLatticeWeights(&det_clat_);
      MinimizeCompactLattice(&det_clat_);
    }
    // Invert the original acoustic scaling
    fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale_),
                      &det_clat_);
  }
  ~DeterminizeLatticeTask() {
    KALDI_VLOG(2) << "Wrote lattice with " << det_clat_.NumStates()
                  << " for key " << key_;
    clat_writer_->Write(key_, det_clat_);
  }
 private:
  const fst::DeterminizeLatticePrunedOptions &opts_;
  std::string key_;
  BaseFloat acoustic_scale_;
  BaseFloat beam_;
  bool minimize_;
  Lattice *lat_; // The lattice we're working on.  Owned locally.
  CompactLattice det_clat_; // The output of our process.  Will be written
  // to clat_writer_ in the destructor.
  CompactLatticeWriter *clat_writer_;
  int32 *num_warn_;

};

} // namespace kaldi


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    
    const char *usage =
        "Determinize lattices, keeping only the best path (sequence of acoustic states)\n"
        "for each input-symbol sequence.  This is a version of lattice-determnize-pruned\n"
        "that accepts the --num-threads option.  These programs do pruning as part of the\n"
        "determinization algorithm, which is more efficient and prevents blowup.\n"
        "See http://kaldi-asr.org/doc/lattices.html for more information on lattices.\n"
        "\n"
        "Usage: lattice-determinize-pruned-parallel [options] lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-determinize-pruned-parallel --acoustic-scale=0.1 --beam=6.0 ark:in.lats ark:det.lats\n";
    
    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat beam = 10.0;
    bool minimize = false;
    TaskSequencerConfig sequencer_config; // has --num-threads option
    fst::DeterminizeLatticePrunedOptions determinize_config; // Options used in DeterminizeLatticePruned--
    // this options class does not have its own Register function as it's viewed as
    // being more part of "fst world", so we register its elements independently.
    determinize_config.max_mem = 50000000;
    determinize_config.max_loop = 0; // was 500000;
    
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("beam", &beam, "Pruning beam [applied after acoustic scaling].");
    po.Register("minimize", &minimize,
                "If true, push and minimize after determinization");
    determinize_config.Register(&po);
    sequencer_config.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);


    // Read as regular lattice-- this is the form the determinization code
    // accepts.
    SequentialLatticeReader lat_reader(lats_rspecifier);
    
    // Write as compact lattice.
    CompactLatticeWriter compact_lat_writer(lats_wspecifier); 
    TaskSequencer<DeterminizeLatticeTask> sequencer(sequencer_config);
    
    int32 n_done = 0, n_warn = 0;

    if (acoustic_scale == 0.0)
      KALDI_ERR << "Do not use a zero acoustic scale (cannot be inverted)";

    for (; !lat_reader.Done(); lat_reader.Next()) {
      std::string key = lat_reader.Key();

      Lattice *lat = lat_reader.Value().Copy(); // will give ownership to "task"
                                                // below
      
      KALDI_VLOG(2) << "Processing lattice " << key;

      DeterminizeLatticeTask *task = new DeterminizeLatticeTask(
          determinize_config, key, acoustic_scale, beam, minimize,
          lat, &compact_lat_writer, &n_warn);
      sequencer.Run(task);
      n_done++;
    }
    sequencer.Wait();
    KALDI_LOG << "Done " << n_done << " lattices, had warnings on " << n_warn
              << " of these.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
