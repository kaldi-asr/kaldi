// chainbin/chain-split-lattices.cc

// Copyright      2015  Johns Hopkins University (author:  Daniel Povey)

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

#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "chain/chain-supervision-splitter.h"
#include "lat/lattice-functions.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-example-utils.h"
#include "fstext/kaldi-fst-io.h"

namespace kaldi {
namespace nnet3 {


/**
   This function does all the processing for one utterance, and outputs the
   supervision objects to 'example_writer'.  Note: if normalization_fst is the
   empty FST (with no states), it skips the final stage of egs preparation and
   you should do it later with nnet3-chain-normalize-egs.
*/

static bool ProcessFile(const chain::SupervisionLatticeSplitter &sup_lat_splitter,
                        const std::string &utt_id,
                        UtteranceSplitter *utt_splitter,
                        TableWriter<fst::VectorFstHolder> *fst_writer,
                        LatticeWriter *lat_writer) {
  std::vector<int32> state_times;

  int32 frame_subsampling_factor = utt_splitter->Config().frame_subsampling_factor;
  int32 num_frames = sup_lat_splitter.NumFrames() * frame_subsampling_factor;
  
  std::vector<ChunkTimeInfo> chunks;

  utt_splitter->GetChunksForUtterance(num_frames, &chunks);

  if (chunks.empty()) {
    KALDI_WARN << "Not producing egs for utterance " << utt_id
               << " because it is too short: "
               << num_frames << " frames.";
    return false;
  }

  for (size_t c = 0; c < chunks.size(); c++) {
    ChunkTimeInfo &chunk = chunks[c];

    int32 start_frame_subsampled = chunk.first_frame / frame_subsampling_factor,
        num_frames_subsampled = chunk.num_frames / frame_subsampling_factor;

    Lattice lat_part;
    chain::Supervision supervision_part;
    sup_lat_splitter.GetFrameRangeSupervision(start_frame_subsampled,
                                              num_frames_subsampled,
                                              &supervision_part,
                                              &lat_part);

    std::ostringstream oss;
    oss << utt_id << "-" << start_frame_subsampled << "-" << num_frames_subsampled;
    std::string key = oss.str();
    
    fst_writer->Write(key, supervision_part.fst);
    lat_writer->Write(key, lat_part);
  }
  return true;
}

} // namespace nnet3 
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Split lattices to chain supervision FSTs\n"
        "\n"
        "Usage:  chain-split-lattices [options] <transition-model> "
        "<phone-lattice-rspecifier> <fst-wspecifier> [<phone-lattice-wspecifier>]\n";

    ExampleGenerationConfig eg_config;  // controls num-frames,
                                        // left/right-context, etc.
    chain::SupervisionOptions sup_opts;

    int32 srand_seed = 0;

    ParseOptions po(usage);
    po.Register("srand", &srand_seed, "Seed for random number generator ");

    eg_config.Register(&po);

    ParseOptions supervision_opts("supervision", &po);
    sup_opts.Register(&supervision_opts);
    
    chain::SupervisionLatticeSplitterOptions sup_lat_splitter_opts;
    sup_lat_splitter_opts.Register(&po);

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        trans_model_rxfilename,
        lattice_rspecifier, fst_wspecifier;
    trans_model_rxfilename = po.GetArg(1);
    lattice_rspecifier = po.GetArg(2);
    fst_wspecifier = po.GetArg(3);

    std::string lattice_wspecifier = po.GetOptArg(4);
    
    eg_config.ComputeDerived();
    UtteranceSplitter utt_splitter(eg_config);

    TransitionModel trans_model;
    ReadKaldiObject(trans_model_rxfilename, &trans_model);

    SequentialLatticeReader lattice_reader(lattice_rspecifier);
    TableWriter<fst::VectorFstHolder> fst_writer(fst_wspecifier);
    LatticeWriter lattice_writer(lattice_wspecifier);

    int32 num_err = 0;
    
    chain::SupervisionLatticeSplitter sup_lat_splitter(
        sup_lat_splitter_opts, sup_opts, trans_model);

    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      const Lattice &lat = lattice_reader.Value();

      sup_lat_splitter.LoadLattice(lat);
      if (!ProcessFile(sup_lat_splitter,
                       key, &utt_splitter, &fst_writer,
                       &lattice_writer))
        num_err++;
    }
    if (num_err > 0)
      KALDI_WARN << num_err << " utterances had errors and could "
          "not be processed.";
    // utt_splitter prints stats in its destructor.
    return utt_splitter.ExitStatus();
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

