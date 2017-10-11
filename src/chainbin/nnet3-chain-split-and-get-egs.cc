// chainbin/nnet3-chain-get-egs.cc

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

namespace kaldi {
namespace nnet3 {


/**
   This function does all the processing for one utterance, and outputs the
   supervision objects to 'example_writer'.  Note: if normalization_fst is the
   empty FST (with no states), it skips the final stage of egs preparation and
   you should do it later with nnet3-chain-normalize-egs.
*/

static bool ProcessFile(const chain::SupervisionOptions &sup_opts,
                        const fst::StdVectorFst &normalization_fst,
                        const GeneralMatrix &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        int32 ivector_period,
                        const TransitionModel &trans_model,
                        const chain::SupervisionLatticeSplitter &sup_lat_splitter,
                        const VectorBase<BaseFloat> *deriv_weights,
                        int32 supervision_length_tolerance,
                        const std::string &utt_id,
                        bool compress,
                        UtteranceSplitter *utt_splitter,
                        NnetChainExampleWriter *example_writer) {

  int32 num_input_frames = feats.NumRows();

  std::vector<int32> state_times;

  int32 frame_subsampling_factor = utt_splitter->Config().frame_subsampling_factor;
  int32 num_output_frames = sup_lat_splitter.NumFrames();

  if (deriv_weights && (std::abs(deriv_weights->Dim() - num_output_frames)
                        > supervision_length_tolerance)) {
    KALDI_WARN << "For utterance " << utt_id
               << ", mismatch between deriv-weights dim and num-output-frames"
               << "; " << deriv_weights->Dim() << " vs " << num_output_frames;
    return false;
  }

  if (!utt_splitter->LengthsMatch(utt_id, num_input_frames, num_output_frames,
                                  supervision_length_tolerance))
    return false;  // LengthsMatch() will have printed a warning.

  std::vector<ChunkTimeInfo> chunks;

  utt_splitter->GetChunksForUtterance(num_input_frames, &chunks);

  if (chunks.empty()) {
    KALDI_WARN << "Not producing egs for utterance " << utt_id
               << " because it is too short: "
               << num_input_frames << " frames.";
    return false;
  }

  for (size_t c = 0; c < chunks.size(); c++) {
    ChunkTimeInfo &chunk = chunks[c];

    int32 start_frame_subsampled = chunk.first_frame / frame_subsampling_factor,
        num_frames_subsampled = chunk.num_frames / frame_subsampling_factor;


    chain::Supervision supervision_part;
    sup_lat_splitter.GetFrameRangeSupervision(start_frame_subsampled,
                                              num_frames_subsampled,
                                              &supervision_part);

    if (normalization_fst.NumStates() > 0 &&
        !chain::AddWeightToSupervisionFst(normalization_fst,
                                          &supervision_part)) {
      KALDI_WARN << "For utterance " << utt_id << ", feature frames "
                 << chunk.first_frame << " to "
                 << (chunk.first_frame + chunk.num_frames)
                 << ", FST was empty after composing with normalization FST. "
                 << "This should be extremely rare (a few per corpus, at most)";
      return false;
    }

    int32 first_frame = 0;  // we shift the time-indexes of all these parts so
                            // that the supervised part starts from frame 0.
    
    NnetChainExample nnet_chain_eg;
    nnet_chain_eg.outputs.resize(1);

    SubVector<BaseFloat> output_weights(
        &(chunk.output_weights[0]),
        static_cast<int32>(chunk.output_weights.size()));

    if (!deriv_weights) {
      NnetChainSupervision nnet_supervision("output", supervision_part,
                                            output_weights,
                                            first_frame,
                                            frame_subsampling_factor);
      nnet_chain_eg.outputs[0].Swap(&nnet_supervision);
    } else {
      Vector<BaseFloat> this_deriv_weights(num_frames_subsampled);
      for (int32 i = 0; i < num_frames_subsampled; i++) {
        int32 t = i + start_frame_subsampled;
        if (t < deriv_weights->Dim())
          this_deriv_weights(i) = (*deriv_weights)(t);
      }
      KALDI_ASSERT(output_weights.Dim() == num_frames_subsampled);
      this_deriv_weights.MulElements(output_weights);
      NnetChainSupervision nnet_supervision("output", supervision_part,
                                            this_deriv_weights,
                                            first_frame,
                                            frame_subsampling_factor);
      nnet_chain_eg.outputs[0].Swap(&nnet_supervision);
    }

    nnet_chain_eg.inputs.resize(ivector_feats != NULL ? 2 : 1);

    int32 tot_input_frames = chunk.left_context + chunk.num_frames +
        chunk.right_context,
        start_frame = chunk.first_frame - chunk.left_context;

    GeneralMatrix input_frames;
    ExtractRowRangeWithPadding(feats, start_frame, tot_input_frames,
                               &input_frames);

    NnetIo input_io("input", -chunk.left_context, input_frames);
    nnet_chain_eg.inputs[0].Swap(&input_io);

    if (ivector_feats != NULL) {
      // if applicable, add the iVector feature.
      // choose iVector from a random frame in the chunk
      int32 ivector_frame = RandInt(start_frame,
                                    start_frame + num_input_frames - 1),
          ivector_frame_subsampled = ivector_frame / ivector_period;
      if (ivector_frame_subsampled < 0)
        ivector_frame_subsampled = 0;
      if (ivector_frame_subsampled >= ivector_feats->NumRows())
        ivector_frame_subsampled = ivector_feats->NumRows() - 1;
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(ivector_frame_subsampled));
      NnetIo ivector_io("ivector", 0, ivector);
      nnet_chain_eg.inputs[1].Swap(&ivector_io);
    }

    if (compress)
      nnet_chain_eg.Compress();

    std::ostringstream os;
    os << utt_id << "-" << chunk.first_frame;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    example_writer->Write(key, nnet_chain_eg);
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
        "Get frame-by-frame examples of data for nnet3+chain neural network\n"
        "training.  This involves breaking up utterances into pieces of a\n"
        "fixed size.  Input will come from non-compact phone lattice.\n"
        "Note: if <normalization-fst> is not supplied the egs will not be\n"
        "ready for training; in that case they should later be processed\n"
        "with nnet3-chain-normalize-egs\n"
        "\n"
        "Usage:  nnet3-chain-split-and-get-egs [options] [<normalization-fst>] <features-rspecifier> <tree> <transition-model> "
        "<phone-lattice-rspecifier> <egs-wspecifier>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "lattice-copy --write-compact=false ark:1.lat ark:- | \\ \n"
        "  nnet3-chain-split-and-get-egs --left-context=25 --right-context=9 --num-frames=20 dir/normalization.fst \\\n"
        "  \"$feats\" dir/tree dir/0.trans_mdl ark,s,cs:- ark:cegs.1.ark\n";

    bool compress = true;
    int32 length_tolerance = 100, online_ivector_period = 1,
          supervision_length_tolerance = 1;

    ExampleGenerationConfig eg_config;  // controls num-frames,
                                        // left/right-context, etc.
    chain::SupervisionOptions sup_opts;

    int32 srand_seed = 0;
    std::string online_ivector_rspecifier, deriv_weights_rspecifier;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs with input features "
                "in compressed format (recommended).  Update: this is now "
                "only relevant if the features being read are un-compressed; "
                "if already compressed, we keep we same compressed format when "
                "dumping-egs.");
    po.Register("ivectors", &online_ivector_rspecifier, "Alias for "
                "--online-ivectors option, for back compatibility");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier of "
                "ivector features, as a matrix.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of "
                "frames between iVectors in matrices supplied to the "
                "--online-ivectors option");
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("length-tolerance", &length_tolerance, "Tolerance for "
                "difference in num-frames between feat and ivector matrices");
    po.Register("supervision-length-tolerance", &supervision_length_tolerance, "Tolerance for "
                "difference in num-frames-subsampled between supervision and deriv weights");
    po.Register("deriv-weights-rspecifier", &deriv_weights_rspecifier,
                "Per-frame weights (only binary - 0 or 1) that specifies "
                "whether a frame's gradient must be backpropagated or not. "
                "Not specifying this is equivalent to specifying a vector of "
                "all 1s.");

    eg_config.Register(&po);

    ParseOptions supervision_opts("supervision", &po);
    sup_opts.Register(&supervision_opts);

    chain::SupervisionLatticeSplitterOptions sup_lat_splitter_opts;
    sup_lat_splitter_opts.Register(&po);

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 5 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        normalization_fst_rxfilename,
        feature_rspecifier,
        tree_rxfilename, trans_model_rxfilename,
        lattice_rspecifier,
        examples_wspecifier;
    if (po.NumArgs() == 5) {
      feature_rspecifier = po.GetArg(1);
      tree_rxfilename = po.GetArg(2);
      trans_model_rxfilename = po.GetArg(3);
      lattice_rspecifier = po.GetArg(4);
      examples_wspecifier = po.GetArg(5);
    } else {
      normalization_fst_rxfilename = po.GetArg(1);
      KALDI_ASSERT(!normalization_fst_rxfilename.empty());
      feature_rspecifier = po.GetArg(2);
      tree_rxfilename = po.GetArg(3);
      trans_model_rxfilename = po.GetArg(4);
      lattice_rspecifier = po.GetArg(5);
      examples_wspecifier = po.GetArg(6);
    }

    eg_config.ComputeDerived();
    UtteranceSplitter utt_splitter(eg_config);

    fst::StdVectorFst normalization_fst;
    if (!normalization_fst_rxfilename.empty()) {
      ReadFstKaldi(normalization_fst_rxfilename, &normalization_fst);
      KALDI_ASSERT(normalization_fst.NumStates() > 0);
      
      if (sup_opts.lm_scale < 0.0 || sup_opts.lm_scale > 1.0) {
        KALDI_ERR << "Invalid lm-scale; must be in [0.0, 1.0)";
      }

      if (sup_opts.lm_scale != 0.0) {
        ScaleFst(1.0 - sup_opts.lm_scale, &normalization_fst);
      }
    }

    // Read as GeneralMatrix so we don't need to un-compress and re-compress
    // when selecting parts of matrices.
    SequentialGeneralMatrixReader feat_reader(feature_rspecifier);
    
    TransitionModel trans_model;
    ReadKaldiObject(trans_model_rxfilename, &trans_model);

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    RandomAccessLatticeReader lattice_reader(
        lattice_rspecifier);
    NnetChainExampleWriter example_writer(examples_wspecifier);
    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);
    RandomAccessBaseFloatVectorReader deriv_weights_reader(
        deriv_weights_rspecifier);

    int32 num_err = 0;

    KALDI_ASSERT(sup_lat_splitter_opts.add_tolerance_to_lat &&
                 sup_opts.frame_subsampling_factor == 1);
        
    chain::SupervisionLatticeSplitter sup_lat_splitter(
        sup_lat_splitter_opts, sup_opts, trans_model);

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const GeneralMatrix &feats = feat_reader.Value();
      if (!lattice_reader.HasKey(key)) {
        KALDI_WARN << "No lattice for key " << key;
        num_err++;
      } else {
        const Lattice &lat = lattice_reader.Value(key);
        const Matrix<BaseFloat> *online_ivector_feats = NULL;
        if (!online_ivector_rspecifier.empty()) {
          if (!online_ivector_reader.HasKey(key)) {
            KALDI_WARN << "No iVectors for utterance " << key;
            num_err++;
            continue;
          } else {
            // this address will be valid until we call HasKey() or Value()
            // again.
            online_ivector_feats = &(online_ivector_reader.Value(key));
          }
        }
        if (online_ivector_feats != NULL &&
            (abs(feats.NumRows() - (online_ivector_feats->NumRows() *
                                    online_ivector_period)) > length_tolerance
             || online_ivector_feats->NumRows() == 0)) {
          KALDI_WARN << "Length difference between feats " << feats.NumRows()
                     << " and iVectors " << online_ivector_feats->NumRows()
                     << "exceeds tolerance " << length_tolerance;
          num_err++;
          continue;
        }
        
        const Vector<BaseFloat> *deriv_weights = NULL;
        if (!deriv_weights_rspecifier.empty()) {
          if (!deriv_weights_reader.HasKey(key)) {
            KALDI_WARN << "No deriv weights for utterance " << key;
            num_err++;
            continue;
          } else {
            // this address will be valid until we call HasKey() or Value()
            // again.
            deriv_weights = &(deriv_weights_reader.Value(key));
          }
        }

        sup_lat_splitter.LoadLattice(lat);

        if (!ProcessFile(sup_opts, normalization_fst, feats,
                         online_ivector_feats, online_ivector_period,
                         trans_model, sup_lat_splitter,
                         deriv_weights, supervision_length_tolerance,
                         key, compress,
                         &utt_splitter, &example_writer))
          num_err++;
      }
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
