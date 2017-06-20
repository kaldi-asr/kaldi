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

static bool ProcessFile(const fst::StdVectorFst &normalization_fst,
                        const GeneralMatrix &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        int32 ivector_period,
                        const chain::Supervision &supervision,
                        const std::string &utt_id,
                        bool compress,
                        UtteranceSplitter *utt_splitter,
                        NnetChainExampleWriter *example_writer) {
  KALDI_ASSERT(supervision.num_sequences == 1);
  int32 num_input_frames = feats.NumRows(),
      num_output_frames = supervision.frames_per_sequence;

  if (!utt_splitter->LengthsMatch(utt_id, num_input_frames, num_output_frames))
    return false;  // LengthsMatch() will have printed a warning.

  std::vector<ChunkTimeInfo> chunks;

  utt_splitter->GetChunksForUtterance(num_input_frames, &chunks);

  if (chunks.empty()) {
    KALDI_WARN << "Not producing egs for utterance " << utt_id
               << " because it is too short: "
               << num_input_frames << " frames.";
    return false;
  }

  int32 frame_subsampling_factor = utt_splitter->Config().frame_subsampling_factor;

  chain::SupervisionSplitter sup_splitter(supervision);

  for (size_t c = 0; c < chunks.size(); c++) {
    ChunkTimeInfo &chunk = chunks[c];

    int32 start_frame_subsampled = chunk.first_frame / frame_subsampling_factor,
        num_frames_subsampled = chunk.num_frames / frame_subsampling_factor;

    chain::Supervision supervision_part;
    sup_splitter.GetFrameRange(start_frame_subsampled,
                               num_frames_subsampled,
                               &supervision_part);

    if (normalization_fst.NumStates() > 0 &&
        !AddWeightToSupervisionFst(normalization_fst,
                                   &supervision_part)) {
      KALDI_WARN << "For utterance " << utt_id << ", feature frames "
                 << chunk.first_frame << " to "
                 << (chunk.first_frame + chunk.num_frames)
                 << ", FST was empty after composing with normalization FST. "
                 << "This should be extremely rare (a few per corpus, at most)";
    }

    int32 first_frame = 0;  // we shift the time-indexes of all these parts so
                            // that the supervised part starts from frame 0.

    SubVector<BaseFloat> output_weights(
        &(chunk.output_weights[0]),
        static_cast<int32>(chunk.output_weights.size()));

    NnetChainSupervision nnet_supervision("output", supervision_part,
                                          output_weights,
                                          first_frame,
                                          frame_subsampling_factor);

    NnetChainExample nnet_chain_eg;
    nnet_chain_eg.outputs.resize(1);
    nnet_chain_eg.outputs[0].Swap(&nnet_supervision);
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

} // namespace nnet2
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
        "fixed size.  Input will come from chain-get-supervision.\n"
        "Note: if <normalization-fst> is not supplied the egs will not be\n"
        "ready for training; in that case they should later be processed\n"
        "with nnet3-chain-normalize-egs\n"
        "\n"
        "Usage:  nnet3-chain-get-egs [options] [<normalization-fst>] <features-rspecifier> "
        "<chain-supervision-rspecifier> <egs-wspecifier>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "chain-get-supervision [args] | \\\n"
        "  nnet3-chain-get-egs --left-context=25 --right-context=9 --num-frames=20 dir/normalization.fst \\\n"
        "  \"$feats\" ark,s,cs:- ark:cegs.1.ark\n"
        "Note: the --frame-subsampling-factor option must be the same as given to\n"
        "chain-get-supervision.\n";

    bool compress = true;
    int32 length_tolerance = 100, online_ivector_period = 1;

    ExampleGenerationConfig eg_config;  // controls num-frames,
                                        // left/right-context, etc.

    int32 srand_seed = 0;
    std::string online_ivector_rspecifier;

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
    eg_config.Register(&po);

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        normalization_fst_rxfilename,
        feature_rspecifier,
        supervision_rspecifier,
        examples_wspecifier;
    if (po.NumArgs() == 3) {
      feature_rspecifier = po.GetArg(1);
      supervision_rspecifier = po.GetArg(2);
      examples_wspecifier = po.GetArg(3);
    } else {
      normalization_fst_rxfilename = po.GetArg(1);
      KALDI_ASSERT(!normalization_fst_rxfilename.empty());
      feature_rspecifier = po.GetArg(2);
      supervision_rspecifier = po.GetArg(3);
      examples_wspecifier = po.GetArg(4);
    }

    eg_config.ComputeDerived();
    UtteranceSplitter utt_splitter(eg_config);

    fst::StdVectorFst normalization_fst;
    if (!normalization_fst_rxfilename.empty()) {
      ReadFstKaldi(normalization_fst_rxfilename, &normalization_fst);
      KALDI_ASSERT(normalization_fst.NumStates() > 0);
    }

    // Read as GeneralMatrix so we don't need to un-compress and re-compress
    // when selecting parts of matrices.
    SequentialGeneralMatrixReader feat_reader(feature_rspecifier);
    chain::RandomAccessSupervisionReader supervision_reader(
        supervision_rspecifier);
    NnetChainExampleWriter example_writer(examples_wspecifier);
    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);

    int32 num_err = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const GeneralMatrix &feats = feat_reader.Value();
      if (!supervision_reader.HasKey(key)) {
        KALDI_WARN << "No pdf-level posterior for key " << key;
        num_err++;
      } else {
        const chain::Supervision &supervision = supervision_reader.Value(key);
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

        if (!ProcessFile(normalization_fst, feats,
                         online_ivector_feats, online_ivector_period,
                         supervision, key, compress,
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
