// chainbin/nnet3-chain-e2e-get-egs.cc

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
#include "fstext/fstext-lib.h"
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

static bool ProcessFile(const ExampleGenerationConfig &opts,
                        const TransitionModel &trans_model,
                        const fst::StdVectorFst &normalization_fst,
                        const MatrixBase<BaseFloat> &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        int32 ivector_period,
                        const fst::StdVectorFst& training_fst,
                        const std::string &utt_id,
                        bool compress,
                        NnetChainExampleWriter *example_writer) {

  // check feats.NumRows() and if it is not equal to an allowed num-frames
  // delete a few frames from beginning or end
  int32 min_diff = 100;
  int32 len_extend_context = 0;
  for (int32 i = 0; i < opts.num_frames.size(); i++)
    if (abs(feats.NumRows() - opts.num_frames[i]) < abs(min_diff))
      min_diff = feats.NumRows() - opts.num_frames[i];

  if (min_diff != 0) {
    KALDI_WARN << "No exact match found for the length of utt " << utt_id
               << " which has length: " << feats.NumRows()
               << " closest allowed length is off by " << min_diff
               << " frames. Will try to fix it..";
    if (abs(min_diff) < 5)  // we assume possibly up to 5 frames from the end can be safely deleted
      len_extend_context = -min_diff;  // let the code below do it
    else  // unexpected
      KALDI_ERR << "Too much length difference for utterance " << utt_id;
  }
  int32 num_input_frames = feats.NumRows(),
        factor = opts.frame_subsampling_factor,
        num_frames_subsampled = (num_input_frames + len_extend_context + factor - 1) / factor,
        num_output_frames = num_frames_subsampled;


  chain::Supervision supervision;
  KALDI_VLOG(2) << "Preparing supervision for utt " << utt_id;
  if (!TrainingGraphToSupervisionE2e(training_fst, trans_model,
                                     num_output_frames, &supervision))
    return false;
  if (normalization_fst.NumStates() > 0 &&
      !AddWeightToSupervisionFst(normalization_fst,
                                 &supervision)) {
    KALDI_WARN << "For utterance " << utt_id
               << ", FST was empty after composing with normalization FST. "
               << "This should be extremely rare (a few per corpus, at most)";
  }

  int32 first_frame = 0;  // we shift the time-indexes of all these parts so
                          // that the supervised part starts from frame 0.

  Vector<BaseFloat> output_weights(num_output_frames, kSetZero);
  output_weights.Set(1.0);

  NnetChainSupervision nnet_supervision("output", supervision,
                                        output_weights,
                                        first_frame,
                                        opts.frame_subsampling_factor);

  NnetChainExample nnet_chain_eg;
  nnet_chain_eg.outputs.resize(1);
  nnet_chain_eg.outputs[0].Swap(&nnet_supervision);
  nnet_chain_eg.inputs.resize(ivector_feats != NULL ? 2 : 1);

  int32 left_context = (opts.left_context_initial >= 0 ?
                        opts.left_context_initial : opts.left_context);
  int32 right_context = (opts.right_context_final >= 0 ?
                         opts.right_context_final : opts.right_context);


  int32 tot_input_frames = left_context + num_input_frames +
                           right_context + len_extend_context;

  Matrix<BaseFloat> input_frames(tot_input_frames, feats.NumCols(),
                                 kUndefined);

  int32 start_frame = first_frame - left_context;
  for (int32 t = start_frame; t < start_frame + tot_input_frames; t++) {
    int32 t2 = t;
    if (t2 < 0) t2 = 0;
    if (t2 >= num_input_frames) t2 = num_input_frames - 1;
    int32 j = t - start_frame;
    SubVector<BaseFloat> src(feats, t2),
        dest(input_frames, j);
    dest.CopyFromVec(src);
  }
  NnetIo input_io("input", -left_context, input_frames);
  nnet_chain_eg.inputs[0].Swap(&input_io);

  if (ivector_feats != NULL) {
    // if applicable, add the iVector feature.
    // choose iVector from a random frame in the utterance
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
  os << utt_id;

  std::string key = os.str(); // key is <utt_id>-<frame_id>

  example_writer->Write(key, nnet_chain_eg);
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
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Get frame-by-frame examples of data for nnet3+chain end2end neural network\n"
        "training."
        "Note: if <normalization-fst> is not supplied the egs will not be\n"
        "ready for training; in that case they should later be processed\n"
        "with nnet3-chain-normalize-egs\n"
        "\n"
        "Usage:  nnet3-chain-get-egs [options] [<normalization-fst>] <features-rspecifier> "
        "<fst-rspecifier> <trans-model> <egs-wspecifier>\n"
        "\n";

    bool compress = true;
    int32 length_tolerance = 100, online_ivector_period = 1;

    ExampleGenerationConfig eg_config;  // controls num-frames,
                                        // left/right-context, etc.

    int32 srand_seed = 0;
    std::string online_ivector_rspecifier;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");
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

    if (po.NumArgs() < 4 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        normalization_fst_rxfilename,
        feature_rspecifier,
        fst_rspecifier,
        trans_model_rxfilename,
        examples_wspecifier;
    if (po.NumArgs() == 4) {
      feature_rspecifier = po.GetArg(1);
      fst_rspecifier = po.GetArg(2),
      trans_model_rxfilename = po.GetArg(3),
      examples_wspecifier = po.GetArg(4);
    } else {
      normalization_fst_rxfilename = po.GetArg(1);
      KALDI_ASSERT(!normalization_fst_rxfilename.empty());
      feature_rspecifier = po.GetArg(2);
      fst_rspecifier = po.GetArg(3),
      trans_model_rxfilename = po.GetArg(4),
      examples_wspecifier = po.GetArg(5);
    }

    eg_config.ComputeDerived();

    fst::StdVectorFst normalization_fst;
    if (!normalization_fst_rxfilename.empty()) {
      ReadFstKaldi(normalization_fst_rxfilename, &normalization_fst);
      KALDI_ASSERT(normalization_fst.NumStates() > 0);
    }

    TransitionModel trans_model;
    ReadKaldiObject(trans_model_rxfilename, &trans_model);

    RandomAccessBaseFloatMatrixReader feat_reader(feature_rspecifier);
    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    NnetChainExampleWriter example_writer(examples_wspecifier);
    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);

    int32 num_err = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string key = fst_reader.Key();
      if (!feat_reader.HasKey(key)) {
        num_err++;
        KALDI_WARN << "No features for utterance " << key;
      } else {
        const Matrix<BaseFloat> &features = feat_reader.Value(key);
        VectorFst<StdArc> fst(fst_reader.Value());
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
            (abs(features.NumRows() - (online_ivector_feats->NumRows() *
                                    online_ivector_period)) > length_tolerance
             || online_ivector_feats->NumRows() == 0)) {
          KALDI_WARN << "Length difference between feats " << features.NumRows()
                     << " and iVectors " << online_ivector_feats->NumRows()
                     << "exceeds tolerance " << length_tolerance;
          num_err++;
          continue;
        }

        if (!ProcessFile(eg_config, trans_model, normalization_fst, features,
                         online_ivector_feats, online_ivector_period,
                         fst, key, compress, &example_writer))
          num_err++;
      }
    }
    if (num_err > 0)
      KALDI_WARN << num_err << " utterances had errors and could "
          "not be processed.";
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
