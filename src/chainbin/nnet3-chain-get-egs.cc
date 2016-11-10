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
                        const MatrixBase<BaseFloat> &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        const chain::Supervision &supervision,
                        const std::string &utt_id,
                        bool compress,
                        int32 left_context,
                        int32 right_context,
                        int32 frames_per_eg,
                        int32 frames_overlap_per_eg,
                        int32 frame_subsampling_factor,
                        int32 cut_zero_frames,
                        int64 *num_frames_written,
                        int64 *num_egs_written,
                        NnetChainExampleWriter *example_writer) {
  KALDI_ASSERT(supervision.num_sequences == 1);
  int32 num_feature_frames = feats.NumRows(),
      num_output_frames = supervision.frames_per_sequence,
      num_feature_frames_subsampled =
                             (num_feature_frames + frame_subsampling_factor - 1)/
                             frame_subsampling_factor;
  if (num_output_frames != num_feature_frames_subsampled) {
    // we tolerate deviations in the num-frames if they are very small (1 output
    // frame).

    if (abs(num_output_frames - num_feature_frames_subsampled) > 1) {
      KALDI_ERR << "Mismatch in num-frames: chain supervision has "
                << num_output_frames
                << " versus features/frame_subsampling_factor = "
                << num_feature_frames << " / " << frame_subsampling_factor
                << " = " << num_feature_frames_subsampled
                << ": check that --frame-subsampling-factor option is set "
                << "the same as to chain-get-supervision.";
    }
    int32 new_num_feature_frames =
        num_output_frames * frame_subsampling_factor;
    // add a few frames at the end to make it match up.
    Matrix<BaseFloat> feats_new(new_num_feature_frames, feats.NumCols(),
                                kUndefined);
    int32 min_feature_frames = std::min<int32>(num_feature_frames,
                                               new_num_feature_frames);
    feats_new.RowRange(0, min_feature_frames).CopyFromMat(
        feats.RowRange(0, min_feature_frames));
    for (int32 i = num_feature_frames; i < new_num_feature_frames; i++)
      feats_new.Row(i).CopyFromVec(feats.Row(num_feature_frames - 1));
    return ProcessFile(normalization_fst, feats_new, ivector_feats,
                       supervision, utt_id, compress, left_context, right_context,
                       frames_per_eg, frames_overlap_per_eg, frame_subsampling_factor,
                       cut_zero_frames, num_frames_written, num_egs_written,
                       example_writer);
  }

  KALDI_ASSERT(frames_per_eg % frame_subsampling_factor == 0);

  int32 frames_per_eg_subsampled = frames_per_eg / frame_subsampling_factor,
      frames_overlap_subsampled = frames_overlap_per_eg / frame_subsampling_factor,
      frames_shift_subsampled = frames_per_eg_subsampled - frames_overlap_subsampled;

  if (num_feature_frames_subsampled < frames_per_eg_subsampled) {
    KALDI_WARN << "Length of features for utterance " << utt_id
               << " is less than than the frames_per_eg (after sub-sampling).";
    return false;
  }

  // we don't do any padding, as it would be a bit tricky to pad the 'chain' supervision.
  // Instead we select ranges of frames that fully fit within the file;  these
  // might slightly overlap with each other or have gaps.
  std::vector<int32> range_starts_subsampled;
  chain::SplitIntoRanges(num_feature_frames_subsampled -
                         frames_overlap_subsampled,
                         frames_shift_subsampled,
                         &range_starts_subsampled);
  // The 'deriv_weights' make sure we don't count frames twice, and also ensure
  // that we tend to avoid having nonzero weights on the derivatives that are
  // too close to the edge of the corresponding 'range' (these derivatives close
  // to the edge are not as accurate as they could be, because when we split we
  // don't know the correct alphas and betas).
  std::vector<Vector<BaseFloat> > deriv_weights;
  if (cut_zero_frames >= 0)
    chain::GetWeightsForRangesNew(frames_per_eg_subsampled,
                                  cut_zero_frames / frame_subsampling_factor,
                                  range_starts_subsampled,
                                  &deriv_weights);
  else
    chain::GetWeightsForRanges(frames_per_eg_subsampled,
                               range_starts_subsampled,
                               &deriv_weights);

  if (range_starts_subsampled.empty()) {
    KALDI_WARN << "No output for utterance " << utt_id
               << " (num-frames=" << num_feature_frames
               << ") because too short for --frames-per-eg="
               << frames_per_eg;
    return false;
  }
  chain::SupervisionSplitter splitter(supervision);

  for (size_t i = 0; i < range_starts_subsampled.size(); i++) {
    int32 range_start_subsampled = range_starts_subsampled[i],
        range_start = range_start_subsampled * frame_subsampling_factor;

    chain::Supervision supervision_part;
    splitter.GetFrameRange(range_start_subsampled,
                           frames_per_eg_subsampled,
                           &supervision_part);

    if (normalization_fst.NumStates() > 0 &&
        !AddWeightToSupervisionFst(normalization_fst,
                                   &supervision_part)) {
      KALDI_WARN << "For utterance " << utt_id << ", frames "
                 << range_start << " to " << (range_start + frames_per_eg)
                 << ", FST was empty after composing with normalization FST. "
                 << "This should be extremely rare (a few per corpus, at most)";
      return false;
    }

    int32 first_frame = 0;  // we shift the time-indexes of all these parts so
                            // that the supervised part starts from frame 0.
    NnetChainSupervision nnet_supervision("output", supervision_part,
                                          deriv_weights[i],
                                          first_frame, frame_subsampling_factor);

    NnetChainExample nnet_chain_eg;
    nnet_chain_eg.outputs.resize(1);
    nnet_chain_eg.outputs[0].Swap(&nnet_supervision);
    nnet_chain_eg.inputs.resize(ivector_feats != NULL ? 2 : 1);

    int32 tot_frames = left_context + frames_per_eg + right_context;
    Matrix<BaseFloat> input_frames(tot_frames, feats.NumCols(), kUndefined);

    // Set up "input_frames".
    for (int32 j = -left_context; j < frames_per_eg + right_context; j++) {
      int32 t = range_start + j;
      if (t < 0) t = 0;
      if (t >= feats.NumRows()) t = feats.NumRows() - 1;
      SubVector<BaseFloat> src(feats, t),
          dest(input_frames, j + left_context);
      dest.CopyFromVec(src);
    }
    NnetIo input_io("input", - left_context,
                    input_frames);
    nnet_chain_eg.inputs[0].Swap(&input_io);

    if (ivector_feats != NULL) {
      // if applicable, add the iVector feature.
      // choose iVector from a random frame in the chunk
      int32 ivector_frame = RandInt(range_start, range_start + frames_per_eg - 1);
      KALDI_ASSERT(ivector_feats->NumRows() > 0);
      if (ivector_frame >= ivector_feats->NumRows())
        ivector_frame = ivector_feats->NumRows() - 1;
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(ivector_frame));
      NnetIo ivector_io("ivector", 0, ivector);
      nnet_chain_eg.inputs[1].Swap(&ivector_io);
    }

    if (compress)
      nnet_chain_eg.Compress();

    std::ostringstream os;
    os << utt_id << "-" << range_start;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    *num_frames_written += frames_per_eg;
    *num_egs_written += 1;

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
    int32 left_context = 0, right_context = 0, num_frames = 1,
        num_frames_overlap = 0, length_tolerance = 100,
        cut_zero_frames = -1,
        frame_subsampling_factor = 1;

    int32 srand_seed = 0;
    std::string ivector_rspecifier;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format (recommended)");
    po.Register("cut-zero-frames", &cut_zero_frames, "Number of frames "
                "(measured before subsampling) to zero the derivative on each "
                "side of a cut point (if set, activates new-style derivative "
                "weights)");
    po.Register("left-context", &left_context, "Number of frames of left "
                "context the neural net requires.");
    po.Register("right-context", &right_context, "Number of frames of right "
                "context the neural net requires.");
    po.Register("num-frames", &num_frames, "Number of frames with labels "
                "that each example contains.  Will be rounded up to a multiple "
                "of --frame-subsampling-factor.");
    po.Register("num-frames-overlap", &num_frames_overlap, "Number of frames of "
                "overlap between each example (could be useful in conjunction "
                "--min-deriv-time and --max-deriv-time, to avoid wasting data). "
                "Each time we shift by --num-frames minus --num-frames-overlap.");
    po.Register("ivectors", &ivector_rspecifier, "Rspecifier of ivector "
                "features, as a matrix.");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(only relevant if --pick-random-ivector=true)");
    po.Register("length-tolerance", &length_tolerance, "Tolerance for "
                "difference in num-frames between feat and ivector matrices");
    po.Register("frame-subsampling-factor", &frame_subsampling_factor, "Used "
                "if the frame-rate at the output will be less than the "
                "frame-rate of the input");

    po.Read(argc, argv);
    
    srand(srand_seed);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    if (num_frames <= 0 || left_context < 0 || right_context < 0 ||
        length_tolerance < 0 || frame_subsampling_factor <= 0)
      KALDI_ERR << "One of the integer options is out of the allowed range.";
    RoundUpNumFrames(frame_subsampling_factor,
                     &num_frames, &num_frames_overlap);

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

    fst::StdVectorFst normalization_fst;
    if (!normalization_fst_rxfilename.empty()) {
      ReadFstKaldi(normalization_fst_rxfilename, &normalization_fst);
      KALDI_ASSERT(normalization_fst.NumStates() > 0);
    }

    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    chain::RandomAccessSupervisionReader supervision_reader(
        supervision_rspecifier);
    NnetChainExampleWriter example_writer(examples_wspecifier);
    RandomAccessBaseFloatMatrixReader ivector_reader(ivector_rspecifier);

    int32 num_done = 0, num_err = 0;
    int64 num_frames_written = 0, num_egs_written = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (!supervision_reader.HasKey(key)) {
        KALDI_WARN << "No pdf-level posterior for key " << key;
        num_err++;
      } else {
        const chain::Supervision &supervision = supervision_reader.Value(key);
        const Matrix<BaseFloat> *ivector_feats = NULL;
        if (!ivector_rspecifier.empty()) {
          if (!ivector_reader.HasKey(key)) {
            KALDI_WARN << "No iVectors for utterance " << key;
            num_err++;
            continue;
          } else {
            // this address will be valid until we call HasKey() or Value()
            // again.
            ivector_feats = &(ivector_reader.Value(key));
          }
        }
        if (ivector_feats != NULL &&
            (abs(feats.NumRows() - ivector_feats->NumRows()) > length_tolerance
             || ivector_feats->NumRows() == 0)) {
          KALDI_WARN << "Length difference between feats " << feats.NumRows()
                     << " and iVectors " << ivector_feats->NumRows()
                     << " exceeds tolerance " << length_tolerance;
          num_err++;
          continue;
        }
        if (ProcessFile(normalization_fst, feats, ivector_feats, supervision,
                        key, compress,
                        left_context, right_context, num_frames,
                        num_frames_overlap, frame_subsampling_factor,
                        cut_zero_frames, &num_frames_written, &num_egs_written,
                        &example_writer))
          num_done++;
        else
          num_err++;
      }
    }

    KALDI_LOG << "Finished generating nnet3-chain examples, "
              << "successfully processed " << num_done
              << " feature files, wrote " << num_egs_written << " examples, "
              << " with " << num_frames_written << " frames in total; "
              << num_err << " files had errors.";
    return (num_egs_written == 0 || num_err > num_done ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
