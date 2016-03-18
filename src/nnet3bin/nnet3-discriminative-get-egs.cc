// nnet3bin/nnet3-discriminative-get-egs.cc

// Copyright      2015  Johns Hopkins University (author:  Daniel Povey)
//           2014-2015  Vimal Manohar

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
#include "nnet3/nnet-discriminative-example.h"
#include "nnet3/discriminative-supervision.h"
#include "nnet3/nnet-example-utils.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace nnet3 {

/**
   This function does all the processing for one utterance, and outputs the
   supervision objects to 'example_writer'.  
*/

static bool ProcessFile(
                        const discriminative::SplitDiscriminativeSupervisionOptions &config,
                        const TransitionModel &tmodel,
                        const MatrixBase<BaseFloat> &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        const discriminative::DiscriminativeSupervision &supervision,
                        const std::string &utt_id,
                        bool compress,
                        int32 left_context,
                        int32 right_context,
                        int32 frames_per_eg,
                        int32 frames_overlap_per_eg,
                        int32 frame_subsampling_factor,
                        int64 *num_frames_written,
                        int64 *num_egs_written,
                        NnetDiscriminativeExampleWriter *example_writer) {
  KALDI_ASSERT(supervision.num_sequences == 1);
  int32 num_feature_frames = feats.NumRows(),
      num_output_frames = supervision.frames_per_sequence,
      num_feature_frames_subsampled =
                             (num_feature_frames + frame_subsampling_factor - 1)/
                             frame_subsampling_factor;
  if (num_output_frames != num_feature_frames_subsampled)
    KALDI_ERR << "Mismatch in num-frames: discriminative supervision has "
              << num_output_frames
              << " versus features/frame_subsampling_factor = "
              << num_feature_frames << " / " << frame_subsampling_factor
              << ": check that --frame-subsampling-factor option is set "
              << "the same as to discriminative-get-supervision.";

  KALDI_ASSERT(frames_per_eg % frame_subsampling_factor == 0);

  int32 frames_per_eg_subsampled = frames_per_eg / frame_subsampling_factor,
      frames_overlap_subsampled = frames_overlap_per_eg / frame_subsampling_factor,
      frames_shift_subsampled = frames_per_eg_subsampled - frames_overlap_subsampled;

  if (frames_per_eg != -1 && num_feature_frames_subsampled < frames_per_eg_subsampled) {
    KALDI_WARN << "No output for utterance " << utt_id
               << " (num-frames=" << num_feature_frames
               << ") because too short for --frames-per-eg="
               << frames_per_eg;
    return false;
  }

  // we don't do any padding, as it would be a bit tricky to pad the discriminative training supervision.
  // Instead we select ranges of frames that fully fit within the file;  these
  // might slightly overlap with each other or have gaps.
  std::vector<int32> range_starts_subsampled;
  if (frames_per_eg != -1) {
    chain::SplitIntoRanges(num_feature_frames_subsampled -
                           frames_overlap_subsampled,
                           frames_shift_subsampled,
                           &range_starts_subsampled);
  } else {
    range_starts_subsampled.push_back(0);
  }
  // The 'deriv_weights' make sure we don't count frames twice, and also ensure
  // that we tend to avoid having nonzero weights on the derivatives that are
  // too close to the edge of the corresponding 'range' (these derivatives close
  // to the edge are not as accurate as they could be, because when we split we
  // don't know the correct alphas and betas).
  std::vector<Vector<BaseFloat> > deriv_weights;
  if (frames_per_eg != -1) {
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
  } else {
    deriv_weights.push_back(Vector<BaseFloat>());
  }

  discriminative::DiscriminativeSupervisionSplitter splitter(config, tmodel, 
                                                             supervision);

  for (size_t i = 0; i < range_starts_subsampled.size(); i++) {

    NnetDiscriminativeExample nnet_discriminative_eg;
    nnet_discriminative_eg.outputs.resize(1);
    int32 range_start_subsampled = range_starts_subsampled[i],
        range_start = range_start_subsampled * frame_subsampling_factor;
    
    if (frames_per_eg != -1) {

      discriminative::DiscriminativeSupervision supervision_part;

      splitter.GetFrameRange(range_start_subsampled,
                             frames_per_eg_subsampled,
                             (i == 0 ? false : true),
                             &supervision_part);

      int32 first_frame = 0;  // we shift the time-indexes of all these parts so
                              // that the supervised part starts from frame 0.
      NnetDiscriminativeSupervision nnet_supervision("output", supervision_part,
                                                     deriv_weights[i],
                                                     first_frame, 
                                                     frame_subsampling_factor);
      nnet_discriminative_eg.outputs[0].Swap(&nnet_supervision);
    } else {
      int32 first_frame = 0;  // we shift the time-indexes of all these parts so
                              // that the supervised part starts from frame 0.
      NnetDiscriminativeSupervision nnet_supervision("output", supervision,
                                                     deriv_weights[i],
                                                     first_frame, 
                                                     frame_subsampling_factor);
      nnet_discriminative_eg.outputs[0].Swap(&nnet_supervision);
    }

    nnet_discriminative_eg.inputs.resize(ivector_feats != NULL ? 2 : 1);

    int32 this_frames_per_eg = frames_per_eg != -1 ? frames_per_eg : supervision.frames_per_sequence;

    int32 tot_frames = left_context + this_frames_per_eg + right_context;
    Matrix<BaseFloat> input_frames(tot_frames, feats.NumCols(), kUndefined);

    // Set up "input_frames".
    for (int32 j = -left_context; j < this_frames_per_eg + right_context; j++) {
      int32 t = range_start + j;
      if (t < 0) t = 0;
      if (t >= feats.NumRows()) t = feats.NumRows() - 1;
      SubVector<BaseFloat> src(feats, t),
          dest(input_frames, j + left_context);
      dest.CopyFromVec(src);
    }
    NnetIo input_io("input", - left_context,
                    input_frames);
    nnet_discriminative_eg.inputs[0].Swap(&input_io);

    if (ivector_feats != NULL) {
      // if applicable, add the iVector feature.
      // try to get closest frame to middle of window to get
      // a representative iVector.
      int32 closest_frame = range_start + this_frames_per_eg / 2;
      KALDI_ASSERT(ivector_feats->NumRows() > 0);
      if (closest_frame >= ivector_feats->NumRows())
        closest_frame = ivector_feats->NumRows() - 1;
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(closest_frame));
      NnetIo ivector_io("ivector", 0, ivector);
      nnet_discriminative_eg.inputs[1].Swap(&ivector_io);
    }

    if (compress)
      nnet_discriminative_eg.Compress();

    std::ostringstream os;
    os << utt_id << "-" << range_start;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    *num_frames_written += this_frames_per_eg;
    *num_egs_written += 1;

    example_writer->Write(key, nnet_discriminative_eg);
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
        "Get frame-by-frame examples of data for nnet3+sequence neural network\n"
        "training.  This involves breaking up utterances into pieces of a\n"
        "fixed size.  Input will come from discriminative-get-supervision.\n"
        "\n"
        "Usage:  nnet3-discriminative-get-egs [options] <model> <features-rspecifier> "
        "<discriminative-supervision-rspecifier> <egs-wspecifier>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "discriminative-get-supervision [args] | \\\n"
        "  nnet3-discriminative-get-egs --left-context=25 --right-context=9 --num-frames=20 \\\n"
        "  \"$feats\" ark,s,cs:- ark:degs.1.ark\n"
        "Note: the --frame-subsampling-factor option must be the same as given to\n"
        "discriminative-get-supervision.\n";

    bool compress = true;
    int32 left_context = 0, right_context = 0, num_frames = 1,
        num_frames_overlap = 0, length_tolerance = 100,
        frame_subsampling_factor = 1;

    std::string ivector_rspecifier;
    discriminative::SplitDiscriminativeSupervisionOptions splitter_config;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format (recommended)");
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
    po.Register("length-tolerance", &length_tolerance, "Tolerance for "
                "difference in num-frames between feat and ivector matrices");
    po.Register("frame-subsampling-factor", &frame_subsampling_factor, "Used "
                "if the frame-rate at the output will be less than the "
                "frame-rate of the input");
    
    ParseOptions splitter_opts("supervision-splitter", &po);
    splitter_config.Register(&splitter_opts);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    if (left_context < 0 || right_context < 0 ||
        length_tolerance < 0 || frame_subsampling_factor <= 0)
      KALDI_ERR << "One of the integer options is out of the allowed range.";

    if (frame_subsampling_factor != 1)
      RoundUpNumFrames(frame_subsampling_factor,
                       &num_frames, &num_frames_overlap);

    std::string model_wxfilename, feature_rspecifier,
                supervision_rspecifier,
                examples_wspecifier;

    model_wxfilename = po.GetArg(1);
    feature_rspecifier = po.GetArg(2);
    supervision_rspecifier = po.GetArg(3);
    examples_wspecifier = po.GetArg(4);

    TransitionModel tmodel;
    { 
      bool binary;
      Input ki(model_wxfilename, &binary);
      tmodel.Read(ki.Stream(), binary);
    }

    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    discriminative::RandomAccessDiscriminativeSupervisionReader supervision_reader(
        supervision_rspecifier);
    NnetDiscriminativeExampleWriter example_writer(examples_wspecifier);
    RandomAccessBaseFloatMatrixReader ivector_reader(ivector_rspecifier);

    int32 num_done = 0, num_err = 0;
    int64 num_frames_written = 0, num_egs_written = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (!supervision_reader.HasKey(key)) {
        KALDI_WARN << "No supervision for key " << key;
        num_err++;
      } else {
        const discriminative::DiscriminativeSupervision &supervision = supervision_reader.Value(key);
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
            (std::abs(feats.NumRows() - ivector_feats->NumRows()) > length_tolerance
             || ivector_feats->NumRows() == 0)) {
          KALDI_WARN << "Length difference between feats " << feats.NumRows()
                     << " and iVectors " << ivector_feats->NumRows()
                     << "exceeds tolerance " << length_tolerance;
          num_err++;
          continue;
        }
        if (ProcessFile(splitter_config, tmodel,
                        feats, ivector_feats, supervision,
                        key, compress, left_context, right_context, num_frames,
                        num_frames_overlap, frame_subsampling_factor,
                        &num_frames_written, &num_egs_written,
                        &example_writer))
          num_done++;
        else {
          KALDI_WARN << "Failed to process utterance into nnet example "
                     << "for key " << key;
          num_err++;
        }
      }
    }

    KALDI_LOG << "Finished generating nnet3-discriminative examples, "
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

