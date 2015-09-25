// ctcbin/nnet3-ctc-get-egs.cc

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
#include "nnet3/nnet-cctc-example.h"

namespace kaldi {
namespace nnet3 {


static bool ProcessFile(const MatrixBase<BaseFloat> &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        const ctc::CctcSupervision &cctc_supervision,
                        const std::string &utt_id,
                        bool compress,
                        int32 left_context,
                        int32 right_context,
                        int32 frames_per_eg,
                        int32 frame_subsampling_factor,
                        int64 *num_frames_written,
                        int64 *num_egs_written,
                        NnetCctcExampleWriter *example_writer) {
  int32 num_feature_frames = feats.NumRows(),
      num_ctc_frames = cctc_supervision.num_frames,
      num_feature_frames_subsampled = num_feature_frames /
      frame_subsampling_factor;
  if (num_ctc_frames != num_feature_frames_subsampled)
    KALDI_ERR << "Mismatch in num-frames: CTC supervision has "
              << num_ctc_frames << " versus features/frame_subsampling_factor = "
              << num_feature_frames << " / " << frame_subsampling_factor
              << ": check that --frame-subsampling-factor option is set "
              << "the same as to ctc-get-supervision.";

  KALDI_ASSERT(frames_per_eg % frame_subsampling_factor == 0);

  int32 frames_per_eg_subsampled = frames_per_eg / frame_subsampling_factor;
    
  // we don't do any padding, as it would be a bit tricky to pad the CTC supervision.
  // Instead we select ranges of frames that fully fit within the file;  these
  // might slightly overlap with each other or have gaps.
  std::vector<int32> range_starts_subsampled;
  ctc::SplitIntoRanges(num_feature_frames_subsampled,
                       frames_per_eg_subsampled,
                       &range_starts_subsampled);

  if (range_starts_subsampled.empty()) {
    KALDI_WARN << "No output for utterance " << utt_id
               << " (num-frames=" << num_feature_frames
               << ") because too short for --frames-per-eg="
               << frames_per_eg;
    return false;
  }
  ctc::CctcSupervisionSplitter splitter(cctc_supervision);
  
  for (size_t i = 0; i < range_starts_subsampled.size(); i++) {
    int32 range_start_subsampled = range_starts_subsampled[i],
        range_start = range_start_subsampled * frames_per_eg;

    ctc::CctcSupervision supervision_part;
    splitter.GetFrameRange(range_start_subsampled,
                           frames_per_eg_subsampled,
                           &supervision_part);
    int32 first_frame = 0;  // we shift the time-indexes of all these parts so
                            // that the supervised part starts from frame 0.
    NnetCctcSupervision nnet_supervision(supervision_part,
                                         "output", first_frame,
                                         frame_subsampling_factor);

    NnetCctcExample nnet_cctc_eg;
    nnet_cctc_eg.outputs.resize(1);
    nnet_cctc_eg.outputs[0].Swap(&nnet_supervision);
    nnet_cctc_eg.inputs.resize(ivector_feats != NULL ? 2 : 1);
    
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
    nnet_cctc_eg.inputs[0].Swap(&input_io);

    if (ivector_feats != NULL) {
      // if applicable, add the iVector feature.
      // try to get closest frame to middle of window to get
      // a representative iVector.
      int32 closest_frame = range_start + frames_per_eg / 2;
      KALDI_ASSERT(ivector_feats->NumRows() > 0);
      if (closest_frame >= ivector_feats->NumRows())
        closest_frame = ivector_feats->NumRows() - 1;
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(closest_frame));
      NnetIo ivector_io("ivector", 0, ivector);
      nnet_cctc_eg.inputs[1].Swap(&ivector_io);
    }

    if (compress)
      nnet_cctc_eg.Compress();
      
    std::ostringstream os;
    os << utt_id << "-" << range_start;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    *num_frames_written += frames_per_eg;
    *num_egs_written += 1;

    example_writer->Write(key, nnet_cctc_eg);
  }
  return true;
}


} // namespace nnet2
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get frame-by-frame examples of data for nnet3+CTC neural network\n"
        "training.  This involves breaking up utterances into pieces of a\n"
        "fixed size.  Input will come from ctc-get-supervision.\n"
        "\n"
        "Usage:  nnet3-ctc-get-egs [options] <features-rspecifier> "
        "<ctc-supervision-rspecifier> <egs-wspecifier>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "ctc-get-supervision [args] | \\\n"
        "  nnet3-ctc-get-egs --left-context=25 --right-context=9 --num-frames=20 \"$feats\" ark,s,cs:- \\\n"
        "  ark:ctc_egs.1.ark\n"
        "Note: the --frame-subsampling-factor option must be the same as given to\n"
        "ctc-get-supervision.\n";

    bool compress = true;
    int32 left_context = 0, right_context = 0, num_frames = 1,
        length_tolerance = 100, frame_subsampling_factor = 1;
        
    std::string ivector_rspecifier;
    
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
    po.Register("ivectors", &ivector_rspecifier, "Rspecifier of ivector "
                "features, as a matrix.");
    po.Register("length-tolerance", &length_tolerance, "Tolerance for "
                "difference in num-frames between feat and ivector matrices");
    po.Register("frame-subsampling-factor", &frame_subsampling_factor, "Used "
                "if the frame-rate in CTC will be less than the frame-rate "
                "of the original alignment");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (num_frames <= 0 || left_context < 0 || right_context < 0 ||
        length_tolerance < 0 || frame_subsampling_factor <= 0)
      KALDI_ERR << "One of the integer options is out of the allowed range.";
    if (num_frames % frame_subsampling_factor != 0) {
      int32 new_num_frames = frame_subsampling_factor *
          (num_frames / frame_subsampling_factor + 1);
      KALDI_LOG << "Rounding up --num-frames=" << num_frames
                << " to a multiple of --frame-subsampling-factor="
                << frame_subsampling_factor
                << ", now --num-frames=" << new_num_frames;
      num_frames = new_num_frames;
    }
    

    std::string feature_rspecifier = po.GetArg(1),
        supervision_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessCctcSupervisionReader supervision_reader(supervision_rspecifier);
    NnetCctcExampleWriter example_writer(examples_wspecifier);
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
        const ctc::CctcSupervision &supervision = supervision_reader.Value(key);
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
                     << "exceeds tolerance " << length_tolerance;
          num_err++;
          continue;
        }
        if (ProcessFile(feats, ivector_feats, supervision, key, compress,
                        left_context, right_context, num_frames,
                        frame_subsampling_factor,
                        &num_frames_written, &num_egs_written,
                        &example_writer))
          num_done++;
        else
          num_err++;
      }
    }

    KALDI_LOG << "Finished generating nnet3-ctc examples, "
              << "successfully processed " << num_done
              << " feature files, wrote " << num_egs_written << " examples, "
              << " with " << num_frames_written << " egs in total; "
              << num_err << " files had errors.";
    return (num_egs_written == 0 || num_err > num_done ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
