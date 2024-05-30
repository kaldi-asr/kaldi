// nnet3bin/nnet3-get-egs-dense-targets.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
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
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {


static bool ProcessFile(const GeneralMatrix &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        int32 ivector_period,
                        const MatrixBase<BaseFloat> &targets,
                        const std::string &utt_id,
                        bool compress,
                        int32 num_targets,
                        int32 length_tolerance,
                        UtteranceSplitter *utt_splitter,
                        NnetExampleWriter *example_writer) {
  int32 num_input_frames = feats.NumRows();
  if (!utt_splitter->LengthsMatch(utt_id, num_input_frames,
                                  targets.NumRows(),
                                  length_tolerance)) {
    return false;
  }
  if (targets.NumRows() == 0)
    return false;
  KALDI_ASSERT(num_targets < 0 || targets.NumCols() == num_targets);

  std::vector<ChunkTimeInfo> chunks;

  utt_splitter->GetChunksForUtterance(num_input_frames, &chunks);

  if (chunks.empty()) {
    KALDI_WARN << "Not producing egs for utterance " << utt_id
               << " because it is too short: "
               << num_input_frames << " frames.";
    return false;
  }

  // 'frame_subsampling_factor' is not used in any recipes at the time of
  // writing, this is being supported to unify the code with the 'chain' recipes
  // and in case we need it for some reason in future.
  int32 frame_subsampling_factor =
      utt_splitter->Config().frame_subsampling_factor;

  for (size_t c = 0; c < chunks.size(); c++) {
    const ChunkTimeInfo &chunk = chunks[c];

    int32 tot_input_frames = (chunk.left_context +
                              chunk.num_frames +
                              chunk.right_context);

    int32 start_frame = chunk.first_frame - chunk.left_context;

    GeneralMatrix input_frames;
    ExtractRowRangeWithPadding(feats, start_frame, tot_input_frames,
                               &input_frames);

    // 'input_frames' now stores the relevant rows (maybe with padding) from the
    // original Matrix or (more likely) CompressedMatrix. If a CompressedMatrix,
    // it does this without un-compressing and re-compressing, so there is no
    // loss of accuracy.

    NnetExample eg;
    // call the regular input "input".
    eg.io.push_back(NnetIo("input", -chunk.left_context, input_frames));

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
      eg.io.push_back(NnetIo("ivector", 0, ivector));
    }

    // Note: chunk.first_frame and chunk.num_frames will both be
    // multiples of frame_subsampling_factor.
    int32 start_frame_subsampled = chunk.first_frame / frame_subsampling_factor,
        num_frames_subsampled = chunk.num_frames / frame_subsampling_factor;

    KALDI_ASSERT(start_frame_subsampled + num_frames_subsampled - 1 <
                 targets.NumRows());


    // Add the labels.
    Matrix<BaseFloat> targets_part(num_frames_subsampled, targets.NumCols());
    for (int32 i = 0; i < num_frames_subsampled; ++i) {
      // Copy the i^th row of the target matrix from the (t+i)^th row of the
      // input targets matrix.
      int32 t = i + start_frame_subsampled;
      if (t >= targets.NumRows())
        t = targets.NumRows() - 1;
      SubVector<BaseFloat> this_target_dest(targets_part, i);
      SubVector<BaseFloat> this_target_src(targets, t);
      this_target_dest.CopyFromVec(this_target_src);
    }

    // Push this created targets matrix into the eg.
    eg.io.push_back(NnetIo("output", 0, targets_part,
                           frame_subsampling_factor));

    if (compress)
      eg.Compress();

    std::ostringstream os;
    os << utt_id << "-" << chunk.first_frame;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    example_writer->Write(key, eg);
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
        "Get frame-by-frame examples of data for nnet3 neural network training.\n"
        "This program is similar to nnet3-get-egs, but the targets here are "
        "dense matrices instead of posteriors (sparse matrices).\n"
        "This is useful when you want the targets to be continuous real-valued "
        "with the neural network possibly trained with a quadratic objective\n"
        "\n"
        "Usage:  nnet3-get-egs-dense-targets --num-targets=<n> [options] "
        "<features-rspecifier> <targets-rspecifier> <egs-out>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet-get-egs-dense-targets --num-targets=26 --left-context=12 \\\n"
        "--right-context=9 --num-frames=8 \"$feats\" \\\n"
        "\"ark:copy-matrix ark:exp/snrs/snr.1.ark ark:- |\"\n"
        "   ark:- \n";


    bool compress = true;
    int32 num_targets = -1, length_tolerance = 100,
        targets_length_tolerance = 2,
        online_ivector_period = 1;

    ExampleGenerationConfig eg_config;  // controls num-frames,
                                        // left/right-context, etc.

    std::string online_ivector_rspecifier;

    ParseOptions po(usage);

    po.Register("compress", &compress, "If true, write egs with input features "
                "in compressed format (recommended).  This is "
                "only relevant if the features being read are un-compressed; "
                "if already compressed, we keep the same compressed format when "
                "dumping egs.");
    po.Register("num-targets", &num_targets, "Output dimension in egs, "
                "only used to check targets have correct dim if supplied.");
    po.Register("ivectors", &online_ivector_rspecifier, "Alias for "
                "--online-ivectors option, for back compatibility");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier of "
                "ivector features, as a matrix.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of "
                "frames between iVectors in matrices supplied to the "
                "--online-ivectors option");
    po.Register("length-tolerance", &length_tolerance, "Tolerance for "
                "difference in num-frames between feat and ivector matrices");
    po.Register("targets-length-tolerance", &targets_length_tolerance,
                "Tolerance for "
                "difference in num-frames (after subsampling) between "
                "feature and target matrices");
    eg_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    eg_config.ComputeDerived();
    UtteranceSplitter utt_splitter(eg_config);

    std::string feature_rspecifier = po.GetArg(1),
        matrix_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    // SequentialGeneralMatrixReader can read either a Matrix or
    // CompressedMatrix (or SparseMatrix, but not as relevant here),
    // and it retains the type.  This way, we can generate parts of
    // the feature matrices without uncompressing and re-compressing.
    SequentialGeneralMatrixReader feat_reader(feature_rspecifier);
    RandomAccessBaseFloatMatrixReader matrix_reader(matrix_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);

    int32 num_err = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const GeneralMatrix &feats = feat_reader.Value();
      if (!matrix_reader.HasKey(key)) {
        KALDI_WARN << "No target matrix for key " << key;
        num_err++;
      } else {
        const Matrix<BaseFloat> &target_matrix = matrix_reader.Value(key);
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

        if (!ProcessFile(feats, online_ivector_feats, online_ivector_period,
                         target_matrix, key, compress, num_targets,
                         targets_length_tolerance,
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
