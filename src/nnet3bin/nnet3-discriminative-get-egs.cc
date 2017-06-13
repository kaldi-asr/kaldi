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

// This function does all the processing for one utterance, and outputs the
// examples to 'example_writer'.
// returns true if we got as far as calling GetChunksForUtterance()
// [in which case stats will be accumulated by class UtteranceSplitter]
static bool ProcessFile(const discriminative::SplitDiscriminativeSupervisionOptions &config,
                        const TransitionModel &tmodel,
                        const MatrixBase<BaseFloat> &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        int32 ivector_period,
                        const discriminative::DiscriminativeSupervision &supervision,
                        const std::string &utt_id,
                        bool compress,
                        UtteranceSplitter *utt_splitter,
                        NnetDiscriminativeExampleWriter *example_writer) {
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
  }

  int32 frame_subsampling_factor = utt_splitter->Config().frame_subsampling_factor;

  discriminative::DiscriminativeSupervisionSplitter splitter(config, tmodel,
                                                             supervision);

  for (size_t c = 0; c < chunks.size(); c++) {
    ChunkTimeInfo &chunk = chunks[c];

    NnetDiscriminativeExample nnet_discriminative_eg;
    nnet_discriminative_eg.outputs.resize(1);

    int32 start_frame_subsampled = chunk.first_frame / frame_subsampling_factor,
        num_frames_subsampled = chunk.num_frames / frame_subsampling_factor;

    discriminative::DiscriminativeSupervision supervision_part;

    splitter.GetFrameRange(start_frame_subsampled,
                           num_frames_subsampled,
                           (c == 0 ? false : true),
                           &supervision_part);

    SubVector<BaseFloat> output_weights(
        &(chunk.output_weights[0]),
        static_cast<int32>(chunk.output_weights.size()));

    int32 first_frame = 0;  // we shift the time-indexes of all these parts so
                            // that the supervised part starts from frame 0.
    NnetDiscriminativeSupervision nnet_supervision("output", supervision_part,
                                                   output_weights,
                                                   first_frame,
                                                   frame_subsampling_factor);
    nnet_discriminative_eg.outputs[0].Swap(&nnet_supervision);

    nnet_discriminative_eg.inputs.resize(ivector_feats != NULL ? 2 : 1);


    int32 tot_input_frames = chunk.left_context + chunk.num_frames +
        chunk.right_context;

    Matrix<BaseFloat> input_frames(tot_input_frames, feats.NumCols(),
                                   kUndefined);

    int32 start_frame = chunk.first_frame - chunk.left_context;
    for (int32 t = start_frame; t < start_frame + tot_input_frames; t++) {
      int32 t2 = t;
      if (t2 < 0) t2 = 0;
      if (t2 >= num_input_frames) t2 = num_input_frames - 1;
      int32 j = t - start_frame;
      SubVector<BaseFloat> src(feats, t2),
          dest(input_frames, j);
      dest.CopyFromVec(src);
    }

    NnetIo input_io("input", -chunk.left_context, input_frames);
    nnet_discriminative_eg.inputs[0].Swap(&input_io);

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
      nnet_discriminative_eg.inputs[1].Swap(&ivector_io);
    }

    if (compress)
      nnet_discriminative_eg.Compress();

    std::ostringstream os;
    os << utt_id << "-" << chunk.first_frame;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

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
        "training.  This involves breaking up utterances into pieces of sizes\n"
        "determined by the --num-frames option.\n"
        "\n"
        "Usage:  nnet3-discriminative-get-egs [options] <model> <features-rspecifier> "
        "<denominator-lattice-rspecifier> <numerator-alignment-rspecifier> <egs-wspecifier>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "  nnet3-discriminative-get-egs --left-context=25 --right-context=9 --num-frames=150,100,90 \\\n"
        "  \"$feats\" \"ark,s,cs:gunzip -c lat.1.gz\" scp:ali.scp ark:degs.1.ark\n";

    bool compress = true;
    int32 length_tolerance = 100, online_ivector_period = 1;

    std::string online_ivector_rspecifier;

    ExampleGenerationConfig eg_config;  // controls num-frames,
                                        // left/right-context, etc.
    discriminative::SplitDiscriminativeSupervisionOptions splitter_config;

    ParseOptions po(usage);

    eg_config.Register(&po);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format (recommended)");
    po.Register("ivectors", &online_ivector_rspecifier, "Alias for --online-ivectors "
                "option, for back compatibility");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier of ivector "
                "features, as a matrix.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of frames "
                "between iVectors in matrices supplied to the --online-ivectors "
                "option");
    po.Register("length-tolerance", &length_tolerance, "Tolerance for "
                "difference in num-frames between feat and ivector matrices");

    splitter_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    eg_config.ComputeDerived();
    UtteranceSplitter utt_splitter(eg_config);

    std::string model_wxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        den_lat_rspecifier = po.GetArg(3),
        num_ali_rspecifier = po.GetArg(4),
        examples_wspecifier = po.GetArg(5);


    TransitionModel tmodel;
    {
      bool binary;
      Input ki(model_wxfilename, &binary);
      tmodel.Read(ki.Stream(), binary);
    }

    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessLatticeReader den_lat_reader(den_lat_rspecifier);
    RandomAccessInt32VectorReader ali_reader(num_ali_rspecifier);
    NnetDiscriminativeExampleWriter example_writer(examples_wspecifier);
    RandomAccessBaseFloatMatrixReader online_ivector_reader(
        online_ivector_rspecifier);

    int32 num_err = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (!den_lat_reader.HasKey(key)) {
        KALDI_WARN << "No denominator lattice for key " << key;
        num_err++;
      } else if (!ali_reader.HasKey(key)) {
        KALDI_WARN << "No numerator alignment for key " << key;
        num_err++;
      } else {
        discriminative::DiscriminativeSupervision supervision;
        if (!supervision.Initialize(ali_reader.Value(key),
                                    den_lat_reader.Value(key),
                                    1.0)) {
          KALDI_WARN << "Failed to convert lattice to supervision "
                     << "for utterance " << key;
          num_err++;
          continue;
        }

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
        if (!ProcessFile(splitter_config, tmodel,
                         feats, online_ivector_feats, online_ivector_period,
                         supervision, key, compress,
                         &utt_splitter, &example_writer))
          num_err++;
      }
    }
    if (num_err > 0)
      KALDI_WARN << num_err << " utterances had errors and could "
          "not be processed.";
    // utt_splitter prints diagnostics.
    return utt_splitter.ExitStatus();
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
