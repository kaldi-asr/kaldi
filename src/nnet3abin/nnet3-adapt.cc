// nnet3abin/nnet3-adapt.cc

// Copyright 2018   Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet3/nnet-nnet.h"
#include "hmm/transition-model.h"
#include "adapt/differentiable-transform-itf.h"
#include "nnet3a/nnet-chaina-training.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::differentiable_transform;
    typedef kaldi::int32 int32;

    const char *usage =
        "This binary supports various modes that manipulate transform objects for\n"
        "the nnet3a/chaina adaptation framework.  See patterns below\n"
        "\n"
        "Usage:  nnet3-adapt [options] init <config-file-in> [<tree-map-in>] <transform-out>\n"
        "(e.g.:  nnet3-adapt --num-classes=201 init init.aconfig  0.ada)\n"
        "  or:   nnet3-adapt init init.aconfig tree.map 0.ada\n"
        "  or:  nnet3-adapt [options] copy <transform-in> <transform-out>\n"
        "(e.g.:  nnet3-adapt copy --binary=false 0.ada 0.txt)\n"
        "   or:  nnet3-adapt info <transform-in>\n"
        "(e.g.:  nnet3-adapt info 0.ada\n"
        "   or:  nnet3-adapt estimate <transform1-in> <transform2-in> ... <transform-out> \n"
        "    .. which sums stats and calls Estimate(), to get the final class-dependent means... \n"
        "(e.g.   nnet3-adapt estimate foo/final/default.{1,2,3,4,5,6}.ada foo/final/default.ada\n"
        "   or:  nnet3-adapt [options] get-transforms <transform-in> <spk2utt-rspecifier> <feats-in> <posteriors-in> <transforms-out>\n"
        "  ... which estimates and dumps speaker-specific transforms as matrices, which\n"
        "      could be applied to the features with transform-feats; if you want\n"
        "      utterance-specific transforms, make spk2utt a one-to-one map.\n"
        "      <transforms-out> is a wspecifier where matrices will be written.\n"
        "(e.g.:  nnet3-adapt final.ada spk2utt ark:- ark:feats.scp ark:1.trans)\n"
        "\n"
        "See also: nnet3-chaina-train\n";

    bool binary_write = true;
    bool remove_pdf_map = false;
    int32 num_classes = -1;
    int32 iter = 0;
    int32 frame_subsampling_factor = 1;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("num-classes", &num_classes,
                "For 'init' command: number of classes the transform will "
                "use (required if <tree-map> is not supplied).");
    po.Register("remove-pdf-map", &remove_pdf_map,
                "For the 'copy' command: if true, the pdf_map will be "
                "removed so that the transform will be based on "
                "pdf-ids.");
    po.Register("iter", &iter, "Only for the 'estimate' command: iteration "
                "of estimation, will always be 0 in most setups.");
    po.Register("frame-subsampling-factor", &frame_subsampling_factor,
                "Factor by which the posteriors we read are subsampled relative "
                "to the features (only for the get-transforms command). "
                "Will correspond to the top-subsampling-factor,"
                "which, in chaina scripts, refers to frame_subsampling_factor "
                "divided by bottom_subsampling_factor");

    po.Read(argc, argv);


    if (po.GetOptArg(1) == "init" && po.NumArgs() == 3) {
      // This block does the "init" command where the tree.map was not provided.
      if (num_classes <= 0)
        KALDI_ERR << "The --num-classes option is required with the "
            "'init' command.";
      std::string config_rxfilename = po.GetArg(2),
          transform_wxfilename = po.GetArg(3);
      bool binary_in;  // should be false.
      Input ki(config_rxfilename, &binary_in);
      DifferentiableTransformMapped transform;

      transform.transform = DifferentiableTransform::ReadFromConfig(
          ki.Stream(), num_classes);

      WriteKaldiObject(transform, transform_wxfilename, binary_write);
      return 0;
    } else if (po.GetOptArg(1) == "init" && po.NumArgs() == 4) {
      // This block does the "init" command where the tree.map was provided.
      std::string config_rxfilename = po.GetArg(2),
          tree_map_rxfilename = po.GetArg(3),
          transform_wxfilename = po.GetArg(4);

      DifferentiableTransformMapped transform;
      { // This block reads transform.pdf_map and sets up num_classes.
        bool binary_in;
        Input ki(tree_map_rxfilename, &binary_in);
        ReadIntegerVector(ki.Stream(), binary_in, &(transform.pdf_map));
        if (transform.pdf_map.empty())
          KALDI_ERR << "Expected <tree-map> to be nonempty vector.";
        int32 expected_num_classes =
            1 + *std::max_element(transform.pdf_map.begin(),
                                  transform.pdf_map.end());
        if (num_classes > 0 && num_classes != expected_num_classes)
          KALDI_ERR << "The --num-classes given via the option " << num_classes
                    << " differs from the expected value given the tree-map: "
                    << expected_num_classes;
        num_classes = expected_num_classes;
      }

      bool binary_in;  // should be false.
      Input ki(config_rxfilename, &binary_in);
      transform.transform = DifferentiableTransform::ReadFromConfig(
          ki.Stream(), num_classes);
      WriteKaldiObject(transform, transform_wxfilename, binary_write);
      return 0;
    } else if (po.GetOptArg(1) == "info" && po.NumArgs() == 2) {
      std::string transform_rxfilename = po.GetArg(2);
      DifferentiableTransformMapped transform;
      ReadKaldiObject(transform_rxfilename, &transform);
      std::cout << transform.Info();
      return 0;
    } else if (po.GetOptArg(1) == "copy" && po.NumArgs() == 3) {
      std::string transform_rxfilename = po.GetArg(2),
          transform_wxfilename = po.GetArg(3);
      DifferentiableTransformMapped transform;
      ReadKaldiObject(transform_rxfilename, &transform);
      if (remove_pdf_map) {
        if (transform.pdf_map.empty()) {
          KALDI_WARN << "--remove-pdf-map option: transform does not have a pdf-map.";
        } else {
          transform.transform->SetNumClasses(transform.pdf_map.size());
          transform.pdf_map.clear();
        }
      }
      WriteKaldiObject(transform, transform_wxfilename, binary_write);
      return 0;
    } else if (po.GetOptArg(1) == "estimate" && po.NumArgs() >= 3) {
      DifferentiableTransformMapped transform;
      std::string transform_rxfilename = po.GetArg(2);
      ReadKaldiObject(transform_rxfilename, &transform);
      for (int32 i = 3; i < po.NumArgs(); i++) {
        std::string other_transform_rxfilename = po.GetArg(i);
        DifferentiableTransformMapped other_transform;
        ReadKaldiObject(other_transform_rxfilename, &other_transform);
        // sum the stats.
        transform.transform->Add(*(other_transform.transform));
      }
      transform.transform->Estimate(iter);
      std::string transform_wxfilename = po.GetArg(po.NumArgs());
      WriteKaldiObject(transform, transform_wxfilename, binary_write);
      return 0;
    } else if (po.GetOptArg(1) == "get-transforms" && po.NumArgs() == 6) {
      std::string transform_rxfilename = po.GetArg(2),
          spk2utt_rspecifier = po.GetArg(3),
          feats_rspecifier = po.GetArg(4),
          post_rspecifier = po.GetArg(5),
          transforms_wspecifier = po.GetArg(6);

      DifferentiableTransformMapped transform;
      ReadKaldiObject(transform_rxfilename, &transform);
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessPosteriorReader post_reader(post_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feats_rspecifier);
      BaseFloatMatrixWriter transform_writer(transforms_wspecifier);
      int32 num_done = 0, num_no_post = 0, num_other_error = 0;

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        std::unique_ptr <SpeakerStatsItf> stats(
            transform.transform->GetEmptySpeakerStats());
        std::string spk = spk2utt_reader.Key();
        bool got_stats = false;
        const std::vector<std::string> &uttlist = spk2utt_reader.Value();
        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            num_other_error++;
            continue;
          }
          if (!post_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt;
            num_no_post++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const Posterior &post_in = post_reader.Value(utt);
          Posterior post_upsampled(feats.NumRows());
          const Posterior *post_to_use = NULL;
          if (frame_subsampling_factor != 1 || !transform.pdf_map.empty()) {
            ConvertPosterior(
                post_in, 1, 0, frame_subsampling_factor, transform.pdf_map,
                transform.transform->NumClasses(), &post_upsampled);
            post_to_use = &post_upsampled;
          } else {
            KALDI_ASSERT(post_in.size() == size_t(feats.NumRows()) &&
                         "Mismatch in posterior vs. feats dimension");
            post_to_use = &post_in;
          }
          transform.transform->TestingAccumulate(feats, *post_to_use, stats.get());
          got_stats = true;
          num_done++;
        }
        if (!got_stats) {
          KALDI_WARN << "Got no stats for speaker " << spk;
        } else {
          stats->Estimate();
          int32 dim = transform.transform->Dim();
          Matrix<BaseFloat> transform_mat(dim, dim + 1, kUndefined);
          transform.transform->GetTransformAsMatrix(*stats, &transform_mat);
          transform_writer.Write(spk, transform_mat);
        }
      }
      KALDI_LOG << "Done " << num_done << " files, " << num_no_post
                << " with no posts, " << num_other_error << " with other errors.";
      return (num_done != 0 && num_done > (num_no_post + num_other_error)) ? 0 : 1;
    } else {
      po.PrintUsage();
      exit(1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


/*
Test script:

cat <<EOF | nnet3-adapt --binary=false --num-classes=200 init - foo.ada
AppendTransform num-transforms=4
  NoOpTransform dim=20
  FmllrTransform dim=20
  MeanOnlyTransform dim=20
  SequenceTransform num-transforms=2
    FmllrTransform dim=20
    MeanOnlyTransform dim=20
EOF
nnet3-adapt --binary=false --num-classes=400 copy foo.ada -


cat <<EOF | nnet3-adapt --binary=false --num-classes=200 init - - | nnet3-adapt info -
AppendTransform num-transforms=4
  NoOpTransform dim=20
  FmllrTransform dim=20
  MeanOnlyTransform dim=20
  SequenceTransform num-transforms=2
    FmllrTransform dim=20
    MeanOnlyTransform dim=20
EOF

 */
