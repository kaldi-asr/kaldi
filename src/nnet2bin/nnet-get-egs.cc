// nnet2bin/nnet-get-egs.cc

// Copyright 2012-2013  Johns Hopkins University (author:  Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet2/nnet-randomize.h"

namespace kaldi {
namespace nnet2 {

// returns an integer randomly drawn with expected value "expected_count"
// (will be either floor(expected_count) or ceil(expected_count)).
// this will go into an infinite loop if expected_count is very huge, but
// it should never be that huge.
// In the normal case, "expected_count" will be between zero and one.
int32 GetCount(double expected_count) {
  KALDI_ASSERT(expected_count >= 0.0);
  int32 ans = 0;
  while (expected_count > 1.0) {
    ans++;
    expected_count--;
  }
  if (WithProb(expected_count))
    ans++;
  return ans;
}

static void ProcessFile(const MatrixBase<BaseFloat> &feats,
                        const Posterior &pdf_post,
                        const Vector<BaseFloat> &spk_info,
                        int32 left_context,
                        int32 right_context,
                        BaseFloat keep_proportion,
                        int64 *num_frames_written,
                        NnetExampleWriter *example_writer) {
  KALDI_ASSERT(feats.NumRows() == static_cast<int32>(pdf_post.size()));
  NnetExample eg;
  Matrix<BaseFloat> input_frames(left_context + 1 + right_context,
                                 feats.NumCols());
  eg.left_context = left_context;
  eg.spk_info = spk_info;
  for (int32 i = 0; i < feats.NumRows(); i++) {
    int32 count = GetCount(keep_proportion); // number of times
    // we'll write this out (1 by default).
    if (count > 0) {
      // Set up "input_frames".
      for (int32 j = -left_context; j <= right_context; j++) {
        int32 j2 = j + i;
        if (j2 < 0) j2 = 0;
        if (j2 >= feats.NumRows()) j2 = feats.NumRows() - 1;
        SubVector<BaseFloat> src(feats, j2), dest(input_frames,
                                                  j + left_context);
        dest.CopyFromVec(src);
      }
      eg.labels = pdf_post[i];
      eg.input_frames = input_frames;
      std::ostringstream os;
      os << ((*num_frames_written)++);
      std::string key = os.str(); // key in the archive is the number of the
      // example.

      for (int32 c = 0; c < count; c++)
        example_writer->Write(key, eg);
    }
  }
}


} // namespace nnet2
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get frame-by-frame examples of data for neural network training.\n"
        "Essentially this is a format change from features and posteriors\n"
        "into a special frame-by-frame format.  To split randomly into\n"
        "different subsets, do nnet-copy-egs with --random=true, but\n"
        "note that this does not randomize the order of frames.\n"
        "Also see nnet-randomize-frames, which uses more memory but also\n"
        "randomizes the order\n"
        "\n"
        "Usage:  nnet-get-egs [options] <features-rspecifier> "
        "<pdf-post-rspecifier> <training-examples-out>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet-get-egs --left-context=8 --right-context=8 \"$feats\" \\\n"
        "  \"ark:gunzip -c exp/nnet/ali.1.gz | ali-to-pdf exp/nnet/1.nnet ark:- ark:- | ali-to-post ark:- ark:- |\" \\\n"
        "   ark:- \n"
        "Note: the --left-context and --right-context would be derived from\n"
        "the output of nnet-info.";
        
    
    int32 left_context = 0, right_context = 0;
    int32 srand_seed = 0;
    BaseFloat keep_proportion = 1.0;
    
    std::string spk_vecs_rspecifier, utt2spk_rspecifier;
    
    ParseOptions po(usage);
    po.Register("spk-vecs", &spk_vecs_rspecifier, "Rspecifier for speaker vectors");
    po.Register("utt2spk", &utt2spk_rspecifier, "Rspecifier for "
                "speaker-to-utterance map (relevant if --spk-vecs option used)");
    po.Register("left-context", &left_context, "Number of frames of left context "
                "the neural net requires.");
    po.Register("right-context", &right_context, "Number of frames of right context "
                "the neural net requires.");
    po.Register("keep-proportion", &keep_proportion, "If <1.0, this program will "
                "randomly keep this proportion of the input samples.  If >1.0, it will "
                "in expectation copy a sample this many times.  It will copy it a number "
                "of times equal to floor(keep-proportion) or ceil(keep-proportion).");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(only relevant if --keep-proportion != 1.0)");
    
    po.Read(argc, argv);

    srand(srand_seed);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        pdf_post_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    // Read in all the training files.
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessPosteriorReader pdf_post_reader(pdf_post_rspecifier);
    RandomAccessBaseFloatVectorReaderMapped vecs_reader(
        spk_vecs_rspecifier, utt2spk_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    
    int32 num_done = 0, num_err = 0;
    int32 spk_dim = -1;
    int64 num_frames_written = 0;
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (!pdf_post_reader.HasKey(key)) {
        KALDI_WARN << "No pdf-level posterior for key " << key;
        num_err++;
      } else {
        const Posterior &pdf_post = pdf_post_reader.Value(key);
        if (pdf_post.size() != feats.NumRows()) {
          KALDI_WARN << "Posterior has wrong size " << pdf_post.size()
                     << " versus " << feats.NumRows();
          num_err++;
          continue;
        }
        Vector<BaseFloat> spk_info;
        
        if (spk_vecs_rspecifier != "") {
          if (!vecs_reader.HasKey(key)) {
            KALDI_WARN << "No speaker vector for key " << key;
            num_err++;
            continue;
          } else {
            spk_info = vecs_reader.Value(key);
          }
          if (spk_dim == -1) spk_dim = spk_info.Dim();
          else if (spk_info.Dim() != spk_dim) {
            KALDI_WARN << "Invalid dimension of speaker vector, "
                << spk_info.Dim() << " (expected "
                << spk_dim << " ).";
            num_err++;
            continue;
          }
        }
        ProcessFile(feats, pdf_post, spk_info,
                    left_context, right_context, keep_proportion,
                    &num_frames_written, &example_writer);
        num_done++;
      }
    }

    KALDI_LOG << "Finished generating examples, "
              << "successfully processed " << num_done
              << " feature files, wrote " << num_frames_written << " examples, "
              << num_err << " files had errors.";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
