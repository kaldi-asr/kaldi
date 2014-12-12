// nnet2bin/nnet-copy-egs.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)
// Copyright 2014  Vimal Manohar

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
#include "nnet2/nnet-example-functions.h"

namespace kaldi {
namespace nnet2 {
// returns an integer randomly drawn with expected value "expected_count"
// (will be either floor(expected_count) or ceil(expected_count)).
// this will go into an infinite loop if expected_count is very huge, but
// it should never be that huge.
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

} // namespace nnet2
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples (typically single frames) for neural network training,\n"
        "possibly changing the binary mode.  Supports multiple wspecifiers, in\n"
        "which case it will write the examples round-robin to the outputs.\n"
        "\n"
        "Usage:  nnet-copy-egs [options] <egs-rspecifier> <egs-wspecifier1> [<egs-wspecifier2> ...]\n"
        "\n"
        "e.g.\n"
        "nnet-copy-egs ark:train.egs ark,t:text.egs\n"
        "or:\n"
        "nnet-copy-egs ark:train.egs ark:1.egs ark:2.egs\n";
        
    bool random = false;
    int32 srand_seed = 0;
    BaseFloat keep_proportion = 1.0;

    // The following config variables, if set, can be used to extract a single
    // frame of labels from a multi-frame example, and/or to reduce the amount
    // of context.
    int32 left_context = -1, right_context = -1;
    // you can set frame to a number to select a single frame with a particular
    // offset, or to 'random' to select a random single frame.
    std::string frame_str;
    
    ParseOptions po(usage);
    po.Register("random", &random, "If true, will write frames to output "
                "archives randomly, not round-robin.");
    po.Register("keep-proportion", &keep_proportion, "If <1.0, this program will "
                "randomly keep this proportion of the input samples.  If >1.0, it will "
                "in expectation copy a sample this many times.  It will copy it a number "
                "of times equal to floor(keep-proportion) or ceil(keep-proportion).");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(only relevant if --random=true or --keep-proportion != 1.0)");
    po.Register("frame", &frame_str, "This option can be used to select a single "
                "frame from each multi-frame example.  Set to a number 0, 1, etc. "
                "to select a frame with a given index, or 'random' to select a "
                "random frame.");
    po.Register("left-context", &left_context, "Can be used to truncate the "
                "feature left-context that we output.");
    po.Register("right-context", &right_context, "Can be used to truncate the "
                "feature right-context that we output.");

    
    po.Read(argc, argv);

    srand(srand_seed);

    int32 frame = -1;  // -1 means don't do any selection (--frame option unse),
                       // --2 means random selection.
    if (frame_str != "") {
      if (!ConvertStringToInteger(frame_str, &frame)) {
        if (frame_str == "random") {
          frame = -2;
        } else {
          KALDI_ERR << "Invalid --frame option: '" << frame_str << "'";
        }
      } else {
        KALDI_ASSERT(frame >= 0);
      }
    }
    // the following derived variables will be used if the frame, left_context,
    // or right_context options were set (the frame option will be more common).
    bool copy_eg = (frame != -1 || left_context != -1 || right_context != -1);
    int32 start_frame = -1, num_frames = -1;
    if (frame != -1) {  // frame >= 0 or frame == -2 meaning random frame
      num_frames = 1;
      start_frame = frame;  // value will be ignored if frame == -2.
    }
    
    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);

    SequentialNnetExampleReader example_reader(examples_rspecifier);

    int32 num_outputs = po.NumArgs() - 1;
    std::vector<NnetExampleWriter*> example_writers(num_outputs);
    for (int32 i = 0; i < num_outputs; i++)
      example_writers[i] = new NnetExampleWriter(po.GetArg(i+2));

    
    int64 num_read = 0, num_written = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      // count is normally 1; could be 0, or possibly >1.
      int32 count = GetCount(keep_proportion);  
      std::string key = example_reader.Key();
      const NnetExample &eg = example_reader.Value();
      for (int32 c = 0; c < count; c++) {
        int32 index = (random ? Rand() : num_written) % num_outputs;
        if (!copy_eg) {
          example_writers[index]->Write(key, eg);
          num_written++;
        } else { // the --frame option or related options were set.
          if (frame == -2)  // --frame=random was set -> choose random frame
            start_frame = RandInt(0, eg.labels.size() - 1);
          if (start_frame == -1 || start_frame < eg.labels.size()) {
            // note: we'd only reach here with start_frame == -1 if the
            // --left-context or --right-context options were set (reducing
            // context).  -1 means use whatever we had in the original eg.
            NnetExample eg_mod(eg, start_frame, num_frames,
                               left_context, right_context);
            example_writers[index]->Write(key, eg_mod);
            num_written++;            
          }
          // else this frame was out of range for this eg; we don't make this an
          // error, because it can happen for truncated multi-frame egs that
          // were created at the end of an utterance.
        }
      }
    }
    
    for (int32 i = 0; i < num_outputs; i++)
      delete example_writers[i];
    KALDI_LOG << "Read " << num_read << " neural-network training examples, wrote "
              << num_written;
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


