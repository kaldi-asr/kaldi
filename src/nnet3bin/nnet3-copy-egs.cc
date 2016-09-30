// nnet3bin/nnet3-copy-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar

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
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {

// rename io-name of eg w.r.t io_names list e.g. input/input-1,output/output-1
// 'input' is renamed to input-1 and 'output' renamed to output-1.
void RenameIoNames(const std::string &io_names,
                   NnetExample *eg_modified) {
  std::vector<std::string> separated_io_names;
  SplitStringToVector(io_names, ",", true, &separated_io_names);
  int32 num_modified_io = separated_io_names.size(),
   io_size = eg_modified->io.size();
  std::vector<std::string> orig_io_list;
  for (int32 io_ind = 0; io_ind < io_size; io_ind++)
    orig_io_list.push_back(eg_modified->io[io_ind].name);
  
  for (int32 ind = 0; ind < num_modified_io; ind++) {
    std::vector<std::string> rename_io_name;
    SplitStringToVector(separated_io_names[ind], "/", true, &rename_io_name);
    // find the io in eg with specific name and rename it to new name.

    int32 rename_io_ind = 
       std::find(orig_io_list.begin(), orig_io_list.end(), rename_io_name[0]) - 
        orig_io_list.begin();

    if (rename_io_ind >= io_size)
      KALDI_ERR << "No io-node with name " << rename_io_name[0]
                << "exists in eg.";
    eg_modified->io[rename_io_ind].name = rename_io_name[1];            
  }
}

// returns an integer randomly drawn with expected value "expected_count"
// (will be either floor(expected_count) or ceil(expected_count)).
int32 GetCount(double expected_count) {
  KALDI_ASSERT(expected_count >= 0.0);
  int32 ans = floor(expected_count);
  expected_count -= ans;
  if (WithProb(expected_count))
    ans++;
  return ans;
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
        "Copy examples (single frames or fixed-size groups of frames) for neural\n"
        "network training, possibly changing the binary mode.  Supports multiple wspecifiers, in\n"
        "which case it will write the examples round-robin to the outputs.\n"
        "\n"
        "Usage:  nnet3-copy-egs [options] <egs-rspecifier> <egs-wspecifier1> [<egs-wspecifier2> ...]\n"
        "\n"
        "e.g.\n"
        "nnet3-copy-egs ark:train.egs ark,t:text.egs\n"
        "or:\n"
        "nnet3-copy-egs ark:train.egs ark:1.egs ark:2.egs\n";

    bool random = false;
    int32 srand_seed = 0;
    int32 frame_shift = 0;
    BaseFloat keep_proportion = 1.0;

    // The following config variables, if set, can be used to extract a single
    // frame of labels from a multi-frame example, and/or to reduce the amount
    // of context.
    int32 left_context = -1, right_context = -1;

    // you can set frame to a number to select a single frame with a particular
    // offset, or to 'random' to select a random single frame.
    std::string frame_str,
      weight_str = "",
      output_str = "";

    ParseOptions po(usage);
    po.Register("random", &random, "If true, will write frames to output "
                "archives randomly, not round-robin.");
    po.Register("frame-shift", &frame_shift, "Allows you to shift time values "
                "in the supervision data (excluding iVector data).  Only really "
                "useful in clockwork topologies (i.e. any topology for which "
                "modulus != 1).  Shifting is done after any frame selection.");
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
    po.Register("weights", &weight_str,
                "Rspecifier maps the output posterior to each example" 
                "If provided, the supervision weight for output is scaled."
                " Scaling supervision weight is the same as scaling to the derivative during training "
                " in case of linear objective."
                "The default is one, which means we are not applying per-example weights.");
    po.Register("outputs", &output_str,
                "Rspecifier maps example to new output in nnet."
                " If provided, the NnetIo with name 'output' in each example "
                " is renamed to new output name.");

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    
    RandomAccessTokenReader output_reader(output_str);
    RandomAccessBaseFloatReader egs_weight_reader(weight_str);

    int32 num_outputs = po.NumArgs() - 1;
    std::vector<NnetExampleWriter*> example_writers(num_outputs);
    for (int32 i = 0; i < num_outputs; i++)
      example_writers[i] = new NnetExampleWriter(po.GetArg(i+2));

    
    int64 num_read = 0, num_written = 0, num_err = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      // count is normally 1; could be 0, or possibly >1.
      int32 count = GetCount(keep_proportion);
      std::string key = example_reader.Key();
      const NnetExample &eg = example_reader.Value();
      for (int32 c = 0; c < count; c++) {
        int32 index = (random ? Rand() : num_written) % num_outputs;
        if (frame_str == "" && left_context == -1 && right_context == -1 &&
            frame_shift == 0) {
          NnetExample eg_modified = eg;
          if (!weight_str.empty()) {
            // scale the supervision weight for egs
            if (!egs_weight_reader.HasKey(key)) {
              KALDI_WARN << "No weight for example key " << key;
              num_err++;
              continue;
            }
            BaseFloat weight = egs_weight_reader.Value(key);
            for (int32 i = 0; i < eg_modified.io.size(); i++) 
              if (eg_modified.io[i].name == "output") 
                eg_modified.io[i].features.Scale(weight);
          }
          if (!output_str.empty()) {
            if (!output_reader.HasKey(key)) {
              KALDI_WARN << "No new output-name for example key " << key;
              num_err++;
              continue;
            }
            std::string new_output_name = output_reader.Value(key);
            // rename output io name to $new_output_name.
            std::string rename_io_names = "output/" + new_output_name;
            RenameIoNames(rename_io_names, &eg_modified);
          }
          example_writers[index]->Write(key, eg_modified);
          num_written++;
        } else { // the --frame option or context options were set.
          NnetExample eg_modified;
          if (SelectFromExample(eg, frame_str, left_context, right_context,
                                frame_shift, &eg_modified)) {
            // this branch of the if statement will almost always be taken (should only
            // not be taken for shorter-than-normal egs from the end of a file.
            if (!weight_str.empty()) {
              // scale the supervision weight for egs
              if (!egs_weight_reader.HasKey(key)) {
                KALDI_WARN << "No weight for example key " << key;
                num_err++;
                continue;
              }
              int32 weight = egs_weight_reader.Value(key);
              for (int32 i = 0; i < eg_modified.io.size(); i++) 
                if (eg_modified.io[i].name == "output") 
                  eg_modified.io[i].features.Scale(weight);
            }
            if (!output_str.empty()) {
              if (!output_reader.HasKey(key)) {
                KALDI_WARN << "No new output-name for example key " << key;
                num_err++;
                continue;
              }
              std::string new_output_name = output_reader.Value(key);
              // rename output io name to $new_output_name.
              std::string rename_io_names = "output/" + new_output_name;
              RenameIoNames(rename_io_names, &eg_modified);
            }
            example_writers[index]->Write(key, eg_modified);
            num_written++;
          }
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


