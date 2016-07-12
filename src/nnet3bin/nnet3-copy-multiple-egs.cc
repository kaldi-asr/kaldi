// nnet3bin/nnet3-copy-multiple-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//                2014  Vimal Manohar
//                2016  Pegah Ghahremani

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
#include <numeric>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {


// rename io-name of eg w.r.t io_names e.g. input/input-1,output/output-1
// input name rename to input-1 and output name rename to output-1.
void RenameIoNames(const std::string &io_names,
                   NnetExample *modified_eg) {
  std::vector<std::string> separated_io_names;
  SplitStringToVector(io_names, ",", true, &separated_io_names);
  int32 num_modified_io = separated_io_names.size(),
   io_size = modified_eg->io.size();
  std::vector<std::string> orig_io_list;
  for (int32 io_ind = 0; io_ind < io_size; io_ind++)
    orig_io_list.push_back(modified_eg->io[io_ind].name);
  
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
    modified_eg->io[rename_io_ind].name = rename_io_name[1];            
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
        "network training, possibly changing the binary mode.  Supports multiple rspecifier and wspecifiers, in\n"
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
      rename_io_names = "";
    int32 num_inputs = 1;
    std::string input_prob_str = "";
    ParseOptions po(usage);
    po.Register("read-input-prob", &input_prob_str,
                "Comma-separated probality. If non-empty, "
                "it will read frames from input archives"
                "randomly with this probability distribution.");
    po.Register("random", &random, "If true, will write frames to output "
                "archives randomly, not round-robin.");
    po.Register("num-input-egs", &num_inputs, "If num_inputs > 1, it will read "
                " num_inputs input examples.");
    po.Register("rename-io-names", &rename_io_names, "Comma-separated list of io-node "
                ", need to rename to new names. e.g. io1/io1-new,io2/io2-new");
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


    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::vector<SequentialNnetExampleReader*> example_readers(num_inputs);
    for (int32 i = 0; i < num_inputs; i++) 
      example_readers[i] = new SequentialNnetExampleReader(po.GetArg(i+1));

    int32 num_outputs = po.NumArgs() - num_inputs;
    std::vector<NnetExampleWriter*> example_writers(num_outputs);
    for (int32 i = 0; i < num_outputs; i++)
      example_writers[i] = new NnetExampleWriter(po.GetArg(i+num_inputs+1));

    std::vector<BaseFloat> input_probs;
    bool use_rand_read = false;
    if (!input_prob_str.empty()) {
       SplitStringToFloats(input_prob_str, ",", true, &input_probs);
       KALDI_ASSERT(input_probs.size() == num_inputs);
       use_rand_read = true;
    }
    int64 num_read = 0, num_written = 0;

    if (use_rand_read) {
      // If true, the input archives reading has not finished. 
      bool not_finished = true;
      int32 num_archives_done = 0;
      while (not_finished) { 
        int32 input_archive_num = RandIntDiscreteDist(input_probs);
        if (!example_readers[input_archive_num]->Done()) {
          // count is normally 1; could be 0, or possibly >1.
          int32 count = GetCount(keep_proportion);
          const std::string &key = example_readers[input_archive_num]->Key();

          std::ostringstream os;
          os << input_archive_num;
          std::string modified_key = key;
          modified_key += "-arch-" + os.str(); 
          const NnetExample &eg = example_readers[input_archive_num]->Value();
          for (int32 c = 0; c < count; c++) {
            int32 index = (random ? Rand() : num_written) % num_outputs;
            if (frame_str == "" && left_context == -1 && right_context == -1 &&
                frame_shift == 0) {
              if (!rename_io_names.empty()) {
                NnetExample modified_eg = eg;
                RenameIoNames(rename_io_names, &modified_eg);
                example_writers[index]->Write(modified_key, modified_eg);
              } else {
                example_writers[index]->Write(modified_key, eg);
              }
              num_written++;
            } else { // the --frame option or context options were set.
              NnetExample eg_modified;
              if (SelectFromExample(eg, frame_str, left_context, right_context,
                                    frame_shift, &eg_modified)) {
                if (!rename_io_names.empty()) 
                  RenameIoNames(rename_io_names, &eg_modified);
                // This branch of the if statement will almost always be taken (should only
                // not be taken for shorter-than-normal egs from the end of a file.
                std::ostringstream os;
                os << input_archive_num;
                modified_key += "-arch-" + os.str(); 
                example_writers[index]->Write(modified_key, eg_modified);
                num_written++;
              }
            }
          }
          num_read++;
          example_readers[input_archive_num]->Next();
        } else {
          // make the probability of reading frame from this archive
          // to be zero
          num_archives_done++;
          input_probs[input_archive_num] = 0.0;
          BaseFloat prob_sum = std::accumulate(input_probs.begin(),
            input_probs.end(), 0.0);
          if (num_archives_done < num_inputs) { // if all archives finishes, prob_sum gets zero.
            not_finished = false;
          } else {
            // normalize the probs for remaining archives to sum to 1.
            for (int32 i = 0; i < num_inputs; i++) 
              input_probs[i] /= prob_sum; 
          } 
        }
      }
    } else {
      // read input archives sequentially.
      for (int32 in = 0; in < num_inputs; in++) {
        SequentialNnetExampleReader *example_reader = example_readers[in];
        for (; !example_reader->Done(); example_reader->Next(), num_read++) {

          // count is normally 1; could be 0, or possibly >1.
          int32 count = GetCount(keep_proportion);
          std::string key = example_reader->Key();
          std::ostringstream os;
          os << in;
          //key += "-arch-" + os.str();
          const NnetExample &eg = example_reader->Value();
          for (int32 c = 0; c < count; c++) {
            int32 index = (random ? Rand() : num_written) % num_outputs;
            if (frame_str == "" && left_context == -1 && right_context == -1 &&
                frame_shift == 0) {
              if (!rename_io_names.empty()) {
                NnetExample modified_eg = eg;
                RenameIoNames(rename_io_names, &modified_eg);
              
                example_writers[index]->Write(key, modified_eg);
              } else {
                example_writers[index]->Write(key, eg);
              }
              num_written++;
            } else { // the --frame option or context options were set.
              NnetExample eg_modified;
              if (SelectFromExample(eg, frame_str, left_context, right_context,
                                    frame_shift, &eg_modified)) {
                if (!rename_io_names.empty()) 
                  RenameIoNames(rename_io_names, &eg_modified);
                // this branch of the if statement will almost always be taken (should only
                // not be taken for shorter-than-normal egs from the end of a file.
               example_writers[index]->Write(key, eg_modified);
                num_written++;
              }
            }
          }
        }
      }
    }
    for (int32 i = 0; i < num_inputs; i++)
      delete example_readers[i];

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


