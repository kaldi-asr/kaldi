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
// rename name of NnetIo with old_name to new_name.
void RenameIoNames(const std::string &old_name,
                   const std::string &new_name,
                   NnetExample *eg_modified) {
  // list of io-names in eg_modified.
  std::vector<std::string> orig_io_list;
  int32 io_size = eg_modified->io.size();
  for (int32 io_ind = 0; io_ind < io_size; io_ind++)
    orig_io_list.push_back(eg_modified->io[io_ind].name);

  // find the io in eg with name 'old_name'.
  int32 rename_io_ind =
     std::find(orig_io_list.begin(), orig_io_list.end(), old_name) -
      orig_io_list.begin();

  if (rename_io_ind >= io_size)
    KALDI_ERR << "No io-node with name " << old_name
              << "exists in eg.";
  eg_modified->io[rename_io_ind].name = new_name;
}

// ranames NnetIo name with name 'output' to new_output_name
// and scales the supervision for 'output' using weight.
void ScaleAndRenameOutput(BaseFloat weight,
                          const std::string &new_output_name,
                          NnetExample *eg) {
  // scale the supervision weight for egs
  for (int32 i = 0; i < eg->io.size(); i++)
    if (eg->io[i].name == "output")
      if (weight != 0.0 && weight != 1.0)
        eg->io[i].features.Scale(weight);
  // rename output io name to 'new_output_name'.
  RenameIoNames("output", new_output_name, eg);
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

/** Returns true if the "eg" contains just a single example, meaning
    that all the "n" values in the indexes are zero, and the example
    has NnetIo members named both "input" and "output"

    Also computes the minimum and maximum "t" values in the "input" and
    "output" NnetIo members.
 */
bool ContainsSingleExample(const NnetExample &eg,
                           int32 *min_input_t,
                           int32 *max_input_t,
                           int32 *min_output_t,
                           int32 *max_output_t) {
  bool done_input = false, done_output = false;
  int32 num_indexes = eg.io.size();
  for (int32 i = 0; i < num_indexes; i++) {
    const NnetIo &io = eg.io[i];
    std::vector<Index>::const_iterator iter = io.indexes.begin(),
                                        end = io.indexes.end();
    // Should not have an empty input/output type.
    KALDI_ASSERT(!io.indexes.empty());
    if (io.name == "input" || io.name == "output") {
      int32 min_t = iter->t, max_t = iter->t;
      for (; iter != end; ++iter) {
        int32 this_t = iter->t;
        min_t = std::min(min_t, this_t);
        max_t = std::max(max_t, this_t);
        if (iter->n != 0) {
          KALDI_WARN << "Example does not contain just a single example; "
                     << "too late to do frame selection or reduce context.";
          return false;
        }
      }
      if (io.name == "input") {
        done_input = true;
        *min_input_t = min_t;
        *max_input_t = max_t;
      } else {
        KALDI_ASSERT(io.name == "output");
        done_output = true;
        *min_output_t = min_t;
        *max_output_t = max_t;
      }
    } else {
      for (; iter != end; ++iter) {
        if (iter->n != 0) {
          KALDI_WARN << "Example does not contain just a single example; "
                     << "too late to do frame selection or reduce context.";
          return false;
        }
      }
    }
  }
  if (!done_input) {
    KALDI_WARN << "Example does not have any input named 'input'";
    return false;
  }
  if (!done_output) {
    KALDI_WARN << "Example does not have any output named 'output'";
    return false;
  }
  return true;
}

/**
   This function filters the indexes (and associated feature rows) in a
   NnetExample, removing any index/row in an NnetIo named "input" with t <
   min_input_t or t > max_input_t and any index/row in an NnetIo named "output" with t <
   min_output_t or t > max_output_t.
   Will crash if filtering removes all Indexes of "input" or "output".
 */
void FilterExample(const NnetExample &eg,
                   int32 min_input_t,
                   int32 max_input_t,
                   int32 min_output_t,
                   int32 max_output_t,
                   NnetExample *eg_out) {
  eg_out->io.clear();
  eg_out->io.resize(eg.io.size());
  for (size_t i = 0; i < eg.io.size(); i++) {
    bool is_input_or_output;
    int32 min_t, max_t;
    const NnetIo &io_in = eg.io[i];
    NnetIo &io_out = eg_out->io[i];
    const std::string &name = io_in.name;
    io_out.name = name;
    if (name == "input") {
      min_t = min_input_t;
      max_t = max_input_t;
      is_input_or_output = true;
    } else if (name == "output") {
      min_t = min_output_t;
      max_t = max_output_t;
      is_input_or_output = true;
    } else {
      is_input_or_output = false;
    }
    if (!is_input_or_output) {  // Just copy everything.
      io_out.indexes = io_in.indexes;
      io_out.features = io_in.features;
    } else {
      const std::vector<Index> &indexes_in = io_in.indexes;
      std::vector<Index> &indexes_out = io_out.indexes;
      indexes_out.reserve(indexes_in.size());
      int32 num_indexes = indexes_in.size(), num_kept = 0;
      KALDI_ASSERT(io_in.features.NumRows() == num_indexes);
      std::vector<bool> keep(num_indexes, false);
      std::vector<Index>::const_iterator iter_in = indexes_in.begin(),
                                          end_in = indexes_in.end();
      std::vector<bool>::iterator iter_out = keep.begin();
      for (; iter_in != end_in; ++iter_in,++iter_out) {
        int32 t = iter_in->t;
        bool is_within_range = (t >= min_t && t <= max_t);
        *iter_out = is_within_range;
        if (is_within_range) {
          indexes_out.push_back(*iter_in);
          num_kept++;
        }
      }
      KALDI_ASSERT(iter_out == keep.end());
      if (num_kept == 0)
        KALDI_ERR << "FilterExample removed all indexes for '" << name << "'";

      FilterGeneralMatrixRows(io_in.features, keep,
                              &io_out.features);
      KALDI_ASSERT(io_out.features.NumRows() == num_kept &&
                   indexes_out.size() == static_cast<size_t>(num_kept));
    }
  }
}


/**
   This function is responsible for possibly selecting one frame from multiple
   supervised frames, and reducing the left and right context as specified.  If
   frame == "" it does not reduce the supervised frames; if frame == "random" it
   selects one random frame; otherwise it expects frame to be an integer, and
   will select only the output with that frame index (or return false if there was
   no such output).

   If left_context != -1 it removes any inputs with t < (smallest output - left_context).
      If left_context != -1 it removes any inputs with t < (smallest output - left_context).

   It returns true if it was able to select a frame.  We only anticipate it ever
   returning false in situations where frame is an integer, and the eg came from
   the end of a file and has a smaller than normal number of supervised frames.

*/
bool SelectFromExample(const NnetExample &eg,
                       std::string frame_str,
                       int32 left_context,
                       int32 right_context,
                       int32 frame_shift,
                       NnetExample *eg_out) {
  static bool warned_left = false, warned_right = false;
  int32 min_input_t, max_input_t,
      min_output_t, max_output_t;
  if (!ContainsSingleExample(eg, &min_input_t, &max_input_t,
                             &min_output_t, &max_output_t))
    KALDI_ERR << "Too late to perform frame selection/context reduction on "
              << "these examples (already merged?)";
  if (frame_str != "") {
    // select one frame.
    if (frame_str == "random") {
      min_output_t = max_output_t = RandInt(min_output_t,
                                                          max_output_t);
    } else {
      int32 frame;
      if (!ConvertStringToInteger(frame_str, &frame))
        KALDI_ERR << "Invalid option --frame='" << frame_str << "'";
      if (frame < min_output_t || frame > max_output_t) {
        // Frame is out of range.  Should happen only rarely.  Calling code
        // makes sure of this.
        return false;
      }
      min_output_t = max_output_t = frame;
    }
  }
  if (left_context != -1) {
    if (!warned_left && min_input_t > min_output_t - left_context) {
      warned_left = true;
      KALDI_WARN << "You requested --left-context=" << left_context
                 << ", but example only has left-context of "
                 <<  (min_output_t - min_input_t)
                 << " (will warn only once; this may be harmless if "
          "using any --*left-context-initial options)";
    }
    min_input_t = std::max(min_input_t, min_output_t - left_context);
  }
  if (right_context != -1) {
    if (!warned_right && max_input_t < max_output_t + right_context) {
      warned_right = true;
      KALDI_WARN << "You requested --right-context=" << right_context
                << ", but example only has right-context of "
                <<  (max_input_t - max_output_t)
                 << " (will warn only once; this may be harmless if "
            "using any --*right-context-final options.";
    }
    max_input_t = std::min(max_input_t, max_output_t + right_context);
  }
  FilterExample(eg,
                min_input_t, max_input_t,
                min_output_t, max_output_t,
                eg_out);
  if (frame_shift != 0) {
    std::vector<std::string> exclude_names;  // we can later make this
    exclude_names.push_back(std::string("ivector")); // configurable.
    ShiftExampleTimes(frame_shift, exclude_names, eg_out);
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
        "Copy examples (single frames or fixed-size groups of frames) for neural\n"
        "network training, possibly changing the binary mode.  Supports multiple wspecifiers, in\n"
        "which case it will write the examples round-robin to the outputs.\n"
        "\n"
        "Usage:  nnet3-copy-egs [options] <egs-rspecifier> <egs-wspecifier1> [<egs-wspecifier2> ...]\n"
        "\n"
        "e.g.\n"
        "nnet3-copy-egs ark:train.egs ark,t:text.egs\n"
        "or:\n"
        "nnet3-copy-egs ark:train.egs ark:1.egs ark:2.egs\n"
        "See also: nnet3-subset-egs, nnet3-get-egs, nnet3-merge-egs, nnet3-shuffle-egs\n";

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
      eg_weight_rspecifier, eg_output_rspecifier;

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
    po.Register("weights", &eg_weight_rspecifier,
                "Rspecifier indexed by the key of egs, providing a weight by "
                "which we will scale the supervision matrix for that eg. "
                "Used in multilingual training.");
    po.Register("outputs", &eg_output_rspecifier,
                "Rspecifier indexed by the key of egs, providing a string-valued "
                "output name, e.g. 'output-0'.  If provided, the NnetIo with "
                "name 'output' will be renamed to the provided name. Used in "
                "multilingual training.");

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1);

    SequentialNnetExampleReader example_reader(examples_rspecifier);

    RandomAccessTokenReader output_reader(eg_output_rspecifier);
    RandomAccessBaseFloatReader egs_weight_reader(eg_weight_rspecifier);
    int32 num_outputs = po.NumArgs() - 1;
    std::vector<NnetExampleWriter*> example_writers(num_outputs);
    for (int32 i = 0; i < num_outputs; i++)
      example_writers[i] = new NnetExampleWriter(po.GetArg(i+2));


    int64 num_read = 0, num_written = 0, num_err = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      bool modify_eg_output = !(eg_output_rspecifier.empty() &&
                                eg_weight_rspecifier.empty());
      // count is normally 1; could be 0, or possibly >1.
      int32 count = GetCount(keep_proportion);
      std::string key = example_reader.Key();
      NnetExample eg_modified_output;
      const NnetExample &eg_orig = example_reader.Value(),
        &eg = (modify_eg_output ? eg_modified_output : eg_orig);
      // Note: in the normal case we just use 'eg'; eg_modified_output is
      // for the case when the --outputs or --weights option is specified
      // (only for multilingual training).
      BaseFloat weight = 1.0;
      std::string new_output_name;
      if (modify_eg_output) { // This branch is only taken for multilingual training.
        eg_modified_output = eg_orig;
        if (!eg_weight_rspecifier.empty()) {
          if (!egs_weight_reader.HasKey(key)) {
            KALDI_WARN << "No weight for example key " << key;
            num_err++;
            continue;
          }
          weight = egs_weight_reader.Value(key);
        }
        if (!eg_output_rspecifier.empty()) {
          if (!output_reader.HasKey(key)) {
            KALDI_WARN << "No new output-name for example key " << key;
            num_err++;
            continue;
          }
          new_output_name = output_reader.Value(key);
        }
      }
      for (int32 c = 0; c < count; c++) {
        int32 index = (random ? Rand() : num_written) % num_outputs;
        if (frame_str == "" && left_context == -1 && right_context == -1 &&
            frame_shift == 0) {
          if (modify_eg_output) // Only for multilingual training
            ScaleAndRenameOutput(weight, new_output_name, &eg_modified_output);
          example_writers[index]->Write(key, eg);
          num_written++;
        } else { // the --frame option or context options were set.
          NnetExample eg_modified;
          if (SelectFromExample(eg, frame_str, left_context, right_context,
                                frame_shift, &eg_modified)) {
            if (modify_eg_output)
              ScaleAndRenameOutput(weight, new_output_name, &eg_modified);
            // this branch of the if statement will almost always be taken (should only
            // not be taken for shorter-than-normal egs from the end of a file.
            example_writers[index]->Write(key, eg_modified);
            num_written++;
          }
        }
      }
    }

    for (int32 i = 0; i < num_outputs; i++)
      delete example_writers[i];
    KALDI_LOG << "Read " << num_read << " neural-network training examples, wrote "
              << num_written << ", "
              << num_err <<  " examples had errors.";
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
