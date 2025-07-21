// chainbin/nnet3-chain-copy-egs.cc

// Copyright 2012-2015  Johns Hopkins University (author:  Daniel Povey)
//           2014-2017  Vimal Manohar
//                2016  Gaofeng Cheng
//                2017  Pegah Ghahremani
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
#include "nnet3/nnet-chain-example.h"

namespace kaldi {
namespace nnet3 {

// renames outputs named "output" to new_name
void RenameOutputs(const std::string &new_name, NnetChainExample *eg) {
  bool found_output = false;
  for (std::vector<NnetChainSupervision>::iterator it = eg->outputs.begin();
       it != eg->outputs.end(); ++it) {
    if (it->name == "output") {
      it->name = new_name;
      found_output = true;
    }
  }

  if (!found_output)
    KALDI_ERR << "No supervision with name 'output'"
              << "exists in eg.";
}

// scales the supervision for 'output' by a factor of "weight"
void ScaleSupervisionWeight(BaseFloat weight, NnetChainExample *eg) {
  if (weight == 1.0) return;

  bool found_output = false;
  for (std::vector<NnetChainSupervision>::iterator it = eg->outputs.begin();
       it != eg->outputs.end(); ++it) {
    if (it->name == "output") {
      it->supervision.weight *= weight;
      found_output = true;
    }
  }

  if (!found_output)
    KALDI_ERR << "No supervision with name 'output'"
              << "exists in eg.";
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

/**
   This function filters the indexes (and associated feature rows) in a
   NnetExample, removing any index/row in an NnetIo named "input" with t <
   min_input_t or t > max_input_t and any index/row in an NnetIo named "output" with t <
   min_output_t or t > max_output_t.
   Will crash if filtering removes all Indexes of "input" or "output".
 */
void FilterExample(int32 min_input_t,
                   int32 max_input_t,
                   int32 min_output_t,
                   int32 max_output_t,
                   NnetChainExample *eg) {
  // process the <NnetIo> inputs
  for (size_t i = 0; i < eg->inputs.size(); i++) {
    int32 min_t, max_t;
    NnetIo &io = eg->inputs[i];
    if (io.name == "input") {
      min_t = min_input_t;
      max_t = max_input_t;
      
      const std::vector<Index> &indexes_in = io.indexes;
      std::vector<Index> indexes_out;
      indexes_out.reserve(indexes_in.size());
      int32 num_indexes = indexes_in.size(), num_kept = 0;
      KALDI_ASSERT(io.features.NumRows() == num_indexes);
      std::vector<bool> keep(num_indexes, false);
      std::vector<Index>::const_iterator iter_in = indexes_in.begin(),
                                          end_in = indexes_in.end();
      std::vector<bool>::iterator iter_out = keep.begin();
      for (; iter_in != end_in; ++iter_in, ++iter_out) {
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
        KALDI_ERR << "FilterExample removed all indexes for '" << io.name << "'";
      io.indexes = indexes_out;

      GeneralMatrix features_out;
      FilterGeneralMatrixRows(io.features, keep, &features_out);
      io.features = features_out;
      KALDI_ASSERT(io.features.NumRows() == num_kept &&
                   indexes_out.size() == static_cast<size_t>(num_kept));
    }
  }
}


/** Returns true if the "eg" contains just a single example, meaning
    that all the "n" values in the indexes are zero, and the example
    has NnetIo members named both "input" and "output"

    Also computes the minimum and maximum "t" values in the "input" and
    "output" NnetIo members.
 */
bool ContainsSingleExample(const NnetChainExample &eg,
                           int32 *min_input_t,
                           int32 *max_input_t,
                           int32 *min_output_t,
                           int32 *max_output_t) {
  bool done_input = false, done_output = false;
  int32 num_indexes_input = eg.inputs.size();
  int32 num_indexes_output = eg.outputs.size();
  for (int32 i = 0; i < num_indexes_input; i++) {
    const NnetIo &input = eg.inputs[i];
    std::vector<Index>::const_iterator iter = input.indexes.begin(),
                                        end = input.indexes.end();
    // Should not have an empty input/output type.
    KALDI_ASSERT(!input.indexes.empty());
    if (input.name == "input") {
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
      done_input = true;
      *min_input_t = min_t;
      *max_input_t = max_t;
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

  for (int32 i = 0; i < num_indexes_output; i++) {
    const NnetChainSupervision &outputs = eg.outputs[i];
    std::vector<Index>::const_iterator iter = outputs.indexes.begin(),
                                        end = outputs.indexes.end();
    // Should not have an empty input/output type.
    KALDI_ASSERT(!outputs.indexes.empty());
    if (outputs.name == "output") {
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
      done_output = true;
      *min_output_t = min_t;
      *max_output_t = max_t;
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

// calculate the frame_subsampling_factor
void CalculateFrameSubsamplingFactor(const NnetChainExample &eg,
                                     int32 *frame_subsampling_factor) {
  *frame_subsampling_factor = eg.outputs[0].indexes[1].t
                              - eg.outputs[0].indexes[0].t;
}

void ModifyChainExampleContext(int32 left_context,
                               int32 right_context,
                               const int32 frame_subsampling_factor,
                               NnetChainExample *eg) {
  static bool warned_left = false, warned_right = false;
  int32 min_input_t, max_input_t,
        min_output_t, max_output_t;
  if (!ContainsSingleExample(*eg, &min_input_t, &max_input_t,
                             &min_output_t, &max_output_t))
    KALDI_ERR << "Too late to perform frame selection/context reduction on "
              << "these examples (already merged?)";
  if (left_context != -1) {
    int32 observed_left_context = min_output_t - min_input_t;
    if (!warned_left && observed_left_context < left_context) {
      warned_left = true;
      KALDI_WARN << "You requested --left-context=" << left_context
                 << ", but example only has left-context of "
                 <<  observed_left_context
                 << " (will warn only once; this may be harmless if "
          "using any --*left-context-initial options)";
    }
    min_input_t = std::max(min_input_t, min_output_t - left_context);
  }
  if (right_context != -1) {
    int32 observed_right_context = max_input_t - max_output_t;

    if (right_context != -1) {
      if (!warned_right && observed_right_context < right_context) {
        warned_right = true;
        KALDI_WARN << "You requested --right-context=" << right_context
                  << ", but example only has right-context of "
                  << observed_right_context
                 << " (will warn only once; this may be harmless if "
            "using any --*right-context-final options.";
      }
      max_input_t = std::min(max_input_t, max_output_t + right_context);
    }
  }
  FilterExample(min_input_t, max_input_t,
                min_output_t, max_output_t,
                eg);
}  // ModifyChainExampleContext

}  // namespace nnet3
}  // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples for nnet3+chain network training, possibly changing the binary mode.\n"
        "Supports multiple wspecifiers, in which case it will write the examples\n"
        "round-robin to the outputs.\n"
        "\n"
        "Usage:  nnet3-chain-copy-egs [options] <egs-rspecifier> <egs-wspecifier1> [<egs-wspecifier2> ...]\n"
        "\n"
        "e.g.\n"
        "nnet3-chain-copy-egs ark:train.cegs ark,t:text.cegs\n"
        "or:\n"
        "nnet3-chain-copy-egs ark:train.cegs ark:1.cegs ark:2.cegs\n";

    bool random = false;
    int32 srand_seed = 0;
    int32 frame_shift = 0;
    int32 frame_subsampling_factor = -1;
    BaseFloat keep_proportion = 1.0;
    int32 left_context = -1, right_context = -1;
    std::string eg_weight_rspecifier, eg_output_name_rspecifier;

    ParseOptions po(usage);
    po.Register("random", &random, "If true, will write frames to output "
                "archives randomly, not round-robin.");
    po.Register("keep-proportion", &keep_proportion, "If <1.0, this program will "
                "randomly keep this proportion of the input samples.  If >1.0, it will "
                "in expectation copy a sample this many times.  It will copy it a number "
                "of times equal to floor(keep-proportion) or ceil(keep-proportion).");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(only relevant if --random=true or --keep-proportion != 1.0)");
    po.Register("frame-shift", &frame_shift, "Allows you to shift time values "
                "in the supervision data (excluding iVector data) - useful in "
                "augmenting data.  Note, the outputs will remain at the closest "
                "exact multiples of the frame subsampling factor");
    po.Register("left-context", &left_context, "Can be used to truncate the "
                "feature left-context that we output.");
    po.Register("right-context", &right_context, "Can be used to truncate the "
                "feature right-context that we output.");
    po.Register("weights", &eg_weight_rspecifier,
                "Rspecifier indexed by the key of egs, providing a weight by "
                "which we will scale the supervision matrix for that eg. "
                "Used in multilingual training.");
    po.Register("outputs", &eg_output_name_rspecifier,
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

    SequentialNnetChainExampleReader example_reader(examples_rspecifier);

    // In the normal case, these would not be used. These are only applicable
    // for multi-task or multilingual training.
    RandomAccessTokenReader output_name_reader(eg_output_name_rspecifier);
    RandomAccessBaseFloatReader egs_weight_reader(eg_weight_rspecifier);

    int32 num_outputs = po.NumArgs() - 1;
    std::vector<NnetChainExampleWriter*> example_writers(num_outputs);
    for (int32 i = 0; i < num_outputs; i++)
      example_writers[i] = new NnetChainExampleWriter(po.GetArg(i+2));

    std::vector<std::string> exclude_names;  // names we never shift times of;
                                            // not configurable for now.
    exclude_names.push_back(std::string("ivector"));

    int64 num_read = 0, num_written = 0, num_err = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      const std::string &key = example_reader.Key();
      NnetChainExample &eg = example_reader.Value();
      if (frame_subsampling_factor == -1)
        CalculateFrameSubsamplingFactor(eg,
                                        &frame_subsampling_factor);
      // count is normally 1; could be 0, or possibly >1.
      int32 count = GetCount(keep_proportion);

      if (!eg_weight_rspecifier.empty()) {
        BaseFloat weight = 1.0;
        if (!egs_weight_reader.HasKey(key)) {
          KALDI_WARN << "No weight for example key " << key;
          num_err++;
          continue;
        }
        weight = egs_weight_reader.Value(key);
        ScaleSupervisionWeight(weight, &eg);
      }
      
      if (!eg_output_name_rspecifier.empty()) {
        if (!output_name_reader.HasKey(key)) {
          KALDI_WARN << "No new output-name for example key " << key;
          num_err++;
          continue;
        }
        std::string new_output_name = output_name_reader.Value(key);
        RenameOutputs(new_output_name, &eg);
      }
      
      if (frame_shift != 0)
        ShiftChainExampleTimes(frame_shift, exclude_names, &eg);
      if (left_context != -1 || right_context != -1)
        ModifyChainExampleContext(left_context, right_context,
                                  frame_subsampling_factor, &eg);
        
      for (int32 c = 0; c < count; c++) {
        int32 index = (random ? Rand() : num_written) % num_outputs;
        example_writers[index]->Write(key, eg);
        num_written++;
      }
    }
    for (int32 i = 0; i < num_outputs; i++)
      delete example_writers[i];
    KALDI_LOG << "Read " << num_read
              << " neural-network training examples, wrote " << num_written;
    return (num_written == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
