// bin/compare-int-vector.cc

// Copyright 2018  Johns Hopkins University (Author: Daniel Povey)

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
#include "matrix/kaldi-vector.h"
#include "transform/transform-common.h"
#include <iomanip>


namespace kaldi {
void AddToCount(int32 location_to_add,
                double value_to_add,
                std::vector<double> *counts) {
  if (location_to_add < 0)
    KALDI_ERR << "Contents of vectors cannot be "
        "negative if --write-tot-counts or --write-diff-counts "
        "options are provided.";
  if (counts->size() <= static_cast<size_t>(location_to_add))
    counts->resize(location_to_add + 1, 0.0);
  (*counts)[location_to_add] += value_to_add;
}

void AddToConfusionMatrix(int32 phone1, int32 phone2,
                          Matrix<double> *counts) {
  if (phone1 < 0 || phone2 < 0)
    KALDI_ERR << "Contents of vectors cannot be "
        "negative if --write-confusion-matrix option is "
        "provided.";
  int32 max_size = std::max<int32>(phone1, phone2) + 1;
  if (counts->NumRows() < max_size)
    counts->Resize(max_size, max_size, kCopyData);
  (*counts)(phone1, phone2) += 1.0;
}


void WriteAsKaldiVector(const std::vector<double> &counts,
                        std::string &wxfilename,
                        bool binary) {
  Vector<BaseFloat> counts_vec(counts.size());
  for (size_t i = 0; i < counts.size(); i++)
    counts_vec(i) = counts[i];
  WriteKaldiObject(counts_vec, wxfilename, binary);
}

}  // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compare vectors of integers (e.g. phone alignments)\n"
        "Prints to stdout fields of the form:\n"
        "<utterance-id>  <num-frames-in-utterance>  <num-frames-that-differ>\n"
        "\n"
        "e.g.:\n"
        " SWB1_A_31410_32892 420 36\n"
        "\n"
        "Usage:\n"
        "compare-int-vector [options] <vector1-rspecifier> <vector2-rspecifier>\n"
        "\n"
        "e.g. compare-int-vector scp:foo.scp scp:bar.scp > comparison\n"
        "E.g. the inputs might come from ali-to-phones.\n"
        "Warnings are printed if the vector lengths differ for a given utterance-id,\n"
        "and in those cases, the number of frames printed will be the smaller of the\n"
        "\n"
        "See also: ali-to-phones, copy-int-vector\n";


    ParseOptions po(usage);

    std::string tot_wxfilename,
        diff_wxfilename,
        confusion_matrix_wxfilename;
    bool binary = true;

    po.Register("binary", &binary, "If true, write in binary mode (only applies "
                "if --write-tot-counts or --write-diff-counts options are supplied).");
    po.Register("write-tot-counts", &tot_wxfilename, "Filename to write "
                "vector of total counts.  These may be summed with 'vector-sum'.");
    po.Register("write-diff-counts", &diff_wxfilename, "Filename to write "
                "vector of counts of phones (or whatever is in the inputs) "
                "that differ from one vector to the other.  Each time a pair differs, "
                "0.5 will be added to each one's location.");
    po.Register("write-confusion-matrix", &confusion_matrix_wxfilename,
                "Filename to write confusion matrix, indexed by [phone1][phone2]."
                "These may be summed by 'matrix-sum'.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string vector1_rspecifier = po.GetArg(1),
        vector2_rspecifier = po.GetArg(2);

    int64 num_done = 0,
        num_not_found = 0,
        num_mismatched_lengths = 0,
        tot_frames = 0, tot_difference = 0;

    std::vector<double> diff_counts;
    std::vector<double> tot_counts;
    Matrix<double> confusion_matrix;

    SequentialInt32VectorReader reader1(vector1_rspecifier);
    RandomAccessInt32VectorReader reader2(vector2_rspecifier);

    for (; !reader1.Done(); reader1.Next(), num_done++) {
      const std::string &key = reader1.Key();
      if (!reader2.HasKey(key)) {
        KALDI_WARN << "No key " << key << " found in second input.";
        num_not_found++;
        continue;
      }
      const std::vector<int32> &value1 = reader1.Value(),
          &value2 = reader2.Value(key);
      size_t len1 = value1.size(), len2 = value2.size();
      if (len1 != len2) {
        KALDI_WARN << "For utterance " << key << ", lengths differ "
                   << len1 << " vs. " << len2;
        num_mismatched_lengths++;
      }
      size_t len = std::min(len1, len2),
          difference = 0;
      for (size_t i = 0; i < len; i++) {
        int32 phone1 = value1[i], phone2 = value2[i];
        if (phone1 != phone2) {
          difference++;
          if (!diff_wxfilename.empty()) {
            AddToCount(phone1, 0.5, &diff_counts);
            AddToCount(phone2, 0.5, &diff_counts);
          }
        }
        if (!tot_wxfilename.empty())
          AddToCount(phone1, 1.0, &tot_counts);
        if (!confusion_matrix_wxfilename.empty())
          AddToConfusionMatrix(phone1, phone2, &confusion_matrix);
      }
      num_done++;
      std::cout << key << " " << len << " " << difference << "\n";
      tot_frames += len;
      tot_difference += difference;
    }

    BaseFloat difference_percent = tot_difference * 100.0 / tot_frames;
    KALDI_LOG << "Computed difference for " << num_done << " utterances, of which "
              << num_mismatched_lengths << " had mismatched lengths; corresponding "
        "utterance not found for " << num_not_found;
    KALDI_LOG << "Average p(different) is " << std::setprecision(4) << difference_percent
              << "%, over " << tot_frames << " frames.";

    if (!tot_wxfilename.empty())
      WriteAsKaldiVector(tot_counts, tot_wxfilename, binary);
    if (!diff_wxfilename.empty())
      WriteAsKaldiVector(diff_counts, diff_wxfilename, binary);
    if (!confusion_matrix_wxfilename.empty())
      WriteKaldiObject(confusion_matrix, confusion_matrix_wxfilename, binary);

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
