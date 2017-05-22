// segmenterbin/intersect-int-vectors.cc

// Copyright 2017   Vimal Manohar

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Intersect two integer vectors and create a new integer vectors "
        "whole ids are defined as the cross-products of the integer "
        "ids from the two vectors.\n"
        "\n"
        "Usage: intersect-int-vectors [options] "
        "<ali-rspecifier1> <ali-rspecifier2> <ali-wspecifier>\n"
        " e.g.: intersect-int-vectors ark:1.ali ark:2.ali ark:-\n"
        "See also: segmentation-init-from-segments, "
        "segmentation-combine-segments\n";

    ParseOptions po(usage);
    
    std::string mapping_rxfilename, mapping_wxfilename;
    int32 length_tolerance = 0;

    po.Register("mapping-in", &mapping_rxfilename, 
                "A file with three columns that define the mapping from "
                "a pair of integers to a third one.");
    po.Register("mapping-out", &mapping_wxfilename,
                "Write a mapping in the same format as --mapping-in, "
                "but let the program decide the mapping to unique integer "
                "ids.");
    po.Register("length-tolerance", &length_tolerance,
                "Tolerance this number of frames of mismatch between the "
                "two integer vector pairs.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ali_rspecifier1 = po.GetArg(1),
      ali_rspecifier2 = po.GetArg(2),
      ali_wspecifier = po.GetArg(3);

    std::map<std::pair<int32, int32>, int32> mapping;
    if (!mapping_rxfilename.empty()) {
      Input ki(mapping_rxfilename);
      std::string line;
      while (std::getline(ki.Stream(), line)) {
        std::vector<std::string> parts;
        SplitStringToVector(line, " ", true, &parts);
        KALDI_ASSERT(parts.size() == 3);

        std::pair<int32, int32> id_pair = std::make_pair(
            std::atoi(parts[0].c_str()), std::atoi(parts[1].c_str()));
        int32 id_new = std::atoi(parts[2].c_str());
        KALDI_ASSERT(id_new >= 0);

        std::map<std::pair<int32, int32>, int32>::iterator it = 
          mapping.lower_bound(id_pair);
        KALDI_ASSERT(it == mapping.end() || it->first != id_pair);

        mapping.insert(it, std::make_pair(id_pair, id_new));
      }
    }

    SequentialInt32VectorReader ali_reader1(ali_rspecifier1);
    RandomAccessInt32VectorReader ali_reader2(ali_rspecifier2);

    Int32VectorWriter ali_writer(ali_wspecifier);

    int32 num_ids = 0, num_err = 0, num_done = 0;

    for (; !ali_reader1.Done(); ali_reader1.Next()) {
      const std::string &key = ali_reader1.Key();

      if (!ali_reader2.HasKey(key)) {
        KALDI_WARN << "Could not find second alignment for key " << key
                   << "in " << ali_rspecifier2;
        num_err++;
        continue;
      }

      const std::vector<int32> &alignment1 = ali_reader1.Value();
      const std::vector<int32> &alignment2 = ali_reader2.Value(key);

      if (static_cast<int32>(alignment1.size()) 
          - static_cast<int32>(alignment2.size()) > length_tolerance) {
        KALDI_WARN << "Mismatch in length of alignments in "
                   << ali_rspecifier1 << " and " << ali_rspecifier2
                   << "; " << alignment1.size() << " vs " 
                   << alignment2.size();
        num_err++;
      }
     
      int32 min_length = std::min(static_cast<int32>(alignment1.size()),
                                  static_cast<int32>(alignment2.size()));
      std::vector<int32> alignment_out(min_length);

      for (size_t i = 0; i < min_length; i++) {
        std::pair<int32, int32> id_pair = std::make_pair(
            alignment1[i], alignment2[i]);

        std::map<std::pair<int32, int32>, int32>::iterator it = 
          mapping.lower_bound(id_pair);

        int32 id_new = -1;
        if (!mapping_rxfilename.empty()) {
          if (it == mapping.end() || it->first != id_pair) {
            KALDI_ERR << "Could not find id-pair (" << id_pair.first 
                      << ", " << id_pair.second 
                      << ") in mapping " << mapping_rxfilename;
          }
          id_new = it->second;
        } else {
          if (it == mapping.end() || it->first != id_pair) {
            id_new = ++num_ids;
            mapping.insert(it, std::make_pair(id_pair, id_new));
          } else {
            id_new = it->second;
          }
        }

        alignment_out[i] = id_new;
      }

      ali_writer.Write(key, alignment_out);
      num_done++;
    }

    KALDI_LOG << "Intersected " << num_done << " int vector pairs; "
              << "failed with " << num_err;
 
    return ((num_done > 0 && num_err < num_done) ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

