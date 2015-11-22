// ivectorbin/merge-vads.cc

// Copyright  2015 David Snyder

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
#include "matrix/kaldi-matrix.h"
#include "util/stl-utils.h"

using namespace kaldi;
using kaldi::int32;

void PrepareMap(const std::string map_rxfilename,
  unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > *map) {
  Input map_input(map_rxfilename);

  // If a map file isn't specified, provide an obvious mapping.  The
  // following mapping assumes "0" corresponds to nonspeech and "1"
  // corresponds to speech. The combination of two VAD decisions only
  // results in a decision of speech if both input frames are
  // classified as speech.
  if (map_rxfilename.empty()) {
    (*map)[std::pair<int32, int32>(0, 0)] = 0;
    (*map)[std::pair<int32, int32>(0, 1)] = 0;
    (*map)[std::pair<int32, int32>(1, 0)] = 0;
    (*map)[std::pair<int32, int32>(1, 1)] = 1;
  } else {
    std::string line;
    while (std::getline(map_input.Stream(), line)) {
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 3) {
        KALDI_ERR << "Bad line. Expected three fields, got: "
                  << line;
      }
      int32 label1 = std::atoi(fields[0].c_str()),
            label2 = std::atoi(fields[1].c_str()),
            result_label = std::atoi(fields[2].c_str());
      (*map)[std::pair<int32, int32>(label1, label2)] = result_label;
    }
  }
}

int main(int argc, char *argv[]) {
  try {
    const char *usage =
      "This program merges two Vector archives of frame-level voice\n"
      "activity decisions. By default, the program assumes that the\n"
      "input vectors consist of floats that are 0.0 if a frame is judged\n"
      "as nonspeech and 1.0 if it is considered speech. The default\n"
      "produces an output Vector that is 1.0 if both inputs are 1.0\n"
      "and 0.0 otherwise. Additional classes (e.g., 2.0 for music) \n"
      "can be handled using the \"map\" option.\n"
      "Usage: merge-vads [options] <vad-rspecifier-1> <vad-rspecifier-2>\n"
      "    <vad-wspecifier>\n"
      "e.g.: merge-vads [options] scp:vad_energy.scp scp:vad_gmm.scp\n"
      "    ark:vad.ark\n";

    ParseOptions po(usage);
    std::string map_rxfilename;
    po.Register("map", &map_rxfilename, "This table specifies a mapping"
    " between the labels of the frame-level decisions in the first and"
    " second input archives to the integer output label.");

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    unordered_map<std::pair<int32, int32>, int32, PairHasher<int32> > map;
    PrepareMap(map_rxfilename, &map);
    SequentialBaseFloatVectorReader first_vad_reader(po.GetArg(1));
    RandomAccessBaseFloatVectorReader second_vad_reader(po.GetArg(2));
    BaseFloatVectorWriter vad_writer(po.GetArg(3));

    int32 num_done = 0, num_err = 0;
    for (;!first_vad_reader.Done(); first_vad_reader.Next()) {
      std::string utt = first_vad_reader.Key();
      Vector<BaseFloat> vad1(first_vad_reader.Value());
      if (!second_vad_reader.HasKey(utt)) {
        KALDI_WARN << "No vector for utterance " << utt;
        num_err++;
        continue;
      }
      Vector<BaseFloat> vad2(second_vad_reader.Value(utt));
      if (vad1.Dim() != vad2.Dim()) {
        KALDI_WARN << "VAD length mismatch for utterance " << utt;
        num_err++;
        continue;
      }
      Vector<BaseFloat> vad_result(vad1.Dim());
      for (int32 i = 0; i < vad1.Dim(); i++) {
        std::pair<int32, int32> key(static_cast<int32>(vad1(i)),
          static_cast<int32>(vad2(i)));
        if (map.find(key) == map.end()) {
          KALDI_ERR << "Map is missing combination "
                    << vad1(i) << " and " << vad2(i);
        }
        vad_result(i) = map[key];
      }

      if (vad_result.Dim() > 0) {
        vad_writer.Write(utt, vad_result);
        num_done++;
      }
    }
    KALDI_LOG << "Merged voice activity detection decisions; "
              << "processed " << num_done << " utterances successfully; "
              << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
