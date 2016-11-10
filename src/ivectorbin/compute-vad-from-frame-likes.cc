// ivectorbin/compute-vad-from-frame-likes.cc

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
#include "util/parse-options.h"
#include "util/stl-utils.h"

namespace kaldi {

/**
   PrepareMap creates a map that specifies the mapping between the input
   and output class labels.  If the string map_rxfilename is empty, then
   the mapping is the identity map (e.g., 0 maps to 0, 1 maps to 1, etc),
   based on the number of classes num_classes.  If map_rxfilename is not
   empty, the mapping is created from that file.  The file is expected to
   be two columns of integers with up to num_classes rows.  If an input
   class is not specified in the file, then the output class label is the
   same as the input.  The first column is the input class and the second
   column is the output class.  For example:
       0 0
       1 1
       2 0
*/
void PrepareMap(const std::string &map_rxfilename, int32 num_classes,
                unordered_map<int32, int32> *map) {
  Input map_input(map_rxfilename);
  for (int32 i = 0; i < num_classes; i++)
    (*map)[i] = i;

  if (!map_rxfilename.empty()) {
    std::string line;
    while (std::getline(map_input.Stream(), line)) {
      if (line.size() == 0) continue;
      int32 start = line.find_first_not_of(" \t");
      int32 end = line.find_first_of('#'); // Ignore trailing comments
      if (start == std::string::npos || start == end) continue;
      end = line.find_last_not_of(" \t", end - 1);
      KALDI_ASSERT(end >= start);
      std::vector<std::string> fields;
      SplitStringToVector(line.substr(start, end - start + 1),
         " \t\n\r", true, &fields);
      if (fields.size() != 2) {
        KALDI_ERR << "Bad line. Expected two fields, got: "
                  << line;
      }
      (*map)[std::atoi(fields[0].c_str())] = std::atoi(fields[1].c_str());
    }
  }

  if (map->size() > num_classes)
    KALDI_ERR << "Map table has " << map->size() << " classes.  "
              << "Expected " << num_classes << " or fewer";
}

/**
   PreparePriors creates a table specifying the priors for each class.
   If priors_str is empty, uniform priors are assumed.  If priors_str is
   nonempty, the comma-separated floats are parsed out.  If present, the
   input of priors_str is of the form:
      0.5,0.25,0.25
*/
void PreparePriors(const std::string &priors_str, int32 num_classes,
                std::vector<BaseFloat> *priors) {
  if (priors_str.empty()) {
    for (int32 i = 0; i < num_classes; i++)
      priors->push_back(log(1.0/num_classes)); // Uniform priors
  } else {
    SplitStringToFloats(priors_str, ",", false, priors);
    for (int32 i = 0; i < priors->size(); i++)
      (*priors)[i] = log((*priors)[i]);
  }

  if (priors->size() != num_classes)
    KALDI_ERR << priors->size() << " priors specified.  Expected "
              << num_classes;
}

}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "This program computes frame-level voice activity decisions from a\n"
      "set of input frame-level log-likelihoods.  Usually, these\n"
      "log-likelihoods are the output of fgmm-global-get-frame-likes.\n"
      "Frames are assigned labels according to the class for which the\n"
      "log-likelihood (optionally weighted by a prior) is maximal.  The\n"
      "class labels are determined by the order of inputs on the command\n"
      "line.  See options for more details.\n"
      "\n"
      "Usage: compute-vad-from-frame-likes [options] <likes-rspecifier-1>\n"
      "    ... <likes-rspecifier-n> <vad-wspecifier>\n"
      "e.g.: compute-vad-from-frame-likes --map=label_map.txt\n"
      "    scp:likes1.scp scp:likes2.scp ark:vad.ark\n"
      "See also: fgmm-global-get-frame-likes, compute-vad, merge-vads\n";

    ParseOptions po(usage);
    std::string map_rxfilename;
    std::string priors_str;

    po.Register("map", &map_rxfilename, "Table that defines the frame-level "
      "labels.  For each row, the first field is the zero-based index of the "
      "input likelihood archive and the second field is the associated "
      "integer label.");

    po.Register("priors", &priors_str, "Comma-separated list that specifies "
      "the priors for each class.  The order of the floats corresponds to "
      "the index of the input archives.  E.g., --priors=0.5,0.2,0.3");

    po.Read(argc, argv);
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    unordered_map<int32, int32> map;
    std::vector<BaseFloat> priors;
    int32 num_classes = po.NumArgs() - 1;
    PrepareMap(map_rxfilename, num_classes, &map);
    PreparePriors(priors_str, num_classes, &priors);

    SequentialBaseFloatVectorReader first_reader(po.GetArg(1));
    std::vector<RandomAccessBaseFloatVectorReader *> readers;
    std::string vad_wspecifier = po.GetArg(po.NumArgs());
    BaseFloatVectorWriter vad_writer(vad_wspecifier);

    for (int32 i = 2; i < po.NumArgs(); i++) {
      RandomAccessBaseFloatVectorReader *reader
        = new RandomAccessBaseFloatVectorReader(po.GetArg(i));
      readers.push_back(reader);
    }

    int32 num_done = 0, num_err = 0;
    for (;!first_reader.Done(); first_reader.Next()) {
      std::string utt = first_reader.Key();
      Vector<BaseFloat> like(first_reader.Value());
      int32 like_dim = like.Dim();
      std::vector<Vector<BaseFloat> > likes;
      likes.push_back(like);
      if (like_dim == 0) {
        KALDI_WARN << "Empty vector for utterance " << utt;
        num_err++;
        continue;
      }
      for (int32 i = 0; i < num_classes - 1; i++) {
        if (!readers[i]->HasKey(utt)) {
          KALDI_WARN << "No vector for utterance " << utt;
          num_err++;
          continue;
        }
        Vector<BaseFloat> other_like(readers[i]->Value(utt));
        if (like_dim != other_like.Dim()) {
          KALDI_WARN << "Dimension mismatch in input vectors in " << utt
                    << ": " << like_dim << " vs. " << other_like.Dim();
          num_err++;
          continue;
        }
        likes.push_back(other_like);
      }

      Vector<BaseFloat> vad_result(like_dim);
      for (int32 i = 0; i < like.Dim(); i++) {
        int32 max_indx = 0;
        BaseFloat max_post = likes[0](i) + priors[0];
        for (int32 j = 0; j < num_classes; j++) {
          BaseFloat other_post = likes[j](i) + priors[j];
          if (other_post > max_post) {
            max_indx = j;
            max_post = other_post;
          }
        }
        unordered_map<int32, int32>::const_iterator iter = map.find(max_indx);
        if (iter == map.end()) {
          KALDI_ERR << "Missing label " << max_indx  << " in map";
        } else {
          vad_result(i) = iter->second;
        }
      }
      vad_writer.Write(utt, vad_result);
      num_done++;
    }

    for (int32 i = 0; i < num_classes - 1; i++)
      delete readers[i];

    KALDI_LOG << "Applied frame-level likelihood-based voice activity "
              << "detection; processed " << num_done
              << " utterances successfully; " << num_err
              << " had empty features.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
