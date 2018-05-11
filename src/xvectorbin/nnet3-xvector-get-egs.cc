// xvectorbin/nnet3-xvector-get-egs.cc

// Copyright 2012-2016  Johns Hopkins University (author:  Daniel Povey)
//                2016  David Snyder

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

#include "util/common-utils.h"
#include "nnet3/nnet-example.h"

namespace kaldi {
namespace nnet3 {

// A struct for holding information about the position and
// duration of each pair of chunks.
struct ChunkPairInfo {
  std::string pair_name;
  int32 output_archive_id;
  int32 start_frame1;
  int32 start_frame2;
  int32 num_frames1;
  int32 num_frames2;
};

// Process the range input file and store it as a map from utterance
// name to vector of ChunkPairInfo structs.
static void ProcessRangeFile(const std::string &range_rxfilename,
                             unordered_map<std::string,
                             std::vector<ChunkPairInfo *> > *utt_to_pairs) {
  Input range_input(range_rxfilename);
  if (!range_rxfilename.empty()) {
    std::string line;
    while (std::getline(range_input.Stream(), line)) {
      ChunkPairInfo *pair = new ChunkPairInfo();
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 7)
        KALDI_ERR << "Expected 7 fields in line of range file, got "
                  << fields.size() << " instead.";

      std::string utt = fields[0],
                  start_frame1_str = fields[3],
                  num_frames1_str = fields[4],
                  start_frame2_str = fields[5],
                  num_frames2_str = fields[6];

      if (!ConvertStringToInteger(fields[1], &(pair->output_archive_id))
          || !ConvertStringToInteger(start_frame1_str, &(pair->start_frame1))
          || !ConvertStringToInteger(start_frame2_str, &(pair->start_frame2))
          || !ConvertStringToInteger(num_frames1_str, &(pair->num_frames1))
          || !ConvertStringToInteger(num_frames2_str, &(pair->num_frames2)))
        KALDI_ERR << "Expected integer for output archive in range file.";
      pair->pair_name = utt + "-" + start_frame1_str + "-" + num_frames1_str
                      + "-" + start_frame2_str + "-" + num_frames2_str;
      unordered_map<std::string, std::vector<ChunkPairInfo*> >::iterator
        got = utt_to_pairs->find(utt);
      if (got == utt_to_pairs->end()) {
        std::vector<ChunkPairInfo* > pairs;
        pairs.push_back(pair);
        utt_to_pairs->insert(std::pair<std::string,
                             std::vector<ChunkPairInfo* > > (utt, pairs));
      } else {
        got->second.push_back(pair);
      }
    }
  }
}

static void WriteExamples(const MatrixBase<BaseFloat> &feats,
                          const std::vector<ChunkPairInfo *> &pairs,
                          const std::string &utt,
                          bool compress,
                          int32 *num_egs_written,
                          std::vector<NnetExampleWriter *> *example_writers) {
  for (std::vector<ChunkPairInfo *>::const_iterator it = pairs.begin();
      it != pairs.end(); ++it) {
    ChunkPairInfo *pair = *it;
    NnetExample eg;
    int32 num_rows = feats.NumRows(),
          feat_dim = feats.NumCols();
    if (num_rows < std::max(pair->num_frames1, pair->num_frames2)) {
      KALDI_WARN << "Unable to create examples for utterance " << utt
                 << ". Requested chunk size of "
                 << std::max(pair->num_frames1, pair->num_frames2)
                 << " but utterance has only " << num_rows << " frames.";
    } else {
      // The requested chunk positions are approximate. It's possible
      // that they slightly exceed the number of frames in the utterance.
      // If that occurs, we can shift the chunks location back slightly.
      int32 shift1 = std::min(0, num_rows - pair->start_frame1
                                 - pair->num_frames1),
            shift2 = std::min(0, num_rows - pair->start_frame2
                                 - pair->num_frames2);
      SubMatrix<BaseFloat> chunk1(feats, pair->start_frame1 + shift1,
                                  pair->num_frames1, 0, feat_dim),
                           chunk2(feats, pair->start_frame2 + shift2,
                                  pair->num_frames2, 0, feat_dim);
      NnetIo nnet_io1 = NnetIo("input", 0, chunk1),
             nnet_io2 = NnetIo("input", 0, chunk2);
      for (std::vector<Index>::iterator indx_it = nnet_io1.indexes.begin();
          indx_it != nnet_io1.indexes.end(); ++indx_it)
        indx_it->n = 0;
      for (std::vector<Index>::iterator indx_it = nnet_io2.indexes.begin();
          indx_it != nnet_io2.indexes.end(); ++indx_it)
        indx_it->n = 1;

      NnetExample eg;
      eg.io.push_back(nnet_io1);
      eg.io.push_back(nnet_io2);
      if (compress)
        eg.Compress();

      if (pair->output_archive_id >= example_writers->size())
        KALDI_ERR << "Requested output index exceeds number of specified "
                  << "output files.";
      (*example_writers)[pair->output_archive_id]->Write(
                         pair->pair_name, eg);
      (*num_egs_written) += 1;
    }
  }
}

// Delete the dynamically allocated memory.
static void Cleanup(unordered_map<std::string,
                    std::vector<ChunkPairInfo *> > *utt_to_pairs,
                    std::vector<NnetExampleWriter *> *writers) {
  for (unordered_map<std::string, std::vector<ChunkPairInfo*> >::iterator
      map_it = utt_to_pairs->begin();
      map_it != utt_to_pairs->end(); ++map_it)
    for (std::vector<ChunkPairInfo*>::iterator
        vec_it = map_it->second.begin(); vec_it != map_it->second.end();
        ++vec_it)
      delete *vec_it;
  for (std::vector<NnetExampleWriter *>::iterator
      it = writers->begin(); it != writers->end(); ++it)
    delete *it;
}

} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Get examples for training an nnet3 neural network for the xvector\n"
        "system.  Each output example contains a pair of feature chunks from\n"
        "the same utterance.  The location and length of the feature chunks\n"
        "are specified in the 'ranges' file.  Each line is interpreted as\n"
        "follows:\n"
        "  <source-utterance> <relative-output-archive-index> "
        "<absolute-archive-index>  <start-frame-index1> <num-frames1> "
        "<start-frame-index2> <num-frames2>\n"
        "where <relative-output-archive-index> is interpreted as a zero-based\n"
        "index into the wspecifiers specified on the command line (<egs-0-out>\n"
        "and so on), and <absolute-archive-index> is ignored by this program.\n"
        "For example:\n"
        "  utt1  3  13  0   65  112  110\n"
        "  utt1  0  10  160 50  214  180\n"
        "  utt2  ...\n"
        "\n"
        "Usage:  nnet3-xvector-get-egs [options] <ranges-filename> "
        "<features-rspecifier> <egs-0-out> <egs-1-out> ... <egs-N-1-out>\n"
        "\n"
        "For example:\n"
        "nnet3-xvector-get-egs ranges.1 \"$feats\" ark:egs_temp.1.ark"
        "  ark:egs_temp.2.ark ark:egs_temp.3.ark\n";

    bool compress = true;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        range_rspecifier = po.GetArg(1),
        feature_rspecifier = po.GetArg(2);
    std::vector<NnetExampleWriter *> example_writers;

    for (int32 i = 3; i <= po.NumArgs(); i++)
      example_writers.push_back(new NnetExampleWriter(po.GetArg(i)));

    unordered_map<std::string, std::vector<ChunkPairInfo *> > utt_to_pairs;
    ProcessRangeFile(range_rspecifier, &utt_to_pairs);
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);

    int32 num_done = 0,
          num_err = 0,
          num_egs_written = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      unordered_map<std::string, std::vector<ChunkPairInfo*> >::iterator
        got = utt_to_pairs.find(key);
      if (got == utt_to_pairs.end()) {
        KALDI_WARN << "Could not create examples from utterance "
                   << key << " because it has no entry in the ranges "
                  <<  "input file.";
        num_err++;
      } else {
        std::vector<ChunkPairInfo *> pairs = got->second;
        WriteExamples(feats, pairs, key, compress, &num_egs_written,
                      &example_writers);
        num_done++;
      }
    }
    Cleanup(&utt_to_pairs, &example_writers);

    KALDI_LOG << "Finished generating examples, "
              << "successfully processed " << num_done
              << " feature files, wrote " << num_egs_written << " examples; "
              << num_err << " files had errors.";
    return (num_egs_written == 0 || num_err > num_done ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
