// nnet3bin/nnet3-xvector-get-egs.cc

// Copyright 2016-2017  Johns Hopkins University (author:  Daniel Povey)
//           2016-2017  Johns Hopkins University (author:  Daniel Garcia-Romero)
//           2016-2017  David Snyder

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
// duration of each chunk.
struct ChunkInfo {
  std::string name;
  int32 output_archive_id;
  int32 start_frame;
  int32 num_frames;
  int32 label;
};

// Process the range input file and store it as a map from utterance
// name to vector of ChunkInfo structs.
static void ProcessRangeFile(const std::string &range_rxfilename,
    unordered_map<std::string, std::vector<ChunkInfo *> > *utt_to_chunks) {
  Input range_input(range_rxfilename);
  if (!range_rxfilename.empty()) {
    std::string line;
    while (std::getline(range_input.Stream(), line)) {
      ChunkInfo *chunk_info = new ChunkInfo();
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 6)
        KALDI_ERR << "Expected 6 fields in line of range file, got "
                  << fields.size() << " instead.";

      std::string utt = fields[0],
                  start_frame_str = fields[3],
                  num_frames_str = fields[4],
                  label_str = fields[5];

      if (!ConvertStringToInteger(fields[1], &(chunk_info->output_archive_id))
        || !ConvertStringToInteger(start_frame_str, &(chunk_info->start_frame))
        || !ConvertStringToInteger(num_frames_str, &(chunk_info->num_frames))
        || !ConvertStringToInteger(label_str, &(chunk_info->label)))
        KALDI_ERR << "Expected integer for output archive in range file.";

      chunk_info->name = utt + "-" + start_frame_str + "-" + num_frames_str
        + "-" + label_str;
      unordered_map<std::string, std::vector<ChunkInfo*> >::iterator
        got = utt_to_chunks->find(utt);

      if (got == utt_to_chunks->end()) {
        std::vector<ChunkInfo* > chunk_infos;
        chunk_infos.push_back(chunk_info);
        utt_to_chunks->insert(std::pair<std::string,
          std::vector<ChunkInfo* > > (utt, chunk_infos));
      } else {
        got->second.push_back(chunk_info);
      }
    }
  }
}

static void WriteExamples(const MatrixBase<BaseFloat> &feats,
    const std::vector<ChunkInfo *> &chunks, const std::string &utt,
    bool compress, int32 num_pdfs, int32 *num_egs_written,
    std::vector<NnetExampleWriter *> *example_writers) {
  for (std::vector<ChunkInfo *>::const_iterator it = chunks.begin();
      it != chunks.end(); ++it) {
    ChunkInfo *chunk = *it;
    NnetExample eg;
    int32 num_rows = feats.NumRows(),
          feat_dim = feats.NumCols();
    if (num_rows < chunk->num_frames) {
      KALDI_WARN << "Unable to create examples for utterance " << utt
                 << ". Requested chunk size of "
                 << chunk->num_frames
                 << " but utterance has only " << num_rows << " frames.";
    } else {
      // The requested chunk positions are approximate. It's possible
      // that they slightly exceed the number of frames in the utterance.
      // If that occurs, we can shift the chunks location back slightly.
      int32 shift = std::min(0, num_rows - chunk->start_frame
                                 - chunk->num_frames);
      SubMatrix<BaseFloat> chunk_mat(feats, chunk->start_frame + shift,
                                  chunk->num_frames, 0, feat_dim);
      NnetIo nnet_input = NnetIo("input", 0, chunk_mat);
      for (std::vector<Index>::iterator indx_it = nnet_input.indexes.begin();
          indx_it != nnet_input.indexes.end(); ++indx_it)
        indx_it->n = 0;

      Posterior label;
      std::vector<std::pair<int32, BaseFloat> > post;
      post.push_back(std::pair<int32, BaseFloat>(chunk->label, 1.0));
      label.push_back(post);
      NnetExample eg;
      eg.io.push_back(nnet_input);
      eg.io.push_back(NnetIo("output", num_pdfs, 0, label));
      if (compress)
        eg.Compress();

      if (chunk->output_archive_id >= example_writers->size())
        KALDI_ERR << "Requested output index exceeds number of specified "
                  << "output files.";
      (*example_writers)[chunk->output_archive_id]->Write(
                         chunk->name, eg);
      (*num_egs_written) += 1;
    }
  }
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
        "system.  Each output example contains a chunk of features from some\n"
        "utterance along with a speaker label.  The location and length of\n"
        "the feature chunks are specified in the 'ranges' file.  Each line\n"
        "is interpreted as follows:\n"
        "  <source-utterance> <relative-output-archive-index> "
        "<absolute-archive-index> <start-frame-index> <num-frames> "
        "<speaker-label>\n"
        "where <relative-output-archive-index> is interpreted as a zero-based\n"
        "index into the wspecifiers provided on the command line (<egs-0-out>\n"
        "and so on), and <absolute-archive-index> is ignored by this program.\n"
        "For example:\n"
        "  utt1  3  13  65  300  3\n"
        "  utt1  0  10  50  400  3\n"
        "  utt2  ...\n"
        "\n"
        "Usage:  nnet3-xvector-get-egs [options] <ranges-filename> "
        "<features-rspecifier> <egs-0-out> <egs-1-out> ... <egs-N-1-out>\n"
        "\n"
        "For example:\n"
        "nnet3-xvector-get-egs ranges.1 \"$feats\" ark:egs_temp.1.ark"
        "  ark:egs_temp.2.ark ark:egs_temp.3.ark\n";

    bool compress = true;
    int32 num_pdfs = -1;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");
    po.Register("num-pdfs", &num_pdfs, "Number of speakers in the training "
                "list.");

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string  range_rspecifier = po.GetArg(1),
        feature_rspecifier = po.GetArg(2);
    std::vector<NnetExampleWriter *> example_writers;

    for (int32 i = 3; i <= po.NumArgs(); i++)
      example_writers.push_back(new NnetExampleWriter(po.GetArg(i)));

    unordered_map<std::string, std::vector<ChunkInfo *> > utt_to_chunks;
    ProcessRangeFile(range_rspecifier, &utt_to_chunks);
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);

    int32 num_done = 0,
          num_err = 0,
          num_egs_written = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      unordered_map<std::string, std::vector<ChunkInfo*> >::iterator
        got = utt_to_chunks.find(key);
      if (got == utt_to_chunks.end()) {
        KALDI_WARN << "Could not create examples from utterance "
                   << key << " because it has no entry in the ranges "
                  <<  "input file.";
        num_err++;
      } else {
        std::vector<ChunkInfo *> chunks = got->second;
        WriteExamples(feats, chunks, key, compress, num_pdfs,
                      &num_egs_written, &example_writers);
        num_done++;
      }
    }

    // Free memory
    for (unordered_map<std::string, std::vector<ChunkInfo*> >::iterator
        map_it = utt_to_chunks.begin();
        map_it != utt_to_chunks.end(); ++map_it) {
      DeletePointers(&map_it->second);
    }
    DeletePointers(&example_writers);

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
