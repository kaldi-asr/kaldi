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
  std::string utt1;
  std::string utt2;
  int32 output_archive_id;
  int32 start_frame1;
  int32 start_frame2;
  int32 num_frames1;
  int32 num_frames2;
};

// Process the range input file and store it as a map from utterance
// name to vector of ChunkPairInfo structs.
static void ProcessRangeFile(const std::string &range_rxfilename,
                             std::vector<ChunkPairInfo *> *pairs) {
  Input range_input(range_rxfilename);
  if (!range_rxfilename.empty()) {
    std::string line;
    while (std::getline(range_input.Stream(), line)) {
      ChunkPairInfo *pair = new ChunkPairInfo();
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 8)
        KALDI_ERR << "Expected 7 fields in line of range file, got "
                  << fields.size() << " instead.";

      std::string utt1 = fields[0],
                  utt2 = fields[1],
                  start_frame1_str = fields[4],
                  num_frames1_str = fields[5],
                  start_frame2_str = fields[6],
                  num_frames2_str = fields[7];
      pair->utt1 = utt1;
      pair->utt2 = utt2;
      if (!ConvertStringToInteger(fields[2], &(pair->output_archive_id))
          || !ConvertStringToInteger(start_frame1_str, &(pair->start_frame1))
          || !ConvertStringToInteger(start_frame2_str, &(pair->start_frame2))
          || !ConvertStringToInteger(num_frames1_str, &(pair->num_frames1))
          || !ConvertStringToInteger(num_frames2_str, &(pair->num_frames2)))
        KALDI_ERR << "Expected integer for output archive in range file.";
      pair->pair_name = utt1 + "-" + start_frame1_str + "-" + num_frames1_str
                      + "-" + utt2
                      + "-" + start_frame2_str + "-" + num_frames2_str;
      pairs->push_back(pair);
    }
  }
}

static void WriteExample(const MatrixBase<BaseFloat> &feat1,
                         const MatrixBase<BaseFloat> &feat2,
                         const ChunkPairInfo *pair,
                         int32 subsample,
                         bool compress,
                         int32 *num_egs_written,
                         std::vector<NnetExampleWriter *> *example_writers) {
    NnetExample eg;
    int32 num_rows1 = feat1.NumRows(),
          feat_dim1 = feat1.NumCols(),
          num_rows2 = feat2.NumRows(),
          feat_dim2 = feat2.NumCols();
    std::string utt1 = pair->utt1,
                utt2 = pair->utt2;

    KALDI_ASSERT(feat_dim1 == feat_dim2);

    if (num_rows1 < pair->num_frames1) {
      KALDI_WARN << "Unable to create examples for utterance "
                 << utt1
                 << ". Requested chunk size of "
                 << pair->num_frames1
                 << " but utterance has only " << num_rows1 << " frames.";
      return;
    }
    if (num_rows2 < pair->num_frames2) {
      KALDI_WARN << "Unable to create examples for utterance "
                 << utt2
                 << ". Requested chunk size of "
                 << pair->num_frames2
                 << " but utterance has only " << num_rows2 << " frames.";
      return;
    }
    // The requested chunk positions are approximate. It's possible
    // that they slightly exceed the number of frames in the utterance.
    // If that occurs, we can shift the chunks location back slightly.
    int32 shift1 = std::min(0, num_rows1 - pair->start_frame1
                              - pair->num_frames1),
          shift2 = std::min(0, num_rows2 - pair->start_frame2
                                 - pair->num_frames2);

    SubMatrix<BaseFloat> chunk1_sub(feat1, pair->start_frame1 + shift1,
                                pair->num_frames1, 0, feat_dim1),
                         chunk2_sub(feat2, pair->start_frame2 + shift2,
                                pair->num_frames2, 0, feat_dim2);
    Matrix<BaseFloat> chunk1_tmp(chunk1_sub);
    Matrix<BaseFloat> chunk2_tmp(chunk2_sub);

    int32 real_chunk_size1 = chunk1_tmp.NumRows() / subsample;
    int32 real_chunk_size2 = chunk2_tmp.NumRows() / subsample;
    Matrix<BaseFloat> chunk1(real_chunk_size1, chunk1_tmp.NumCols());
    Matrix<BaseFloat> chunk2(real_chunk_size2, chunk2_tmp.NumCols());

    std::vector<int32> index_vector1;
    for (int32 i = 0; i < chunk1_tmp.NumRows(); i++)
      index_vector1.push_back(i);

    std::vector<int32> index_vector2;
    for (int32 i = 0; i < chunk2_tmp.NumRows(); i++)
      index_vector2.push_back(i);

    std::random_shuffle(index_vector1.begin(), index_vector1.end());
    for (int32 i = 0; i < real_chunk_size1; i++)
      chunk1.Row(i).CopyFromVec(chunk1_tmp.Row(i));

    std::random_shuffle(index_vector2.begin(), index_vector2.end());
    for (int32 i = 0; i < real_chunk_size2; i++)
      chunk2.Row(i).CopyFromVec(chunk2_tmp.Row(i));

    NnetIo nnet_io1 = NnetIo("input", 0, chunk1),
           nnet_io2 = NnetIo("input", 0, chunk2);
    for (std::vector<Index>::iterator indx_it = nnet_io1.indexes.begin();
        indx_it != nnet_io1.indexes.end(); ++indx_it)
      indx_it->n = 0;
    for (std::vector<Index>::iterator indx_it = nnet_io2.indexes.begin();
        indx_it != nnet_io2.indexes.end(); ++indx_it)
      indx_it->n = 1;

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

// Delete the dynamically allocated memory.
static void Cleanup(std::vector<ChunkPairInfo *> *pairs,
                    std::vector<NnetExampleWriter *> *writers) {
  for (std::vector<ChunkPairInfo *>::iterator
      vec_it = pairs->begin(); vec_it != pairs->end();
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
    int32 subsample = 5;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");
    po.Register("subsample", &subsample, "TODO");

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

    std::vector<ChunkPairInfo *> pairs;
    ProcessRangeFile(range_rspecifier, &pairs);
    RandomAccessBaseFloatMatrixReader feat_reader1(feature_rspecifier);
    RandomAccessBaseFloatMatrixReader feat_reader2(feature_rspecifier);
    int32 num_done = 0,
          num_err = 0,
          num_egs_written = 0;
    for (int32 i = 0; i < pairs.size(); i++) {
      ChunkPairInfo *pair = pairs[i];
      const Matrix<BaseFloat> &feat1(feat_reader1.Value(pair->utt1));
      const Matrix<BaseFloat> &feat2(feat_reader2.Value(pair->utt2));
      WriteExample(feat1, feat2, pair, subsample, compress, &num_egs_written,
                      &example_writers);
      num_done++;
    }
    Cleanup(&pairs, &example_writers);

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
