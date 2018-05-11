// fvectorbin/fvector-get-egs.cc

// Copyright 2012-2016  Johns Hopkins University (author:  Daniel Povey)

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Get examples for training an nnet3 neural network for the fvector\n"
        "system.  Each output example contains a pair of feature chunks.\n"
        "Usage:  fvector-get-egs [options] <chunk-rspecifier> <egs-wspecifier>\n"
        "For example:\n"
        "fvector-get-egs scp:perturbed_chunks.scp ark:egs.ark";

    bool compress = true;
    BaseFloat frame_length_ms = 25; // in milliseconds
    BaseFloat frame_shift_ms = 10; // in milliseconds
    BaseFloat samp_freq=16000;
    int left_padding=0;
    int right_padding=0;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");
    po.Register("frame-length", &frame_length_ms, "Frame length in milliseconds");
    po.Register("frame-shift", &frame_shift_ms, "Frame shift in milliseconds");
    po.Register("sample-frequency", &samp_freq, "Waveform data sample frequency ("
                "must match the waveform file, if specified there)");
    po.Register("left-padding", &left_padding, "When we use convolutional NN,"
                "we tend to pad on the time axis with repeats of the first frame.");
    po.Register("right-padding", &right_padding, "When we use convolutional NN,"
                "we tend to pad on the time axis with repeats of the last frame.");


    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1);
    NnetExampleWriter example_writer(po.GetArg(2));
    
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    int32 num_read = 0,
          num_egs_written = 0;
    for (; !feature_reader.Done(); feature_reader.Next(), num_read++) {
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &feats = feature_reader.Value();
      //Please take care. Here, the 'feats' is a 2-lines matrix which is generated
      //by fvector-add-noise.cc. The 2-lines matrix represents two perturbed 
      //vectors(e.g 100ms wavform) which come from the same source signal.
      //chunk1 and chunk2 corresponds to one line respectively.
      SubVector<BaseFloat> chunk1(feats, 0),
                           chunk2(feats, 1);

      //According to frame_length and frame_shift, cut the chunk into few pieces
      //so that it is similiar with normal feature extract procedure.
      int num_rows = ((int)(((chunk1.Dim() * 1.0 / samp_freq) * 1000 - frame_length_ms) / 
                            frame_shift_ms) + 1);
      int num_cols = (int)(samp_freq / 1000.0 * frame_length_ms);
      Matrix<BaseFloat> chunk1_matrix(num_rows, num_cols),
                        chunk2_matrix(num_rows, num_cols);
      for (MatrixIndexT i = 0; i < num_rows; i++) {
        chunk1_matrix.Row(i).CopyFromVec(chunk1.Range(i*frame_shift_ms*samp_freq/1000, num_cols));
        chunk2_matrix.Row(i).CopyFromVec(chunk2.Range(i*frame_shift_ms*samp_freq/1000, num_cols));
      }
      Matrix<BaseFloat> chunk1_matrix_out(chunk1_matrix),
                        chunk2_matrix_out(chunk2_matrix);
      if((left_padding !=0) || (right_padding != 0)) {
        int32 tot_num_rows = num_rows+left_padding+right_padding;
        chunk1_matrix_out.Resize(tot_num_rows, num_cols, kUndefined);
        chunk2_matrix_out.Resize(tot_num_rows, num_cols, kUndefined);
        for(int32 row = 0; row < tot_num_rows; row++) {
          int32 row_in = row - left_padding;
          if (row_in < 0) {
            row_in = 0;
          } else if (row_in >= num_rows ) {
            row_in = num_rows -1;
          }
          SubVector<BaseFloat> vec_chunk1_in(chunk1_matrix, row_in),
                               vec_chunk1_out(chunk1_matrix_out, row),
                               vec_chunk2_in(chunk2_matrix, row_in),
                               vec_chunk2_out(chunk2_matrix_out, row);
          vec_chunk1_out.CopyFromVec(vec_chunk1_in);
          vec_chunk2_out.CopyFromVec(vec_chunk2_in);
        }
      }
      //generate the NnetIo
      NnetIo nnet_io1 = NnetIo("input", -left_padding, chunk1_matrix_out),
             nnet_io2 = NnetIo("input", -left_padding, chunk2_matrix_out);
      //modify the n index, so that in a mini-batch Nnet3Example, the adjacent
      //two NnetIos come from the same source signal.
      for (std::vector<Index>::iterator indx_it = nnet_io1.indexes.begin();
        indx_it != nnet_io1.indexes.end(); ++indx_it) {
        indx_it->n = 0;
      }
      for (std::vector<Index>::iterator indx_it = nnet_io2.indexes.begin();
        indx_it != nnet_io2.indexes.end(); ++indx_it) {
        indx_it->n = 1;
      }
      NnetExample eg;
      eg.io.push_back(nnet_io1);
      eg.io.push_back(nnet_io2);
      if (compress) {
        eg.Compress();
      }
      example_writer.Write(key, eg);
      num_egs_written += 1;
    }
    KALDI_LOG << "Finished generating examples, "
              << "successfully convert " << num_egs_written << " chunks into examples out of "
              << num_read << " chunks";
    return (num_egs_written == 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
