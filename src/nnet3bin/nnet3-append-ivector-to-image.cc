// nnet3bin/nnet3-append-ivector-to-image.cc

// Copyright      2017  Johns Hopkins University (author:  Daniel Povey)

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
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Appends image ivector after each pixel of num_channels\n"
        "\n"
        "Usage:  nnet3-append-ivector-to-image [options] <rspecifier1> "
        "<rspecifier2>\n"
        "\n"
        "e.g.:\n"
        "nnet3-append-ivector-to-image scp:images.scp \\\n"
        "scp:ivector.scp ark:egs.ark\n";

    int32 num_channels = 3,
    ivector_dim = 30;

    ParseOptions po(usage);
    po.Register("num-channels", &num_channels, "Number of channels.");
    po.Register("ivector-dimension", &ivector_dim, "Height of patches.");
    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }


    std::string image_rspecifier = po.GetArg(1),
        ivector_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader  image_reader(image_rspecifier);
    RandomAccessBaseFloatMatrixReader ivector_reader(ivector_rspecifier);
    BaseFloatMatrixWriter matrix_writer(examples_wspecifier);
    int32 num_done = 0, num_err = 0;

    for (; !image_reader.Done(); image_reader.Next()) {
      std::string key = image_reader.Key();
      const Matrix<BaseFloat> &image_feats = image_reader.Value();

      if (image_feats.NumCols() % num_channels != 0) {
        KALDI_ERR << "Number of columns of image "
            << key << " must be a multiple of the number of channels.";
      }

      int32 num_cols = image_feats.NumCols() / num_channels,
          num_rows = image_feats.NumRows();

      int32 new_num_cols = num_cols*(ivector_dim + num_channels);
      Matrix<BaseFloat> image_ivec_feats(num_rows,new_num_cols);

      if (!ivector_reader.HasKey(key)) {
        KALDI_WARN << "Could not find input for key " << key
                     << " for ivector";
        num_err++;
      }
      else {
        const Matrix<BaseFloat> &ivector_feats = ivector_reader.Value(key);
        image_ivec_feats.SetZero();
        for (int32 i = 0; i < num_rows; i++) {
          for (int32 j = 0; j < num_cols*num_channels; j++) {
            int32 num_ivecs = j/num_channels;
            int32 remainder = j%num_channels;
            if(j%num_channels !=0 || j == 0){
              image_ivec_feats(i,num_ivecs*(ivector_dim + num_channels) + remainder) = image_feats(i,j);
            }
            else{
              for(int32 icol = 0; icol < ivector_dim; icol++) {
                image_ivec_feats(i,(num_ivecs-1)*(ivector_dim + num_channels) + num_channels + icol) = ivector_feats(0,icol);
              }
              image_ivec_feats(i,num_ivecs*(ivector_dim + num_channels)) = image_feats(i,j);
            }
          }
          for(int32 icol = 0; icol < ivector_dim; icol++) {
                image_ivec_feats(i,new_num_cols - ivector_dim + icol) = ivector_feats(0,icol);
          }
        }
        matrix_writer.Write(key, image_ivec_feats);
        num_done++;
      }
    }

    if (num_err > 0)
      KALDI_WARN << num_err << " utterances had errors and could "
          "not be processed.";
    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
