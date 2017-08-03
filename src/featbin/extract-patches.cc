// featbin/extract-patches.cc

// Copyright 2017 David Snyder

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Extract all possible nxm patches from an archive of images.  "
        "Images must be stored as matrices and channels are required\n"
        "to be arranged in a particular order.  Suppose an input image "
        "is 32x96 and we've specified that num-channels=3.  Then we will\n"
        "interpret the the input to mean that columns 0 to 31 correspond "
        "to the first channel, 32 to 63 to the second channel, and 64 to\n"
        "95 for the last channel.  The output is a new matrix, where each "
        "row corresponds to a patch, which includes all channels.\n"
        "Usage:  "
        "extract-patches [options...] <images-rspecifier> "
        " <patches-wspecifier>\n";

    int32 num_channels = 3,
        patch_height = 8,
        patch_width = 8;

    ParseOptions po(usage);
    po.Register("num-channels", &num_channels, "Number of channels.");
    po.Register("patch-height", &patch_height, "Height of patches.");
    po.Register("patch-width", &patch_width, "Width of patches.");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1),
        wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader image_reader(rspecifier);
    BaseFloatMatrixWriter patch_writer(wspecifier);

    for (; !image_reader.Done(); image_reader.Next()) {
      const Matrix<BaseFloat> &image = image_reader.Value();
      std::string key = image_reader.Key();
      if (image.NumRows() < patch_width
          || image.NumCols() < num_channels * patch_height) {
        KALDI_ERR << "Image " << key
          << " is too small for the requested patch size.";
      }

      if (image.NumCols() % num_channels != 0) {
        KALDI_ERR << "Number of columns of image "
            << key << " must be a multiple of the number of channels.";
      }

      int32 num_cols = image.NumCols() / num_channels,
          num_rows = image.NumRows();
      int32 patch_num_rows = (num_rows - patch_width + 1)
          * (num_cols - patch_height + 1),
          patch_num_cols = patch_height * patch_width * num_channels;


      // The output patches.  Each row corresponds to a patch.
      Matrix<BaseFloat> patches(patch_num_rows, patch_num_cols);

      // Iterate over the channels
      for (int32 c = 0; c < num_channels; c++) {

        // We will copy the patches corresponding to channel c here.
        SubMatrix<BaseFloat> sub_patches(patches, 0, patch_num_rows,
            c * patch_height * patch_width, patch_height * patch_width);

        // Now iterate over the rows and columns of the part of the image
        // corresponding to channel c.
        for (int32 i = 0; i < num_rows - patch_width + 1; i++) {
          for (int32 j = 0; j < num_cols - patch_height + 1; j++) {
            int32 dest_row = i * (num_cols - patch_height + 1) + j;

            // Select patch corresponding to channel c.
            //SubMatrix<BaseFloat> patch(image, i, patch_height, c * num_cols + j, patch_width);
            Matrix<BaseFloat> patch(patch_width, patch_height);
            patch.SetZero();
            for(int32 prow = 0; prow < patch_width; prow++) {
              for(int32 pcol = 0; pcol < patch_height; pcol++) {
                patch(prow,pcol) = image(i+prow,(j+pcol)*num_channels + c);
              }
            }
            // Serialize the patch.
            Vector<BaseFloat> vec_patch(patch_height * patch_width);
            vec_patch.CopyRowsFromMat(patch);

            // Copy the patch into a row of the output matrix.
            SubVector<BaseFloat> sub_vec_patch(sub_patches, dest_row);
            sub_vec_patch.CopyFromVec(vec_patch);
          }
        }
      }
      patch_writer.Write(key, patches);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
