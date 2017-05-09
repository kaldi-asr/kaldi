// bin/image-preprocess.cc

// Copyright 2017   Johns Hopkins University (Author: Daniel Povey)

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
#include "transform/transform-common.h"

namespace kaldi {

void PadImageHelper(VectorBase<BaseFloat> &src,
                    VectorBase<BaseFloat> *dest) {
  KALDI_ASSERT(src.Dim() == dest->Dim());
  int32 dim = src.Dim();
  for (int32 i = 0; i < dim; i++) {
    int32 count = 1;
    BaseFloat total = src(i);
    if (i > 0) { total += src(i-1); count++; }
    if (i + 1 < dim) { total += src(i+1); count++; }
    (*dest)(i) = total / count;
  }
}

// Pads an image with the specified number of pixels.  The padding method
// is a little bit like fuzzy reflection; if a pixel is n pixels away
// from the image's edge, its value will be the average of all
// pixels that are in the image, that are within a square of side n.
// or.. something like that.  Not exactly that.  See the code.
void PadImage(int32 num_channels,
              int32 horizontal_padding, int32 vertical_padding,
              Matrix<BaseFloat> *image) {
  Matrix<BaseFloat> orig_image(*image);
  KALDI_ASSERT(image->NumCols() % num_channels == 0);
  int32 orig_width = image->NumRows(),
      orig_height = image->NumCols() / num_channels,
      new_width = orig_width + 2 * horizontal_padding,
      new_height = orig_height + 2 * vertical_padding;
  image->Resize(new_width, new_height * num_channels);

  // first fill in the non-padded part of the image.
  SubMatrix<BaseFloat> image_part(*image, horizontal_padding, orig_width,
                                  vertical_padding * num_channels,
                                  orig_height * num_channels);
  image_part.CopyFromMat(orig_image);

  // vertical padding
  for (int32 c = 0; c < num_channels; c++) {
    Vector<BaseFloat> src(orig_width),
        dest(orig_width);
    // first vertical padding on the top (this terminology is only correct if we
    // assume the top has the lowest numbered indexes).
    int32 src_height = vertical_padding;
    for (int32 w = 0; w < orig_width; w++)
      src(w) = (*image)(horizontal_padding + w, src_height * num_channels + c);
    for (int32 h = vertical_padding - 1; h >= 0; h--) {
      PadImageHelper(src, &dest);
      for (int32 w = 0; w < orig_width; w++)
        (*image)(horizontal_padding + w, h * num_channels + c) = dest(w);
      src.Swap(&dest);
    }
    // Now vertical padding on the bottom
    src_height = orig_height + vertical_padding - 1;
    for (int32 w = 0; w < orig_width; w++)
      src(w) = (*image)(horizontal_padding + w, src_height * num_channels + c);
    for (int32 h = orig_height + vertical_padding; h < new_height; h++) {
      PadImageHelper(src, &dest);
      for (int32 w = 0; w < orig_width; w++)
        (*image)(horizontal_padding + w, h * num_channels + c) = dest(w);
      src.Swap(&dest);
    }
  }

  // horizontal padding
  for (int32 c = 0; c < num_channels; c++) {
    Vector<BaseFloat> src(new_height),
        dest(new_height);
    // first horizontal padding on the left.
    int32 src_width = horizontal_padding;
    for (int32 h = 0; h < new_height; h++)
      src(h) = (*image)(src_width, h * num_channels + c);
    for (int32 w = horizontal_padding - 1; w >= 0; w--) {
      PadImageHelper(src, &dest);
      for (int32 h = 0; h < new_height; h++)
        (*image)(w, h * num_channels + c) = dest(h);
      src.Swap(&dest);
    }
    // now horizontal padding on the right.
    src_width = orig_width + horizontal_padding - 1;
    for (int32 h = 0; h < new_height; h++)
      src(h) = (*image)(src_width, h * num_channels + c);
    for (int32 w = orig_width + horizontal_padding; w < new_width; w++) {
      PadImageHelper(src, &dest);
      for (int32 h = 0; h < new_height; h++)
        (*image)(w, h * num_channels + c) = dest(h);
      src.Swap(&dest);
    }
  }
}


void SubtractMean(int32 num_channels, Matrix<BaseFloat> *image) {
  KALDI_ASSERT(image->NumCols() % num_channels == 0);
  int32 width = image->NumRows(),
      height = image->NumCols() / num_channels;
  for (int32 c = 0; c < num_channels; c++) {
    double sum = 0.0;
    for (int32 w = 0; w < width; w++) {
      for (int32 h = 0; h < height; h++) {
        sum += (*image)(w, h * num_channels + c);
      }
    }
    BaseFloat average = static_cast<BaseFloat>(sum / (width * height));
    for (int32 w = 0; w < width; w++) {
      for (int32 h = 0; h < height; h++) {
        (*image)(w, h * num_channels + c) -= average;
      }
    }
  }
}

} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "This is a special-purpose program used in image preprocessing for\n"
        "padding pixels on the boundaries of a supplied image.  It also supports\n"
        "optional mean subtraction.\n"
        "\n"
        "Usage: image-preprocess [options] <matrix-in-rspecifier> <matrix-out-wspecifier>\n";

    bool binary = true;
    bool compress = false;
    bool subtract_mean = false;
    int32 compression_method_in = 1;
    int32 horizontal_padding = 0,
        vertical_padding = 0;
    int32 num_channels = 1;
    ParseOptions po(usage);

    po.Register("binary", &binary,
                "Write in binary mode (only relevant if output is a wxfilename)");
    po.Register("compress", &compress, "If true, write output in compressed form");
    po.Register("subtract-mean", &subtract_mean, "If true, subtract the mean per "
                "channel of the output.");
    po.Register("num-channels", &num_channels, "The number of channels (colors) "
                "in the input image");
    po.Register("horizontal-padding", &horizontal_padding,
                "Number of pixels to pad on the horizontal axis (left and "
                "right). Padding is based on interpolation of nearby values.");
    po.Register("vertical-padding", &vertical_padding,
                "Number of pixels to pad on the vertical axis (top and "
                "bottom).  Padding is based on interpolation of nearby values.");
    po.Register("compression-method", &compression_method_in,
                "Only relevant if --compress=true; the method (1 through 7) to "
                "compress the matrix.  Search for CompressionMethod in "
                "src/matrix/compressed-matrix.h.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }


    std::string matrix_rspecifier = po.GetArg(1),
        out_wspecifier = po.GetArg(2);

    CompressionMethod compression_method = static_cast<CompressionMethod>(
        compression_method_in);

    int num_done = 0;
    BaseFloatMatrixWriter matrix_writer(compress ? "" : out_wspecifier);
    CompressedMatrixWriter compressed_writer(compress ? out_wspecifier : "");

    SequentialBaseFloatMatrixReader reader(matrix_rspecifier);
    for (; !reader.Done(); reader.Next(), num_done++) {
      Matrix<BaseFloat> image = reader.Value();

      if (horizontal_padding != 0 || vertical_padding != 0)
        PadImage(num_channels, horizontal_padding, vertical_padding,
                 &image);

      if (subtract_mean)
        SubtractMean(num_channels, &image);
      if (compress) {
        CompressedMatrix cmat(image, compression_method);
        compressed_writer.Write(reader.Key(), cmat);
      } else {
        matrix_writer.Write(reader.Key(), image);
      }
    }
    KALDI_LOG << "Processed " << num_done << " images.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


/*

cat <<EOF | image-preprocess --num-channels=2 --horizontal-padding=2 --vertical-padding=2  ark:- ark,t:-
foo [ 0.5 0.1 0.4 0.1
 0.5 0.1 0.4 0.1
 0.5 0.1 0.35 0.1 ]
EOF



*/
