// nnet3bin/nnet3-egs-augment-image.cc

// Copyright      2017  Johns Hopkins University (author:  Daniel Povey)
//                2017  Yiwen Shao

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
#include "hmm/transition-model.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {
namespace nnet3 {

struct ImageAugmentationConfig {
  int32 num_channels;
  BaseFloat horizontal_flip_prob;
  BaseFloat horizontal_shift;
  BaseFloat vertical_shift;

  ImageAugmentationConfig():
      num_channels(1),
      horizontal_flip_prob(0.0),
      horizontal_shift(0.0),
      vertical_shift(0.0) { }


  void Register(ParseOptions *po) {
    po->Register("num-channels", &num_channels, "Number of colors in the image."
                 "It is is important to specify this (helps interpret the image "
                 "correctly.");
    po->Register("horizontal-flip-prob", &horizontal_flip_prob,
                 "Probability of doing horizontal flip");
    po->Register("horizontal-shift", &horizontal_shift,
                 "Maximum allowed horizontal shift as proportion of image "
                 "width.  Padding is with closest pixel.");
    po->Register("vertical-shift", &vertical_shift,
                 "Maximum allowed vertical shift as proportion of image "
                 "height.  Padding is with closest pixel.");
  }

  void Check() const {
    KALDI_ASSERT(num_channels >= 1);
    KALDI_ASSERT(horizontal_flip_prob >= 0 &&
                 horizontal_flip_prob <= 1);
    KALDI_ASSERT(horizontal_shift >= 0 && horizontal_shift <= 1);
    KALDI_ASSERT(vertical_shift >= 0 && vertical_shift <= 1);
  }
};


// Flips the image horizontally.
void HorizontalFlip(MatrixBase<BaseFloat> *image) {
  int32 num_rows = image->NumRows();
  Vector<BaseFloat> temp(image->NumCols());
  for (int32 r = 0; r < num_rows / 2; r++) {
    SubVector<BaseFloat> row_a(*image, r), row_b(*image,
                                                 num_rows - r - 1);
    temp.CopyFromVec(row_a);
    row_a.CopyFromVec(row_b);
    row_b.CopyFromVec(temp);
  }
}

 // Shifts the image horizontally by 'horizontal_shift' (+ve == to the right).
void HorizontalShift(int32 horizontal_shift,
                     MatrixBase<BaseFloat> *image) {
  int32 num_rows = image->NumRows(), num_cols = image->NumCols();
  for (int32 r = 0; r < num_rows; r++) {
    int32 current_row_n, origin_row_n;
    // +ve == to the right, do shifting from right to left; otherwise, from left to right
    if (horizontal_shift > 0) {
      current_row_n = r;
    } else {
      current_row_n = num_rows - 1 - r;
    }
    // use the neareast value to fill points out of boundary
    if (current_row_n + horizontal_shift > num_rows - 1) {
      origin_row_n = num_rows - 1;
    } else if (current_row_n + horizontal_shift < 0) {
      origin_row_n = 0;
    } else {
      origin_row_n = current_row_n + horizontal_shift;
    }

    SubVector<BaseFloat> current_row(*image, current_row_n), origin_row(*image, origin_row_n);
    current_row.CopyFromVec(origin_row);
  }
}

/* Shifts the image vertically by 'vertical_shift' (+ve == to the top).*/
void VerticalShift(int32 vertical_shift,
                   int32 num_channels,
                   MatrixBase<BaseFloat> *image) {
  int32 num_rows = image->NumRows(),
      num_cols = image->NumCols(), height = num_cols / num_channels;
  KALDI_ASSERT(num_cols % num_channels == 0);
  for (int32 r = 0; r < num_rows; r++) {
    BaseFloat *this_row = image->RowData(r);
    for (int32 c = 0; c < height; c++) {
      int32 current_index, origin_index;
      // +ve == to the top, do shifting from top to bottom; otherwise, bottom to top
      if (vertical_shift > 0) {
        current_index = height - 1 - c;
      } else {
        current_index = c;
      }
      // use the neareast value to fill points out of boundary
      if (current_index + vertical_shift > height - 1) {
        origin_index = height -1;
      } else if (current_index + vertical_shift < 0) {
        origin_index = 0;
      } else {
        origin_index = current_index + vertical_shift;
      }
      for (int32 ch = 0; ch < num_channels; ch++) {
        this_row[num_channels * current_index + ch] =
            this_row[num_channels * origin_index + ch];
      }
    }
  }
}



/**
  This function randomly modifies (perturbs) the image.
  @param [in] config  Configuration class that says how
                      to perturb the image.
  @param [in,out] image  The image matrix to be modified.
                     image->NumRows() is the width (number of x values) in
                     the image; image->NumCols() is the height times number
                     of channels/colors (channel varies the fastest).
 */
void PerturbImage(const ImageAugmentationConfig &config,
                  MatrixBase<BaseFloat> *image) {
  config.Check();
  int32 image_width = image->NumRows(),
      num_channels = config.num_channels,
      image_height = image->NumCols() / num_channels;
  if (image->NumCols() % num_channels != 0) {
    KALDI_ERR << "Number of columns in image must divide the number "
        "of channels";
  }
  if (WithProb(config.horizontal_flip_prob)) {
    HorizontalFlip(image);
  }
  { // horizontal shift
    int32 horizontal_shift_max =
        static_cast<int32>(0.5 + config.horizontal_shift * image_width);
    if (horizontal_shift_max > image_width - 1)
      horizontal_shift_max = image_width - 1;  // would be very strange.
    int32 horizontal_shift = RandInt(-horizontal_shift_max,
                                     horizontal_shift_max);
    if (horizontal_shift != 0)
      HorizontalShift(horizontal_shift, image);
  }

  { // vertical shift
    int32 vertical_shift_max =
        static_cast<int32>(0.5 + config.vertical_shift * image_height);
    if (vertical_shift_max > image_height - 1)
      vertical_shift_max = image_height - 1;  // would be very strange.
    int32 vertical_shift = RandInt(-vertical_shift_max,
                                     vertical_shift_max);
    if (vertical_shift != 0)
      VerticalShift(vertical_shift, num_channels, image);
  }

}


/**
   This function does image perturbation as directed by 'config'
   The example 'eg' is expected to contain a NnetIo member with the
   name 'input', representing an image.
 */
void PerturbImageInNnetExample(
    const ImageAugmentationConfig &config,
    NnetExample *eg) {
  int32 io_size = eg->io.size();
  bool found_input = false;
  for (int32 i = 0; i < io_size; i++) {
    NnetIo &io = eg->io[i];
    if (io.name == "input") {
      found_input = true;
      Matrix<BaseFloat> image;
      io.features.GetMatrix(&image);
      // note: 'GetMatrix' may uncompress if it was compressed.
      // We won't recompress, but this won't matter because this
      // program is intended to be used as part of a pipe, we
      // likely won't be dumping the perturbed data to disk.
      PerturbImage(config, &image);

      // modify the 'io' object.
      io.features = image;
    }
  }
  if (!found_input)
    KALDI_ERR << "Nnet example to perturb had no NnetIo object named 'input'";
}


} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Copy examples (single frames or fixed-size groups of frames) for neural\n"
        "network training, doing image augmentation inline (copies after possibly\n"
        "modifying of each image, randomly chosen according to configuration\n"
        "parameters).\n"
        "E.g.:\n"
        "  nnet3-egs-augment-image --horizontal-flip-prob=0.5 --horizontal-shift=0.1\\\n"
        "       --vertical-shift=0.1 --srand=103 --num-channels=3 ark:- ark:-\n"
        "\n"
        "Requires that each eg contain a NnetIo object 'input', with successive\n"
        "'t' values representing different x offsets , and the feature dimension\n"
        "representing the y offset and the channel (color), with the channel\n"
        "varying the fastest.\n"
        "See also: nnet3-copy-egs\n";


    int32 srand_seed = 0;

    ImageAugmentationConfig config;

    ParseOptions po(usage);
    po.Register("srand", &srand_seed, "Seed for the random number generator");
    config.Register(&po);

    po.Read(argc, argv);

    srand(srand_seed);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string examples_rspecifier = po.GetArg(1),
        examples_wspecifier = po.GetArg(2);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);


    int64 num_done = 0;
    for (; !example_reader.Done(); example_reader.Next(), num_done++) {
      std::string key = example_reader.Key();
      NnetExample eg(example_reader.Value());
      PerturbImageInNnetExample(config, &eg);
      example_writer.Write(key, eg);
    }
    KALDI_LOG << "Perturbed" << num_done << " neural-network training images.";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
