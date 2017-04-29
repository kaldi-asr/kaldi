// nnet3bin/nnet3-egs-augment-image.cc

// Copyright      2017  Johns Hopkins University (author:  Daniel Povey)
//                2017  Hossein Hadian

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

void ApplyAffineTransform(MatrixBase<BaseFloat> &transform,
                          int32 num_channels,
                          MatrixBase<BaseFloat> *image) {
  int32 num_rows = image->NumRows(),
        num_cols = image->NumCols(),
        height = num_cols / num_channels;
  KALDI_ASSERT(num_cols % num_channels == 0);
  Matrix<BaseFloat> original_image(*image);
  for (int32 r = 0; r < num_rows; r++) {
    for (int32 c = 0; c < height; c++) {
      BaseFloat r_old = transform(0, 0) * r +
                                          transform(0, 1) * c + transform(0, 2);
      BaseFloat c_old = transform(1, 0) * r +
                                          transform(1, 1) * c + transform(1, 2);

      // we are going to do bilinear interpolation between 4 closest points
      // to the point [r_old, c_old]. r1  <=  r_old  <=  r2
      //                              c1  <=  c_old  <=  c2

      int32 r1 = static_cast<int32>(floor(r_old));
      int32 c1 = static_cast<int32>(floor(c_old));
      int32 r2 = r1 + 1;
      int32 c2 = c1 + 1;

      // TODO(hhadian): it might be better to use the following 'if' only when
      // none of the 4 points are available. For now, we only check one of them.
      if (r1 < 0 || c1 < 0 || r1 > num_rows - 1 ||
          c1 > num_cols - 1) {  // use the nearest point
        int32 nearest_row = (r > 0) ? (r - 1) : (r + 1);
        for (int32 ch = 0; ch < num_channels; ch++) {
          (*image)(r, num_channels * c + ch) =
                                   (*image)(nearest_row, num_channels * c + ch);
        }
      } else {
        if (r2 > num_rows - 1)
          r2 = num_rows - 1;
        if (c2 > num_cols - 1)
          c2 = num_cols - 1;
        for (int32 ch = 0; ch < num_channels; ch++) {
          // find the values at the 4 points
          BaseFloat p11 = original_image(r1, num_channels * c1 + ch),
                    p12 = original_image(r1, num_channels * c2 + ch),
                    p21 = original_image(r2, num_channels * c1 + ch),
                    p22 = original_image(r2, num_channels * c2 + ch);
          BaseFloat interpolate = (r1 + 1 - r_old) * (c1 + 1 - c_old) * p11 +
                                  (r_old - r1) * (c1 + 1 - c_old) * p12 +
                                  (r1 + 1 - r_old) * (c_old - c1) * p21 +
                                  (r_old - r1) * (c_old - c1) * p22;
          (*image)(r, num_channels * c + ch) = interpolate;
        }
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

  // We do an affine transform which
  // handles flipping, translation, rotation, magnification, and shear.

  // Prepare the affine transform for geometric transformation
  Matrix<BaseFloat> transform_mat(3, 3, kUndefined);
  transform_mat.SetUnit();

  Matrix<BaseFloat> shift_mat(3, 3, kUndefined);
  shift_mat.SetUnit();
  // translation (shift) mat:
  // [ 1   0  x_shift
  //   0   1  y_shift
  //   0   0  1       ]

  BaseFloat horizontal_shift = (2.0 * RandUniform() - 1.0) *
                               config.horizontal_shift * image_width;
  BaseFloat vertical_shift = (2.0 * RandUniform() - 1.0) *
                             config.vertical_shift * image_height;
  shift_mat(0, 2) = floor(horizontal_shift);  // best to floor shifts to avoid blurring
  shift_mat(1, 2) = floor(vertical_shift);
  // since we will center the image before applying the transform,
  // horizontal flipping is simply achieved by setting [0, 0] to -1:
  if (WithProb(config.horizontal_flip_prob))
    shift_mat(0, 0) = -1;


  Matrix<BaseFloat> rotation_mat(3, 3, kUndefined);
  rotation_mat.SetUnit();
  // rotation mat:
  // [ cos(theta)  -sin(theta)  0
  //   sin(theta)  cos(theta)   0
  //   0           0            1 ]


  Matrix<BaseFloat> shear_mat(3, 3, kUndefined);
  shear_mat.SetUnit();
  // shear mat:
  // [ 1    -sin(shear)   0
  //   0     cos(shear)   0
  //   0     0            1 ]


  Matrix<BaseFloat> zoom_mat(3, 3, kUndefined);
  zoom_mat.SetUnit();
  // zoom mat:
  // [ x_zoom   0   0
  //   0   y_zoom   0
  //   0     0      1 ]

  // transform_mat = rotation_mat * shift_mat * shear_mat * zoom_mat:
  transform_mat.AddMatMat(1.0, shift_mat, kNoTrans,
                               shear_mat, kNoTrans, 0.0);
  transform_mat.AddMatMatMat(1.0, rotation_mat, kNoTrans,
                               transform_mat, kNoTrans,
                               zoom_mat, kNoTrans, 0.0);

  if (transform_mat.IsUnit())  // nothing to do
    return;

  // we should now change the origin of transform to the center of
  // the image (necessary for flipping, zoom, shear, and rotation)
  // we do this by using two translations: one before the main transform
  // and one after.
  Matrix<BaseFloat> set_origin_mat(3, 3, kUndefined);
  set_origin_mat.SetUnit();
  set_origin_mat(0, 2) = image_width / 2.0 - 0.5;
  set_origin_mat(1, 2) = image_height / 2.0 - 0.5;

  Matrix<BaseFloat> reset_origin_mat(3, 3, kUndefined);
  reset_origin_mat.SetUnit();
  reset_origin_mat(0, 2) = -image_width / 2.0 + 0.5;
  reset_origin_mat(1, 2) = -image_height / 2.0 + 0.5;

  // transform_mat = set_origin_mat * transform_mat * reset_origin_mat
  transform_mat.AddMatMatMat(1.0, set_origin_mat, kNoTrans,
                                  transform_mat, kNoTrans,
                                  reset_origin_mat, kNoTrans, 0.0);
  ApplyAffineTransform(transform_mat, config.num_channels, image);
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
