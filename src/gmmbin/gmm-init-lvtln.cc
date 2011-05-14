// gmmbin/gmm-init-lvtln.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "transform/lvtln.h"


int main(int argc, char *argv[])
{
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize lvtln transforms\n"
        "Usage:  gmm-init-lvtln [options] <lvtln-out>\n"
        "e.g.: \n"
        " gmm-init-lvtln --dim = 13 --num-classes = 21 --default-class = 10 1.lvtln\n";

    bool binary = true;
    int32 dim = 13;
    int32 default_class = 10;
    int32 num_classes = 21;

    int32 num_blocks = 1;
    bool male_female = false;  // if true, initialize as prototype male-female models
    // (like very small VTLN warp factor).   In this case, "num-blocks" becomes
    // relevant: set e.g. to 3 if using deltas+delta-deltas.
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("dim", &dim, "feature dimension");
    po.Register("num-classes", &num_classes, "Number of transforms to be trained");
    po.Register("default-class", &default_class, "Number of transforms to be trained");
    po.Register("num-blocks", &num_blocks, "Number of blocks of features (e.g. 3 if deltas+delta-deltas); relevent only if male-female = true.");
    po.Register("male-female", &male_female, "If true, initialize to special two-class transform approximating a delta male-female difference.");

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string lvtln_wxfilename = po.GetArg(1);

    if (!male_female) {
      LinearVtln lvtln(dim, num_classes, default_class);
      {
        Output ko(lvtln_wxfilename, binary);
        lvtln.Write(ko.Stream(), binary);
      }
    } else {
      if (dim % num_blocks != 0 || num_blocks <= 0)
        KALDI_ERR << "gmm-init-lvtln: num-blocks has invalid value " << num_blocks
                  << ": must divide dimension and be >= 1.";
      LinearVtln lvtln(dim, 2, default_class);
      Matrix<BaseFloat> M_male(dim, dim);
      Matrix<BaseFloat> M_female(dim, dim);
      int32  block_size = dim / num_blocks;
      M_male.SetUnit();
      M_female.SetUnit();
      for (int32 b = 0; b < num_blocks; b++) {
        SubMatrix<BaseFloat> male_block(M_male, b*block_size, block_size,
                                        b*block_size, block_size);
        SubMatrix<BaseFloat> female_block(M_female, b*block_size, block_size,
                                          b*block_size, block_size);
        for (int32 i = 0; i < block_size-1; i++) {
          BaseFloat delta = (0.1 * i) / block_size;  // goes from 0 to 0.1.
          male_block(i, i+1) = delta;
          male_block(i+1, i) = -delta;
          female_block(i, i+1) = -delta;
          female_block(i+1, i) = delta;
        }
      }
      lvtln.SetTransform(0, M_male);
      lvtln.SetTransform(1, M_female);
      {
        Output ko(lvtln_wxfilename, binary);
        lvtln.Write(ko.Stream(), binary);
      }
    }
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}

