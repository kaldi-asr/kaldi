// gmmbin/gmm-init-lvtln.cc

// Copyright 2009-2011  Microsoft Corporation
//                2014  Johns Hopkins University (author: Daniel Povey)

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
#include "transform/lvtln.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize lvtln transforms\n"
        "Usage:  gmm-init-lvtln [options] <lvtln-out>\n"
        "e.g.: \n"
        " gmm-init-lvtln --dim=13 --num-classes=21 --default-class=10 1.lvtln\n";

    bool binary = true;
    int32 dim = 13;
    int32 default_class = 10;
    int32 num_classes = 21;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("dim", &dim, "feature dimension");
    po.Register("num-classes", &num_classes, "Number of transforms to be trained");
    po.Register("default-class", &default_class, "Index of default transform, "
                "to be used if no data is available for training");

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string lvtln_wxfilename = po.GetArg(1);

    // We'll set the transforms separately using gmm-train-lvtln-special
    LinearVtln lvtln(dim, num_classes, default_class);
    WriteKaldiObject(lvtln, lvtln_wxfilename, binary);
    
    KALDI_LOG << "Initialized LVTLN object and wrote it to "
              << PrintableWxfilename(lvtln_wxfilename);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
