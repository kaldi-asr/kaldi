// durmodbin/durmod-info.cc

// Copyright 2015 Hossein Hadian

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
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "util/parse-options.h"
#include "tree/build-tree.h"
#include "durmod/kaldi-durmod.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Print info about the phone duration model.\n"
        "Usage:  durmod-info [options] <duration-model>\n"
        "e.g.: \n"
        "  durmod-info durmodel.mdl";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_filename = po.GetArg(1);
    PhoneDurationModel durmodel;
    ReadKaldiObject(model_filename, &durmodel);
    PhoneDurationEgsMaker egs_maker(durmodel);

    std::cout << "feature-dim: " << egs_maker.FeatureDim()
              << std::endl
              << "num-binary-features: " << egs_maker.NumBinaryFeatures()
              << std::endl
              << "num-phone-identities: " << egs_maker.NumPhoneIdentities()
              << std::endl
              << "max-duration: " << durmodel.MaxDuration()
              << std::endl
              << "left-context: " << durmodel.LeftContext()
              << std::endl
              << "right-context: " << durmodel.RightContext()
              << std::endl
              << "full-context-size: " << durmodel.FullContextSize()
              << std::endl;
    KALDI_LOG << "Done.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
