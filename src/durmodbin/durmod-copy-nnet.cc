// durmodbin/durmod-copy-nnet.cc
// Copyright 2015 Johns Hopkins University

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
        "Copy the raw nnet.\n"
        "Usage:  durmod-copy-nnet [options] <dur-model> <raw-nnet>\n"
        "e.g.: \n"
        "  durmod-copy-nnet 0.durmod 0.nnet";
    ParseOptions po(usage);
    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_filename = po.GetArg(1),
                nnet_filename = po.GetArg(2);
    PhoneDurationModel durmodel;
    ReadKaldiObject(model_filename, &durmodel);
    WriteKaldiObject(durmodel.GetNnet(), nnet_filename, false);
    KALDI_LOG << "Done writing the nnet.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
