// durmodbin/nnet3-durmodel-init.cc

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
  using nnet3::Nnet;

  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Init the nnet3-duration model.\n"
        "Usage:  nnet3-durmodel-init [options] <durmodel-in> <raw-nnet3-in>"
        " <nnet3-dur-model-out>\n"
        "e.g.: \n"
        "  nnet3-durmodel-init durmodel.mdl nnet.raw 0.mdl";

    bool binary_write = true;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string in_durmodel_filename = po.GetArg(1),
                in_nnet_filename = po.GetArg(2),
                out_model_filename = po.GetArg(3);

    PhoneDurationModel durmodel;
    ReadKaldiObject(in_durmodel_filename, &durmodel);

    Nnet nnet;
    ReadKaldiObject(in_nnet_filename, &nnet);

    NnetPhoneDurationModel nnet_dur_model(durmodel, nnet);
    WriteKaldiObject(nnet_dur_model, out_model_filename, binary_write);

    KALDI_LOG << "Done writing the model.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
