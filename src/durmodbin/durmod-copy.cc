// durmodbin/durmod-copy.cc
// Author: Hossein Hadian

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
  using nnet3::Nnet;
  try {
    const char *usage =
        "Copy the model. If it is used with --raw=true then only the raw nnet"
        " inside the model is copied. Also --set-raw-nnet can be used to "
        " set the raw nnet inside the model.\n"
        "Usage:  durmod-copy [options] <in-dur-model> <out-dur-model>\n"
        "e.g.:\n"
        "  durmod-copy-nnet --binary=false 0.mdl 1.mdl\n"
        "";

    bool binary_write = true, raw = false;
    std::string raw_nnet = "";

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("raw", &raw, "If true, write only 'raw' neural net.");
    po.Register("set-raw-nnet", &raw_nnet,
                "Set the raw nnet inside the model to the one provided in "
                "the option string.");

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string in_model_filename = po.GetArg(1),
                out_model_filename = po.GetArg(2);
    PhoneDurationModel durmodel;
    ReadKaldiObject(in_model_filename, &durmodel);

    if (!raw_nnet.empty()) {
      Nnet nnet;
      ReadKaldiObject(raw_nnet, &nnet);
      durmodel.SetNnet(nnet);
    }

    if (raw)
      WriteKaldiObject(durmodel.GetNnet(), out_model_filename, binary_write);
    else
      WriteKaldiObject(durmodel, out_model_filename, binary_write);

    KALDI_LOG << "Done writing the nnet.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
