// durmodbin/durmod-init.cc

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
        "Init the phone duration model.\n"
        "Usage:  durmod-init [options] <roots-file> <binary-feature-questions>"
        " <dur-model>\n"
        "e.g.: \n"
        "  durmod-init roots.int extra_questions.int durmodel.mdl";

    bool binary_write = true;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    PhoneDurationModelOptions opts;
    opts.Register(&po);
    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string roots_filename = po.GetArg(1),
        questions_filename = po.GetArg(2),
        model_filename = po.GetArg(3);
    std::vector<std::vector<int32> > roots;
    {
      Input ki(roots_filename.c_str());
      std::vector<bool> is_shared, is_split;
      ReadRootsFile(ki.Stream(), &roots, &is_shared, &is_split);
    }
    KALDI_LOG << "Read " << roots.size() << " roots (phone sets)";
    std::vector<std::vector<int32> > questions;
    if (!ReadIntegerVectorVectorSimple(questions_filename, &questions)) {
      KALDI_ERR << "Cannot read the phonetic questions from "
                << questions_filename;
    }
    KALDI_LOG << "Read " << questions.size() << " phonetic questions";
    PhoneDurationModel durmod(opts, roots, questions);
    PhoneDurationEgsMaker egs_maker(durmod);
    int32 hidden_dim1 = egs_maker.FeatureDim() * 6;
    durmod.InitNnet(egs_maker.FeatureDim(), hidden_dim1,
                    egs_maker.OutputDim() * 1.5, egs_maker.OutputDim());
    // TODO(hhadian): print some nice info
    WriteKaldiObject(durmod, model_filename, binary_write);
    KALDI_LOG << "Done writing the model.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
