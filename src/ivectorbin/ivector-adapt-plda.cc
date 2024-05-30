// ivectorbin/ivector-adapt-plda.cc

// Copyright 2013-2014  Daniel Povey

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
#include "ivector/plda.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Adapt a PLDA object using unsupervised adaptation-data iVectors from a different\n"
        "domain to the training data.\n"
        "\n"
        "Usage: ivector-adapt-plda [options] <plda-in> <ivectors-rspecifier> <plda-out>\n"
        "e.g.: ivector-adapt-plda plda ark:ivectors.ark plda.adapted\n";

    ParseOptions po(usage);

    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    PldaUnsupervisedAdaptorConfig config;
    config.Register(&po);



    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
        ivector_rspecifier = po.GetArg(2),
        plda_wxfilename = po.GetArg(3);

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

    SequentialBaseFloatVectorReader ivector_reader(ivector_rspecifier);

    int32 num_done = 0;
    PldaUnsupervisedAdaptor adaptor;
    for (; !ivector_reader.Done(); ivector_reader.Next(), num_done++)
      adaptor.AddStats(1.0, ivector_reader.Value());

    adaptor.UpdatePlda(config, &plda);

    WriteKaldiObject(plda, plda_wxfilename, binary);

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
