// ivectorbin/logistic-regression-copy.cc

// Copyright 2014  Daniel Povey

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
#include "ivector/logistic-regression.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Copy a logistic-regression model, possibly changing the binary mode;\n"
        "also supports the --scale-priors option which can scale the prior probabilities\n"
        "the model assigns to different classes (e.g., you can remove the effect of\n"
        "unbalanced training data by scaling by the inverse of the class priors in the\n"
        "training data)\n"
        "Usage: logistic-regression-copy [options] <model-in> <model-out>\n"
        "e.g.: echo '[ 2.6 1.7 3.9 1.24 7.5 ]' | logistic-regression-copy --scale-priors=- \\\n"
        "  1.model scaled_priors.mdl\n";

    ParseOptions po(usage);

    bool binary = true;
    std::string scale_priors_rxfilename;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("scale-priors", &scale_priors_rxfilename, "(extended) filename for file "
                "containing a vector of prior-scales (e.g. inverses of training priors)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_rxfilename = po.GetArg(1),
        model_wxfilename = po.GetArg(2);


    LogisticRegression model;
    ReadKaldiObject(model_rxfilename, &model);

    if (scale_priors_rxfilename != "") {
      Vector<BaseFloat> prior_scales;
      ReadKaldiObject(scale_priors_rxfilename, &prior_scales);
      model.ScalePriors(prior_scales);
    }

    WriteKaldiObject(model, model_wxfilename, binary);

    KALDI_LOG << "Wrote model to " << PrintableWxfilename(model_wxfilename);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
