// ivectorbin/ivector-subtract-global-mean.cc

// Copyright 2013  Daniel Povey

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


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Copies a table of iVectors but subtracts the global mean as\n"
        "it does so.  The mean may be specified as the first argument; if not,\n"
        "the sum of the input iVectors is used.\n"
        "\n"
        "Usage: ivector-subtract-global-mean <ivector-rspecifier> <ivector-wspecifier>\n"
        "or: ivector-subtract-global-mean <mean-rxfliename> <ivector-rspecifier> <ivector-wspecifier>\n"
        "e.g.: ivector-subtract-global-mean scp:ivectors.scp ark:-\n"
        "or: ivector-subtract-global-mean mean.vec scp:ivectors.scp ark:-\n"
        "See also: ivector-mean\n";

    ParseOptions po(usage);

    bool subtract_mean = true;
    po.Register("subtract-mean", &subtract_mean,
                "If true, subtract mean; if false, just copy the input.");

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    int64 num_done = 0;

    if (po.NumArgs() == 2) {
      std::string ivector_rspecifier = po.GetArg(1),
          ivector_wspecifier = po.GetArg(2);

      Vector<double> sum;

      std::vector<std::pair<std::string, Vector<BaseFloat>*> > ivectors;

      SequentialBaseFloatVectorReader ivector_reader(ivector_rspecifier);
      BaseFloatVectorWriter ivector_writer(ivector_wspecifier);


      for (; !ivector_reader.Done(); ivector_reader.Next()) {
        std::string key = ivector_reader.Key();
        const Vector<BaseFloat> &ivector = ivector_reader.Value();
        if (sum.Dim() == 0) sum.Resize(ivector.Dim());
        sum.AddVec(1.0, ivector);
        num_done++;
        ivectors.push_back(std::make_pair(key, new Vector<BaseFloat>(ivector)));
      }

      KALDI_LOG << "Read " << num_done << " iVectors.";

      if (num_done != 0) {
        KALDI_LOG << "Norm of iVector mean was " << (sum.Norm(2.0) / num_done);
        for (size_t i = 0; i < ivectors.size(); i++) {
          std::string key = ivectors[i].first;
          Vector<BaseFloat> *ivector = ivectors[i].second;
          if (subtract_mean)
            ivector->AddVec(-1.0 / num_done, sum);
          ivector_writer.Write(key, *ivector);
          delete ivector;
          ivectors[i].second = NULL;
        }
      }
    } else {
      // po.NumArgs() == 3
      std::string mean_rxfilename = po.GetArg(1),
          ivector_rspecifier = po.GetArg(2),
          ivector_wspecifier = po.GetArg(3);
      Vector<BaseFloat> mean;
      ReadKaldiObject(mean_rxfilename, &mean);

      SequentialBaseFloatVectorReader ivector_reader(ivector_rspecifier);
      BaseFloatVectorWriter ivector_writer(ivector_wspecifier);
      for (; !ivector_reader.Done(); ivector_reader.Next()) {
        std::string key = ivector_reader.Key();
        Vector<BaseFloat> ivector = ivector_reader.Value();
        ivector.AddVec(-1.0, mean);
        ivector_writer.Write(key, ivector);
        num_done++;
      }
    }
    KALDI_LOG << "Wrote " << num_done << " mean-subtracted iVectors";
    return (num_done != 0 ? 0 : 1);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
