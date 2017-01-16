// gmmbin/gmm-global-copy.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

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
#include "gmm/diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Copy a diagonal-covariance GMM\n"
        "Usage:  gmm-global-copy [options] <model-in> <model-out>\n"
        "  or    gmm-global-copy [options] <model-wspecifier> <model-wspecifier>\n"
        "e.g.: gmm-global-copy --binary=false 1.model - | less";

    bool binary_write = true;
    ParseOptions po(usage);
    po.Register("binary", &binary_write, 
                "Write in binary mode (only relevant if output is a wxfilename)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // all these "fn"'s are either rspecifiers or filenames.

    bool in_is_rspecifier =
        (ClassifyRspecifier(model_in_filename, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(model_out_filename, NULL, NULL, NULL)
         != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix archives with regular files (copying gmm models)";

    if (!in_is_rspecifier) {
      DiagGmm gmm;
      {
        bool binary_read;
        Input ki(model_in_filename, &binary_read);
        gmm.Read(ki.Stream(), binary_read);
      }
      WriteKaldiObject(gmm, model_out_filename, binary_write);

      KALDI_LOG << "Written model to " << model_out_filename;
    } else {
      SequentialDiagGmmReader gmm_reader(model_in_filename);
      DiagGmmWriter gmm_writer(model_out_filename);
  
      int32 num_done = 0;
      for (; !gmm_reader.Done(); gmm_reader.Next(), num_done++) {
        gmm_writer.Write(gmm_reader.Key(), gmm_reader.Value());
      }

      KALDI_LOG << "Wrote " << num_done << " GMM models to "               << model_out_filename;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


