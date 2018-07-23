// featbin/wav-copy.cc

// Copyright 2013-2014  Daniel Povey
//                2016  Aalto University (author: Peter Smit)

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
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Copy wave file or archives of wave files\n"
        "\n"
        "Usage: wav-copy [options] <wav-rspecifier> <wav-wspecifier>\n"
        "  or:  wav-copy [options] <wav-rxfilename> <wav-wxfilename>\n"
        "e.g. wav-copy scp:wav.scp ark:-\n"
        "     wav-copy wav.ark:123456 -\n"
        "See also: wav-to-duration extract-segments\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_in_fn = po.GetArg(1),
        wav_out_fn = po.GetArg(2);

    bool in_is_rspecifier = (ClassifyRspecifier(wav_in_fn, NULL, NULL)
                             != kNoRspecifier),
        out_is_wspecifier = (ClassifyWspecifier(wav_out_fn, NULL, NULL, NULL)
                              != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix archives with regular files";

    if (in_is_rspecifier) {
      int32 num_done = 0;

      SequentialTableReader<WaveHolder> wav_reader(wav_in_fn);
      TableWriter<WaveHolder> wav_writer(wav_out_fn);

      for (; !wav_reader.Done(); wav_reader.Next()) {
        wav_writer.Write(wav_reader.Key(), wav_reader.Value());
        num_done++;
      }
      KALDI_LOG << "Copied " << num_done << " wave files";
      return (num_done != 0 ? 0 : 1);
    } else {
      bool binary = true;
      Input ki(wav_in_fn, &binary);
      Output ko(wav_out_fn, binary, false);
      WaveHolder wh;
      if (!wh.Read(ki.Stream())) {
        KALDI_ERR << "Read failure from "
                  << PrintableRxfilename(wav_in_fn);
      }
      if (!WaveHolder::Write(ko.Stream(), true, wh.Value())) {
        KALDI_ERR << "Write failure to "
                  << PrintableWxfilename(wav_out_fn);
      }
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

