// Copyright 2015 Vimal Manohar

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

using namespace kaldi;

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage = "Splits speakers using diarization assigments\n"
                        "Usage: split-speakers-on-diarization-assigments <utt2spk-rspecifier> <diar-rspecifier> <utt2spk-wspecifier>\n"
                        " e.g.: split-speakers-on-diarization-assigments ark,t:data/dev/utt2spk ark,t:exp/diarization_dev/diarization.txt ark,t:exp/diarization_dev/utt2spk\n"
                        "\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string utt2spk_rspecifier = po.GetArg(1);
    std::string diar_rspecifier = po.GetArg(2);
    std::string utt2spk_wspecifier = po.GetArg(3);

    SequentialTokenReader utt2spk_reader(utt2spk_rspecifier);
    RandomAccessInt32Reader diar_reader(diar_rspecifier);
    TokenWriter utt2spk_writer(utt2spk_wspecifier);

    int32 num_done = 0, num_err = 0;
    for (; !utt2spk_reader.Done(); utt2spk_reader.Next()) {
      std::string utt = utt2spk_reader.Key();
      std::string spk = utt2spk_reader.Value();

      if (!diar_reader.HasKey(utt)) {
        KALDI_WARN << "No speaker assignment for utterance " << utt;
        num_err++;
        continue;
      } else {
        int32 spk_id = diar_reader.Value(utt);
        std::ostringstream oss;
        oss << spk << "-" << spk_id;
        utt2spk_writer.Write(utt, oss.str());
      }
      num_done++;
    }

    KALDI_LOG << "Done splitting speaker for " << num_done << " utterances; "
              << "failed for " << num_err;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
