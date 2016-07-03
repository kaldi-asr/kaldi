// kwsbin/transcripts-to-fsts.cc

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)

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
#include "fstext/kaldi-fst-io.h"
#include "fstext/fstext-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    typedef kaldi::int32 int32;
    typedef kaldi::uint64 uint64;

    const char *usage =
        "Build a linear acceptor for each transcription. Read in the transcriptions in archive\n"
        "format and write out the linear acceptors in archive format with the same key.\n"
        "\n"
        "Usage: transcripts-to-fsts [options]  transcriptions-rspecifier fsts-wspecifier\n"
        " e.g.: transcripts-to-fsts ark:train.tra ark:train.fsts\n";

    ParseOptions po(usage);

    std::string left_compose = "";
    std::string right_compose = "";
    bool project_input = false;
    bool project_output = false;

    po.Register("left-compose", &left_compose, "Compose the given FST to the left");
    po.Register("right-compose", &right_compose, "Compose the given FST to the right");
    po.Register("project-input", &project_input, "Project input labels if true");
    po.Register("project-output", &project_output, "Project input labels if true");

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string transcript_rspecifier = po.GetArg(1),
        fst_wspecifier = po.GetOptArg(2);


    SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
    TableWriter<VectorFstHolder> fst_writer(fst_wspecifier);

    // Read the possible given FSTs
    VectorFst<StdArc> *lfst = NULL;
    VectorFst<StdArc> *rfst = NULL;
    if (left_compose != "") {
      lfst = ReadFstKaldi(left_compose);
    }
    if (right_compose != "") {
      rfst = ReadFstKaldi(right_compose);
    }

    int32 n_done = 0;
    for (; !transcript_reader.Done(); transcript_reader.Next()) {
      std::string key = transcript_reader.Key();
      std::vector<int32> transcript = transcript_reader.Value();
      transcript_reader.FreeCurrent();

      VectorFst<StdArc> fst;
      MakeLinearAcceptor(transcript, &fst);

      if (lfst != NULL) {
        VectorFst<StdArc> composed_fst;
        Compose(*lfst, fst, &composed_fst);
        fst = composed_fst;
      }
      
      if (rfst != NULL) {
        VectorFst<StdArc> composed_fst;
        Compose(fst, *rfst, &composed_fst);
        fst = composed_fst;
      }

      if (project_input) {
        Project(&fst, PROJECT_INPUT);
      }

      if (project_output) {
        Project(&fst, PROJECT_OUTPUT);
      }

      fst_writer.Write(key, fst);

      n_done++;
    }

    delete lfst;
    delete rfst;

    KALDI_LOG << "Done " << n_done << " transcriptions";
    return (n_done != 0 ? 0 : 1);    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
