// bin/ali-to-phones.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "fst/fstlib.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert model-level alignments to phone-sequences (in integer, not text, form)\n"
        "Usage:  ali-to-phones  [options] <model> <alignments-rspecifier> <phone-transcript-wspecifier>\n"
        "e.g.: \n"
        " ali-to-phones 1.mdl ark:1.ali ark:phones.tra\n";
    ParseOptions po(usage);
    bool per_frame = false;
    bool write_lengths = false;
    po.Register("per-frame", &per_frame, "If true, write out the frame-level phone alignment (else phone sequence)");
    po.Register("write-lengths", &write_lengths, "If true, write the #frames for each phone (different format)");
    
    po.Read(argc, argv);

    KALDI_ASSERT(!(per_frame && write_lengths) && "Incompatible options.");
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        alignments_rspecifier = po.GetArg(2),
        phones_wspecifier = po.GetArg(3);

    TransitionModel trans_model;
    ReadKaldiObject(model_filename, &trans_model);

    SequentialInt32VectorReader reader(alignments_rspecifier);
    std::string empty;
    Int32VectorWriter phones_writer(write_lengths ? empty : phones_wspecifier);
    Int32PairVectorWriter pair_writer(write_lengths ? phones_wspecifier : empty);

    int32 n_done = 0;
    
    for (; !reader.Done(); reader.Next()) {
      std::string key = reader.Key();
      const std::vector<int32> &alignment = reader.Value();

      std::vector<std::vector<int32> > split;
      SplitToPhones(trans_model, alignment, &split);

      if (!write_lengths) {
        std::vector<int32> phones;
        for (size_t i = 0; i < split.size(); i++) {
          KALDI_ASSERT(split[i].size() > 0);
          int32 tid = split[i][0],
              tstate = trans_model.TransitionIdToTransitionState(tid),
              phone = trans_model.TransitionStateToPhone(tstate);
          int32 num_repeats = split[i].size();
          KALDI_ASSERT(num_repeats!=0);
          if (per_frame)
            for(int32 j = 0; j < num_repeats; j++)
              phones.push_back(phone);
          else
            phones.push_back(phone);
        }
        phones_writer.Write(key, phones);        
      } else {
        std::vector<std::pair<int32, int32> > pairs;
        for (size_t i = 0; i < split.size(); i++) {
          KALDI_ASSERT(split[i].size() > 0);
          int32 tid = split[i][0],
              tstate = trans_model.TransitionIdToTransitionState(tid),
              phone = trans_model.TransitionStateToPhone(tstate);
          int32 num_repeats = split[i].size();
          KALDI_ASSERT(num_repeats!=0);
          pairs.push_back(std::make_pair(phone, num_repeats));
        }
        pair_writer.Write(key, pairs);
      }
      n_done++;
    }
    KALDI_LOG << "Done " << n_done << " utterances."; 
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


