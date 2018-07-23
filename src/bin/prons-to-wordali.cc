// bin/prons-to-wordali.cc

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
#include "fstext/fstext-utils.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using fst::VectorFst;
  using fst::StdArc;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Caution: this program relates to older scripts and is deprecated,\n"
        "for modern scripts see egs/wsj/s5/steps/{get_ctm,get_train_ctm}.sh\n"
        "Given per-utterance pronunciation information as output by \n"
        "words-to-prons, and per-utterance phone alignment information\n"
        "as output by ali-to-phones --write-lengths, output word alignment\n"
        "information that can be turned into the ctm format.\n"
        "Outputs is pairs of (word, #frames), or if --per-frame is given,\n"
        "just the word for each frame.\n"
        "Note: zero word-id usually means optional silence.\n"
        "Format is standard format for archives of vector<pair<int32, int32> >\n"
        "i.e. :\n"
        "utt-id  600 22 ; 1028 32 ; 0 41\n"
        "where 600, 1028 and 0 are the word-ids, and 22, 32 and 41 are the\n"
        "lengths.\n"
        "\n"
        "Usage:  prons-to-wordali [options] <prons-rspecifier>"
        " <phone-lengths-rspecifier> <wordali-wspecifier>\n"
        "e.g.: \n"
        " ali-to-phones 1.mdl ark:1.ali ark:- | \\\n"
        "  phones-to-prons L_align.fst 46 47 ark:- 'ark:sym2int.pl -f 2- words.txt text|' \\\n"
        "  ark:- | prons-to-wordali ark:- \\\n"
        "    \"ark:ali-to-phones --write-lengths 1.mdl ark:1.ali ark:-|\" ark:1.wali\n";
    
    ParseOptions po(usage);
    bool per_frame = false;
    po.Register("per-frame", &per_frame, "If true, write out the frame-level word alignment (else word sequence)");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string prons_rspecifier = po.GetArg(1),
        phone_lengths_rspecifier = po.GetArg(2),
        wordali_wspecifier = po.GetArg(3);
        
                
    SequentialInt32VectorVectorReader prons_reader(prons_rspecifier);
    RandomAccessInt32PairVectorReader phones_reader(phone_lengths_rspecifier);

    std::string empty;
    Int32PairVectorWriter pair_writer(per_frame ? empty : wordali_wspecifier);
    Int32VectorWriter frame_writer(per_frame ? wordali_wspecifier : empty);

    int32 n_done = 0, n_err = 0;
    
    for (; !prons_reader.Done(); prons_reader.Next()) {
      std::string key = prons_reader.Key();
      const std::vector<std::vector<int32> > &prons = prons_reader.Value();
      if (!phones_reader.HasKey(key)) {
        KALDI_WARN << "Not processing utterance " << key << " because no phone "
                   << "alignment found.";
        n_err++;
        continue;
      }
      // first member of each pair is phone; second is length in
      // frames.
      const std::vector<std::pair<int32, int32> > &phones =
          phones_reader.Value(key);

      std::vector<std::pair<int32, int32> > word_alignment;

      size_t p = 0; // index into "phones".
      for (size_t i = 0; i < prons.size(); i++) {
        if (!(prons[i].size() >= 1)) {
          KALDI_WARN << "Invalid, empty pronunciation.";
          n_err++;
          continue;
        }
        int32 word = prons[i][0], word_len = 0;
        for (size_t j = 1; j < prons[i].size(); j++, p++) {
          if (!(static_cast<size_t>(p) < phones.size() &&
                prons[i][j] == phones[p].first) ) {
            KALDI_WARN << "For key " << key << ", mismatch between prons and phones.";
            n_err++;
            continue;
          }
          word_len += phones[p].second;
        }
        word_alignment.push_back(std::make_pair(word, word_len));
      }
      if (static_cast<size_t>(p) != phones.size()) {
        KALDI_WARN << "For key " << key << ", mismatch between prons and phones (wrong #phones)";
        n_err++;
        continue;
      }

      if (!per_frame) {
        pair_writer.Write(key, word_alignment);
      } else {
        std::vector<int32> word_per_frame;
        for (size_t i = 0; i < word_alignment.size(); i++) {
          int32 word = word_alignment[i].first,
              len = word_alignment[i].second;
          for (int32 j = 0; j < len; j++)
            word_per_frame.push_back(word);
        }
        frame_writer.Write(key, word_per_frame);
      }
      n_done++;
    }
    KALDI_LOG << "Done " << n_done << " utterances; " << n_err << " had errors.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


