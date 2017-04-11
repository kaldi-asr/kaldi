// bin/phones-to-prons.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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
#include "fstext/fstext-lib.h"

// Create FST that accepts the phone sequence, with any number
// of word-start and word-end symbol in between each phone.
void CreatePhonesAltFst(const std::vector<int32> &phones,
                        int32 word_start_sym,
                        int32 word_end_sym,
                        fst::VectorFst<fst::StdArc> *ofst) {
  using fst::StdArc;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Weight Weight;

  ofst->DeleteStates();
  StateId cur_s = ofst->AddState();
  ofst->SetStart(cur_s); // will be 0.
  for (size_t i = 0; i < phones.size(); i++) {
    StateId next_s = ofst->AddState();
    // add arc to next state.
    ofst->AddArc(cur_s, StdArc(phones[i], phones[i], Weight::One(),
                               next_s));
    cur_s = next_s;
  }
  for (StateId s = 0; s <= cur_s; s++) {
    ofst->AddArc(s, StdArc(word_end_sym, word_end_sym,
                                    Weight::One(), s));
    ofst->AddArc(s, StdArc(word_start_sym, word_start_sym,
                                    Weight::One(), s));
  }
  ofst->SetFinal(cur_s, Weight::One());
  {
    fst::OLabelCompare<StdArc> olabel_comp;
    ArcSort(ofst, olabel_comp);
  }
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using fst::VectorFst;
  using fst::StdArc;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert pairs of (phone-level, word-level) transcriptions to\n"
        "output that indicates the phones assigned to each word.\n"
        "Format is standard format for archives of vector<vector<int32> >\n"
        "i.e. :\n"
        "utt-id  600 4 7 19 ; 512 4 18 ; 0 1\n"
        "where 600, 512 and 0 are the word-ids (0 for non-word phones, e.g.\n"
        "optional-silence introduced by the lexicon), and the phone-ids\n"
        "follow the word-ids.\n"
        "Note: L_align.fst must have word-start and word-end symbols in it\n"
        "\n"
        "Usage:  phones-to-prons [options] <L_align.fst> <word-start-sym> "
        "<word-end-sym> <phones-rspecifier> <words-rspecifier> <prons-wspecifier>\n"
        "e.g.: \n"
        " ali-to-phones 1.mdl ark:1.ali ark:- | \\\n"
        "  phones-to-prons L_align.fst 46 47 ark:- "
        "'ark:sym2int.pl -f 2- words.txt text|' ark:1.prons\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }


    std::string lex_fst_filename = po.GetArg(1),
        word_start_sym_str = po.GetArg(2),
        word_end_sym_str = po.GetArg(3),
        phones_rspecifier = po.GetArg(4),
        words_rspecifier = po.GetArg(5),
        prons_wspecifier = po.GetArg(6);

    int32 word_start_sym, word_end_sym;

    if (!ConvertStringToInteger(word_start_sym_str, &word_start_sym)
        || word_start_sym <= 0)
      KALDI_ERR << "Invalid word start symbol (expecting integer >= 0): "
                 << word_start_sym_str;
    if (!ConvertStringToInteger(word_end_sym_str, &word_end_sym)
        || word_end_sym <= 0 || word_end_sym == word_start_sym)
      KALDI_ERR << "Invalid word end symbol (expecting integer >= 0"
                 << ", different from word start symbol): "
                 << word_end_sym_str;

    // L should be lexicon with word start and end symbols marked.
    VectorFst<StdArc> *L = fst::ReadFstKaldi(lex_fst_filename);
    {
      // Make sure that L is sorted on output symbol (words).
      fst::OLabelCompare<StdArc> olabel_comp;
      ArcSort(L, olabel_comp);
    }

    SequentialInt32VectorReader phones_reader(phones_rspecifier);
    RandomAccessInt32VectorReader words_reader(words_rspecifier);

    int32 n_done = 0, n_err = 0;

    std::string empty;
    Int32VectorVectorWriter prons_writer(prons_wspecifier);

    for (; !phones_reader.Done(); phones_reader.Next()) {
      std::string key = phones_reader.Key();
      const std::vector<int32> &phones = phones_reader.Value();
      if (!words_reader.HasKey(key)) {
        KALDI_WARN << "Not processing utterance " << key << " because no word "
                   << "transcription found.";
        n_err++;
        continue;
      }
      const std::vector<int32> &words = words_reader.Value(key);

      // convert word alignment to acceptor and compose it with lexicon.
      // phn2word will have phones (and word start/end symbols) on its
      // input, and words on its output.  It will enode the alternative
      // pronunciations of this word-sequence, with word start and end
      // symbols at the appropriate places.
      VectorFst<StdArc> phn2word;
      {
        VectorFst<StdArc> words_acceptor;
        MakeLinearAcceptor(words, &words_acceptor);
        Compose(*L, words_acceptor, &phn2word);
      }
      if (phn2word.Start() == fst::kNoStateId) {
        KALDI_WARN << "Phone to word FST for utterance " << key
                   << "is empty (either decoding for this utterance did "
                   << "not reach end-state, or mismatched lexicon.)";
        n_err++;
        continue;
      }

      VectorFst<StdArc> phones_alt_fst;
      CreatePhonesAltFst(phones, word_start_sym, word_end_sym, &phones_alt_fst);

      // phnx2word will have phones and word-start and word-end symbols
      // on the input side, and words on the output side.
      VectorFst<StdArc> phnx2word;
      Compose(phones_alt_fst, phn2word, &phnx2word);

      if (phnx2word.Start() == fst::kNoStateId) {
        KALDI_WARN << "phnx2word FST for utterance " << key
                   << "is empty (either decoding for this utterance did "
                   << "not reach end-state, or mismatched lexicon.)";
        if (g_kaldi_verbose_level >= 2) {
          KALDI_LOG << "phn2word FST is below:";
          fst::FstPrinter<StdArc> fstprinter(phn2word, NULL, NULL, NULL, false, true, "\t");
          fstprinter.Print(&std::cerr, "standard error");
          KALDI_LOG << "phone sequence is: ";
          for (size_t i = 0; i < phones.size(); i++)
            std::cerr << phones[i] << ' ';
          std::cerr << '\n';
        }
        continue;
      }

      // Now get the best path in phnx2word.
      VectorFst<StdArc> phnx2word_best;
      ShortestPath(phnx2word, &phnx2word_best);

      // Now get seqs of phones and words.
      std::vector<int32> phnx, words2;
      StdArc::Weight garbage;
      if (!fst::GetLinearSymbolSequence(phnx2word_best,
                                        &phnx, &words2, &garbage))
        KALDI_ERR << "phnx2word is not a linear transducer (code error?)";
      if (words2 != words)
        KALDI_ERR << "words have changed! (code error?)";

      // Now, "phnx" should be the phone sequence with start and end
      // symbols included.  At this point we break it up into segments,
      // and try to match it up with words.
      std::vector<std::vector<int32> > prons;
      if (!ConvertPhnxToProns(phnx, words,
                              word_start_sym, word_end_sym,
                              &prons)) {
        KALDI_WARN << "Error converting phones and words to prons "
                   << " (mismatched or non-marked lexicon or partial "
                   << " alignment?)";
        n_err++;
        continue;
      }
      prons_writer.Write(key, prons);
      n_done++;
    }
    KALDI_LOG << "Done " << n_done << " utterances; " << n_err << " had errors.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
