// latbin/nbest-to-ctm.cc

// Copyright 2012  Johns Hopkins University (Author: Daniel Povey)

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
#include "lat/lattice-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Takes as input lattices which must be linear (single path),\n"
        "and must be in CompactLattice form where the transition-ids on the arcs\n"
        "have been aligned with the word boundaries... typically the input will\n"
        "be a lattice that has been piped through lattice-1best and then\n"
        "lattice-word-align.  It outputs ctm format (with integers in place of words),\n"
        "assuming the frame length is 0.01 seconds by default (change this with the\n"
        "--frame-length option).  Note: the output is in the form\n"
        "utterance-id 1 <begin-time> <end-time> <word-id>\n"
        "and you can post-process this to account for segmentation issues and to \n"
        "convert ints to words; note, the times are relative to start of the utterance.\n"
        "\n"
        "Usage: nbest-to-ctm [options] <aligned-linear-lattice-rspecifier> <ctm-wxfilename>\n"
        "e.g.: lattice-1best --acoustic-weight=0.08333 ark:1.lats | \\\n"
        "      lattice-align-words data/lang/phones/word_boundary.int exp/dir/final.mdl ark:- ark:- | \\\n"
        "      nbest-to-ctm ark:- 1.ctm\n";
    
    ParseOptions po(usage);

    BaseFloat frame_shift = 0.01;
    int32 precision = 2;
    po.Register("frame-shift", &frame_shift, "Time in seconds between frames.\n");
    po.Register("precision", &precision,
                "Number of decimal places for start duration times\n");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        ctm_wxfilename = po.GetArg(2);

    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    
    int32 n_done = 0, n_err = 0;

    Output ko(ctm_wxfilename, false); // false == non-binary write mode.
    ko.Stream() << std::fixed;  // Set to "fixed" floating point model, where precision() specifies
    // the #digits after the decimal point.
    ko.Stream().precision(precision);
    
    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();

      std::vector<int32> words, times, lengths;

      if (!CompactLatticeToWordAlignment(clat, &words, &times, &lengths)) {
        n_err++;
        KALDI_WARN << "Format conversion failed for key " << key;
      } else {
        KALDI_ASSERT(words.size() == times.size() &&
                     words.size() == lengths.size());
        for (size_t i = 0; i < words.size(); i++) {
          if (words[i] == 0)  // Don't output anything for <eps> links, which
            continue; // correspond to silence....
          ko.Stream() << key << " 1 " << (frame_shift * times[i]) << ' '
                      << (frame_shift * lengths[i]) << ' ' << words[i] <<std::endl;
        }
        n_done++;
      }
    }
    ko.Close(); // Note: we don't normally call Close() on these things,
    // we just let them go out of scope and it happens automatically.
    // We do it this time in order to avoid wrongly printing out a success message
    // if the stream was going to fail to close
            
    KALDI_LOG << "Converted " << n_done << " linear lattices to ctm format; "
              << n_err  << " had errors.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
