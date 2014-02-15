// bin/analyze-counts.cc

// Copyright 2012-2014 Brno University of Technology (Author: Karel Vesely)

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

/** @brief Sums the pdf vectors to counts, this is used to obtain prior counts for hybrid decoding.
*/
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

#include <iomanip>
#include <algorithm>
#include <numeric>

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::uint64 uint64;
  try {
    const char *usage =
        "Counts element frequencies from integer vector table.\n"
        "(eg. for example to get pdf-counts to estimate DNN-output priros, or data analysis)\n"
        "Verbosity : level 1 => print frequencies and histogram\n"
        "\n"
        "Usage:  analyze-counts  [options] <alignments-rspecifier> <counts-wxfilname>\n"
        "e.g.: \n"
        " analyze-counts ark:1.ali prior.counts\n"
        " Show phone counts by:\n"
        " ali-to-phone --per-frame=true ark:1.ali ark:- | analyze-counts --verbose=1 ark:- - >/dev/null\n";
    
    ParseOptions po(usage);
    
    bool binary = false;
    std::string symbol_table_filename = "";
    
    po.Register("binary", &binary, "write in binary mode");
    po.Register("symbol-table", &symbol_table_filename, "Read symbol table for display of counts");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string alignments_rspecifier = po.GetArg(1),
        wxfilename = po.GetArg(2);

    SequentialInt32VectorReader reader(alignments_rspecifier);

    // Get the counts
    std::vector<uint64> counts;
    int32 num_done = 0;
    for (; !reader.Done(); reader.Next()) {
      std::string key = reader.Key();
      std::vector<int32> alignment = reader.Value();

      for (size_t i = 0; i < alignment.size(); i++) {
        int32 value = alignment[i];
        if(value >= counts.size()) {
          counts.resize(value+1);
        }
        counts[value]++; // Accumulate
      }

      num_done++;
    }

    // We need at least one occurence for each tgt, so there is no nan during decoding
    std::vector<uint64> counts_nozero(counts);
    for(size_t i = 0; i < counts.size(); i++) {
      if(counts_nozero[i] == 0) {
        KALDI_WARN << "Zero count for element " << i << ", force setting to one."
                   << " This avoids divide-by-zero when used counts used in decoding.";
        counts_nozero[i]++;
      }
    }

    // Write
    Output ko(wxfilename, binary);
    WriteIntegerVector(ko.Stream(), binary, counts_nozero);

    ////
    //// THE REST IS FOR ANALYSIS, IT GETS PRINTED TO LOG
    ////
    if (symbol_table_filename != "" || (kaldi::g_kaldi_verbose_level >= 1)) {

      // load the symbol table
      fst::SymbolTable *elem_syms = NULL;
      if (symbol_table_filename != "") {
          elem_syms = fst::SymbolTable::ReadText(symbol_table_filename);
          if (!elem_syms)
            KALDI_ERR << "Could not read symbol table from file " << symbol_table_filename;
      }
      
      // sort the counts
      std::vector<std::pair<int32,int32> > sorted_counts;
      for (int32 i = 0; i < counts.size(); i++) {
        sorted_counts.push_back(std::make_pair(counts[i], i));
      }
      std::sort(sorted_counts.begin(), sorted_counts.end());
      
      // print
      std::ostringstream os;
      int32 sum = std::accumulate(counts.begin(),counts.end(), 0);
      os << "Printing...\n### The sorted count table," << std::endl;
      os << "count\t(norm),\tid\t(symbol):" << std::endl;
      for (int32 i=0; i<sorted_counts.size(); i++) {
        os << sorted_counts[i].first << "\t(" 
           << static_cast<float>(sorted_counts[i].first) / sum << "),\t"
           << sorted_counts[i].second << "\t" 
           << (elem_syms != NULL ? std::string("(")+elem_syms->Find(sorted_counts[i].second)+")" : "")
           << std::endl;
      }
      os << "\n#total " << sum 
         << " (" << static_cast<float>(sum)/100/3600 << "h)" 
         << std::endl;
      KALDI_LOG << os.str();
    }

    KALDI_LOG << "Summed " << num_done << " int32 vectors to counts.";
    KALDI_LOG << "Counts written to " << wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


