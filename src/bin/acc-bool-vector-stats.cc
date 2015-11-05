// bin/acc-bool-vector-stats.cc

// Copyright 2015   Vimal Manohar

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
#include "decision-tree/tree-stats.h"

namespace kaldi { 

template<>
// Specialization for bool vector
bool BasicVectorVectorHolder<bool>::Read(std::istream &is) {
  t_.clear();
  bool is_binary;
  if (!InitKaldiInputStream(is, &is_binary)) {
    KALDI_WARN << "Failed reading binary header\n";
    return false;
  }
  if (!is_binary) {
    // In text mode, we terminate with newline.
    try {  // catching errors from ReadBasicType..
      std::vector<bool> v;  // temporary vector
      while (1) {
        int i = is.peek();
        if (i == -1) {
          KALDI_WARN << "Unexpected EOF";
          return false;
        } else if (static_cast<char>(i) == '\n') {
          if (!v.empty()) {
            KALDI_WARN << "No semicolon before newline (wrong format)";
            return false;
          } else { is.get(); return true; }
        } else if (std::isspace(i)) {
          is.get();
        } else if (static_cast<char>(i) == ';') {
          t_.push_back(v);
          v.clear();
          is.get();
        } else {  // some object we want to read...
          bool b;
          ReadBasicType(is, false, &b);  // throws on error.
          v.push_back(b);
        }
      }
    } catch(std::exception &e) {
      KALDI_WARN << "BasicVectorVectorHolder::Read, read error";
      if (!IsKaldiError(e.what())) { std::cerr << e.what(); }
      return false;
    }
  } else {  // binary mode.
    size_t filepos = is.tellg();
    try {
      int32 size;
      ReadBasicType(is, true, &size);
      t_.resize(size);
      for (std::vector<std::vector<bool> >::iterator iter = t_.begin();
          iter != t_.end();
          ++iter) {
        int32 size2;
        ReadBasicType(is, true, &size2);
        iter->resize(size2);
        for (std::vector<bool>::iterator iter2 = iter->begin();
            iter2 != iter->end();
            ++iter2) {
          bool b;
          ReadBasicType(is, true, &b);
          *iter2 = b;
        }
      }
      return true;
    } catch (...) {
      KALDI_WARN << "Read error or unexpected data at archive entry beginning at file position " << filepos;
      return false;
    }
  }
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::decision_tree_classifier;

    typedef kaldi::int32 int32;  

    typedef SequentialTableReader<BasicVectorVectorHolder<bool> >  SequentialBooleanVectorVectorReader;

    const char *usage =
        "Turn posteriors representing quantized feats into into boolean vector.\n"
        "Usage: acc-bool-vector-stats <bits-rspecifier> <alignment-rspecifier> <stats-wxfilename\n"
        " e.g.: acc-bool-vector-stats ark:bits.1.ark ark:ali.1.ark -\n"
        "See also: acc-tree-stats, quantize-feats, post-to-bits\n";
    
    bool binary = true;
    int32 num_classes = -1;
    bool ignore_irrelevant_classes = false;

    ParseOptions po(usage); 

    po.Register("num-classes", &num_classes, "Number of classes. Must match "
                "the alignment file i.e. if C is number of classes, "
                "then alignment[i] < C, ");
    po.Register("ignore-irrelevant-classes", &ignore_irrelevant_classes,
                "If set true, ignore frames for which the aligned class is "
                ">= num-classes. Otherwise exits with an error.");
                
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
     
    if (num_classes < 0) {
      KALDI_ERR << "--num-classes must be specified and must be > 0";
    }


    std::string bits_rspecifier = po.GetArg(1),
                alignment_rspecifier = po.GetArg(2),
                accs_out_wxfilename = po.GetArg(3);

    SequentialBooleanVectorVectorReader bits_reader(bits_rspecifier);
    RandomAccessInt32VectorReader alignment_reader(alignment_rspecifier);
    
    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0;
    
    BooleanTreeStats tree_stats(num_classes);

    for (; !bits_reader.Done(); bits_reader.Next()) {
      const std::string &key = bits_reader.Key();

      if (!alignment_reader.HasKey(key)) {
        KALDI_WARN << "Could not find alignment for key " << key;
        num_no_alignment++;
        continue;
      }

      const std::vector<int32> &alignment = alignment_reader.Value(key);

      const std::vector<std::vector<bool> > &bits_vectors = bits_reader.Value();

      int32 num_frames = bits_vectors.size();
        
      if (alignment.size() != num_frames) {
        KALDI_WARN << "Alignments has wrong size "
                   << (alignment.size())<<" vs. "<< num_frames;
        num_other_error++;
        continue;
      }

      for (size_t i = 0; i < num_frames; i++) {
        if (alignment[i] >= num_classes) {
          if (ignore_irrelevant_classes)
            KALDI_WARN << "Found " << alignment[i] << " in alignment when "
                       << "--num-classes is specified as " << num_classes;
          else
            KALDI_ERR << "Found " << alignment[i] << " in alignment when "
                      << "--num-classes is specified as " << num_classes;
        }
        const std::vector<bool> &bits = bits_vectors[i];

        tree_stats.Accumulate(bits, alignment[i]);
      }
      
      num_done++;
      if (num_done % 1000 == 0)
        KALDI_LOG << "Processed " << num_done << " utterances.";
    }

    Output ko(accs_out_wxfilename, binary);
    tree_stats.Write(ko.Stream(), binary);

    KALDI_LOG << "Accumulated stats for " << num_done << " files, "
              << num_no_alignment << " failed due to no alignment, "
              << num_other_error << " failed for other reasons; "
              << "wrote a total of " << tree_stats.NumStats() << " stats";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

