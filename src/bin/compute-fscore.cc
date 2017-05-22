// bin/compute-fscore.cc

// Copyright 2016   Vimal Manohar

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

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Compute F1-score, precision, recall etc.\n"
        "Takes two alignment files and computes statistics\n"
        "\n"
        "Usage: compute-fscore [options] <ref-rspecifier> <hyp-rspecifier>\n"
        " e.g.: compute-fscore ark:data/train/text ark:hyp_text\n";

    ParseOptions po(usage);

    std::string mode = "strict";
    std::string mask_rspecifier;

    po.Register("mode", &mode,
                "Scoring mode: \"present\"|\"all\"|\"strict\":\n"
                "  \"present\" means score those we have transcriptions for\n"
                "  \"all\" means treat absent transcriptions as empty\n"
                "  \"strict\" means die if all in ref not also in hyp");
    po.Register("mask", &mask_rspecifier,
                "Only score on frames where mask is 1");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string ref_rspecifier = po.GetArg(1);
    std::string hyp_rspecifier = po.GetArg(2);

    if (mode != "strict" && mode != "present" && mode != "all") {
      KALDI_ERR << "--mode option invalid: expected \"present\"|\"all\"|\"strict\", got "
                << mode;
    }

    int64 num_tp = 0, num_fp = 0, num_tn = 0, num_fn = 0, num_frames = 0;
    int32 num_absent_sents = 0; 

    // Both text and integers are loaded as vector of strings,
    SequentialInt32VectorReader ref_reader(ref_rspecifier);
    RandomAccessInt32VectorReader hyp_reader(hyp_rspecifier);
    RandomAccessInt32VectorReader mask_reader(mask_rspecifier);
    
    // Main loop, accumulate WER stats,
    for (; !ref_reader.Done(); ref_reader.Next()) {
      const std::string &key = ref_reader.Key();
      const std::vector<int32> &ref_ali = ref_reader.Value();
      std::vector<int32> hyp_ali;
      if (!hyp_reader.HasKey(key)) {
        if (mode == "strict")
          KALDI_ERR << "No hypothesis for key " << key << " and strict "
              "mode specifier.";
        num_absent_sents++;
        if (mode == "present")  // do not score this one.
          continue;
      } else {
        hyp_ali = hyp_reader.Value(key);
      }
      
      std::vector<int32> mask_ali;
      if (!mask_rspecifier.empty()) {
        if (!mask_reader.HasKey(key)) {
          if (mode == "strict")
            KALDI_ERR << "No hypothesis for key " << key << " and strict "
                "mode specifier.";
          num_absent_sents++;
          if (mode == "present")  // do not score this one.
            continue;
        } else {
          mask_ali = mask_reader.Value(key);
        }
      }

      for (int32 i = 0; i < ref_ali.size(); i++) {
        if ( (i < hyp_ali.size() && hyp_ali[i] != 0 && hyp_ali[i] != 1) || 
             (i < ref_ali.size() && ref_ali[i] != 0 && ref_ali[i] != 1) || 
             (i < mask_ali.size() && mask_ali[i] != 0 && mask_ali[i] != 1) ) {
          KALDI_ERR << "Expecting alignment to be 0s or 1s";
        }

        if (!mask_rspecifier.empty() && (std::abs(static_cast<int64>(ref_ali.size()) - static_cast<int64>(mask_ali.size())) > 2) )
          KALDI_ERR << "Length mismatch: mask vs ref";

        if (!mask_rspecifier.empty() && (i > mask_ali.size() || mask_ali[i] == 0)) continue;
        num_frames++;

        if (ref_ali[i] == 1 && i > hyp_ali.size()) { num_fn++; continue; }
        if (ref_ali[i] == 0 && i > hyp_ali.size()) { num_tn++; continue; }
        
        if (ref_ali[i] == 1 && hyp_ali[i] == 1) num_tp++;
        else if (ref_ali[i] == 0 && hyp_ali[i] == 1) num_fp++;
        else if (ref_ali[i] == 1 && hyp_ali[i] == 0) num_fn++;
        else if (ref_ali[i] == 0 && hyp_ali[i] == 0) num_tn++;
        else 
          KALDI_ERR << "Unknown condition";
      }
    }

    // Print the ouptut,
    std::cout.precision(2);
    std::cerr.precision(2);

    BaseFloat precision = static_cast<BaseFloat>(num_tp) / (num_tp + num_fp);
    BaseFloat recall = static_cast<BaseFloat>(num_tp) / (num_tp + num_fn);

    std::cout << "F1 " << 2 * precision * recall / (precision + recall) << "\n";
    std::cout << "Precision " << precision << "\n";
    std::cout << "Recall " << recall << "\n";
    std::cout << "Specificity "
      << static_cast<BaseFloat>(num_tn) / (num_tn + num_fp) << "\n";
    std::cout << "Accuracy "
      << static_cast<BaseFloat>(num_tp + num_tn) / num_frames << "\n";

    std::cerr << "TP " << num_tp << "\n";
    std::cerr << "FP " << num_fp << "\n";
    std::cerr << "TN " << num_tn << "\n";
    std::cerr << "FN " << num_fn << "\n";
    std::cerr << "Length " << num_frames << "\n";

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

