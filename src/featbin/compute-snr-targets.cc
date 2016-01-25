// featbin/compute-snr-targets.cc

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
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compute snr targets using clean and noisy speech features\n"
        "Usage: compute-snr-targets [options] <clean-feature-rspecifier> [<noisy-feature-rspecifier>|<noise-feature-rspecifier>] <targets-wspecifier>\n"
        "e.g.: compute-snr-targets scp:clean.scp scp:noisy.scp ark:targets.ark\n";

    std::string target_type = "FbankMask";
    std::string ali_rspecifier;
    std::string silence_phones_str;
    std::string floor_str = "-inf", ceiling_str = "inf";
    int32 length_tolerance = 0;

    ParseOptions po(usage);
    po.Register("target_type", &target_type, "Target type can be FbankMask or IRM");
    po.Register("ali-rspecifier", &ali_rspecifier, "If provided, all the "
                "energy in the silence region of clean file is considered noise");
    po.Register("silence-phones", &silence_phones_str, "Comma-separated list of "
                "silence phones");
    po.Register("floor", &floor_str, "If specified, the target is floored at "
                "this value");
    po.Register("ceiling", &ceiling_str, "If specified, the target is ceiled "
                "at this value");
    po.Register("length-tolerance", &length_tolerance, "Tolerate differences "
                "in utterance lengths of these many frames");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::vector<int32> silence_phones;
    if (!silence_phones_str.empty()) { 
      if (!SplitStringToIntegers(silence_phones_str, ":", false, &silence_phones)) {
        KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
      }
      std::sort(silence_phones.begin(), silence_phones.end());
    }

    double floor = kLogZeroDouble, ceiling = -kLogZeroDouble;

    if (floor_str != "-inf")
      if (!ConvertStringToReal(floor_str, &floor)) {
        KALDI_ERR << "Invalid --floor value " << floor_str;
      }

    if (ceiling_str != "inf")
      if (!ConvertStringToReal(ceiling_str, &ceiling)) {
        KALDI_ERR << "Invalid --ceiling value " << ceiling_str;
      }

    int32 num_done = 0, num_err = 0, num_success = 0;
    
    // Copying tables of features.
    std::string clean_rspecifier = po.GetArg(1),    
                noisy_rspecifier = po.GetArg(2),
                targets_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader noisy_reader(noisy_rspecifier);
    RandomAccessBaseFloatMatrixReader clean_reader(clean_rspecifier);
    BaseFloatMatrixWriter kaldi_writer(targets_wspecifier);

    RandomAccessInt32VectorReader alignment_reader(ali_rspecifier);

    int64 num_sil_frames = 0;
    int64 num_speech_frames = 0;

    for (; !noisy_reader.Done(); noisy_reader.Next(), num_done++) {
      const std::string &key = noisy_reader.Key();
      Matrix<double> noisy_energy(noisy_reader.Value());

      std::string uniq_key = key;
      if (!clean_reader.HasKey(uniq_key)) {
        KALDI_WARN << "Could not find uniq key " << uniq_key << " "
                   << "in clean feats " << clean_rspecifier;
        num_err++;
        continue;
      }

      Matrix<double> clean_energy(clean_reader.Value(uniq_key));
      if (target_type == "Snr" || target_type == "FbankMask") {
        if (!ali_rspecifier.empty()) {
          if (!alignment_reader.HasKey(uniq_key)) {
            KALDI_WARN << "Could not find uniq key " << uniq_key
                       << " in alignment " << ali_rspecifier;
            num_err++;
            continue;
          }
          const std::vector<int32> &ali = alignment_reader.Value(key);

          if (std::abs(static_cast<int32> (ali.size()) - clean_energy.NumRows()) > length_tolerance) { 
            KALDI_WARN << "Mismatch in number of frames in alignment "
              << "and feats; " << static_cast<int32>(ali.size())
              << " vs " << clean_energy.NumRows();
            num_err++;
            continue;
          }

          int32 length = std::min(static_cast<int32>(ali.size()), clean_energy.NumRows()); 
          KALDI_ASSERT(ali.size() >= length);

          if (clean_energy.NumRows() < length) clean_energy.Resize(length, clean_energy.NumCols(), kCopyData);
          if (noisy_energy.NumRows() < length) noisy_energy.Resize(length, noisy_energy.NumCols(), kCopyData);

          for (int32 i = 0; i < length; i++) {
            if (std::binary_search(silence_phones.begin(), silence_phones.end(), ali[i])) {
              if (target_type == "Snr") 
                noisy_energy.Row(i).LogAddExpVec(1.0, clean_energy.Row(i)); // Actually noise energy
              clean_energy.Row(i).Set(kLogZeroDouble);
              num_sil_frames++;
            } else num_speech_frames++;
          }
        }

        clean_energy.AddMat(-1.0, noisy_energy);  
        if (ceiling_str != "inf") {
          clean_energy.ApplyCeiling(ceiling);
        }

        if (floor_str != "-inf") {
          clean_energy.ApplyFloor(floor);
        }
        
        kaldi_writer.Write(key, Matrix<BaseFloat>(clean_energy));
      } else if (target_type == "Irm") {
        Matrix<double> total_energy(noisy_energy);  // Actually noise feats
        
        total_energy.LogAddExpMat(1.0, clean_energy, kNoTrans);

        if (!ali_rspecifier.empty()) {
          if (!alignment_reader.HasKey(uniq_key)) {
            KALDI_WARN << "Could not find uniq key " << uniq_key
                       << "in alignment " << ali_rspecifier;
            num_err++;
            continue;
          }
          const std::vector<int32> &ali = alignment_reader.Value(key);
          
          if (std::abs(static_cast<int32> (ali.size()) - clean_energy.NumRows()) > length_tolerance) { 
            KALDI_WARN << "Mismatch in number of frames in alignment "
                       << "and feats; " << static_cast<int32>(ali.size())
                       << " vs " << clean_energy.NumRows();
            num_err++;
            continue;
          }
          
          int32 length = std::min(static_cast<int32>(ali.size()), clean_energy.NumRows()); 
          KALDI_ASSERT(ali.size() >= length);
          
          if (clean_energy.NumRows() < length) clean_energy.Resize(length, clean_energy.NumCols(), kCopyData);
          if (noisy_energy.NumRows() < length) noisy_energy.Resize(length, noisy_energy.NumCols(), kCopyData);

          for (int32 i = 0; i < clean_energy.NumRows(); i++) {
            if (std::binary_search(silence_phones.begin(), silence_phones.end(), ali[i])) {
              clean_energy.Row(i).Set(kLogZeroDouble);
              num_sil_frames++;
            } else num_speech_frames++;
          }
        }

        clean_energy.AddMat(-1.0, total_energy);
        if (ceiling_str != "inf") {
          clean_energy.ApplyCeiling(ceiling);
        }

        if (floor_str != "-inf") {
          clean_energy.ApplyFloor(floor);
        }

        kaldi_writer.Write(key, Matrix<BaseFloat>(clean_energy));
      } else {
        KALDI_ERR << "Unsupported target-type " << target_type;
      }
      num_success++;
    }
    KALDI_LOG << "Computed SNR targets for " << num_success 
              << " out of " << num_done << " utterances; failed for "
              << num_err;
    KALDI_LOG << "Got [ " << num_speech_frames << "," 
              << num_sil_frames << "] frames of silence and speech";
    return (num_success > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
