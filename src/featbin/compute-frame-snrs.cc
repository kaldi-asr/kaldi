// featbin/compute-frame-snrs.cc

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
#include "matrix/kaldi-matrix.h"

namespace kaldi {

void ComputeFrameSnrs(const Matrix<BaseFloat> &clean_fbank, 
                      const Matrix<BaseFloat> &fbank,
                      Vector<BaseFloat> *frame_snrs) {
  int32 min_len = frame_snrs->Dim();

  for (size_t t = 0; t < min_len; t++) {
    SubVector<BaseFloat> clean_fbank_t(clean_fbank, t);
    SubVector<BaseFloat> fbank_t(fbank, t);

    BaseFloat clean_energy_t = clean_fbank_t.LogSumExp();
    BaseFloat total_energy_t = fbank_t.LogSumExp();

    BaseFloat noise_energy_t = (total_energy_t > clean_energy_t ? 
                                LogSub(total_energy_t, clean_energy_t) : 
                                LogSub(clean_energy_t, total_energy_t) );

    (*frame_snrs)(t) = clean_energy_t - noise_energy_t;
  }
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;
    
    const char *usage =
        "Compute frame-level log-SNRs from time-frequency bin predictions and "
        "the corrupted fbank features. \n"
        "Optionally write clean feats as output.\n"
        "Usage: compute-frame-snrs <feats-rspecifier> <mask-rspecifier> <frame-snr-wspecifier> [<clean-feats-wspecifier>]\n"
        " e.g.:   compute-frame-snrs scp:data/train_fbank/feats.scp \"ark:nnet3-compute exp/nnet3/final.raw scp:data/train_hires/feats.scp ark:- |\" ark:-\n";
    
    int32 length_tolerance = 0;
    std::string prediction_type = "FbankMask";

    ParseOptions po(usage);

    po.Register("length-tolerance", &length_tolerance,
                "If length is different, trim as shortest up to a frame "
                " difference of length-tolerance, otherwise exclude segment.");
    po.Register("prediction_type", &prediction_type, 
                "Prediction type can be FbankMask or IRM");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string fbank_rspecifier = po.GetArg(1),
                mask_rspecifier = po.GetArg(2),
                frame_snr_wspecifier = po.GetArg(3),
                clean_fbank_wspecifier;
    if (po.NumArgs() == 4)
      clean_fbank_wspecifier = po.GetArg(4);

    SequentialBaseFloatMatrixReader fbank_reader(fbank_rspecifier);
    RandomAccessBaseFloatMatrixReader mask_reader(mask_rspecifier);
    BaseFloatVectorWriter frame_snr_writer(frame_snr_wspecifier);
    BaseFloatMatrixWriter clean_fbank_writer(clean_fbank_wspecifier);

    int32 num_done = 0, num_fail = 0;

    for (; !fbank_reader.Done(); fbank_reader.Next()) {
      const Matrix<BaseFloat> &fbank = fbank_reader.Value();
      const std::string &utt = fbank_reader.Key();

      if (!mask_reader.HasKey(utt)) {
        KALDI_WARN << "No mask features for utt " << utt;
        num_fail++;
        continue;
      }

      const Matrix<BaseFloat> &mask = mask_reader.Value(utt);
      
      if (mask.NumCols() != fbank.NumCols()) {
        KALDI_ERR << "Dimension mismatch between fbank and mask; "
                  << fbank.NumCols() << " vs " << mask.NumCols() 
                  << " for utt " << utt;
      }
      
      int32 min_len = 0, max_len = 0;
      if (mask.NumRows() < fbank.NumRows()) {
        min_len = mask.NumRows();
        max_len = fbank.NumRows();
      } else {
        min_len = fbank.NumRows();
        max_len = mask.NumRows();
      }
      
      if (max_len - min_len > length_tolerance || min_len == 0) {
        KALDI_WARN << "Length mismatch " << max_len << " vs. " << min_len
          << (utt.empty() ? "" : " for utt ") << utt
          << " exceeds tolerance " << length_tolerance;
        num_fail++;
        continue;
      }
      
      if (max_len - min_len > 0) {
        KALDI_VLOG(2) << "Length mismatch " << max_len << " vs. " << min_len
          << (utt.empty() ? "" : " for utt ") << utt
          << " exceeds tolerance " << length_tolerance;
      }

      KALDI_ASSERT(max_len == min_len);

      Vector<BaseFloat> frame_snrs(min_len);

      Matrix<BaseFloat> clean_fbank(mask.NumRows(), mask.NumCols());
      
      if (prediction_type == "Irm") {
        clean_fbank.CopyFromMat(mask);  
        // clean_fbank temporarily stores the mask
        clean_fbank.AddMat(2.0, fbank, kNoTrans);
        clean_fbank.Scale(0.5); 
        // clean_fbank has been computed
      } else if (prediction_type == "FbankMask") {
        clean_fbank.CopyFromMat(mask);  
        // clean_fbank temporarily stores the mask
        clean_fbank.AddMat(1.0, fbank, kNoTrans);
        // clean_fbank has been computed
      } else if (prediction_type == "Snr") {
        Matrix<BaseFloat> zeros(mask.NumRows(), mask.NumCols());
        clean_fbank.CopyFromMat(mask);
        clean_fbank.Scale(-1.0);  
        clean_fbank.LogAddExpMat(1.0, zeros);
        clean_fbank.Scale(-1.0);    // irm has been computed
        clean_fbank.AddMat(2.0, fbank, kNoTrans);
        clean_fbank.Scale(0.5); 
        // clean_fbank has been computed
      } else {
        KALDI_ERR << "Unknown prediction-type" << prediction_type;
      }

      ComputeFrameSnrs(clean_fbank, fbank, &frame_snrs);
      if (clean_fbank_wspecifier != "") {
        clean_fbank_writer.Write(utt, clean_fbank);
      }
      
      frame_snr_writer.Write(utt, frame_snrs);
      num_done++;
    }

    KALDI_LOG << "Computed frame snr for " << num_done << " utterances; "
              << "failed for " << num_fail;

    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
