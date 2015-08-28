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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;
    
    const char *usage =
        "Compute frame-level SNRs from log-SNR of time-frequency bins and the "
        "corrupted fbank features\n"
        "Usage: compute-frame-snrs <feats1-rspecifier> <feats2-rspecifier> <snr-wspecifier>\n"
        " or:   compute-frame-snrs \"ark:nnet3-compute exp/nnet3/final.raw scp:data/train_hires/feats.scp ark:- |\" scp:data/train_fbank/feats.scp -\n";
    
    ParseOptions po(usage);

    int32 length_tolerance = 0;
    po.Register("length-tolerance", &length_tolerance,
                "If length is different, trim as shortest up to a frame "
                " difference of length-tolerance, otherwise exclude segment.");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }
    
    
    std::string log_snr_rspecifier = po.GetArg(1),
                fbank_rspecifier = po.GetArg(2),
                frame_snr_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader log_snr_reader(log_snr_rspecifier);
    RandomAccessBaseFloatMatrixReader fbank_reader(fbank_rspecifier);
    BaseFloatVectorWriter frame_snr_writer(frame_snr_wspecifier);

    int32 num_done = 0, num_fail = 0;

    for (; !log_snr_reader.Done(); log_snr_reader.Next()) {
      const Matrix<BaseFloat> &log_snr = log_snr_reader.Value();
      const std::string &utt = log_snr_reader.Key();

      if (!fbank_reader.HasKey(utt)) {
        KALDI_WARN << "No fbank features for utt " << utt;
        num_fail++;
        continue;
      }

      const Matrix<BaseFloat> &fbank = fbank_reader.Value(utt);
      
      if (log_snr.NumCols() != fbank.NumCols()) {
        KALDI_ERR << "Dimension mismatch between log_snr and fbank; "
                  << log_snr.NumCols() << " vs " << fbank.NumCols() 
                  << " for utt " << utt;
      }
      
      int32 min_len = 0, max_len = 0;
      if (log_snr.NumRows() < fbank.NumRows()) {
        min_len = log_snr.NumRows();
        max_len = fbank.NumRows();
      } else {
        min_len = fbank.NumRows();
        max_len = log_snr.NumRows();
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

      Vector<BaseFloat> frame_snr(min_len);

      SubMatrix<BaseFloat> pred(log_snr, 0, min_len, 0, log_snr.NumCols());
      SubMatrix<BaseFloat> temp(fbank, 0, min_len, 0, log_snr.NumCols());

      for (size_t t = 0; t < min_len; t++) {
        SubVector<BaseFloat> pred_t(pred, t);
        SubVector<BaseFloat> fbank_t(temp, t);

        frame_snr(t) = Exp(pred_t.LogSumExp() - fbank_t.LogSumExp());
      }

      frame_snr_writer.Write(utt, frame_snr);
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
