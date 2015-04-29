// ivectorbin/select-top-chunks.cc

// Copyright 2015   Vimal Manohar (Johns Hopkins University)

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
#include "feat/feature-functions.h"

namespace kaldi {
  template <typename T> 
  class OtherVectorComparator {
    public:
      OtherVectorComparator(const std::vector<T> &vec, bool descending = true) 
                      : vec_(vec), descending_(descending) { }

      bool operator() (int32 a, int32 b) {
        if (descending_) return vec_[a] > vec_[b];
        else return vec_[a] < vec_[b];
      }

      inline void SetDescending() { descending_ = true; }
      inline void SetAscending() { descending_ = false; }

    private:
      const std::vector<T> &vec_;
      bool descending_;
  }; 

  template class OtherVectorComparator<BaseFloat>; 
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Select a subset of chunks of frames of the input files, based on the log-energy\n"
        "of the frames\n"
        "Usage: select-top-chunks [options] <feats-rspecifier> "
        "<feats-wspecifier>\n"
        "e.g. : select-top-chunks --frames-proportion=0.1 --window-size=100"
        "scp:feats.scp ark:-\n";

    BaseFloat frames_proportion = 1.0, select_frames = 0;
    bool select_bottom_frames = false;
    int32 window_size = 100;    // 100 frames = 1 second assuming a shift of 10ms
    std::string energies_rspecifier = "";
    
    ParseOptions po(usage);
    po.Register("frames-proportion", &frames_proportion,
                "Select only the top / bottom proportion frames by the feature value");
    po.Register("select-bottom-frames", &select_bottom_frames,
                "If true, select only the bottom frames.");
    po.Register("select-frames", &select_frames,
                "Select these many frames, instead of looking at a proportion. "
                "Overrides frame-proportion if provided");
    po.Register("window-size", &window_size, 
                "Size of window to consider at once");
    po.Register("energies", &energies_rspecifier,
                "Read log-energies from a separate archive instead of treating "
                "the first dimension as log-energy");


    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feat_rspecifier = po.GetArg(1),
                feat_wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);
    RandomAccessBaseFloatMatrixReader energies_reader(energies_rspecifier);

    int32 num_done = 0, num_err = 0; 
    long long num_select = 0, num_frames = 0;

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (feats.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for utterance " << utt;
        num_err++;
        continue;
      }
      num_frames += feats.NumRows();

      int32 num_chunks = ( feats.NumRows() + 0.5 * window_size ) / window_size;
      int32 chunk_size = feats.NumRows() / num_chunks;

      int32 this_select = 0;
      if (select_frames == 0) {
        this_select = frames_proportion * num_chunks + 0.5;
      } else {
        this_select = (static_cast<BaseFloat>(select_frames) + 0.5) / chunk_size ;
      }
      if (this_select == 0) this_select = 1;
      
      if (this_select >= num_chunks) {
        feat_writer.Write(feat_reader.Key(), feats);
        num_done++;
        num_select += feats.NumRows();
        continue;
      }

      Vector<BaseFloat> log_energy(feats.NumRows());

      if (energies_rspecifier != "") {
        if (!energies_reader.HasKey(utt)) {
          KALDI_WARN << "log-energy not found for utterance " << utt;
          num_err++;
          continue;
        }
        log_energy.CopyColFromMat(energies_reader.Value(utt), 0);
      } else
        log_energy.CopyColFromMat(feats, 0);

      std::vector<BaseFloat> chunk_energies(num_chunks, 0.0);
      for (int32 i = 0; i < num_chunks; i++) {
        SubVector<BaseFloat> chunk_energy(log_energy, i*chunk_size, chunk_size);
        chunk_energies[i] = chunk_energy.Sum() / chunk_energy.Dim();
      }
  
      std::vector<int32> idx(num_chunks);
      for (int32 i = 0; i < num_chunks; i++) {
        idx[i] = i;
      }

      OtherVectorComparator<BaseFloat> comparator(chunk_energies);
      if (select_bottom_frames) comparator.SetAscending();

      sort(idx.begin(), idx.end(), comparator);
      idx.resize(this_select);

      Matrix<BaseFloat> selected_feats(this_select * chunk_size,
                                          feats.NumCols());

      int32 n = 0;
      for (std::vector<int32>::const_iterator it = idx.begin();
            it != idx.end(); ++it, n++) {
        SubMatrix<BaseFloat> src_feats(feats, *it * chunk_size, chunk_size, 0, feats.NumCols());
        SubMatrix<BaseFloat> dst_feats(selected_feats, n*chunk_size, chunk_size, 0, feats.NumCols());
        dst_feats.CopyFromMat(src_feats);
      }

      feat_writer.Write(feat_reader.Key(), selected_feats);
      num_done++;
      num_select += this_select;
    }
   
    KALDI_LOG << "Done selecting " << num_select 
              << " top frames out out " 
              << num_frames << " frames ; processed "
              << num_done << " utterances, "
              << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

