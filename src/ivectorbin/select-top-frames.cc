// ivectorbin/select-top-frames.cc

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
      OtherVectorComparator(const Vector<T> &vec, bool descending = true) 
                      : vec_(vec), descending_(descending) { }

      bool operator() (int32 a, int32 b) {
        if (descending_) return vec_(a) > vec_(b);
        else return vec_(a) < vec_(b);
      }

      inline void SetDescending() { descending_ = true; }
      inline void SetAscending() { descending_ = false; }

    private:
      const Vector<T> &vec_;
      bool descending_;
  }; 

  template class OtherVectorComparator<BaseFloat>; 
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Select a subset of frames of the input files, based on the log-energy\n"
        "of the frames\n"
        "Usage: select-top-frames [options] <feats-rspecifier> "
        "<feats-wspecifier>\n"
        "e.g. : select-top-frames --top-frames-proportion=0.1 --window-size=1000 "
        "scp:feats.scp ark:-\n";

    BaseFloat top_frames_proportion = 1.0;
    BaseFloat bottom_frames_proportion = 0.0;
    int32 window_size = 1000;
    
    ParseOptions po(usage);
    po.Register("top-frames-proportion", &top_frames_proportion,
                "Select only the top frames by the feature value");
    po.Register("bottom-frames-proportion", &bottom_frames_proportion,
                "Select only the bottom frames by the feature value");
    po.Register("window-size", &window_size, 
                "Size of window to consider at once");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feat_rspecifier = po.GetArg(1),
                feat_wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

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
      
      int32 this_select = 0;
      if (bottom_frames_proportion == 0.0)
        this_select = top_frames_proportion * feats.NumRows() + 0.5;
      else if (top_frames_proportion == 0.0)
        this_select = bottom_frames_proportion * feats.NumRows() + 0.5;
      
      if (this_select >= feats.NumRows()) {
        feat_writer.Write(feat_reader.Key(), feats);
        num_done++;
        num_select += feats.NumRows();
        continue;
      }

      Vector<BaseFloat> log_energy(feats.NumRows());
      log_energy.CopyColFromMat(feats, 0);
  
      std::vector<int32> idx(feats.NumRows());
      for (int32 i = 0; i < feats.NumRows(); i++)
        idx[i] = i;
      
      OtherVectorComparator<BaseFloat> comparator(log_energy);
      if (top_frames_proportion == 0.0) comparator.SetAscending();

      sort(idx.begin(), idx.end(), comparator);
      
      idx.resize(this_select);

      Matrix<BaseFloat> selected_feats(this_select, feats.NumCols());
      selected_feats.CopyRows(feats, idx);

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
      
      


