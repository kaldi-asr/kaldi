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

  void SmoothVector(int32 window_size, Vector<BaseFloat> *weights) {
    Vector<BaseFloat> weights_tmp(weights->Dim());
    weights_tmp.CopyFromVec(*weights);

    for (int32 i = 0; i < weights_tmp.Dim(); i++) {
      int32 left_index = std::max(0, i - window_size);
      int32 right_index = std::min(i + window_size, weights_tmp.Dim() - 1);
      SubVector<BaseFloat> this_weights(weights_tmp,
          left_index, right_index - left_index);
      (*weights)(i) = this_weights.Sum() / this_weights.Dim();
    }
  }

  void SmoothMask(int32 window_size, int32 select_class, BaseFloat threshold, 
                  Vector<BaseFloat> *mask) {
    Vector<BaseFloat> mask_tmp(mask->Dim());
    mask_tmp.CopyFromVec(*mask);

    for (int32 i = 0; i < mask_tmp.Dim(); i++) {
      int32 left_index = std::max(0, i - window_size);
      int32 right_index = std::min(i + window_size, mask_tmp.Dim() - 1);

      int32 mask_sum = 0;
      for (int32 j = left_index; j <= right_index; j++) {
        mask_sum += (mask_tmp(j) == select_class ? 1 : 0);
      }
      (*mask)(i) = (static_cast<BaseFloat>(mask_sum) / (right_index - left_index + 1) >= threshold ? select_class : -1.0);
    }
  }

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
        "<feats-wspecifier> [<mask-wspecifier>]\n"
        "e.g. : select-top-chunks --frames-proportion=0.1 --window-size=100"
        "scp:feats.scp ark:-\n";

    BaseFloat frames_proportion = 1.0;
    int32 window_size = 100;    // 100 frames = 1 second assuming a shift of 10ms
    std::string mask_rspecifier, weights_rspecifier, weights_next_rspecifier;
    int32 select_frames = -1, select_frames_next = -1;
    int32 select_class = 1;
    int32 dim_as_weight = -1;
    bool select_bottom_frames = false, select_bottom_frames_next = false;
    bool smooth_weights = false, smooth_mask = false;
    int32 smoothing_window = 4;
    BaseFloat selection_threshold = 0.5;
    
    ParseOptions po(usage);
    po.Register("frames-proportion", &frames_proportion,
                "Select only the top / bottom proportion frames by the feature value");
    po.Register("select-class", &select_class,
                "Select frames of this class in the mask");
    po.Register("num-select-frames", &select_frames,
                "Select these many frames, instead of looking at a proportion. "
                "Overrides frame-proportion if provided and >= 0");
    po.Register("num-select-frames-next", &select_frames_next,
                "Second level of selection of frames among frames selected "
                "using num-frames");
    po.Register("window-size", &window_size, 
                "Size of window to consider at once");
    po.Register("weights", &weights_rspecifier,
                "Read weights from an archive to do selection");
    po.Register("weights_next", &weights_next_rspecifier,
                "Read weights from an archive to do a second level of selection");
    po.Register("selection-mask", &mask_rspecifier,
                "Selection mask on the frames. These are chosen for the chunk "
                "based on majority");
    po.Register("select-bottom-frames", &select_bottom_frames, 
                "Select the bottom frames instead of top frames");
    po.Register("select-bottom-frames-next", &select_bottom_frames_next,
                "Select bottom frames for second level of selection");
    po.Register("use-dim-as-weight", &dim_as_weight,
                "Use a particular dimension of feature (e.g. C0) as the weight "
                "when --weights is not specified");
    po.Register("smoothing-window", &smoothing_window, 
                "Size of smoothing window. Applicable if --smooth-vectors=true");
    po.Register("smooth-weights", &smooth_weights,
                "Smooth weights over a window");
    po.Register("smooth-mask", &smooth_mask,
                "Smooth mask over a window");
    po.Register("selection-threshold", &selection_threshold,
                "Select chunks that have this fraction of frames to be "
                "the select-class");

    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feat_rspecifier = po.GetArg(1),
                feat_wspecifier = po.GetArg(2),
                mask_wspecifier;

    if (po.NumArgs() == 3)
      mask_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);
    
    RandomAccessBaseFloatVectorReader weights_reader(weights_rspecifier);
    RandomAccessBaseFloatVectorReader weights_next_reader(weights_next_rspecifier);
    RandomAccessBaseFloatVectorReader mask_reader(mask_rspecifier);

    BaseFloatVectorWriter mask_writer(mask_wspecifier);

    int32 num_done = 0, num_err = 0; 
    long long num_select = 0, num_frames = 0, num_filtered = 0, 
              num_select_second_level = 0, num_filtered_second_level = 0,
              num_masked = 0;

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
      if (num_chunks == 0) {
        KALDI_WARN << "No chunks found for utterance " << utt;
        num_err++;
        continue;
      }

      // Find chunk size to use
      int32 chunk_size = feats.NumRows() / num_chunks;

      Vector<BaseFloat> weights;
      Vector<BaseFloat> mask;
      Vector<BaseFloat> weights_next;

      // Read weights if specified
      if (weights_rspecifier != "") {
        if (!weights_reader.HasKey(utt)) {
          KALDI_WARN << "weights not found for utterance " << utt;
          num_err++;
          continue;
        }
        weights = (weights_reader.Value(utt));
      }

      if (dim_as_weight > 0) {
        KALDI_ASSERT(dim_as_weight < feats.NumCols());
        weights.CopyColFromMat(feats, dim_as_weight);
      }

      // Read mask if specified
      if (mask_rspecifier != "") {
        if (!mask_reader.HasKey(utt)) {
          KALDI_WARN << "mask not found for utterance " << utt;
          num_err++;
          continue;
        }
        mask = (mask_reader.Value(utt));
      }

      // Read second level weights if specified
      if (weights_next_rspecifier != "") {
        if (!weights_next_reader.HasKey(utt)) {
          KALDI_WARN << "second-level weights not found for utterance " << utt;
          num_err++;
          continue;
        }
        weights_next = (weights_next_reader.Value(utt));
      }

      std::vector<BaseFloat> chunk_weights;
      std::vector<BaseFloat> chunk_weights_next;
      std::vector<int32> chunk_mask;

      if (weights_rspecifier != "")
        chunk_weights.resize(num_chunks, 0.0);

      if (weights_next_rspecifier != "") 
        chunk_weights_next.resize(num_chunks, 0.0);

      if (mask_rspecifier != "") 
        chunk_mask.resize(num_chunks, 0);
      else
        chunk_mask.resize(num_chunks, 1);

      if (smooth_weights) {
        SmoothVector(smoothing_window, &weights);
        SmoothVector(smoothing_window, &weights_next);
      }

      if (smooth_mask) {
        SmoothMask(smoothing_window, select_class, selection_threshold, &mask);
      }

      // Find average weight for each chunk
      for (int32 i = 0; i < num_chunks; i++) {
        if (weights_rspecifier != "") {
          SubVector<BaseFloat> this_chunk_weights(weights, i*chunk_size, chunk_size);
          chunk_weights[i] = this_chunk_weights.Sum() / chunk_size;
        }
        if (weights_next_rspecifier != "") {
          SubVector<BaseFloat> this_chunk_weights_next(weights_next, i*chunk_size, chunk_size);
          chunk_weights_next[i] = this_chunk_weights_next.Sum() / chunk_size;
        }
        if (mask_rspecifier != "") { 
          SubVector<BaseFloat> this_chunk_mask(mask, i*chunk_size, chunk_size);
          int32 mask_sum = 0;
          for (int32 j = 0; j < this_chunk_mask.Dim(); j++) {
            if (this_chunk_mask(j) == select_class) {
              mask_sum++;
              num_masked++;
            }
          }
          chunk_mask[i] = (static_cast<BaseFloat>(mask_sum) / chunk_size >= selection_threshold ? 1 : 0);
        }
      }
  
      std::vector<int32> idx;
      for (int32 i = 0; i < num_chunks; i++) {
        if ( chunk_mask[i] == 1 )
          idx.push_back(i);
      }

      int32 this_select = 0;

      if (select_frames < 0) {
        if (frames_proportion == 1.0) 
          this_select = idx.size();
        else
          this_select = frames_proportion * idx.size() + 0.5;
      } else 
        this_select = (static_cast<BaseFloat>(select_frames) + 0.5) / chunk_size ;

      // No chunk selected. Just select one instead.
      if (this_select == 0) this_select = 1;    
      
      num_filtered += idx.size() * chunk_size;

      if (this_select < idx.size()) {
        // Need to select frames because this_select is less than the 
        // number of chunks found
        
        if (chunk_weights.size() > 0) {
          // Select only top frames according to chunk_weights
          OtherVectorComparator<BaseFloat> comparator(chunk_weights);
          if (select_bottom_frames) comparator.SetAscending();

          sort(idx.begin(), idx.end(), comparator);
        }
        idx.resize(this_select);
      } else {
        this_select = idx.size();
      }

      num_select += this_select * chunk_size;
      num_filtered_second_level += idx.size() * chunk_size;

      int32 this_select_next = idx.size();
      if (select_frames_next > 0)
        this_select_next = (static_cast<BaseFloat>(select_frames_next) + 0.5) / chunk_size;
      if (this_select_next == 0) this_select_next = 1;

      if (this_select_next < idx.size()) {
        // Need to select frames at second level because this_select is 
        // less than the number of chunks retained after first level
        // selection
        
        if (chunk_weights_next.size() > 0) {
          // Select only top frames according to chunk_weights
          OtherVectorComparator<BaseFloat> comparator(chunk_weights_next);
          if (select_bottom_frames_next) comparator.SetAscending();

          sort(idx.begin(), idx.end(), comparator);
        }
        idx.resize(this_select_next);
      } else {
        this_select_next = idx.size();
      }

      num_select_second_level += this_select_next * chunk_size;

      Matrix<BaseFloat> selected_feats(this_select_next * chunk_size, feats.NumCols());
      Vector<BaseFloat> output_mask(feats.NumRows());

      int32 n = 0;
      for (std::vector<int32>::const_iterator it = idx.begin();
          it != idx.end(); ++it, n++) {
        KALDI_VLOG(2) << utt << " " << *it * chunk_size << " " << *it * chunk_size + chunk_size;
        SubMatrix<BaseFloat> src_feats(feats, *it * chunk_size, chunk_size, 0, feats.NumCols());
        SubMatrix<BaseFloat> dst_feats(selected_feats, n*chunk_size, chunk_size, 0, feats.NumCols());
        dst_feats.CopyFromMat(src_feats);

        if (mask_wspecifier != "") {
          SubVector<BaseFloat> this_mask(output_mask, *it * chunk_size, chunk_size);
          this_mask.Set(1.0);
        }
      }

      feat_writer.Write(feat_reader.Key(), selected_feats);
      if (mask_wspecifier != "")
        mask_writer.Write(feat_reader.Key(), output_mask);

      num_done++;
    }
   
    KALDI_LOG << "Done selecting " << num_select_second_level 
              << " top frames out of " 
              << num_filtered_second_level << " frames at second level; "
              << num_select << " top frames were selected at first level "
              << "out of " << num_filtered 
              << " frames filtered through the mask out of "
              << num_frames << " frames ; "
              << " unmasked " << num_masked << " frames; processed "
              << num_done << " utterances, "
              << num_err << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

