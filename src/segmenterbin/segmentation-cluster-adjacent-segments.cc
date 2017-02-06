// segmenterbin/segmentation-merge.cc

// Copyright 2017   Vimal Manohar (Johns Hopkins University)

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
#include "segmenter/segmentation-utils.h"
#include "tree/clusterable-classes.h"

namespace kaldi {
namespace segmenter {

BaseFloat Distance(const Segment &seg1, const Segment &seg2, 
                   const MatrixBase<BaseFloat> &feats,
                   BaseFloat var_floor,
                   int32 length_tolerance = 2) {
  int32 start1 = seg1.start_frame;
  int32 end1 = seg1.end_frame;

  int32 start2 = seg2.start_frame;
  int32 end2 = seg2.end_frame;

  if (end1 > feats.NumRows() + length_tolerance) {
    KALDI_ERR << "Segment end > feature length; " << end1 
              << " vs " << feats.NumRows();
  }

  GaussClusterable stats1(feats.NumCols(), var_floor);
  for (int32 i = start1; i < std::min(end1, feats.NumRows()); i++) {
    stats1.AddStats(feats.Row(i));
  }
  Vector<BaseFloat> means1(stats1.x_stats());
  means1.Scale(1.0 / stats1.count());
  Vector<BaseFloat> vars1(stats1.x2_stats());
  vars1.Scale(1.0 / stats1.count());
  vars1.AddVec2(-1.0, means1);
  vars1.ApplyFloor(var_floor);

  GaussClusterable stats2(feats.NumCols(), var_floor);
  for (int32 i = start2; i < std::min(end2, feats.NumRows()); i++) {
    stats2.AddStats(feats.Row(i));
  }
  Vector<BaseFloat> means2(stats2.x_stats());
  means2.Scale(1.0 / stats2.count());
  Vector<BaseFloat> vars2(stats2.x2_stats());
  vars2.Scale(1.0 / stats2.count());
  vars2.AddVec2(-1.0, means2);
  vars2.ApplyFloor(var_floor);

  double ans = 0.0;
  for (int32 i = 0; i < feats.NumCols(); i++) {
    ans += (vars1(i) / vars2(i) + vars2(i) / vars1(i) 
            + (means2(i) - means1(i)) * (means2(i) - means1(i)) 
            * (1.0 / vars1(i) + 1.0 / vars2(i))); 
  }

  return ans;
}

int32 ClusterAdjacentSegments(const MatrixBase<BaseFloat> &feats,
                              BaseFloat absolute_distance_threshold,
                              BaseFloat delta_distance_threshold,
                              BaseFloat var_floor,
                              int32 length_tolerance,
                              Segmentation *segmentation) {
  if (segmentation->Dim() <= 3) {
    // Very unusual case. 
    // TODO: Do something more reasonable.
    return 1;
  }

  
  SegmentList::iterator it = segmentation->Begin(), 
    next_it = segmentation->Begin();
  ++next_it;
  
  // Vector storing for each segment, whether there is a change point at the 
  // beginning of the segment.
  std::vector<bool> is_change_point(segmentation->Dim(), false);
  is_change_point[0] = true;

  Vector<BaseFloat> distances(segmentation->Dim() - 1);
  int32 i = 0;

  for (; next_it != segmentation->End(); ++it, ++next_it, i++) {
    // Distance between segment i and i + 1
    distances(i) = Distance(*it, *next_it, feats, 
                            var_floor, length_tolerance);

    if (i > 2) {
      if (distances(i-1) - distances(i-2) > delta_distance_threshold &&
          distances(i) - distances(i-1) < -delta_distance_threshold) {
        is_change_point[i-1] = true;
      }
    } else {
      if (distances(i) - distances(i-1) > absolute_distance_threshold)
        is_change_point[i] = true;
    }
  }
    
  int32 num_classes = 0;
  for (i = 0, it = segmentation->Begin(); 
       it != segmentation->End(); ++it, i++) {
    if (is_change_point[i]) {
      num_classes++;
    }
    it->SetLabel(num_classes);
  }

  return num_classes;
  /*
  BaseFloat prev_dist = Distance(*it, *next_it, feats,
                                 var_floor, length_tolerance);

  if (segmentation->Dim() == 2) {
    it->SetLabel(1);
    if (prev_dist < absolute_distance_threshold * feats.NumCols()
        && next_it->start_frame <= it->end_frame) {
      // Similar segments merged.
      next_it->SetLabel(it->Label());
    } else {
      // Segments not merged.
      next_it->SetLabel(it->Label() + 1);
    }

    return next_it->Label();;
  }

  // The algorithm is a simple peak detection.
  // Consider three segments that are pointed by the iterators 
  // prev_it, it, next_it.
  // If Distance(prev_it, it) > Consider 
  ++it;
  ++next_it;
  bool next_segment_is_new_cluster = false;

  for (; next_it != segmentation->End(); ++it, ++next_it) {
    SegmentList::iterator prev_it(it);
    --prev_it;

    // Compute distance between this and next segment.
    BaseFloat dist = Distance(*it, *next_it, feats, var_floor,
                              length_tolerance);

    // Possibly merge current segment if previous.
    if (next_segment_is_new_cluster ||
        (prev_it->end_frame + 1 >= it->start_frame &&
         prev_dist < absolute_distance_threshold * feats.NumCols())) {
      // Previous and current segment are next to each other.
      // Merge current segment with previous.
      it->SetLabel(prev_it->Label());
      
      KALDI_VLOG(3) << "Merging clusters " << *prev_it << " and " << *it
                    << " ; dist = " << prev_dist;
    } else {
      it->SetLabel(prev_it->Label() + 1);
      KALDI_VLOG(3) << "Not merging merging cluster " << *prev_it 
                    << " and " << *it << " ; dist = " << prev_dist;
    }

    // Decide if the current segment must be merged with next.
    if (prev_it->end_frame + 1 >= it->start_frame &&
        it->end_frame + 1 >= next_it->start_frame) {
      // All 3 segments are adjacent.
      if (dist - prev_dist > delta_distance_threshold * feats.NumCols()) {
        // Next segment is very different from the current and previous segment.
        // So create a new cluster for the next segment.
        next_segment_is_new_cluster = true;
      } else {
        next_segment_is_new_cluster = false;
      }
    }
    
    prev_dist = dist;
  }
  
  SegmentList::iterator prev_it(it);
  --prev_it;
  if (next_segment_is_new_cluster ||
      (prev_it->end_frame + 1 >= it->start_frame &&
       prev_dist < absolute_distance_threshold * feats.NumCols())) {
    // Merge current segment with previous.
    it->SetLabel(prev_it->Label());
    
    KALDI_VLOG(3) << "Merging clusters " << *prev_it << " and " << *it
                  << " ; dist = " << prev_dist;
  } else {
    it->SetLabel(prev_it->Label() + 1);
  }

  return it->Label();
  */
}

}  // end segmenter
}  // end kaldi 


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace segmenter;

    const char *usage =
        "Merge adjacent segments that are similar to each other.\n"
        "\n"
        "Usage: segmentation-cluster-adjacent-segments [options] "
        "<segmentation-rspecifier> <feats-rspecifier> <segmentation-wspecifier>\n"
        " e.g.: segmentation-cluster-adjacent-segments ark:foo.seg ark:feats.ark ark,t:-\n"
        "See also: segmentation-merge, segmentation-merge-recordings, "
        "segmentation-post-process --merge-labels\n";

    bool binary = true;
    int32 length_tolerance = 2;
    BaseFloat var_floor = 0.01;
    BaseFloat absolute_distance_threshold = 3.0;
    BaseFloat delta_distance_threshold = 0.0002;

    ParseOptions po(usage);

    po.Register("binary", &binary,
                "Write in binary mode "
                "(only relevant if output is a wxfilename)");
    po.Register("length-tolerance", &length_tolerance,
                "Tolerate length difference between segmentation and "
                "features if its less than this many frames.");
    po.Register("variance-floor", &var_floor,
                "Variance floor of Gaussians used in computing distances "
                "for clustering.");
    po.Register("absolute-distance-threshold", &absolute_distance_threshold,
                "Maximum per-dim distance below which segments will not be "
                "be merged.");
    po.Register("delta-distance-threshold", &delta_distance_threshold,
                "If the delta-distance is below this value, then it will "
                "be treated as 0.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string segmentation_in_fn = po.GetArg(1),
      feats_in_fn = po.GetArg(2),
      segmentation_out_fn = po.GetArg(3);

    // all these "fn"'s are either rspecifiers or filenames.
    bool in_is_rspecifier =
        (ClassifyRspecifier(segmentation_in_fn, NULL, NULL)
         != kNoRspecifier),
        out_is_wspecifier =
        (ClassifyWspecifier(segmentation_out_fn, NULL, NULL, NULL)
         != kNoWspecifier);
    
    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix regular files and archives";

    if (!in_is_rspecifier) {
      Segmentation segmentation;
      ReadKaldiObject(segmentation_in_fn, &segmentation);

      Matrix<BaseFloat> feats;
      ReadKaldiObject(feats_in_fn, &feats);

      Sort(&segmentation);
      int32 num_clusters = ClusterAdjacentSegments(
          feats, absolute_distance_threshold, delta_distance_threshold,
          var_floor, length_tolerance,
          &segmentation);

      KALDI_LOG << "Clustered segments; got " << num_clusters << " clusters.";
      WriteKaldiObject(segmentation, segmentation_out_fn, binary);

      return 0;
    } else {
      int32 num_done = 0, num_err = 0;

      SequentialSegmentationReader segmentation_reader(segmentation_in_fn);
      RandomAccessBaseFloatMatrixReader feats_reader(feats_in_fn);
      SegmentationWriter segmentation_writer(segmentation_out_fn);

      for (; !segmentation_reader.Done(); segmentation_reader.Next()) {
        Segmentation segmentation(segmentation_reader.Value());
        const std::string &key = segmentation_reader.Key();

        if (!feats_reader.HasKey(key)) {
          KALDI_WARN << "Could not find key " << key << " in " 
                     << "feats-rspecifier " << feats_in_fn;
          num_err++;
          continue;
        }

        const MatrixBase<BaseFloat> &feats = feats_reader.Value(key);

        Sort(&segmentation);
        int32 num_clusters = ClusterAdjacentSegments(
            feats, absolute_distance_threshold, delta_distance_threshold,
            var_floor, length_tolerance,
            &segmentation);
        KALDI_VLOG(2) << "For key " << key << ", got " << num_clusters 
                      << " clusters.";

        segmentation_writer.Write(key, segmentation);
        num_done++;
      }

      KALDI_LOG << "Clustered segments from " << num_done << " recordings "
                << "failed with " << num_err;
      return (num_done != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


