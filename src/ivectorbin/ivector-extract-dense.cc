// ivectorbin/ivector-extract-dense.cc

// Copyright 2016  David Snyder
//           2016  Matthew Maciejewski

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

#include <iomanip>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "util/kaldi-thread.h"
#include "util/stl-utils.h"

namespace kaldi {

void GetChunkRange(int32 num_rows, int32 chunk_num, int32 chunk_size,
  int32 period, int32 min_chunk, int32 *chunk_start, int32 *chunk_end) {
  int32 offset = std::min(chunk_size, num_rows - chunk_num * period);

  // If the chunk is less than the target minimum chunk, shift its starting
  // point to the left.
  int32 adjust = offset < min_chunk ? min_chunk - offset : 0;
  *chunk_start = std::max(chunk_num * period - adjust, 0);
  *chunk_end = chunk_num * period + offset;
}

int32 ProcessSegment(const std::string line, std::string *segment,
  std::string *recording, BaseFloat *start, BaseFloat *end) {
  std::vector<std::string> split_line;
  // Split the line by space or tab and check the number of fields in each
  // line. There must be 4 fields--segment name , reacording wav file name,
  // start time, end time; 5th field (channel info) is optional.
  SplitStringToVector(line, " \t\r", true, &split_line);
  if (split_line.size() != 4 && split_line.size() != 5)
    return 0;
  (*segment) = split_line[0];
  (*recording) = split_line[1];
  std::string start_str = split_line[2],
              end_str = split_line[3];
  // Convert the start time and endtime to real from string. Segment is
  // ignored if start or end time cannot be converted to real.
  if (!ConvertStringToReal(start_str, start))
    return 0;
  if (!ConvertStringToReal(end_str, end))
    return 0;
  // start time must not be negative; start time must not be greater than
  // end time, except if end time is -1
  if (*start < 0 || (*end != -1.0 && *end <= 0) || ((*start >= *end)
    && (*end > 0)))
    return 0;
  return 1;
}

int32 NumChunks(int32 num_rows, int32 chunk_size, int32 period) {
  int32 num_chunks;
  if (chunk_size >= period) {
    num_chunks = std::max(static_cast<int32>(ceil((num_rows
      - chunk_size + period) / static_cast<BaseFloat>(period))), 1);
  } else {
    num_chunks = static_cast<int32>(num_rows / period) + 1;
  }
  return num_chunks;
}

void IvectorExtractFromChunk(const IvectorExtractor &extractor,
  std::string utt, const Matrix<BaseFloat> &feat_temp,
  const Posterior &post, int32 chunk_start, int32 chunk_end,
  Vector<BaseFloat> *ivector_out, double *tot_auxf_change) {

  bool need_2nd_order_stats = false;
  Vector<double> ivector(extractor.IvectorDim());
  double auxf_change;

  SubMatrix<BaseFloat> sub_feat(feat_temp, chunk_start,
    chunk_end - chunk_start, 0, feat_temp.NumCols());
  Posterior sub_post = std::vector<std::vector<std::pair<int32, BaseFloat> > >
            (&post[chunk_start], &post[chunk_end]);

  IvectorExtractorUtteranceStats utt_stats(extractor.NumGauss(),
                                           extractor.FeatDim(),
                                           need_2nd_order_stats);
  utt_stats.AccStats(sub_feat, sub_post);

  ivector(0) = extractor.PriorOffset();

  if (tot_auxf_change != NULL) {
    double old_auxf = extractor.GetAuxf(utt_stats, ivector);
    extractor.GetIvectorDistribution(utt_stats, &ivector, NULL);
    double new_auxf = extractor.GetAuxf(utt_stats, ivector);
    auxf_change = new_auxf - old_auxf;
  } else {
    extractor.GetIvectorDistribution(utt_stats, &ivector, NULL);
  }

  if (tot_auxf_change != NULL) {
    double T = TotalPosterior(sub_post);
    *tot_auxf_change += auxf_change;
    KALDI_VLOG(2) << "Auxf change for utterance " << utt << " was "
                  << (auxf_change / T) << " per frame over " << T
                  << " frames (weighted)";
  }
  // We actually write out the offset of the iVectors from the mean of the
  // prior distribution; this is the form we'll need it in for scoring.  (most
  // formulations of iVectors have zero-mean priors so this is not normally an
  // issue).
  ivector(0) -= extractor.PriorOffset();
  KALDI_VLOG(2) << "Ivector norm for utterance " << utt
                << " was " << ivector.Norm(2.0);
  ivector_out->CopyFromVec(ivector);
}

}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Extract iVectors for sessions using a sliding window over segments,\n"
        "using a trained iVector extractor and features and Gaussian-level posteriors\n"
        "Usage:  ivector-extract [options] <model-in> <segments-rxfilename> "
        "<feature-rspecifier> <posteriors-rspecifier> <ivector-wspecifier> "
        "<ivector-ranges-wspecifier> <ivector-weights-wspecifier>\n"
        "e.g.: \n"
        " fgmm-global-gselect-to-post 1.ubm '$feats' 'ark:gunzip -c gselect.1.gz|' ark:- | \\\n"
        "  ivector-extract-dense final.ie segments '$feats' ark,s,cs:- ark,t:ivectors.1.ark \\\n"
        "   ark,t:ivector_ranges.1.ark ark,t:ivector_weights.1.ark\n";

    ParseOptions po(usage);
    bool compute_objf_change = true;
    int32 chunk_size = 100,
          min_chunk = 20,
	  period = 50;
    double frame_shift = 0.01;
    std::string segment_rxfilename;
    IvectorEstimationOptions opts;
    TaskSequencerConfig sequencer_config;
    po.Register("compute-objf-change", &compute_objf_change,
                "If true, compute the change in objective function from using "
                "nonzero iVector (a potentially useful diagnostic).  Combine "
                "with --verbose=2 for per-utterance information");
    po.Register("chunk-size", &chunk_size,
		"Size of the sliding window in frames.");
    po.Register("period", &period, "How frequently we compute a new iVector.");
    po.Register("frame-shift", &frame_shift, "Frame shift in milliseconds.");
    po.Register("segment-rxfilename", &segment_rxfilename,
		"Supply if input features were extracted from segments.");
    po.Register("min-chunk-size", &min_chunk, "Minimum size (in frames) after "
                "splitting segments larger than chunk-size.");

    opts.Register(&po);
    sequencer_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5 && po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_extractor_rxfilename = po.GetArg(1),
      feature_rspecifier = po.GetArg(2),
      posterior_rspecifier = po.GetArg(3);
    IvectorExtractor extractor;
    ReadKaldiObject(ivector_extractor_rxfilename, &extractor);
    RandomAccessPosteriorReader posterior_reader(posterior_rspecifier);
    double tot_auxf_change = 0.0, tot_t = 0.0;
    double *auxf_ptr = (compute_objf_change ? &tot_auxf_change : NULL );
    int32 num_done = 0, num_err = 0;

    if (po.NumArgs() == 5) {
      std::string ivector_wspecifier = po.GetArg(4),
        utt2spk_wspecifier = po.GetArg(5);
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      BaseFloatVectorWriter ivector_writer(ivector_wspecifier);
      TokenWriter utt2spk_writer(utt2spk_wspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        const Matrix<BaseFloat> &feat = feature_reader.Value();
        Posterior post = posterior_reader.Value(utt);
        if (static_cast<int32>(post.size()) != feat.NumRows()) {
          KALDI_WARN << "Size mismatch between posterior " << post.size()
            << " and features " << feat.NumRows() << " for utterance " << utt;
          num_err++;
          continue;
        }
        double this_t = opts.acoustic_weight * TotalPosterior(post),
          max_count_scale = 1.0;
        if (opts.max_count > 0 && this_t > opts.max_count) {
          max_count_scale = opts.max_count / this_t;
          KALDI_LOG << "Scaling stats for utterance " << utt
            << " by scale " << max_count_scale << " due to --max-count="
            << opts.max_count;
          this_t = opts.max_count;
        }
        ScalePosterior(opts.acoustic_weight * max_count_scale,
                       &post);
        int32 num_chunks = NumChunks(feat.NumRows(), chunk_size, period);
        for (int32 i = 0; i < num_chunks; i++) {
          Vector<BaseFloat> ivector(extractor.IvectorDim());
          int32 chunk_start, chunk_end;
          GetChunkRange(feat.NumRows(), i, chunk_size, period, min_chunk,
            &chunk_start, &chunk_end);
          IvectorExtractFromChunk(extractor, utt, feat, post, chunk_start,
            chunk_end, &ivector, auxf_ptr);
          std::stringstream ss_segment;
          ss_segment << utt << "-"
            << std::setw(6) << std::setfill('0') << chunk_start << std::setw(1)
            << "-" << std::setw(6) << std::setfill('0') << chunk_end;
          std::string segment = ss_segment.str();
          ivector_writer.Write(segment, ivector);
          utt2spk_writer.Write(segment, utt);
          tot_t += this_t;
          num_done++;
        }
      }
    } else {
      std::string segments_rxfilename = po.GetArg(4),
      ivector_wspecifier = po.GetArg(5),
      segments_wxfilename = po.GetArg(6),
      utt2spk_wspecifier = po.GetArg(7);
      Input ki(segments_rxfilename);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
      BaseFloatVectorWriter ivector_writer(ivector_wspecifier);
      TokenWriter utt2spk_writer(utt2spk_wspecifier);
      Output segments_output(segments_wxfilename, false);
      std::string line;
      while (std::getline(ki.Stream(), line)) {
        std::string segment,
                    recording;
        BaseFloat start,
              end;
        if (!ProcessSegment(line, &segment, &recording, &start, &end)) {
          KALDI_WARN << "Invalid line in segments file: " << line;
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &feat = feature_reader.Value(segment);
        Posterior post = posterior_reader.Value(segment);
        if (static_cast<int32>(post.size()) != feat.NumRows()) {
          KALDI_WARN << "Size mismatch between posterior " << post.size()
            << " and features " << feat.NumRows() << " for utterance "
            << segment;
          num_err++;
          continue;
        }
        double this_t = opts.acoustic_weight * TotalPosterior(post),
          max_count_scale = 1.0;
        if (opts.max_count > 0 && this_t > opts.max_count) {
          max_count_scale = opts.max_count / this_t;
          KALDI_LOG << "Scaling stats for utterance " << segment
            << " by scale " << max_count_scale << " due to --max-count="
            << opts.max_count;
          this_t = opts.max_count;
        }
        ScalePosterior(opts.acoustic_weight * max_count_scale,
                       &post);
        int32 num_chunks = NumChunks(feat.NumRows(), chunk_size, period);
        for (int32 i = 0; i < num_chunks; i++) {
          Vector<BaseFloat> ivector(extractor.IvectorDim());
          int32 seg_start = static_cast<int32>(start / frame_shift),
            chunk_start, chunk_end;
          GetChunkRange(feat.NumRows(), i, chunk_size, period, min_chunk,
            &chunk_start, &chunk_end);
          IvectorExtractFromChunk(extractor, segment, feat, post, chunk_start,
            chunk_end, &ivector, auxf_ptr);

          std::stringstream ss_subsegment;
          ss_subsegment << recording << "-" << std::setw(6)
            << std::setfill('0') << seg_start + chunk_start << std::setw(1)
            << "-" << std::setw(6) << std::setfill('0')
            << seg_start + chunk_end;
          std::string subsegment = ss_subsegment.str();

          ivector_writer.Write(subsegment, ivector);
          utt2spk_writer.Write(subsegment, recording);
          segments_output.Stream() << subsegment << " " << recording << " "
            << start + chunk_start * frame_shift << " "
            << start + chunk_end * frame_shift << "\n";
          tot_t += this_t;
          num_done++;
        }
      }
    }
    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.  Total (weighted) frames " << tot_t;
    if (compute_objf_change)
      KALDI_LOG << "Overall average objective-function change from estimating "
                << "ivector was " << (tot_auxf_change / tot_t) << " per frame "
                << " over " << tot_t << " (weighted) frames.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
