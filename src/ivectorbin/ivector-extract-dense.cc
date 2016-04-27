// ivectorbin/ivector-extract-dense.cc

// Copyright 2013  Daniel Povey
// Copyright 2016  Matthew Maciejewski

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
#include "gmm/am-diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "thread/kaldi-task-sequence.h"
#include "util/stl-utils.h"

namespace kaldi {

struct Segments {
  std::string segment;
  double start, end;
};

void PrepareSegments (const std::string segments_rxfilename,
		      unordered_map<std::string, std::vector<Segments*> > *record2seg) {
  
  Input ki(segments_rxfilename);

  std::string line;
  while (std::getline(ki.Stream(), line)) {
    std::vector<std::string> split_line;
    // Split the line by space or tab and check the number of fields in each
    // line. There must be 4 fields--segment name , reacording wav file name,
    // start time, end time; 5th field (channel info) is optional.
    SplitStringToVector(line, " \t\r", true, &split_line);
    if (split_line.size() != 4 && split_line.size() != 5) {
      KALDI_WARN << "Invalid line in segments file: " << line;
      continue;
    }
    std::string segment = split_line[0],
        recording = split_line[1],
        start_str = split_line[2],
        end_str = split_line[3];

    // Convert the start time and endtime to real from string. Segment is
    // ignored if start or end time cannot be converted to real.
    double start, end;
    if (!ConvertStringToReal(start_str, &start)) {
      KALDI_WARN << "Invalid line in segments file [bad start]: " << line;
      continue;
    }
    if (!ConvertStringToReal(end_str, &end)) {
      KALDI_WARN << "Invalid line in segments file [bad end]: " << line;
      continue;
    }
    // start time must not be negative; start time must not be greater than
    // end time, except if end time is -1
    if (start < 0 || (end != -1.0 && end <= 0) || ((start >= end) && (end > 0))) {
      KALDI_WARN << "Invalid line in segments file [empty or invalid segment]: "
                 << line;
      continue;
    }
    
    unordered_map<std::string, std::vector<Segments*> >::const_iterator
	iter = record2seg->find(recording);
    if (iter == record2seg->end()) {
      record2seg->insert(std::make_pair<std::string, std::vector<Segments*> >
				(recording, std::vector<Segments*>()));
    }
    Segments *seg = new Segments();
    seg->segment = segment;
    seg->start = start;
    seg->end = end;
    (*record2seg)[recording].push_back(seg);
  }

}
	
void IvectorExtract(const IvectorExtractor &extractor,
                   std::string utt,
                   const Matrix<BaseFloat> &feats_temp,
                   const Posterior &posterior,
                   Vector<BaseFloat> *ivector_out,
                   double *tot_auxf_change) {

  bool need_2nd_order_stats = false;
  Vector<double> ivector(extractor.IvectorDim());
  double auxf_change;
  Matrix<double> feats(feats_temp);

  IvectorExtractorUtteranceStats utt_stats(extractor.NumGauss(),
                                           extractor.FeatDim(),
                                           need_2nd_order_stats);
    
  utt_stats.AccStats(feats_temp, posterior);
  
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
    double T = TotalPosterior(posterior);
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
	  period = 50;
    double frame_shift = 0.01;
    IvectorEstimationOptions opts;
    TaskSequencerConfig sequencer_config;
    po.Register("compute-objf-change", &compute_objf_change,
                "If true, compute the change in objective function from using "
                "nonzero iVector (a potentially useful diagnostic).  Combine "
                "with --verbose=2 for per-utterance information");
    po.Register("chunk-size", &chunk_size,
		"Size of the sliding window in frames.");
    po.Register("period", &period,
		"Offset of each window in frames.");
    po.Register("frame-shift", &frame_shift,
		"With of frames. Used to compute ivector ranges.");
    
    opts.Register(&po);
    sequencer_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_extractor_rxfilename = po.GetArg(1),
	segments_rxfilename = po.GetArg(2),
	feature_rspecifier = po.GetArg(3),
	posterior_rspecifier = po.GetArg(4),
	ivectors_wspecifier = po.GetArg(5),
	ivector_ranges_wspecifier = po.GetArg(6),
	ivector_weights_wspecifier = po.GetArg(7);

    double tot_auxf_change = 0.0, tot_t = 0.0;
    int32 num_done = 0, num_err = 0;
    
    IvectorExtractor extractor;
    ReadKaldiObject(ivector_extractor_rxfilename, &extractor);
  
    unordered_map<std::string, std::vector<Segments*> > record2seg;
    PrepareSegments(segments_rxfilename, &record2seg);

    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posterior_reader(posterior_rspecifier);
    BaseFloatMatrixWriter ivectors_writer(ivectors_wspecifier);
    TokenVectorWriter ivector_ranges_writer(ivector_ranges_wspecifier);
    TokenVectorWriter ivector_weights_writer(ivector_weights_wspecifier);

    for (unordered_map<std::string, std::vector<Segments*> >::iterator iter = record2seg.begin();
         iter != record2seg.end(); ++iter) {
      std::vector<Matrix<BaseFloat> > ivector_list;
      std::vector<std::string> ivector_ranges;
      std::vector<std::string> ivector_weights;
      int32 tot_size = 0;
      for (int32 seg_num = 0; seg_num < iter->second.size(); seg_num++) {
        std::string utt = iter->second[seg_num]->segment;
        if (!posterior_reader.HasKey(utt)) {
          KALDI_WARN << "No posteriors for utterance " << utt;
          num_err++;
          continue;
        }
        const Matrix<BaseFloat> &mat = feature_reader.Value(utt);
        Posterior posterior = posterior_reader.Value(utt);
      
        if (static_cast<int32>(posterior.size()) != mat.NumRows()) {
          KALDI_WARN << "Size mismatch between posterior " << posterior.size()
                     << " and features " << mat.NumRows() << " for utterance "
                     << utt;
          num_err++;
          continue;
        }

        double *auxf_ptr = (compute_objf_change ? &tot_auxf_change : NULL );

        double this_t = opts.acoustic_weight * TotalPosterior(posterior),
            max_count_scale = 1.0;
        if (opts.max_count > 0 && this_t > opts.max_count) {
          max_count_scale = opts.max_count / this_t;
          KALDI_LOG << "Scaling stats for utterance " << utt << " by scale "
                    << max_count_scale << " due to --max-count="
                    << opts.max_count;
          this_t = opts.max_count;
        }
        ScalePosterior(opts.acoustic_weight * max_count_scale,
                       &posterior);
        // note: now, this_t == sum of posteriors.

        int32 num_chunks = std::max(static_cast<int32>(ceil((mat.NumRows() - chunk_size
					+ period) / static_cast<BaseFloat>(period))), 1);
        Matrix<BaseFloat> ivectors(num_chunks, extractor.IvectorDim());
        for (int32 i = 0; i < num_chunks; i++) {
          Vector<BaseFloat> ivector(extractor.IvectorDim());
          int32 window = std::min(chunk_size, mat.NumRows() - i * period);
          SubMatrix<BaseFloat> sub_mat(mat, i * period, window, 0, mat.NumCols());
          IvectorExtract(extractor, utt, Matrix<BaseFloat>(sub_mat),
  		         std::vector<std::vector<std::pair<int32, BaseFloat> > >
  		         (&posterior[i * period], &posterior[i * period + window]),
      		         &ivector, auxf_ptr);
          ivectors.CopyRowFromVec(ivector, i);

	  std::stringstream ss;
	  double start = iter->second[seg_num]->start;
	  ss << start + i * period * frame_shift << ","
	     << start + (i * period + window) * frame_shift;
	  ivector_ranges.push_back(ss.str());
	  ss.str(std::string());
	  ss << window;
	  ivector_weights.push_back(ss.str());
        }
	ivector_list.push_back(ivectors);
	tot_size += num_chunks;

        tot_t += this_t;
        num_done++;

	delete iter->second[seg_num];
      }
      Matrix<BaseFloat> recording_ivectors(tot_size, extractor.IvectorDim());
      int32 start_ind = 0;
      for (std::vector<Matrix<BaseFloat> >::iterator vec_iter = ivector_list.begin();
	   vec_iter != ivector_list.end(); ++vec_iter) {
        SubMatrix<BaseFloat> recording_ivectors_sub(recording_ivectors,
			start_ind, vec_iter->NumRows(), 0, extractor.IvectorDim());
	recording_ivectors_sub.CopyFromMat(*vec_iter);
	start_ind += vec_iter->NumRows();
      }
      ivectors_writer.Write(iter->first, recording_ivectors);
      ivector_ranges_writer.Write(iter->first, ivector_ranges);
      ivector_weights_writer.Write(iter->first, ivector_weights);
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
