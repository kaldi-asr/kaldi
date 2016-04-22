// ivectorbin/ivector-extract-dense.cc

// Copyright 2013  Daniel Povey

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

namespace kaldi {

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
        "Extract iVectors for utterances, using a trained iVector extractor,\n"
        "and features and Gaussian-level posteriors\n"
        "Usage:  ivector-extract [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <ivector-wspecifier>\n"
        "e.g.: \n"
        " fgmm-global-gselect-to-post 1.ubm '$feats' 'ark:gunzip -c gselect.1.gz|' ark:- | \\\n"
        "  ivector-extract final.ie '$feats' ark,s,cs:- ark,t:ivectors.1.ark\n";

    ParseOptions po(usage);
    bool compute_objf_change = true;
    int32 chunk_size = 100,
	  period = 50;
    IvectorEstimationOptions opts;
    TaskSequencerConfig sequencer_config;
    po.Register("compute-objf-change", &compute_objf_change,
                "If true, compute the change in objective function from using "
                "nonzero iVector (a potentially useful diagnostic).  Combine "
                "with --verbose=2 for per-utterance information");
    po.Register("chunk-size", &chunk_size,
		"Todo");
    po.Register("period", &period,
		"Todo");
    
    opts.Register(&po);
    sequencer_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_extractor_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posterior_rspecifier = po.GetArg(3),
        ivectors_wspecifier = po.GetArg(4);


    // g_num_threads affects how ComputeDerivedVars is called when we read the
    // extractor.
    g_num_threads = sequencer_config.num_threads; 
    IvectorExtractor extractor;
    ReadKaldiObject(ivector_extractor_rxfilename, &extractor);

    double tot_auxf_change = 0.0, tot_t = 0.0;
    int32 num_done = 0, num_err = 0;
  
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posterior_reader(posterior_rspecifier);
    BaseFloatMatrixWriter ivectors_writer(ivectors_wspecifier);
  
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!posterior_reader.HasKey(utt)) {
        KALDI_WARN << "No posteriors for utterance " << utt;
        num_err++;
        continue;
      }
      const Matrix<BaseFloat> &mat = feature_reader.Value();
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

      int32 num_chunks = ceil((mat.NumRows() - chunk_size + period) / static_cast<BaseFloat>(period));
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
      }
      ivectors_writer.Write(utt, ivectors);

      tot_t += this_t;
      num_done++;
    }
    // Destructor of "sequencer" will wait for any remaining tasks.

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
