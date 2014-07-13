// ivectorbin/ivector-extract-online.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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

// This class will be used to parallelize over multiple threads the job
// that this program does.  The work happens in the operator (), the
// output happens in the destructor.
class IvectorExtractTask {
 public:
  IvectorExtractTask(const IvectorExtractor &extractor,
                     std::string utt,
                     const Matrix<BaseFloat> &feats,
                     const Posterior &posterior,
                     BaseFloatVectorWriter *writer,
                     double *tot_auxf_change):
      extractor_(extractor), utt_(utt), feats_(feats), posterior_(posterior),
      writer_(writer), tot_auxf_change_(tot_auxf_change) { }

  void operator () () {
    bool need_2nd_order_stats = false;
    
    IvectorExtractorUtteranceStats utt_stats(extractor_.NumGauss(),
                                             extractor_.FeatDim(),
                                             need_2nd_order_stats);
      
    utt_stats.AccStats(feats_, posterior_);

    ivector_.Resize(extractor_.IvectorDim());
    ivector_(0) = extractor_.PriorOffset();

    if (tot_auxf_change_ != NULL) {
      double old_auxf = extractor_.GetAuxf(utt_stats, ivector_);
      extractor_.GetIvectorDistribution(utt_stats, &ivector_, NULL);
      double new_auxf = extractor_.GetAuxf(utt_stats, ivector_);
      auxf_change_ = new_auxf - old_auxf;
    } else {
      extractor_.GetIvectorDistribution(utt_stats, &ivector_, NULL);
    }
  }
  ~IvectorExtractTask() {
    if (tot_auxf_change_ != NULL) {
      int32 T = posterior_.size();
      *tot_auxf_change_ += auxf_change_;
      KALDI_VLOG(2) << "Auxf change for utterance " << utt_ << " was "
                    << (auxf_change_ / T) << " per frame over " << T
                    << " frames.";
    }
    // We actually write out the offset of the iVector's from the mean of the
    // prior distribution; this is the form we'll need it in for scoring.  (most
    // formulations of iVectors have zero-mean priors so this is not normally an
    // issue).
    ivector_(0) -= extractor_.PriorOffset();
    KALDI_VLOG(2) << "Ivector norm for utterance " << utt_
                  << " was " << ivector_.Norm(2.0);
    writer_->Write(utt_, Vector<BaseFloat>(ivector_));
  }
 private:
  const IvectorExtractor &extractor_;
  std::string utt_;
  Matrix<BaseFloat> feats_;
  Posterior posterior_;
  BaseFloatVectorWriter *writer_;
  double *tot_auxf_change_; // if non-NULL we need the auxf change.
  Vector<double> ivector_;
  double auxf_change_;
};



}


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Extract iVectors for utterances, using a trained iVector extractor,\n"
        "and features and Gaussian-level posteriors.  This version extracts an\n"
        "iVector every n frames (see the --ivector-period option), by including\n"
        "all frames up to that point in the utterance.  This is designed to\n"
        "correspond with what will happen in a streaming decoding scenario;\n"
        "the iVectors would be used in neural net training.  The iVectors are\n"
        "output as an archive of matrices, indexed by utterance-id; each row\n"
        "corresponds to an iVector.\n"
        "\n"
        "Usage:  ivector-extract-online [options] <model-in> <feature-rspecifier>"
        "<posteriors-rspecifier> <ivector-wspecifier>\n"
        "e.g.: \n"
        " gmm-global-get-post 1.dubm '$feats' ark:- | \\\n"
        "  ivector-extract-online --ivector-period=10 final.ie '$feats' ark,s,cs:- ark,t:ivectors.1.ark\n";

    ParseOptions po(usage);
    int32 num_cg_iters = 15;
    int32 ivector_period = 10;
    g_num_threads = 8;

    po.Register("num-cg-iters", &num_cg_iters,
                "Number of iterations of conjugate gradient descent to perform "
                "each time we re-estimate the iVector.");
    po.Register("ivector-period", &ivector_period,
                "Controls how frequently we re-estimate the iVector as we get "
                "more data.");
    po.Register("num-threads", &g_num_threads,
                "Number of threads to use for computing derived variables "
                "of iVector extractor, at process start-up.");
    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string ivector_extractor_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        ivectors_wspecifier = po.GetArg(4);
    
    IvectorExtractor extractor;
    ReadKaldiObject(ivector_extractor_rxfilename, &extractor);
    
    double tot_objf_impr = 0.0, tot_t = 0.0;
    int32 num_done = 0, num_err = 0;
    
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
    BaseFloatMatrixWriter ivector_writer(ivectors_wspecifier);
    

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!posteriors_reader.HasKey(utt)) {
        KALDI_WARN << "No posteriors for utterance " << utt;
        num_err++;
        continue;
      }
      const Matrix<BaseFloat> &feats = feature_reader.Value();
      const Posterior &posterior = posteriors_reader.Value(utt);
      
      if (static_cast<int32>(posterior.size()) != feats.NumRows()) {
        KALDI_WARN << "Size mismatch between posterior " << posterior.size()
                   << " and features " << feats.NumRows() << " for utterance "
                   << utt;
        num_err++;
        continue;
      }


      Matrix<BaseFloat> ivectors;      
      double objf_impr_per_frame;
      objf_impr_per_frame = EstimateIvectorsOnline(feats, posterior, extractor,
                                                   ivector_period, num_cg_iters,
                                                   &ivectors);
      
      BaseFloat offset = extractor.PriorOffset();
      for (int32 i = 0 ; i < ivectors.NumRows(); i++)
        ivectors(i, 0) -= offset;
      
      double tot_post = TotalPosterior(posterior);

      KALDI_VLOG(2) << "For utterance " << utt << " objf impr/frame is "
                    << objf_impr_per_frame << " per frame, over "
                    << tot_post << " frames (weighted).";

      ivector_writer.Write(utt, ivectors);
      
      tot_t += tot_post;
      tot_objf_impr += objf_impr_per_frame * tot_post;

      num_done++;
    }

    KALDI_LOG << "Estimated iVectors for " << num_done << " files, " << num_err
              << " with errors.";
    KALDI_LOG << "Average objective-function improvement was "
              << (tot_objf_impr / tot_t) << " per frame, over "
              << tot_t << " frames (weighted).";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
