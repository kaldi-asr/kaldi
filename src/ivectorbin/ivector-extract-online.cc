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
#include "util/kaldi-thread.h"

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
        "See also ivector-extract-online2\n"
        "\n"
        "Usage:  ivector-extract-online [options] <model-in> <feature-rspecifier>"
        "<posteriors-rspecifier> <ivector-wspecifier>\n"
        "e.g.: \n"
        " gmm-global-get-post 1.dubm '$feats' ark:- | \\\n"
        "  ivector-extract-online --ivector-period=10 final.ie '$feats' ark,s,cs:- ark,t:ivectors.1.ark\n";

    ParseOptions po(usage);
    int32 num_cg_iters = 15;
    int32 ivector_period = 10;
    BaseFloat max_count = 0.0;
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
    po.Register("max-count", &max_count,
                "If >0, when the count of posteriors exceeds max-count we will "
                "start using a stronger prior term.  Can make iVectors from "
                "longer than normal utterances look more 'typical'.  Interpret "
                "this value as a number of frames multiplied by your "
                "posterior scale (so typically 0.1 times a number of frames).");
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

    double tot_objf_impr = 0.0, tot_t = 0.0, tot_length = 0.0,
        tot_length_utt_end = 0.0;
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
                                                   max_count, &ivectors);

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
      tot_length_utt_end += ivectors.Row(ivectors.NumRows() - 1).Norm(2.0) *
          tot_post;
      for (int32 i = 0; i < ivectors.NumRows(); i++)
        tot_length += ivectors.Row(i).Norm(2.0) * tot_post / ivectors.NumRows();

      num_done++;
    }

    KALDI_LOG << "Estimated iVectors for " << num_done << " files, " << num_err
              << " with errors.";
    KALDI_LOG << "Average objective-function improvement was "
              << (tot_objf_impr / tot_t) << " per frame, over "
              << tot_t << " frames (weighted).";
    KALDI_LOG << "Average iVector length was " << (tot_length / tot_t)
              << " and at utterance-end was " << (tot_length_utt_end / tot_t)
              << ", over " << tot_t << " frames (weighted); "
              << " expected length is " << sqrt(extractor.IvectorDim());

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
