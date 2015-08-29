// gmmbin/gmm-acc-pdf-stats-segmentation.cc

// Copyright 2015   Vimal Manohar

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
#include "gmm/mle-am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "segmenter/segmenter.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace segmenter;

  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Initialize an AmGmm or some pdfs of it from segmentation\n"
        "Usage:  gmm-init-from-segmentation [options] <model-in> <feature-rspecifier> "
        "<segmentation-rspecifier> <model-out>\n"
        "e.g.:\n gmm-init-from-segmentation --pdfs=0:2 1.mdl scp:train.scp ark:1.seg 2.mdl\n";

    ParseOptions po(usage);
    MleDiagGmmOptions gmm_opts;

    bool binary = true;
    int32 num_gauss = 100;
    int32 num_gauss_init = 0;
    int32 num_iters = 50;
    int32 num_frames = 200000;
    int32 srand_seed = 0;
    int32 num_threads = 4;
    int32 label = -1;
    
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("num-gauss", &num_gauss, "Number of Gaussians in the model");
    po.Register("num-gauss-init", &num_gauss_init, "Number of Gaussians in "
                "the model initially (if nonzero and less than num_gauss, "
                "we'll do mixture splitting)");
    po.Register("num-iters", &num_iters, "Number of iterations of training");
    po.Register("num-frames", &num_frames, "Number of feature vectors to store in "
                "memory and train on (randomly chosen from the input features)");
    po.Register("srand", &srand_seed, "Seed for random number generator ");
    po.Register("num-threads", &num_threads, "Number of threads used for "
                "statistics accumulation");

    gmm_opts.Register(&po);

    po.Read(argc, argv);
    
    srand(srand_seed);    

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        segmentation_rspecifier = po.GetArg(3),
        model_wxfilename = po.GetArg(4);

    // Read class2pdf map
    unordered_map<int32, int32> class2pdf;
    if (class2pdf_rxfilename != "") {
      Input ki;
      if (!ki.OpenTextMode(class2pdf_rxfilename)) 
        KALDI_ERR << "Unable to open file " << class2pdf_rxfilename 
                  << " for reading in text mode";
      std::istream &is = ki.Stream();
      std::string line;
      while (std::getline(is, line)) {
        std::vector<int32> v;
        if (!SplitStringToIntegers(line, " \t\r", true, &v) || v.size() != 2) {
          KALDI_ERR << "Unable to parse line " << line << " in " 
                    << class2pdf_rxfilename;
        }
        class2pdf.insert(std::make_pair(v[0], v[1]));
      }

      if (!is.eof()) {
        KALDI_ERR << "Did not reach EOF. Could not read file " << class2pdf_rxfilename
                  << " successfully";
      }
    }

    // Seed AmDiagGmm
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_filename, &binary);
      TransitionModel trans_model;
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }
    MleAccumGmm

    double tot_like = 0.0;
    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessSegmentationReader segmentation_reader(segmentation_rspecifier);

    int32 num_done = 0, num_err = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      if (!segmentation_reader.HasKey(key)) {
        KALDI_WARN << "No segmentation for utterance " << key;
        num_err++;
        continue;
      } 
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      const Segmentation &segmentation = segmentation_reader.Value(key);

      BaseFloat tot_like_this_file = 0.0;
      BaseFloat tot_t_this_file = 0.0;

      for (std::forward_list<Segment>::const_iterator it = segmentation.Begin();
            it != segmentation.End(); ++it) {
        int32 pdf_id;
        if (class2pdf_rxfilename != "")
          pdf_id = it->Label();
        else 
          pdf_id = class2pdf.at(it->Label());

        if ( (pdfs_str != "" && std::binary_search(pdfs.begin(), pdfs.end(), pdf_id)) 
            || (pdfs_str == "") ) {
          // Pdf needs to be initialized
          KALDI_ASSERT(pdf_id < NumPdfs() && pdf_id >= 0);
          if (gauss_clusterable[pdf_id] == NULL) {
            gauss_clusterable[pdf_id] = new GaussClusterable(mat.NumCols(), gmm_opts.min_variance);
          }
          for (int32 i = it->start_frame; i <= it->end_frame; i++)
            gauss_clusterable[pdf_id]->AddStats(mat.Row(i), 1.0);
        }
      }
      num_done++;
    }

    if (pdfs_str != "") {
      for (std::vector<int32>::const_iterator it = pdfs.begin();
            it != pdfs.end(); ++it) {
        if (init_am_gmm) {
          DiagGmm gmm(*gauss_clusterable[pdf_id], var_floor);
          // Initialize am_gmm from scratch
          am_gmm.Init(

              }

        if (*it < NumPdfs()) {
          DiagGmm &gmm = am_gmm.GetPdf(*it);
        } else {
          DiagGmm gmm;
          am_gmm.AddPdf(
        (*it)
      }
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_err
              << " with errors.";

    KALDI_LOG << "Overall avg like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";

    {
      Output ko(accs_wxfilename, binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";
    if (num_done != 0)
      return 0;
    else
      return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


