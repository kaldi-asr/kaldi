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
        "Accumulate pdf stats for GMM training from segmentation.\n"
        "Usage:  gmm-acc-pdf-stats-segmentation [options] <model-in> <feature-rspecifier> "
        "<segmentation-rspecifier> <stats-out>\n"
        "e.g.:\n gmm-acc-stats-ali 1.mdl scp:train.scp ark:1.seg 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;
    std::string class2pdf_rxfilename, pdfs_str;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("class2pdf", &class2pdf_rxfilename, 
                "Map from class label to pdf id");
    po.Register("pdfs", &pdfs_str,
                "Only accumulate stats for these pdfs");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        segmentation_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);

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

    std::vector<int32> pdfs;
    if (pdfs_str != "") {
      if (!SplitStringToIntegers(pdfs_str, ":", true, &pdfs)) {
        KALDI_ERR << "Unable to parse string " << pdfs_str;
      }
    }

    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_filename, &binary);
      TransitionModel trans_model;
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    AccumAmDiagGmm gmm_accs;
    gmm_accs.Init(am_gmm, kGmmMeans|kGmmVariances|kGmmWeights);

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
            || (pdfs_str == "" && pdf_id < am_gmm.NumPdfs() && pdf_id >=0) ) {
          KALDI_ASSERT(pdf_id >= 0 && pdf_id < am_gmm.NumPdfs());
          for (int32 i = it->start_frame; i <= it->end_frame; i++)
            tot_like_this_file += gmm_accs.AccumulateForGmm(am_gmm, mat.Row(i),
                pdf_id, 1.0);
          tot_t_this_file = it->end_frame - it->start_frame + 1;
        }
      }
      tot_like += tot_like_this_file;
      tot_t += tot_t_this_file;

      if (num_done % 50 == 0) {
        KALDI_LOG << "Processed " << num_done << " utterances; for utterance "
          << key << " avg. like is "
          << (tot_like/tot_t)
          << " over " << tot_t <<" frames.";
      }
      num_done++;
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

