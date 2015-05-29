// gmmbin/gmm-est-segmentation.cc

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

namespace kaldi {
namespace segmenter {

void MleAmDiagGmmUpdateSubsetPdfs (const MleDiagGmmOptions &config,
                         const AccumAmDiagGmm &am_diag_gmm_acc,
                         const std::vector<int32> *pdfs, 
                         GmmFlagsType flags,
                         AmDiagGmm *am_gmm,
                         BaseFloat *obj_change_out,
                         BaseFloat *count_out) {
  KALDI_ASSERT(am_diag_gmm_acc.Dim() == am_gmm->Dim());
  KALDI_ASSERT(am_gmm != NULL);
  KALDI_ASSERT(am_diag_gmm_acc.NumAccs() == am_gmm->NumPdfs());
  if (obj_change_out != NULL) *obj_change_out = 0.0;
  if (count_out != NULL) *count_out = 0.0;

  BaseFloat tot_obj_change = 0.0, tot_count = 0.0;
  int32 tot_elems_floored = 0, tot_gauss_floored = 0,
      tot_gauss_removed = 0;

  if (pdfs != NULL) {
    for (std::vector<int32>::const_iterator it = pdfs->begin();
          it != pdfs->end(); ++it) {
      BaseFloat obj_change, count;
      int32 elems_floored, gauss_floored, gauss_removed;
      MleDiagGmmUpdate(config, am_diag_gmm_acc.GetAcc(*it), flags,
                       &(am_gmm->GetPdf(*it)),
                       &obj_change, &count, &elems_floored,
                       &gauss_floored, &gauss_removed);
      KALDI_LOG << "Count for pdf " << *it << " is " << count;
      tot_obj_change += obj_change;
      tot_count += count;
      tot_elems_floored += elems_floored;
      tot_gauss_floored += gauss_floored;
      tot_gauss_removed += gauss_removed;
    }
  } else {
    for (int32 i = 0; i < am_diag_gmm_acc.NumAccs(); i++) {
      BaseFloat obj_change, count;
      int32 elems_floored, gauss_floored, gauss_removed;

      MleDiagGmmUpdate(config, am_diag_gmm_acc.GetAcc(i), flags,
          &(am_gmm->GetPdf(i)),
          &obj_change, &count, &elems_floored,
          &gauss_floored, &gauss_removed);
      KALDI_LOG << "Count for pdf " << i << " is " << count;

      tot_obj_change += obj_change;
      tot_count += count;
      tot_elems_floored += elems_floored;
      tot_gauss_floored += gauss_floored;
      tot_gauss_removed += gauss_removed;
    }
  }

  if (obj_change_out != NULL) *obj_change_out = tot_obj_change;
  if (count_out != NULL) *count_out = tot_count;
  KALDI_LOG << tot_elems_floored << " variance elements floored in "
            << tot_gauss_floored << " Gaussians, out of "
            <<  am_gmm->NumGauss();
  if (config.remove_low_count_gaussians) {
    KALDI_LOG << "Removed " << tot_gauss_removed
              << " Gaussians due to counts < --min-gaussian-occupancy="
              <<  config.min_gaussian_occupancy
              << " and --remove-low-count-gaussians=true";
  }
}

}
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace segmenter;

  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Accumulate pdf stats for GMM training from segmentation "
        "and update GMM\n"
        "Usage:  gmm-est-segmentation [options] <model-in> <feature-rspecifier> "
        "<segmentation-rspecifier> <model-out>\n"
        "e.g.:\n gmm-acc-stats-ali 1.mdl scp:train.scp ark:1.seg 2.mdl\n";

    ParseOptions po(usage);
    bool binary = true;
    std::string class2pdf_rxfilename, pdfs_str;
    MleDiagGmmOptions gmm_opts;
    int32 mixup = 0;
    std::string mixup_per_pdf_str, mixup_rxfilename;
    int32 mixdown = 0;
    BaseFloat perturb_factor = 0.01;
    BaseFloat power = 0.2;
    BaseFloat min_count = 20.0;
    std::string update_flags_str = "mvw";
    std::string occs_out_filename;
    int32 num_iters = 3;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("class2pdf", &class2pdf_rxfilename, 
                "Map from class label to pdf id");
    po.Register("pdfs", &pdfs_str,
                "Only accumulate stats for these pdfs");
    po.Register("mix-up", &mixup, "Increase number of mixture components to "
                "this overall target.");
    po.Register("mix-up-per-pdf", &mixup_per_pdf_str,
                "Mix-up per pdf specified as comma separated list");
    po.Register("mix-up-rxfilename", &mixup_rxfilename,
                "Mix-up per pdf specified in a table");
    po.Register("min-count", &min_count,
                "Minimum per-Gaussian count enforced while mixing up and down.");
    po.Register("mix-down", &mixdown, "If nonzero, merge mixture components to this "
                "target.");
    po.Register("power", &power, "If mixing up, power to allocate Gaussians to"
                " states.");
    po.Register("update-flags", &update_flags_str, "Which GMM parameters to "
                "update: subset of mvwt.");
    po.Register("perturb-factor", &perturb_factor, "While mixing up, perturb "
                "means by standard deviation times this factor.");
    po.Register("write-occs", &occs_out_filename, "File to write pdf "
                "occupation counts to.");
    po.Register("num-iters", &num_iters, "Number of iterations of ML estimation");

    gmm_opts.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    kaldi::GmmFlagsType update_flags =
        StringToGmmFlags(update_flags_str);

    std::string model_in_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        segmentation_rspecifier = po.GetArg(3),
        model_out_filename = po.GetArg(4);

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
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    std::vector<int32> components_per_pdf(am_gmm.NumPdfs());
    for (int32 i = 0; i < am_gmm.NumPdfs(); i++) {
      components_per_pdf[i] = am_gmm.GetPdf(i).NumGauss();
    }
    
    std::vector<int32> target_components_per_pdf(am_gmm.NumPdfs(), -1);
    if (mixup_per_pdf_str != "") {
      std::vector<int32> mixup_per_pdf;
      if (!SplitStringToIntegers(mixup_per_pdf_str, ":", true, &mixup_per_pdf)
          && mixup_per_pdf.size() != am_gmm.NumPdfs()) {
        KALDI_ERR << "Unable to parse string " << mixup_per_pdf_str 
                  << " or it has wrong size (!= " << am_gmm.NumPdfs() << ")";
      }
      for (int32 i = 0; i < am_gmm.NumPdfs(); i++) {
        target_components_per_pdf[i] = mixup_per_pdf[i];
      }
    } else if (mixup_rxfilename != "") {
      Input ki(mixup_rxfilename);
      std::string line;
      while (std::getline(ki.Stream(), line)) {
        std::vector<std::string> split_line; 
        // Split the line by space or tab and check the number of fields in each
        // line. There must be 4 fields--segment name , reacording wav file name,
        // start time, end time; 5th field (channel info) is optional.
        SplitStringToVector(line, " \t\r", true, &split_line);
        if (split_line.size() != 2) {
          KALDI_ERR << "Invalid line in file: " << line;
        }

        int32 pdf_id, num_mix;
        if (!ConvertStringToInteger(split_line[0], &pdf_id)) {
          KALDI_ERR << "Invalid line in file [bad pdf_id]: " << line;
        }
        if (!ConvertStringToInteger(split_line[1], &num_mix)) {
          KALDI_ERR << "Invalid line in file [bad num_mix]: " << line;
        }
        target_components_per_pdf[pdf_id] = num_mix;
      }
    }

    std::vector<int32> components_incr_per_pdf(am_gmm.NumPdfs(), 0);
    for (int32 i = 0 ; i < am_gmm.NumPdfs(); i++) {
      components_incr_per_pdf[i] = std::ceil((target_components_per_pdf[i] - components_per_pdf[i]) / (num_iters / 2));
    }

    RandomAccessSegmentationReader segmentation_reader(segmentation_rspecifier);

    for (int32 n = 0; n < num_iters; n++) {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

      AccumAmDiagGmm gmm_accs;
      gmm_accs.Init(am_gmm, update_flags);
      
      double tot_like = 0.0;
      kaldi::int64 tot_t = 0;
    
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

        for (SegmentList::const_iterator it = segmentation.Begin();
            it != segmentation.End(); ++it) {
          int32 pdf_id;
          if (class2pdf_rxfilename != "")
            try {
              pdf_id = class2pdf.at(it->Label());
            } catch (const std::out_of_range& oor) {
              KALDI_VLOG(2) << "Out of Range error: " << oor.what() << '\n';
              continue;
            }
          else 
            pdf_id = it->Label();
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

        num_done++;
      }

      KALDI_LOG << "In iteration " << n << ", done " << num_done << " files, " << num_err
                << " with errors.";

      KALDI_LOG << "In iteration " << n << ", overall avg like per frame (Gaussian only) = "
                << (tot_like/tot_t) << " over " << tot_t << " frames.";

      KALDI_ASSERT(tot_t > 0);

      BaseFloat objf_impr, count;
      MleAmDiagGmmUpdateSubsetPdfs(gmm_opts, gmm_accs, pdfs.size() > 0 ? &pdfs : NULL, update_flags,
                                   &am_gmm, &objf_impr, &count);

      KALDI_LOG << "GMM update: In iteration " << n << ", overall "
                << (objf_impr/count)
                << " objective function improvement per frame over "
                <<  count <<  " frames";

      KALDI_ASSERT(count > 0);

      if (mixup != 0 || mixdown != 0 || 
          (n == num_iters - 1 && !occs_out_filename.empty()) ) {
        // get pdf occupation counts
        Vector<BaseFloat> pdf_occs;
        pdf_occs.Resize(gmm_accs.NumAccs());
        for (int i = 0; i < gmm_accs.NumAccs(); i++)
          pdf_occs(i) = gmm_accs.GetAcc(i).occupancy().Sum();

        if (mixdown != 0)
          am_gmm.MergeByCount(pdf_occs, mixdown, power, min_count);

        if (mixup != 0)
          am_gmm.SplitByCount(pdf_occs, mixup, perturb_factor,
              power, min_count);

        if (n == num_iters - 1 && !occs_out_filename.empty()) {
          bool binary = false;
          WriteKaldiObject(pdf_occs, occs_out_filename, binary);
        }
      }

      if (mixup_per_pdf_str != "" || mixup_rxfilename != "") {
        if (pdfs_str != "") {
          for (std::vector<int32>::const_iterator it = pdfs.begin();
              it != pdfs.end(); ++it) {
            components_per_pdf[*it] += components_incr_per_pdf[*it];
            if (target_components_per_pdf[*it] > 0 &&
                components_per_pdf[*it] > target_components_per_pdf[*it])
              components_per_pdf[*it] = target_components_per_pdf[*it];
            am_gmm.GetPdf(*it).Split(components_per_pdf[*it], gmm_opts.min_variance);
          }
        } else {
          for (int32 i = 0; i < am_gmm.NumPdfs(); i++) {
            components_per_pdf[i] += components_incr_per_pdf[i];
            if (target_components_per_pdf[i] > 0 &&
                components_per_pdf[i] > target_components_per_pdf[i])
              components_per_pdf[i] = target_components_per_pdf[i];
            am_gmm.GetPdf(i).Split(components_per_pdf[i], gmm_opts.min_variance);
          }
        }
      }
      if (num_done == 0) return 1;
    }

    {
      Output ko(model_out_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


