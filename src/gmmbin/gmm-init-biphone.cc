// gmmbin/gmm-init-biphone.cc

// Copyright 2017   Hossein Hadian

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
#include "tree/event-map.h"
#include "tree/context-dep.h"
#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"

namespace kaldi {
// This function reads a file like:
// 1 2 3
// 4 5
// 6 7 8
// where each line is a list of integer id's of phones (that should have their pdfs shared).
void ReadSharedPhonesList(std::string rxfilename, std::vector<std::vector<int32> > *list_out) {
  list_out->clear();
  Input input(rxfilename);
  std::istream &is = input.Stream();
  std::string line;
  while (std::getline(is, line)) {
    list_out->push_back(std::vector<int32>());
    if (!SplitStringToIntegers(line, " \t\r", true, &(list_out->back())))
      KALDI_ERR << "Bad line in shared phones list: " << line << " (reading "
                << PrintableRxfilename(rxfilename) << ")";
    std::sort(list_out->rbegin()->begin(), list_out->rbegin()->end());
    if (!IsSortedAndUniq(*(list_out->rbegin())))
      KALDI_ERR << "Bad line in shared phones list (repeated phone): " << line
                << " (reading " << PrintableRxfilename(rxfilename) << ")";
  }
}

EventMap
*GetFullBiphoneStubMap(const std::vector<std::vector<int32> > &phone_sets,
                       const std::vector<int32> &phone2num_pdf_classes,
                       const std::vector<int32> &ci_phones_list,
                       const std::vector<std::vector<int32> > &bi_counts,
                       int32 biphone_min_count,
                       const std::vector<int32> &mono_counts,
                       int32 mono_min_count) {

  {  // Check the inputs
    KALDI_ASSERT(!phone_sets.empty());
    std::set<int32> all_phones;
    for (size_t i = 0; i < phone_sets.size(); i++) {
      KALDI_ASSERT(IsSortedAndUniq(phone_sets[i]));
      KALDI_ASSERT(!phone_sets[i].empty());
      for (size_t j = 0; j < phone_sets[i].size(); j++) {
        KALDI_ASSERT(all_phones.count(phone_sets[i][j]) == 0);  // Check not present.
        all_phones.insert(phone_sets[i][j]);
      }
    }
  }


  int32 numpdfs_per_phone = phone2num_pdf_classes[1];
  int32 current_pdfid = 0;
  std::map<EventValueType, EventMap*> level1_map;  // key is 1

  for (size_t i = 0; i < ci_phones_list.size(); i++) {
    std::map<EventValueType, EventAnswerType> level2_map;
    level2_map[0] = current_pdfid++;
    if (numpdfs_per_phone == 2) level2_map[1] = current_pdfid++;
    level1_map[ci_phones_list[i]] = new TableEventMap(kPdfClass, level2_map);
  }

  // If there is not enough data for a biphone, we will revert to monophone
  // and if there is not enough data for the monophone either, we will revert
  // to zerophone (which is like a global garbage pdf) after initializing it.
  int32 zerophone_pdf = -1;
  // If a monophone state is created for a phone-set, the corresponding pdf will
  // be stored in this vector.
  std::vector<int32> monophone_pdf(phone_sets.size(), -1);

  for (size_t i = 0; i < phone_sets.size(); i++) {

    if (numpdfs_per_phone == 1) {
      // Create an event map for level2:
      std::map<EventValueType, EventAnswerType> level2_map;  // key is 0
      level2_map[0] = current_pdfid++;  // no-left-context case
      for (size_t j = 0; j < phone_sets.size(); j++) {
        int32 pdfid = current_pdfid++;
        std::vector<int32> pset = phone_sets[j];  // All these will have a
                                                  // shared pdf with id=pdfid
        for (size_t k = 0; k < pset.size(); k++)
          level2_map[pset[k]] = pdfid;
      }
      std::vector<int32> pset = phone_sets[i];  // All these will have a
                                                // shared event-map child
      for (size_t k = 0; k < pset.size(); k++)
        level1_map[pset[k]] = new TableEventMap(0, level2_map);
    } else {
      KALDI_ASSERT(numpdfs_per_phone == 2);
      std::vector<int32> right_phoneset = phone_sets[i];  // All these will have a shared
                                                // event-map child
      // Create an event map for level2:
      std::map<EventValueType, EventMap*> level2_map;  // key is 0
      {  // Handle CI phones
        std::map<EventValueType, EventAnswerType> level3_map;  // key is kPdfClass
        level3_map[0] = current_pdfid++;
        level3_map[1] = current_pdfid++;
        level2_map[0] = new TableEventMap(kPdfClass, level3_map);  // no-left-context case
        for (size_t i = 0; i < ci_phones_list.size(); i++)  // ci-phone left-context cases
          level2_map[ci_phones_list[i]] = new TableEventMap(kPdfClass, level3_map);
      }
      for (size_t j = 0; j < phone_sets.size(); j++) {
        std::vector<int32> left_phoneset = phone_sets[j];  // All these will have a
        // shared subtree with 2 pdfids
        std::map<EventValueType, EventAnswerType> level3_map;  // key is kPdfClass
        if (bi_counts.empty() ||
            bi_counts[left_phoneset[0]][right_phoneset[0]] >= biphone_min_count) {
          level3_map[0] = current_pdfid++;
          level3_map[1] = current_pdfid++;
        } else if (mono_counts.empty() ||
                   mono_counts[right_phoneset[0]] > mono_min_count) {
          //  Revert to mono.
          KALDI_VLOG(2) << "Reverting to mono for biphone (" << left_phoneset[0]
                        << "," << right_phoneset[0] << ")";
          if (monophone_pdf[i] == -1) {
            KALDI_VLOG(1) << "Reserving mono PDFs for phone-set " << i;
            monophone_pdf[i] = current_pdfid++;
            current_pdfid++; // num-pdfs-per-phone is 2
          }
          level3_map[0] = monophone_pdf[i];
          level3_map[1] = monophone_pdf[i] + 1;
        } else {
          KALDI_VLOG(2) << "Reverting to zerophone for biphone ("
                        << left_phoneset[0]
                        << "," << right_phoneset[0] << ")";
          // Revert to zerophone
          if (zerophone_pdf == -1) {
            KALDI_VLOG(1) << "Reserving zero PDFs.";
            zerophone_pdf = current_pdfid++;
            current_pdfid++; // num-pdfs-per-phone is 2
          }
          level3_map[0] = zerophone_pdf;
          level3_map[1] = zerophone_pdf + 1;
        }

        for (size_t k = 0; k < left_phoneset.size(); k++) {
          int32 left_phone = left_phoneset[k];
          level2_map[left_phone] = new TableEventMap(kPdfClass, level3_map);
        }
      }
      for (size_t k = 0; k < right_phoneset.size(); k++) {
        std::map<EventValueType, EventMap*> level2_copy;
        for (auto const& kv: level2_map)
          level2_copy[kv.first] = kv.second->Copy(std::vector<EventMap*>());
        int32 right_phone = right_phoneset[k];
        level1_map[right_phone] = new TableEventMap(0, level2_copy);
      }
    }

  }
  KALDI_LOG << "Num PDFs: " << current_pdfid;
  return new TableEventMap(1, level1_map);
}


ContextDependency*
BiphoneContextDependencyFull(std::vector<std::vector<int32> > phone_sets,
                             const std::vector<int32> phone2num_pdf_classes,
                             const std::vector<int32> &ci_phones_list,
                             const std::vector<std::vector<int32> > &bi_counts,
                             int32 biphone_min_count,
                             const std::vector<int32> &mono_counts,
                             int32 mono_min_count) {
  // Remove all the CI phones from the phone sets
  std::set<int32> ci_phones;
  for (size_t i = 0; i < ci_phones_list.size(); i++)
    ci_phones.insert(ci_phones_list[i]);
  for (int32 i = phone_sets.size() - 1; i >= 0; i--) {
    for (int32 j = phone_sets[i].size() - 1; j >= 0; j--) {
      if (ci_phones.find(phone_sets[i][j]) != ci_phones.end()) {  // Delete it
        phone_sets[i].erase(phone_sets[i].begin() + j);
        if (phone_sets[i].empty())   // If empty, delete the whole entry
          phone_sets.erase(phone_sets.begin() + i);
      }
    }
  }

  std::vector<bool> share_roots(phone_sets.size(), false);  // Don't share roots
  // N is context size, P = position of central phone (must be 0).
  int32 P = 1, N = 2;
  EventMap *pdf_map = GetFullBiphoneStubMap(phone_sets,
                                            phone2num_pdf_classes,
                                            ci_phones_list, bi_counts,
                                            biphone_min_count, mono_counts,
                                            mono_min_count);
  return new ContextDependency(N, P, pdf_map);
}


} // end namespace kaldi

/* This function reads the counts of biphones and monophones from a text file
   generated for chain flat-start training. On each line there is either a
   biphone count or a monophone count:
   <left-phone-id> <right-phone-id> <count>
   <monophone-id> <count>
   The phone-id's are according to phones.txt.

   It's more efficient to load the biphone counts into a map because
   most entries are zero, but since there are not many biphones, a 2-dim vector
   is OK. */
static void ReadPhoneCounts(std::string &filename, int32 num_phones,
                            std::vector<int32> *mono_counts,
                            std::vector<std::vector<int32> > *bi_counts) {
  // The actual phones start from id = 1 (so the last phone has id = num_phones).
  mono_counts->resize(num_phones + 1, 0);
  bi_counts->resize(num_phones + 1, std::vector<int>(num_phones + 1, 0));
  std::ifstream infile(filename);
  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    int a, b;
    long c;
    if ((std::istringstream(line) >> a >> b >> c)) {
      // It's a biphone count.
      KALDI_ASSERT(a >= 0 && a <= num_phones);  // 0 means no-left-context
      KALDI_ASSERT(b > 0 && b <= num_phones);
      KALDI_ASSERT(c >= 0);
      (*bi_counts)[a][b] = c;
    } else if ((std::istringstream(line) >> b >> c)) {
      // It's a monophone count.
      KALDI_ASSERT(b > 0 && b <= num_phones);
      KALDI_ASSERT(c >= 0);
      (*mono_counts)[b] = c;
    } else {
      KALDI_ERR << "Bad line in phone stats file: " << line;
    }
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize a biphone context-dependency tree with all the\n"
        "leaves (i.e. a full tree). Intended for end-to-end tree-free models.\n"
        "Usage:  gmm-init-biphone <topology-in> <dim> <model-out> <tree-out> \n"
        "e.g.: \n"
        " gmm-init-biphone topo 39 bi.mdl bi.tree\n";

    bool binary = true;
    std::string shared_phones_rxfilename, phone_counts_rxfilename;
    int32 min_biphone_count = 100, min_mono_count = 20;
    std::string ci_phones_str;
    std::vector<int32> ci_phones;  // Sorted, uniqe vector of
    // context-independent phones.

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("shared-phones", &shared_phones_rxfilename,
                "rxfilename containing, on each line, a list of phones "
                "whose pdfs should be shared.");
    po.Register("ci-phones", &ci_phones_str, "Colon-separated list of "
                "integer indices of context-independent phones.");
    po.Register("phone-counts", &phone_counts_rxfilename,
                "rxfilename containing, on each line, a biphone/phone and "
                "its count in the training data.");
    po.Register("min-biphone-count", &min_biphone_count, "Minimum number of "
                "occurrences of a biphone in training data to reserve pdfs "
                "for it.");
    po.Register("min-monophone-count", &min_mono_count, "Minimum number of "
                "occurrences of a monophone in training data to reserve pdfs "
                "for it.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }


    std::string topo_filename = po.GetArg(1);
    int dim = 0;
    if (!ConvertStringToInteger(po.GetArg(2), &dim) || dim <= 0 || dim > 10000)
      KALDI_ERR << "Bad dimension:" << po.GetArg(2)
                << ". It should be a positive integer.";
    std::string model_filename = po.GetArg(3);
    std::string tree_filename = po.GetArg(4);

    if (!ci_phones_str.empty()) {
      SplitStringToIntegers(ci_phones_str, ":", false, &ci_phones);
      std::sort(ci_phones.begin(), ci_phones.end());
      if (!IsSortedAndUniq(ci_phones) || ci_phones.empty() || ci_phones[0] == 0)
        KALDI_ERR << "Invalid --ci-phones option: " << ci_phones_str;
    }

    Vector<BaseFloat> glob_inv_var(dim);
    glob_inv_var.Set(1.0);
    Vector<BaseFloat> glob_mean(dim);
    glob_mean.Set(1.0);

    HmmTopology topo;
    bool binary_in;
    Input ki(topo_filename, &binary_in);
    topo.Read(ki.Stream(), binary_in);

    const std::vector<int32> &phones = topo.GetPhones();

    std::vector<int32> phone2num_pdf_classes(1 + phones.back());
    for (size_t i = 0; i < phones.size(); i++) {
      phone2num_pdf_classes[phones[i]] = topo.NumPdfClasses(phones[i]);
      // For now we only support 1 or 2 pdf's per phone
      KALDI_ASSERT(phone2num_pdf_classes[phones[i]] == 1 ||
                   phone2num_pdf_classes[phones[i]] == 2);
    }

    std::vector<int32> mono_counts;
    std::vector<std::vector<int32> > bi_counts;
    if (!phone_counts_rxfilename.empty()) {
      ReadPhoneCounts(phone_counts_rxfilename, phones.size(),
                      &mono_counts, &bi_counts);
      KALDI_LOG << "Loaded mono/bi phone counts.";
    }


    // Now the tree:
    ContextDependency *ctx_dep = NULL;
    std::vector<std::vector<int32> > shared_phones;
    if (shared_phones_rxfilename == "") {
      shared_phones.resize(phones.size());
      for (size_t i = 0; i < phones.size(); i++)
        shared_phones[i].push_back(phones[i]);
    } else {
      ReadSharedPhonesList(shared_phones_rxfilename, &shared_phones);
      // ReadSharedPhonesList crashes on error.
    }
    ctx_dep = BiphoneContextDependencyFull(shared_phones, phone2num_pdf_classes,
                                           ci_phones, bi_counts,
                                           min_biphone_count,
                                           mono_counts, min_mono_count);

    int32 num_pdfs = ctx_dep->NumPdfs();

    AmDiagGmm am_gmm;
    DiagGmm gmm;
    gmm.Resize(1, dim);
    {  // Initialize the gmm.
      Matrix<BaseFloat> inv_var(1, dim);
      inv_var.Row(0).CopyFromVec(glob_inv_var);
      Matrix<BaseFloat> mu(1, dim);
      mu.Row(0).CopyFromVec(glob_mean);
      Vector<BaseFloat> weights(1);
      weights.Set(1.0);
      gmm.SetInvVarsAndMeans(inv_var, mu);
      gmm.SetWeights(weights);
      gmm.ComputeGconsts();
    }

    for (int i = 0; i < num_pdfs; i++)
      am_gmm.AddPdf(gmm);

    // Now the transition model:
    TransitionModel trans_model(*ctx_dep, topo);

    {
      Output ko(model_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_gmm.Write(ko.Stream(), binary);
    }

    // Now write the tree.
    ctx_dep->Write(Output(tree_filename, binary).Stream(),
                   binary);

    delete ctx_dep;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
