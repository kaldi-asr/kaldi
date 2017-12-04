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
                       const std::vector<bool> &share_roots) {

  {  // Checking inputs.
    KALDI_ASSERT(!phone_sets.empty() && share_roots.size() == phone_sets.size());
    std::set<int32> all_phones;
    for (size_t i = 0; i < phone_sets.size(); i++) {
      KALDI_ASSERT(IsSortedAndUniq(phone_sets[i]));
      KALDI_ASSERT(!phone_sets[i].empty());
      for (size_t j = 0; j < phone_sets[i].size(); j++) {
        KALDI_ASSERT(all_phones.count(phone_sets[i][j]) == 0);  // check not present.
        all_phones.insert(phone_sets[i][j]);
      }
    }
  }
  int32 numpdfs_per_phone = phone2num_pdf_classes[1];

  int32 current_pdfid = 0;

  std::map<EventValueType, EventMap*> level1_map; // key is 1
  for (size_t i = 0; i < phone_sets.size(); i++) {

    if (numpdfs_per_phone == 1) {
      // create an event map for level2:
      std::map<EventValueType, EventAnswerType> level2_map; // key is 0
      level2_map[0] = current_pdfid++; // no-left-context case
      for (size_t j = 0; j < phone_sets.size(); j++) {
        int32 pdfid = current_pdfid++;
        std::vector<int32> pset = phone_sets[j]; // all these will have a
                                                 // shared pdf with id=pdfid
        for (size_t k = 0; k < pset.size(); k++)
          level2_map[pset[k]] = pdfid;
      }
      std::vector<int32> pset = phone_sets[i]; // all these will have a
      // shared event-map child
      //EventMap* lvl2_eventmap =
      for (size_t k = 0; k < pset.size(); k++)
        level1_map[pset[k]] = new TableEventMap(0, level2_map);

    } else {

      KALDI_ASSERT(numpdfs_per_phone == 2);
      int32 base_pdfid = current_pdfid;
      std::vector<int32> pset = phone_sets[i]; // all these will have a shared event-map child
      for (size_t k = 0; k < pset.size(); k++) {
        // create an event map for level2:
        std::map<EventValueType, EventMap*> level2_map; // key is 0
        {
          std::map<EventValueType, EventAnswerType> level3_map; // key is -1
          level3_map[0] = current_pdfid++;
          level3_map[1] = current_pdfid++;
          level2_map[0] = new TableEventMap(kPdfClass, level3_map); // no-left-context case
        }
        for (size_t j = 0; j < phone_sets.size(); j++) {
          std::map<EventValueType, EventAnswerType> level3_map; // key is -1
          level3_map[0] = current_pdfid++;
          level3_map[1] = current_pdfid++;

          std::vector<int32> ipset = phone_sets[j]; // all these will have a shared subtree with 2 pdfids
          for (size_t ik = 0; ik < ipset.size(); ik++) {
            level2_map[ipset[ik]] = new TableEventMap(kPdfClass, level3_map);
          }
        }
        level1_map[pset[k]] = new TableEventMap(0, level2_map); //lvl2_eventmap;
        if (k != pset.size() - 1)
          current_pdfid = base_pdfid;
      } /////////   k

    }
  }

  return new TableEventMap(1, level1_map);
}


ContextDependency*
BiphoneContextDependencyFull(const std::vector<std::vector<int32> > phone_sets,
                             const std::vector<int32> phone2num_pdf_classes) {
  std::vector<bool> share_roots(phone_sets.size(), false);  // don't share roots
  // N is context size, P = position of central phone (must be 0).
  int32 P = 1, N = 2;
  EventMap *pdf_map = GetFullBiphoneStubMap(phone_sets,
                                            phone2num_pdf_classes, share_roots);
  return new ContextDependency(N, P, pdf_map);
}


} // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialize biphone GMM with all the leaves. Intended for e2e experiments.\n"
        "Usage:  gmm-init-biphone <topology-in> <model-out> <tree-out> \n"
        "e.g.: \n"
        " gmm-init-biphone topo 39 bi.mdl bi.tree\n";

    bool binary = true;
    std::string shared_phones_rxfilename;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("shared-phones", &shared_phones_rxfilename,
                "rxfilename containing, on each line, a list of phones whose pdfs should be shared.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }


    std::string topo_filename = po.GetArg(1);
    std::string model_filename = po.GetArg(2);
    std::string tree_filename = po.GetArg(3);

    int32 dim = 10;
    Vector<BaseFloat> glob_inv_var(dim);
    glob_inv_var.Set(1.0);
    Vector<BaseFloat> glob_mean(dim);
    glob_mean.Set(1.0);

    HmmTopology topo;
    bool binary_in;
    Input ki(topo_filename, &binary_in);
    topo.Read(ki.Stream(), binary_in);

    const std::vector<int32> &phones = topo.GetPhones();

    std::vector<int32> phone2num_pdf_classes (1 + phones.back());
    for (size_t i = 0; i < phones.size(); i++) {
      phone2num_pdf_classes[phones[i]] = topo.NumPdfClasses(phones[i]);
      // for now we only support 1 or 2 pdf per phone
      KALDI_ASSERT(phone2num_pdf_classes[phones[i]] == 1 ||
                   phone2num_pdf_classes[phones[i]] == 2);
    }

    // Now the tree:
    ContextDependency *ctx_dep = NULL;
    std::vector<std::vector<int32> > shared_phones;
    if (shared_phones_rxfilename == "") {
      for (size_t i = 0; i < phones.size(); i++)
        shared_phones[i].push_back(phones[i]);
    } else {
      ReadSharedPhonesList(shared_phones_rxfilename, &shared_phones);
      // ReadSharedPhonesList crashes on error.
    }
    ctx_dep = BiphoneContextDependencyFull(shared_phones, phone2num_pdf_classes);

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
