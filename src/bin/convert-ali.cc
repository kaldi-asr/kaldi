// bin/convert-ali.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"

namespace kaldi {

void ReadPhoneMap(std::string phone_map_rxfilename,
                  std::vector<int32> *phone_map) {
  // phone map file has format e.g.:
  // 1 1
  // 2 1
  // 3 2
  // 4 2
  std::vector<std::vector<int32> > vec;  // vector of vectors, each with two elements
  // (if file has right format). first is old phone, second is new phone
  if (!ReadIntegerVectorVectorSimple(phone_map_rxfilename, &vec))
    KALDI_ERR << "Error reading phone map from " <<
        PrintableRxfilename(phone_map_rxfilename);
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i].size() != 2 || vec[i][0]<=0 || vec[i][1]<=0 ||
       (vec[i][0]<static_cast<int32>(phone_map->size()) &&
        (*phone_map)[vec[i][0]] != -1))
      KALDI_ERR << "Error reading phone map from "
                 <<   PrintableRxfilename(phone_map_rxfilename)
                 << " (bad line " << i << ")";
    if (vec[i][0]>=static_cast<int32>(phone_map->size()))
      phone_map->resize(vec[i][0]+1, -1);
    KALDI_ASSERT((*phone_map)[vec[i][0]] == -1);
    (*phone_map)[vec[i][0]] = vec[i][1];
  }
}


} // end namespace kaldi


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert alignments from one decision-tree/model to another\n"
        "Usage:  convert-ali  [options] old-model new-model new-tree old-alignments-rspecifier new-alignments-wspecifier\n"
        "e.g.: \n"
        " convert-ali old.mdl new.mdl new.tree ark:old.ali ark:new.ali\n";


    std::string phone_map_rxfilename;
    ParseOptions po(usage);
    po.Register("phone-map", &phone_map_rxfilename,
                "File name containing old->new phone mapping (each line is: old-integer-id new-integer-id)");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string old_model_filename = po.GetArg(1);
    std::string new_model_filename = po.GetArg(2);
    std::string new_tree_filename = po.GetArg(3);
    std::string old_alignments_rspecifier = po.GetArg(4);
    std::string new_alignments_wspecifier = po.GetArg(5);

    std::vector<int32> phone_map;
    if (phone_map_rxfilename != "") {  // read phone map.
      ReadPhoneMap(phone_map_rxfilename,
                   &phone_map);
    }
    
    SequentialInt32VectorReader alignment_reader(old_alignments_rspecifier);
    Int32VectorWriter alignment_writer(new_alignments_wspecifier);

    TransitionModel old_trans_model;
    ReadKaldiObject(old_model_filename, &old_trans_model);

    TransitionModel new_trans_model;
    ReadKaldiObject(new_model_filename, &new_trans_model);

    if (!(old_trans_model.GetTopo() == new_trans_model.GetTopo()))
      KALDI_WARN << "Toplogies of models are not equal: "
                 << "conversion may not be correct or may fail.";
    
    
    ContextDependency new_ctx_dep;  // the tree.
    ReadKaldiObject(new_tree_filename, &new_ctx_dep);

    int num_success = 0, num_fail = 0;

    for (; !alignment_reader.Done(); alignment_reader.Next()) {
      std::string key = alignment_reader.Key();
      const std::vector<int32> &old_alignment = alignment_reader.Value();
      std::vector<int32> new_alignment;
      if (ConvertAlignment(old_trans_model,
                          new_trans_model,
                          new_ctx_dep,
                          old_alignment,
                          (phone_map_rxfilename != "" ? &phone_map : NULL),
                          &new_alignment)) {
        alignment_writer.Write(key, new_alignment);
        num_success++;
      } else {
        KALDI_WARN << "Could not convert alignment for key " << key
                   <<" (possibly truncated alignment?)";
        num_fail++;
      }
    }

    KALDI_LOG << "Succeeded converting alignments for " << num_success
              <<" files, failed for " << num_fail;

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


