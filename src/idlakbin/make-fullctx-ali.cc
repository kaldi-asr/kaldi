// bin/make-fullctx-ali.cc

// Copyright 2013 CereProc Ltd.  (Author: Matthew Aylett)

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
//

// Full context alignments have the form
// <phone contexts> <full context> <state no> as an integer vector for
// each frame.
// Full context is output by cex idlak voice build module and assigns a
// full context for each phone.
// the align idlak voice build module produces a standard quin phone alignment
// by transition id.
// This program takes this alignment, extracts the phone context and state no
// for each frame and merges it with the phone full context information.
// The process is complicated by the fact that initial and final pauses may be
// missing from the data while assummed in the full context information.
// Current policy is to remove context information for leading and final pauses
// if they are not in the alignment and set all context information to 0.
// Models in any built tree for silence data are then ignored.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "idlakfeat/hmm-utils-idlak.h"
#include "hmm/tree-accu.h"  // for ReadPhoneMap

#define PHONECONTEXT 5
#define MIDCONTEXT 2

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  int32 phone_index, j, k, last_phone;
  std::vector< std::vector<int32> >::iterator v;
  try {
    const char *usage =
        "Merges output from context extraction with a standard alignment\n"
        "Usage: make-fullctx-ali model old-alignments-rspecifier \n"
        " full-contexts-rspecifier fullcontext-alignments-wspecifier\n"
        "e.g.: \n"
        " make-fullctx-ali model.mdl ark:old.ali ark,t:cex.ark ark:new.ali\n";


    std::string phone_map_rxfilename;
    int phonecontext = 5;
    int midcontext = 2;
    int maxsilphone = 1;
    ParseOptions po(usage);
    po.Register("phone-context", &phonecontext, "Size of the phone context, e.g. 3 for triphone.");
    po.Register("mid-context", &midcontext, "Position of the middle phone, e.g. 1 for triphone.");
    po.Register("max-sil-phone", &maxsilphone, "Maximum value of silence phone");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1);
    std::string old_alignments_rspecifier = po.GetArg(2);
    std::string contexts_rspecifier = po.GetArg(3);
    std::string new_alignments_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    ReadKaldiObject(model_filename, &trans_model);
    SequentialInt32VectorReader alignment_reader(old_alignments_rspecifier);
    SequentialInt32VectorVectorReader contexts_reader(contexts_rspecifier);
    Int32VectorVectorWriter alignment_writer(new_alignments_wspecifier);

    int num_success = 0, num_fail = 0;

    for (; !contexts_reader.Done() && !alignment_reader.Done();
         contexts_reader.Next(), alignment_reader.Next()) {
      std::string ctxkey = alignment_reader.Key();
      const std::vector< std::vector<int32> > &contexts =
          contexts_reader.Value();
      std::string key = alignment_reader.Key();
      const std::vector<int32> &old_alignment = alignment_reader.Value();

      // Check the keys are the same
      if (ctxkey != key) {
        KALDI_WARN << "Could not merge alignment and contexts for key " << key
                   <<" (missing data?)";
        num_fail++;
        continue;
      }
      /*std::cout << ctxkey << " " << key << "\n";*/
      std::vector< std::vector<int32> > phone_windows;
      std::vector<int32> phones;
      std::vector<int32> states;
      std::vector< std::vector<int32> > output;
      std::vector<int32> curphone;

      for (size_t i = 0; i < old_alignment.size(); i++)
        states.push_back(trans_model.TransitionIdToHmmState(old_alignment[i]));

      if (GetPhoneWindows(trans_model,
                          old_alignment,
                          phonecontext,
                          midcontext,
                          true,
                          &phone_windows,
                          &phones)) {
	  last_phone = -1;
	  phone_index = 0;
	  for (v = phone_windows.begin(), j = 0;
	       v != phone_windows.end();
	       v++, j++) {
	      curphone = *v;
	      if ( curphone[midcontext] != last_phone ) {
		  if (last_phone >= 0)  phone_index++;
	      }
	      // check we have a silence at start of utt if not
	      // skip silence in context data
	      if (!phone_index && curphone[midcontext] > maxsilphone)
		  phone_index++;
	      if (phone_index >= contexts.size()) {
		  break;
	      }
	      /*std::cout << phone_index << " " << curphone[midcontext]  << "\n";*/
	      // first item in context is the mid point phone - ignore it
	      for (k = 1; k < contexts[phone_index].size(); k++)
		  curphone.push_back(contexts[phone_index][k]);
	      curphone.push_back(states[j]);
	      output.push_back(curphone);
	      last_phone = curphone[midcontext];
	  }
	  if (curphone[midcontext] > maxsilphone) phone_index++;
	  if (phone_index + 1 != contexts.size()) {
	      KALDI_WARN << "Merge of alignment and contexts failed for key " << key
			 <<" mismatching number of phones contexts:"
			 << contexts.size()
			 <<" alignment:" << phone_index + 1;
	      num_fail++;
	      continue;
	  }

	  alignment_writer.Write(key, output);
	  num_success++;
      } else {
	  KALDI_WARN << "Could not convert alignment for key " << key
		     <<" (possibly truncated alignment?)";
	  num_fail++;
      }
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


