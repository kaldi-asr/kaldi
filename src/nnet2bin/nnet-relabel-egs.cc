// nnet2bin/nnet-relabel-egs.cc

// Copyright 2014   Vimal Manohar

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

/** @brief Relabels neural network egs with the read pdf-id alignments
*/

#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet2/nnet-example.h"

namespace kaldi {
  
  // this functions splits an egs key like <utt_id>-<frame_id> into 
  // separate utterance id and frame id on the last delimiter.
  // Returns false if the delimiter is not found in the key.
  bool SplitEgsKey(const std::string &key, 
                    std::string *utt_id, int32 *frame_id) {
    size_t start = 0, found = 0, end = key.size();
    utt_id->clear();
 
    found = key.find_last_of("-", end);
    // start != end condition is for when the delimiter is at the end
    
    if (found != start && start != end && found < end) {
      *utt_id = key.substr(start, found - start);
      std::istringstream tmp(key.substr(found + 1, end));
      tmp >> *frame_id;
      return true;
    }

    return false;
  }
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet2;

  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Relabel neural network egs with the read pdf-id alignments, "
        "zero-based..\n"
        "Usage: nnet-relabel-egs [options] <pdf-aligment-rspecifier> "
        "<egs_rspecifier1> ... <egs_rspecifierN> "
        "<egs_wspecifier1> ... <egs_wspecifierN>\n"
        "e.g.: \n"
        " nnet-relabel-egs ark:1.ali egs_in/egs.1.ark egs_in/egs.2.ark "
        "egs_out/egs.1.ark egs_out/egs.2.ark\n"
        "See also: nnet-get-egs, nnet-copy-egs, steps/nnet2/relabel_egs.sh\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    // Here we expect equal number of input egs archive and output egs archives. 
    // So the total number of arguments including the alignment specifier must be odd.
    if (po.NumArgs() < 3 || po.NumArgs() % 2 == 0) {
      po.PrintUsage();
      exit(1);
    }

    std::string alignments_rspecifier = po.GetArg(1);
    int32 num_archives = (po.NumArgs() - 1) / 2;
    
    SequentialInt32VectorReader ali_reader(alignments_rspecifier);

    unordered_map<std::string, std::vector<int32>* > utt_to_pdf_ali;

    // Keep statistics
    int32 num_ali = 0;
    int64 num_frames_ali = 0, num_frames_egs = 0, 
          num_frames_missing = 0, num_frames_relabelled = 0;

    // Read alignments and put the pointer in an unordered map
    // indexed by the key. This is so that we can efficiently find the 
    // alignment corresponding to the utterance to 
    // which a particular frame belongs
    for (; !ali_reader.Done(); ali_reader.Next(), num_ali++) {
      std::string key = ali_reader.Key();
      std::vector<int32> *alignment = new std::vector<int32>(ali_reader.Value());
      std::pair<std::string, std::vector<int32>* > map(key, alignment);
      utt_to_pdf_ali.insert(map);
      num_frames_ali += alignment->size();
    }

    // Read archives of egs sequentially
    for (int32 i = 0; i < num_archives; i++) {
      std::string egs_rspecifier(po.GetArg(i+2));
      std::string egs_wspecifier(po.GetArg(i+2+num_archives));

      SequentialNnetExampleReader egs_reader(egs_rspecifier);
      NnetExampleWriter egs_writer(egs_wspecifier);

      for (; !egs_reader.Done(); egs_reader.Next(), num_frames_egs++) {
      
        std::string key(egs_reader.Key());

        std::string utt_id;
        int32 frame_id;

        if (!SplitEgsKey(key, &utt_id, &frame_id)) {
          KALDI_ERR << "Unable to split key " << key << " on delimiter - " 
                    << " into utterance id and frame id";
        }
        NnetExample eg(egs_reader.Value());

        if (utt_to_pdf_ali.find(utt_id) == utt_to_pdf_ali.end()) {
          KALDI_WARN << "Unable to find utterance id " << utt_id;
          egs_writer.Write(key, eg);
          num_frames_missing++;
          continue;
        }
        const std::vector<int32> *alignment = utt_to_pdf_ali[utt_id];

        int32 num_frames_in_eg = eg.labels.size();
        for (int32 t_offset = 0; t_offset < num_frames_in_eg; t_offset++) {
          int32 t = frame_id + t_offset;
          if (t >= static_cast<int32>(alignment->size())) {
            KALDI_ERR << "Time index " << t << " out of range for alignment, "
                      << "should be < " << alignment->size();
          }
          if (eg.GetLabelSingle(t_offset) != (*alignment)[t])
            num_frames_relabelled++; 
          eg.SetLabelSingle(t_offset, (*alignment)[t]);
        }
        egs_writer.Write(key, eg);
      }
    }

    unordered_map<std::string, std::vector<int32>*>::iterator iter;
    
    for (iter = utt_to_pdf_ali.begin(); iter != utt_to_pdf_ali.end(); ++iter)
      delete iter->second;
    
    KALDI_LOG << "Read " << num_ali << " alignments containing a total of " 
              << num_frames_ali << " frames; labelled " 
              << num_frames_egs - num_frames_missing << " frames out of " 
              << num_frames_egs << " examples; labels changed for " 
              << num_frames_relabelled << " of those frames.\n.";

    return (num_frames_missing > 0.5  * num_frames_egs);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

