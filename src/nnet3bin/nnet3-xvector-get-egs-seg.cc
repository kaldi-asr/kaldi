// nnet3bin/nnet3-xvector-get-egs-seg.cc

// Copyright 2016-2017  Johns Hopkins University (author:  Hang Lyu)

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

#include <sstream>
#include "util/common-utils.h"
#include "nnet3/nnet-example.h"

namespace kaldi {
namespace nnet3 {

// Process the utt2label file and store it as a map from utt_id to
// label
static void ProcessUttToLabelFile(const std::string &utt2label_rxfilename, 
    unordered_map<std::string, int32, StringHasher> *utt_to_label) {
  Input utt2label_input(utt2label_rxfilename);
  if (!utt2label_rxfilename.empty()) {
    std::string line;
    while (std::getline(utt2label_input.Stream(), line)) {
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 2) {
        KALDI_ERR << "Expected 2 fields in line of utt2label file, got "
                  << fields.size() << " instead.";
      }
      std::string utt_id = fields[0];
      int32 label_id;
      if (!ConvertStringToInteger(fields[1], &label_id)) {
        KALDI_ERR << "Expect integer for label_id";
      }
      // Add to map
      (*utt_to_label)[utt_id] = label_id;
    } // end of while 
  } // end of if
}


static void WriteExample(const MatrixBase<BaseFloat> &feats,
  const int32 this_pdf_id, const std::string &key,
  bool compress, int32 num_pdfs, int32 &num_egs_written,
  NnetExampleWriter *example_writer) {

  NnetIo nnet_input = NnetIo("input", 0, feats);
  for (std::vector<Index>::iterator indx_it = nnet_input.indexes.begin();
      indx_it != nnet_input.indexes.end(); ++indx_it) {
    indx_it->n = 0;
  }
  Posterior label;
  std::vector<std::pair<int32, BaseFloat> > post;
  post.push_back(std::pair<int32, BaseFloat>(this_pdf_id, 1.0));
  label.push_back(post);
  NnetExample eg;
  eg.io.push_back(nnet_input);
  eg.io.push_back(NnetIo("output", num_pdfs, 0, label));
  if (compress) {
    eg.Compress();
  }

  example_writer->Write(key, eg);
  num_egs_written++;
}


} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "In previous xvector setup, the different examples in the same archive\n"
        "are equvalent length. Now, we hope the length of different examples in\n"
        "the same archive is different, so that the randomness of data is increased\n"
        "For this binary, it deals with the variable length feats.scp file.\n"
        "Generate the variable length egs into egs.ark file.\n"
        "Usage: nnet3-xvector-get-egs [options] <utt2label-rxfilename>"
        "<features-rspecifier> <egs-wspecifier>\n"
        "For example:\n"
        "nnet3-xvector-get-egs-seg data/utt2label scp:data/feats.scp ark,t:egs.ark\n";

    bool compress = true;
    int32 num_pdfs = -1;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");
    po.Register("num-pdfs", &num_pdfs, "Number of speakers in the training "
                "list.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string utt2label_rxfilename = po.GetArg(1),
                feature_rspecifier = po.GetArg(2),
                egs_wspecifier = po.GetArg(3);

    unordered_map<std::string, int32, StringHasher> utt_to_label;
    ProcessUttToLabelFile(utt2label_rxfilename, &utt_to_label);

    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    NnetExampleWriter egs_writer(egs_wspecifier);

    int32 num_done = 0,
          num_err = 0,
          num_egs_written = 0;

    if (num_pdfs == -1) {
      for(unordered_map<std::string, int32, StringHasher>::const_iterator 
          iter= utt_to_label.begin(); iter != utt_to_label.end(); iter++) {
        if(num_pdfs < iter->second) {
          num_pdfs = iter->second;
        }
      }
      num_pdfs = num_pdfs + 1;
    }

    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      
      // Now the format is <uttid>_<startpoint>_<length> 
      unordered_map<std::string, int32, StringHasher>::iterator got_label;
      got_label = utt_to_label.find(key);
      if (got_label == utt_to_label.end()) {
        KALDI_WARN << "Could not find the label of this utterance:" << key;
        num_err++;
      } else {
        int32 this_pdf_id = got_label->second;
        KALDI_ASSERT(this_pdf_id < num_pdfs);
        WriteExample(feats, this_pdf_id, key, compress, num_pdfs,
            num_egs_written, &egs_writer);
        num_done++;
      }
    }

    KALDI_LOG << "Finished generating examples, "
              << "successfully processed " << num_done
              << " feature files, wrote " << num_egs_written << " examples; "
              << num_err << " files had errors.";
    return (num_egs_written == 0 || num_err > num_done ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
