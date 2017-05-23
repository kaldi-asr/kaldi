// nnet3bin/nnet3-get-egs-simple.cc

// Copyright      2017  Johns Hopkins University (author:  Daniel Povey)

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
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h"

namespace kaldi {

// This function attempts to split a command-line argument of the
// form 'name=rspecifier', on the first equals sign.  Will
// call KALDI_ERR if something seems wrong.
void SplitArgOnEquals(const std::string &arg,
                      std::string *name,
                      std::string *rspecifier) {
  size_t pos = arg.find_first_of('=');
  if (pos == std::string::npos) {
    KALDI_ERR << "Bad command line argument (expecting '='): "
              << arg;
  }
  *name = std::string(arg, 0, pos);
  *rspecifier = std::string(arg, pos + 1);
  if (!IsToken(*name)) {
    KALDI_ERR << "Bad command line argument (expecting a valid name before '='): "
              << arg;
  }
  if (ClassifyRspecifier(*rspecifier, NULL, NULL) == kNoRspecifier) {
    KALDI_ERR << "Bad command line argument (expecting an rspecifier after '='): "
              << arg;
  }
}

} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get frame-by-frame examples of data for nnet3 neural network training.\n"
        "This is like nnet3-get-egs, but does not split up its inputs into pieces\n"
        "and allows more general generation of egs.  E.g. this is usable for image\n"
        "recognition tasks.\n"
        "\n"
        "Usage:  nnet3-get-egs-simple [options] <name1>=<rspecifier1> "
        "<name2>=<rspecifier2> ...\n"
        "\n"
        "e.g.:\n"
        "nnet3-get-egs-simple input=scp:images.scp \\\n"
        "output='ark,o:ali-to-post ark:labels.txt ark:- | post-to-smat --dim=10 ark:- ark:-' ark:egs.ark\n"
        "\n"
        "See also: nnet3-get-egs\n";


    ParseOptions po(usage);


    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }


    std::string examples_wspecifier = po.GetArg(po.NumArgs());
    NnetExampleWriter example_writer(examples_wspecifier);


    std::string first_name, first_reader_name;
    SplitArgOnEquals(po.GetArg(1), &first_name, &first_reader_name);
    SequentialGeneralMatrixReader first_reader(first_reader_name);

    std::vector<std::string> other_names;

    std::vector<RandomAccessGeneralMatrixReader*> other_readers;

    for (int32 i = 2; i < po.NumArgs(); i++) {
      std::string name, rspecifier;
      SplitArgOnEquals(po.GetArg(i), &name, &rspecifier);
      other_names.push_back(name);
      other_readers.push_back(new RandomAccessGeneralMatrixReader(rspecifier));
    }

    int32 num_done = 0, num_err = 0;

    for (; !first_reader.Done(); first_reader.Next()) {
      std::string key = first_reader.Key();
      NnetExample eg;
      const GeneralMatrix &feats = first_reader.Value();
      int32 t = 0;  // first 't' value; each row of the matrix gets
                    // its own 't' value.
      eg.io.push_back(NnetIo(first_name, t, feats));
      bool all_ok = true;
      for (size_t i = 0; i < other_names.size(); i++) {
        if (!other_readers[i]->HasKey(key)) {
          KALDI_WARN << "Could not find input for key " << key
                     << " for io-name=" << other_names[i];
          all_ok = false;
          break;
        }
        const GeneralMatrix &other_feats = other_readers[i]->Value(key);
        eg.io.push_back(NnetIo(other_names[i], t, other_feats));
      }
      if (!all_ok) {
        num_err++;
      } else {
        example_writer.Write(key, eg);
        num_done++;
      }
    }
    if (num_err > 0)
      KALDI_WARN << num_err << " utterances had errors and could "
          "not be processed.";
    return (num_done > 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
