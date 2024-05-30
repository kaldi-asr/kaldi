// featbin/subset-feats.cc

// Copyright 2009-2011  Microsoft Corporation
//           2014 Hainan Xu

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
#include "matrix/kaldi-matrix.h"

using namespace kaldi;

int32 CopyIncludedFeats(std::string filename,
                        SequentialBaseFloatMatrixReader *kaldi_reader,
                        BaseFloatMatrixWriter *kaldi_writer) {
  unordered_set<std::string, StringHasher> include_set;
  bool binary;
  Input ki(filename, &binary);
  KALDI_ASSERT(!binary);
  std::string line;
  while (std::getline(ki.Stream(), line)) {
    std::vector<std::string> split_line;
    SplitStringToVector(line, " \t\r", true, &split_line);
    KALDI_ASSERT(!split_line.empty() &&
        "Empty line encountered in input from --include option");
    include_set.insert(split_line[0]);
  }

  int32 num_total = 0;
  size_t num_success = 0;
  for (; !kaldi_reader->Done(); kaldi_reader->Next(), num_total++) {
    if (include_set.count(kaldi_reader->Key()) > 0) {
      kaldi_writer->Write(kaldi_reader->Key(), kaldi_reader->Value());
      num_success++;
    }
  }

  KALDI_LOG << " Wrote " << num_success << " out of " << num_total
            << " utterances.";
  return (num_success != 0 ? 0 : 1);
}

int32 CopyExcludedFeats(std::string filename,
                        SequentialBaseFloatMatrixReader *kaldi_reader,
                        BaseFloatMatrixWriter *kaldi_writer) {
  unordered_set<std::string, StringHasher> exclude_set;
  bool binary;
  Input ki(filename, &binary);
  KALDI_ASSERT(!binary);
  std::string line;
  while (std::getline(ki.Stream(), line)) {
    std::vector<std::string> split_line;
    SplitStringToVector(line, " \t\r", true, &split_line);
    KALDI_ASSERT(!split_line.empty() &&
        "Empty line encountered in input from --include option");
    exclude_set.insert(split_line[0]);
  }

  int32 num_total = 0;
  size_t num_success = 0;
  for (; !kaldi_reader->Done(); kaldi_reader->Next(), num_total++) {
    if (exclude_set.count(kaldi_reader->Key()) == 0) {
      kaldi_writer->Write(kaldi_reader->Key(), kaldi_reader->Value());
      num_success++;
    }
  }

  KALDI_LOG << " Wrote " << num_success << " out of " << num_total
            << " utterances.";
  return (num_success != 0 ? 0 : 1);
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy a subset of features (by default, the first n feature files)\n"
        "Usually used where only a small amount of data is needed\n"
        "Note: if you want a specific subset, it's usually best to\n"
        "filter the original .scp file with utils/filter_scp.pl\n"
        "(possibly with the --exclude option).  The --include and --exclude\n"
        "options of this program are intended for specialized uses.\n"
        "The --include and --exclude options are mutually exclusive, \n"
        "and both cause the --n option to be ignored.\n"
        "Usage: subset-feats [options] <in-rspecifier> <out-wspecifier>\n"
        "e.g.: subset-feats --n=10 ark:- ark:-\n"
        "or:  subset-feats --include=include_uttlist ark:- ark:-\n"
        "or:  subset-feats --exclude=exclude_uttlist ark:- ark:-\n"
        "See also extract-feature-segments, select-feats, subsample-feats\n";

    ParseOptions po(usage);

    int32 n = 10;
    std::string include_rxfilename;
    std::string exclude_rxfilename;
    po.Register("n", &n, "If nonnegative, copy the first n feature files.");
    po.Register("include", &include_rxfilename,
                        "Text file, the first field of each"
                        " line being interpreted as an "
                        "utterance-id whose features will be included");
    po.Register("exclude", &exclude_rxfilename,
                        "Text file, the first field of each "
                        "line being interpreted as an utterance-id"
                        " whose features will be excluded");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    KALDI_ASSERT(n >= 0);

    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);

    if (include_rxfilename != "") {
      if (n != 10) {
        KALDI_ERR << "Should not have both --include and --n option!";
      }
      if (exclude_rxfilename != "") {
        KALDI_ERR << "should not have both --exclude and --include option!";
      }
      return CopyIncludedFeats(include_rxfilename,
                               &kaldi_reader, &kaldi_writer);
    }
    else if (exclude_rxfilename != "") {
      if (n != 10) {
        KALDI_ERR << "Should not have both --exclude and --n option!";
      }
      return CopyExcludedFeats(exclude_rxfilename,
                               &kaldi_reader, &kaldi_writer);
    }

    if (n == 0) {
      KALDI_ERR << "Invalid option --n=0. Should be at least 1";
    }

    int32 k = 0;
    for (; !kaldi_reader.Done() && k < n; kaldi_reader.Next(), k++)
      kaldi_writer.Write(kaldi_reader.Key(), kaldi_reader.Value());

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


