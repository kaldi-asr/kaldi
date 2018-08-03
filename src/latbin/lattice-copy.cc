// latbin/lattice-copy.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {
  int32 CopySubsetLattices(std::string filename,
      SequentialLatticeReader *lattice_reader,
      LatticeWriter *lattice_writer,
      bool include = true, bool ignore_missing = false,
      bool sorted = false) {
    unordered_set<std::string, StringHasher> subset;
    std::set<std::string> subset_list;

    bool binary;
    Input ki(filename, &binary);
    KALDI_ASSERT(!binary);
    std::string line;
    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> split_line;
      SplitStringToVector(line, " \t\r", true, &split_line);
      if(split_line.empty()) {
        KALDI_ERR << "Unable to parse line \"" << line << "\" encountered in input in " << filename;
      }
      subset.insert(split_line[0]);
      subset_list.insert(split_line[0]);
    }

    int32 num_total = 0;
    size_t num_success = 0;
    for (; !lattice_reader->Done(); lattice_reader->Next(), num_total++) {
      if (include && sorted && subset_list.size() > 0
              && lattice_reader->Key() > *(subset_list.rbegin())) {
        KALDI_LOG << "The utterance " << lattice_reader->Key()
                  << " is larger than "
                  << "the last key in the include list. Not reading further.";
        KALDI_LOG << "Wrote " << num_success << " utterances";
        return 0;
      }

      if (include && subset.count(lattice_reader->Key()) > 0) {
        lattice_writer->Write(lattice_reader->Key(), lattice_reader->Value());
        num_success++;
      } else if (!include && subset.count(lattice_reader->Key()) == 0) {
        lattice_writer->Write(lattice_reader->Key(), lattice_reader->Value());
        num_success++;
      }
    }

    KALDI_LOG << "Wrote " << num_success << " out of " << num_total
      << " utterances.";

    if (ignore_missing) return 0;

    return (num_success != 0 ? 0 : 1);
  }

  int32 CopySubsetLattices(std::string filename,
      SequentialCompactLatticeReader *lattice_reader,
      CompactLatticeWriter *lattice_writer,
      bool include = true, bool ignore_missing = false,
      bool sorted = false) {
    unordered_set<std::string, StringHasher> subset;
    std::set<std::string> subset_list;

    bool binary;
    Input ki(filename, &binary);
    KALDI_ASSERT(!binary);
    std::string line;
    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> split_line;
      SplitStringToVector(line, " \t\r", true, &split_line);
      if(split_line.empty()) {
        KALDI_ERR << "Unable to parse line \"" << line << "\" encountered in input in " << filename;
      }
      subset.insert(split_line[0]);
      subset_list.insert(split_line[0]);
    }

    int32 num_total = 0;
    size_t num_success = 0;
    for (; !lattice_reader->Done(); lattice_reader->Next(), num_total++) {
      if (include && sorted && subset_list.size() > 0
              && lattice_reader->Key() > *(subset_list.rbegin())) {
        KALDI_LOG << "The utterance " << lattice_reader->Key()
                  << " is larger than "
                  << "the last key in the include list. Not reading further.";
        KALDI_LOG << "Wrote " << num_success << " utterances";
        return 0;
      }

      if (include && subset.count(lattice_reader->Key()) > 0) {
        lattice_writer->Write(lattice_reader->Key(), lattice_reader->Value());
        num_success++;
      } else if (!include && subset.count(lattice_reader->Key()) == 0) {
        lattice_writer->Write(lattice_reader->Key(), lattice_reader->Value());
        num_success++;
      }
    }

    KALDI_LOG << " Wrote " << num_success << " out of " << num_total
      << " utterances.";

    if (ignore_missing) return 0;

    return (num_success != 0 ? 0 : 1);
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Copy lattices (e.g. useful for changing to text mode or changing\n"
        "format to standard from compact lattice.)\n"
        "The --include and --exclude options can be used to copy only a subset "
        "of lattices, where are the --include option specifies the "
        "whitelisted utterances that would be copied and --exclude option "
        "specifies the blacklisted utterances that would not be copied.\n"
        "Only one of --include and --exclude can be supplied.\n"
        "Usage: lattice-copy [options] lattice-rspecifier lattice-wspecifier\n"
        " e.g.: lattice-copy --write-compact=false ark:1.lats ark,t:text.lats\n"
        "See also: lattice-scale, lattice-to-fst, and\n"
        "   the script egs/wsj/s5/utils/convert_slf.pl\n";

    ParseOptions po(usage);
    bool write_compact = true, ignore_missing = false;
    std::string include_rxfilename;
    std::string exclude_rxfilename;

    po.Register("write-compact", &write_compact, "If true, write in normal (compact) form.");
    po.Register("include", &include_rxfilename,
                "Text file, the first field of each "
                "line being interpreted as the "
                "utterance-id whose lattices will be included");
    po.Register("exclude", &exclude_rxfilename,
                "Text file, the first field of each "
                "line being interpreted as an utterance-id "
                "whose lattices will be excluded");
    po.Register("ignore-missing", &ignore_missing,
                "Exit with status 0 even if no lattices are copied");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2);

    RspecifierOptions opts;
    ClassifyRspecifier(lats_rspecifier, NULL, &opts);
    bool sorted = opts.sorted;

    int32 n_done = 0;

    if (write_compact) {
      SequentialCompactLatticeReader lattice_reader(lats_rspecifier);
      CompactLatticeWriter lattice_writer(lats_wspecifier);

      if (include_rxfilename != "") {
        if (exclude_rxfilename != "") {
          KALDI_ERR << "should not have both --exclude and --include option!";
        }
        return CopySubsetLattices(include_rxfilename,
            &lattice_reader, &lattice_writer,
            true, ignore_missing, sorted);
      } else if (exclude_rxfilename != "") {
        return CopySubsetLattices(exclude_rxfilename,
            &lattice_reader, &lattice_writer,
            false, ignore_missing);
      }

      for (; !lattice_reader.Done(); lattice_reader.Next(), n_done++)
        lattice_writer.Write(lattice_reader.Key(), lattice_reader.Value());
    } else {
      SequentialLatticeReader lattice_reader(lats_rspecifier);
      LatticeWriter lattice_writer(lats_wspecifier);

      if (include_rxfilename != "") {
        if (exclude_rxfilename != "") {
          KALDI_ERR << "should not have both --exclude and --include option!";
        }
        return CopySubsetLattices(include_rxfilename,
            &lattice_reader, &lattice_writer,
            true, ignore_missing, sorted);
      } else if (exclude_rxfilename != "") {
        return CopySubsetLattices(exclude_rxfilename,
            &lattice_reader, &lattice_writer,
            true, ignore_missing);
      }

      for (; !lattice_reader.Done(); lattice_reader.Next(), n_done++)
        lattice_writer.Write(lattice_reader.Key(), lattice_reader.Value());
    }
    KALDI_LOG << "Done copying " << n_done << " lattices.";

    if (ignore_missing) return 0;

    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
