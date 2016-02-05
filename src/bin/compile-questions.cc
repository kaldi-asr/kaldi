// bin/compile-questions.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "hmm/hmm-topology.h"
#include "tree/build-tree-questions.h"


namespace kaldi {
int32 ProcessTopo(const HmmTopology &topo, const std::vector<std::vector<int32> > &questions) {
  std::vector<int32> seen_phones;  // ids of phones seen in questions.
  for (size_t i = 0; i < questions.size(); i++)
    for (size_t j= 0; j < questions[i].size(); j++) seen_phones.push_back(questions[i][j]);
  SortAndUniq(&seen_phones);
  // topo_phones is also sorted and uniq; a list of phones defined in the topology.
  const std::vector<int32> &topo_phones = topo.GetPhones();
  if (seen_phones != topo_phones) {
    std::ostringstream ss_seen, ss_topo;
    WriteIntegerVector(ss_seen, false, seen_phones);
    WriteIntegerVector(ss_topo, false, topo_phones);
    KALDI_WARN << "ProcessTopo: phones seen in questions differ from those in topology: "
               << ss_seen.str() << " vs. " << ss_topo.str();
    if (seen_phones.size() > topo_phones.size()) {
      KALDI_ERR << "ProcessTopo: phones are asked about that are undefined in the topology.";
    } // we accept the reverse (not asking about all phones), even though it's very bad.
  }

  int32 max_num_pdf_classes = 0;
  for (size_t i = 0; i < topo_phones.size(); i++) {
    int32 p = topo_phones[i];
    int32 num_pdf_classes = topo.NumPdfClasses(p);
    max_num_pdf_classes = std::max(num_pdf_classes, max_num_pdf_classes);
  }
  KALDI_LOG << "Max # pdf classes is " << max_num_pdf_classes;
  return max_num_pdf_classes;
}

} // end namespace kaldi.

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Compile questions\n"
        "Usage:  compile-questions [options] <topo> <questions-text-file> <questions-out>\n"
        "e.g.: \n"
        " compile-questions questions.txt questions.qst\n";
    bool binary = true;
    int32 P = 1, N = 3;
    int32 num_iters_refine = 0,
        leftmost_questions_truncate = -1;


    ParseOptions po(usage);
    po.Register("binary", &binary,
                "Write output in binary mode");
    po.Register("context-width", &N,
                "Context window size [must match acc-tree-stats].");
    po.Register("central-position", &P,
                "Central position in phone context window [must match acc-tree-stats]");
    po.Register("num-iters-refine", &num_iters_refine,
                "Number of iters of refining questions at each node.  >0 --> questions "
                "not refined");
    po.Register("leftmost-questions-truncate", &leftmost_questions_truncate,
                "If > 0, the questions for the left-most context position will be "
                "truncated to the specified number.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        topo_filename = po.GetArg(1),
        questions_rxfilename = po.GetArg(2),
        questions_out_filename = po.GetArg(3);

    HmmTopology topo;  // just needed for checking, and to get the
    // largest number of pdf-classes for any phone.
    ReadKaldiObject(topo_filename, &topo);

    std::vector<std::vector<int32> > questions;  // sets of phones.
    if (!ReadIntegerVectorVectorSimple(questions_rxfilename, &questions))
      KALDI_ERR << "Could not read questions from "
                 << PrintableRxfilename(questions_rxfilename);
    for (size_t i = 0; i < questions.size(); i++) {
      std::sort(questions[i].begin(), questions[i].end());
      if (!IsSortedAndUniq(questions[i]))
        KALDI_ERR << "Questions contain duplicate phones";
    }
    size_t nq = static_cast<int32>(questions.size());
    SortAndUniq(&questions);
    if (questions.size() != nq)
      KALDI_WARN << (nq-questions.size())
                 << " duplicate questions present in " << questions_rxfilename;

    // ProcessTopo checks that all phones in the topo are
    // represented in at least one questions (else warns), and
    // returns the max # pdf classes in any given phone (normally
    // 3).
    int32 max_num_pdfclasses = ProcessTopo(topo, questions);

    Questions qo;

    QuestionsForKey phone_opts(num_iters_refine);
    // the questions-options corresponding to keys 0, 1, .. N-1 which
    // represent the phonetic context positions (including the central phone).
    for (int32 n = 0; n < N; n++) {
      KALDI_LOG << "Setting questions for phonetic-context position "<< n;
      if (n == 0 && leftmost_questions_truncate > 0 &&
          leftmost_questions_truncate < questions.size()) {
        KALDI_LOG << "Truncating " << questions.size() << " to "
                  << leftmost_questions_truncate << " for position 0.";
        phone_opts.initial_questions.assign(
            questions.begin(), questions.begin() + leftmost_questions_truncate);
      } else {
        phone_opts.initial_questions = questions;
      }
      qo.SetQuestionsOf(n, phone_opts);
    }

    QuestionsForKey pdfclass_opts(num_iters_refine);
    std::vector<std::vector<int32> > pdfclass_questions(max_num_pdfclasses-1);
    for (int32 i = 0; i < max_num_pdfclasses - 1; i++)
      for (int32 j = 0; j <= i; j++)
        pdfclass_questions[i].push_back(j);
    // E.g. if max_num_pdfclasses == 3,  pdfclass_questions is now [ [0], [0, 1] ].
    pdfclass_opts.initial_questions = pdfclass_questions;
    KALDI_LOG << "Setting questions for hmm-position [hmm-position ranges from 0 to "<< (max_num_pdfclasses-1) <<"]";
    qo.SetQuestionsOf(kPdfClass, pdfclass_opts);

    WriteKaldiObject(qo, questions_out_filename, binary);
    KALDI_LOG << "Wrote questions to "<<questions_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
