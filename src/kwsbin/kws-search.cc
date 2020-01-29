// kwsbin/kws-search.cc

// Copyright 2012-2015  Johns Hopkins University (Authors: Guoguo Chen,
//                                                         Daniel Povey.
//                                                         Yenda Trmal)

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
#include "fstext/kaldi-fst-io.h"
#include "kws/kaldi-kws.h"

namespace kaldi {

typedef KwsLexicographicArc Arc;
typedef Arc::Weight Weight;
typedef Arc::StateId StateId;

// encode ilabel, olabel pair as a single 64bit (output) symbol
uint64 EncodeLabel(StateId ilabel, StateId olabel) {
  return (static_cast<int64>(olabel) << 32) + static_cast<int64>(ilabel);
}

// extract the osymbol from the 64bit symbol. That represents the utterance id
// in this setup -- we throw away the isymbol which is typically 0 or an
// disambiguation symbol
StateId DecodeLabelUid(uint64 osymbol) {
  return static_cast<StateId>(osymbol >> 32);
}

// this is a mapper adapter that helps converting
// between the StdArc FST (i.e. tropical semiring FST)
// to the KwsLexicographic FST. Structure will be kept,
// the weights converted/recomputed
class VectorFstToKwsLexicographicFstMapper {
 public:
  typedef fst::StdArc FromArc;
  typedef FromArc::Weight FromWeight;
  typedef KwsLexicographicArc ToArc;
  typedef KwsLexicographicWeight ToWeight;

  VectorFstToKwsLexicographicFstMapper() {}

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel,
                 arc.olabel,
                 (arc.weight == FromWeight::Zero() ?
                  ToWeight::Zero() :
                  ToWeight(arc.weight.Value(),
                           StdLStdWeight::One())),
                 arc.nextstate);
  }

  fst::MapFinalAction FinalAction() const {
    return fst::MAP_NO_SUPERFINAL;
  }

  fst::MapSymbolsAction InputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  fst::MapSymbolsAction OutputSymbolsAction() const {
    return fst::MAP_COPY_SYMBOLS;
  }

  uint64 Properties(uint64 props) const { return props; }
};

struct ActivePath {
  std::vector<KwsLexicographicArc::Label> path;
  KwsLexicographicArc::Weight weight;
  KwsLexicographicArc::Label last;
};

bool GenerateActivePaths(const KwsLexicographicFst &proxy,
                       std::vector<ActivePath> *paths,
                       KwsLexicographicFst::StateId cur_state,
                       std::vector<KwsLexicographicArc::Label> cur_path,
                       KwsLexicographicArc::Weight cur_weight) {
  for (fst::ArcIterator<KwsLexicographicFst> aiter(proxy, cur_state);
       !aiter.Done(); aiter.Next()) {
    const Arc &arc = aiter.Value();
    Weight temp_weight = Times(arc.weight, cur_weight);

    cur_path.push_back(arc.ilabel);

    if ( arc.olabel != 0 ) {
      ActivePath path;
      path.path = cur_path;
      path.weight = temp_weight;
      path.last = arc.olabel;
      paths->push_back(path);
    } else {
      GenerateActivePaths(proxy, paths,
                        arc.nextstate, cur_path, temp_weight);
    }
    cur_path.pop_back();
  }

  return true;
}
}  // namespace kaldi

typedef kaldi::TableWriter< kaldi::BasicVectorHolder<double> >
                                                        VectorOfDoublesWriter;
void OutputDetailedStatistics(const std::string &kwid,
                        const kaldi::KwsLexicographicFst &keyword,
                        const unordered_map<uint32, uint64> &label_decoder,
                        VectorOfDoublesWriter *output ) {
  std::vector<kaldi::ActivePath> paths;

  if (keyword.Start() == fst::kNoStateId)
    return;

  kaldi::GenerateActivePaths(keyword, &paths, keyword.Start(),
                  std::vector<kaldi::KwsLexicographicArc::Label>(),
                  kaldi::KwsLexicographicArc::Weight::One());

  for (int i = 0; i < paths.size(); ++i) {
    std::vector<double> out;
    double score;
    int32 tbeg, tend, uid;

    uint64 osymbol = label_decoder.find(paths[i].last)->second;
    uid = kaldi::DecodeLabelUid(osymbol);
    tbeg = paths[i].weight.Value2().Value1().Value();
    tend = paths[i].weight.Value2().Value2().Value();
    score = paths[i].weight.Value1().Value();

    out.push_back(uid);
    out.push_back(tbeg);
    out.push_back(tend);
    out.push_back(score);

    for (int j = 0; j < paths[i].path.size(); ++j) {
      out.push_back(paths[i].path[j]);
    }
    output->Write(kwid, out);
  }
}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using std::vector;
    typedef kaldi::int32 int32;
    typedef kaldi::uint32 uint32;
    typedef kaldi::uint64 uint64;

    const char *usage =
        "Search the keywords over the index. This program can be executed\n"
        "in parallel, either on the index side or the keywords side; we use\n"
        "a script to combine the final search results. Note that the index\n"
        "archive has a single key \"global\".\n\n"
        "Search has one or two outputs. The first one is mandatory and will\n"
        "contain the seach output, i.e. list of all found keyword instances\n"
        "The file is in the following format:\n"
        "kw_id utt_id beg_frame end_frame neg_logprob\n"
        " e.g.: \n"
        "KW105-0198 7 335 376 1.91254\n\n"
        "The second parameter is optional and allows the user to gather more\n"
        "statistics about the individual instances from the posting list.\n"
        "Remember \"keyword\" is an FST and as such, there can be multiple\n"
        "paths matching in the keyword and in the lattice index in that given\n"
        "time period. The stats output will provide all matching paths\n"
        "each with the appropriate score. \n"
        "The format is as follows:\n"
        "kw_id utt_id beg_frame end_frame neg_logprob 0 w_id1 w_id2 ... 0\n"
        " e.g.: \n"
        "KW105-0198 7 335 376 16.01254 0 5766 5659 0\n"
        "\n"
        "Usage: kws-search [options] <index-rspecifier> <keywords-rspecifier> "
        "<results-wspecifier> [<stats_wspecifier>]\n"
        " e.g.: kws-search ark:index.idx ark:keywords.fsts "
                           "ark:results ark:stats\n";

    ParseOptions po(usage);

    int32 n_best = -1;
    int32 keyword_nbest = -1;
    bool strict = true;
    double negative_tolerance = -0.1;
    double keyword_beam = -1;
    int32 frame_subsampling_factor = 1;

    po.Register("frame-subsampling-factor", &frame_subsampling_factor,
                "Frame subsampling factor. (Default value 1)");
    po.Register("nbest", &n_best, "Return the best n hypotheses.");
    po.Register("keyword-nbest", &keyword_nbest,
                "Pick the best n keywords if the FST contains "
                "multiple keywords.");
    po.Register("strict", &strict, "Affects the return status of the program.");
    po.Register("negative-tolerance", &negative_tolerance,
                "The program will print a warning if we get negative score "
                "smaller than this tolerance.");
    po.Register("keyword-beam", &keyword_beam,
                "Prune the FST with the given beam if the FST contains "
                "multiple keywords.");

    if (n_best < 0 && n_best != -1) {
      KALDI_ERR << "Bad number for nbest";
      exit(1);
    }
    if (keyword_nbest < 0 && keyword_nbest != -1) {
      KALDI_ERR << "Bad number for keyword-nbest";
      exit(1);
    }
    if (keyword_beam < 0 && keyword_beam != -1) {
      KALDI_ERR << "Bad number for keyword-beam";
      exit(1);
    }

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string index_rspecifier = po.GetArg(1),
        keyword_rspecifier = po.GetArg(2),
        result_wspecifier = po.GetArg(3),
        stats_wspecifier = po.GetOptArg(4);

    RandomAccessTableReader< VectorFstTplHolder<KwsLexicographicArc> >
                                                index_reader(index_rspecifier);
    SequentialTableReader<VectorFstHolder> keyword_reader(keyword_rspecifier);
    VectorOfDoublesWriter result_writer(result_wspecifier);
    VectorOfDoublesWriter stats_writer(stats_wspecifier);


    // Index has key "global"
    KwsLexicographicFst index = index_reader.Value("global");

    // First we have to remove the disambiguation symbols. But rather than
    // removing them totally, we actually move them from input side to output
    // side, making the output symbol a "combined" symbol of the disambiguation
    // symbols and the utterance id's.
    // Note that in Dogan and Murat's original paper, they simply remove the
    // disambiguation symbol on the input symbol side, which will not allow us
    // to do epsilon removal after composition with the keyword FST. They have
    // to traverse the resulting FST.
    int32 label_count = 1;
    unordered_map<uint64, uint32> label_encoder;
    unordered_map<uint32, uint64> label_decoder;
    for (StateIterator<KwsLexicographicFst> siter(index);
                                           !siter.Done(); siter.Next()) {
      StateId state_id = siter.Value();
      for (MutableArcIterator<KwsLexicographicFst>
           aiter(&index, state_id); !aiter.Done(); aiter.Next()) {
        KwsLexicographicArc arc = aiter.Value();
        // Skip the non-final arcs
        if (index.Final(arc.nextstate) == Weight::Zero())
          continue;
        // Encode the input and output label of the final arc, and this is the
        // new output label for this arc; set the input label to <epsilon>
        uint64 osymbol = EncodeLabel(arc.ilabel, arc.olabel);
        arc.ilabel = 0;
        if (label_encoder.find(osymbol) == label_encoder.end()) {
          arc.olabel = label_count;
          label_encoder[osymbol] = label_count;
          label_decoder[label_count] = osymbol;
          label_count++;
        } else {
          arc.olabel = label_encoder[osymbol];
        }
        aiter.SetValue(arc);
      }
    }
    ArcSort(&index, fst::ILabelCompare<KwsLexicographicArc>());

    int32 n_done = 0;
    int32 n_fail = 0;
    for (; !keyword_reader.Done(); keyword_reader.Next()) {
      std::string key = keyword_reader.Key();
      VectorFst<StdArc> keyword = keyword_reader.Value();
      keyword_reader.FreeCurrent();

      // Process the case where we have confusion for keywords
      if (keyword_beam != -1) {
        Prune(&keyword, keyword_beam);
      }
      if (keyword_nbest != -1) {
        VectorFst<StdArc> tmp;
        ShortestPath(keyword, &tmp, keyword_nbest, true, true);
        keyword = tmp;
      }

      KwsLexicographicFst keyword_fst;
      KwsLexicographicFst result_fst;
      Map(keyword, &keyword_fst, VectorFstToKwsLexicographicFstMapper());
      Compose(keyword_fst, index, &result_fst);

      if (stats_wspecifier != "") {
        KwsLexicographicFst matched_seq(result_fst);
        OutputDetailedStatistics(key,
                                 matched_seq,
                                 label_decoder,
                                 &stats_writer);
      }

      Project(&result_fst, PROJECT_OUTPUT);
      Minimize(&result_fst, (KwsLexicographicFst *) nullptr, kDelta, true);
      ShortestPath(result_fst, &result_fst, n_best);
      RmEpsilon(&result_fst);

      // No result found
      if (result_fst.Start() == kNoStateId)
        continue;

      // Got something here
      double score;
      int32 tbeg, tend, uid;
      for (ArcIterator<KwsLexicographicFst>
           aiter(result_fst, result_fst.Start()); !aiter.Done(); aiter.Next()) {
        const KwsLexicographicArc &arc = aiter.Value();

        // We're expecting a two-state FST
        if (result_fst.Final(arc.nextstate) != Weight::One()) {
          KALDI_WARN << "The resulting FST does not have "
                     << "the expected structure for key " << key;
          n_fail++;
          continue;
        }

        uint64 osymbol = label_decoder[arc.olabel];
        uid = static_cast<int32>(DecodeLabelUid(osymbol));
        tbeg = arc.weight.Value2().Value1().Value();
        tend = arc.weight.Value2().Value2().Value();
        score = arc.weight.Value1().Value();

        if (score < 0) {
          if (score < negative_tolerance) {
            KALDI_WARN << "Score out of expected range: " << score;
          }
          score = 0.0;
        }
        vector<double> result;
        result.push_back(uid);
        result.push_back(tbeg * frame_subsampling_factor);
        result.push_back(tend * frame_subsampling_factor);
        result.push_back(score);
        result_writer.Write(key, result);
      }

      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " keywords";
    if (strict == true)
      return (n_done != 0 ? 0 : 1);
    else
      return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
