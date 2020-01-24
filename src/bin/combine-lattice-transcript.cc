// bin/compile-reference-graph.cc

// This binary is a part of the implementation of lattice-based SST training approach
// introduced in http://arxiv.org/abs/1905.13150 by Joachim Fainberg, et al.

// Copyright 2019       Brno University of Technology (author: Martin Kocour)

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/training-graph-compiler.h"
#include "lat/kaldi-lattice.h"

using namespace kaldi;
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;
using fst::MutableFst;
using fst::MakeLinearAcceptor;
using fst::StateIterator;
using fst::ArcIterator;
using std::vector;
using std::string;

bool FilterWords(const vector<int32> &trans, const CompactLattice &clat, vector<int32> *words) {
    if (words == NULL) {
        return false;
    }
    *words = trans; // This will copy transcriptions to words
    SortAndUniq(words);

    vector<int32> clat_isymbols; // Lattice int32 input symbols

    CompactLattice::StateId start = clat.Start();
    if (start == fst::kNoStateId) {
        return false; // Lattice is empty
    }
    for (StateIterator<CompactLattice> siter(clat); !siter.Done(); siter.Next()) {
        const auto s = siter.Value();
        if (s != start) {
            for (ArcIterator<CompactLattice> aiter(clat, s); !aiter.Done(); aiter.Next()) {
                const CompactLatticeArc &arc = aiter.Value();
                clat_isymbols.push_back((int32) arc.ilabel);
            }
        }
    }
    // Concatenate both vectors
    vector<int32> tmp;
    tmp.reserve(words->size() + clat_isymbols.size());
    tmp.insert(tmp.end(), words->begin(), words->end());
    tmp.insert(tmp.end(), clat_isymbols.begin(), clat_isymbols.end());
    *words = tmp;
    SortAndUniq(words);
    return true;
}

template<class Arc>
void MakeEditTransducer(const vector<int32> &words, MutableFst<Arc> *ofst) {
    typedef typename Arc::StateId StateId;
    typedef typename Arc::Weight Weight;

    vector<int32> labels(words);
    SortAndUniq(&labels);

    ofst->DeleteStates();
    StateId cur_state = ofst->AddState();
    ofst->SetStart(cur_state);

    // Add (w_i, w_j) arcs
    for (size_t i = 0; i < labels.size(); i++) {
        for (size_t j = 0; j < labels.size(); j++) {
            Arc arc;
            if (labels[j] == labels[i]) {
                arc = Arc(labels[i], labels[j], Weight(-1.0), cur_state);
            } else {
                arc = Arc(labels[i], labels[j], Weight::One(), cur_state);
            }
            ofst->AddArc(cur_state, arc);
        }
    }

    // Add (eps, w) and (w, eps) arcs
    int32 eps = 0;
    for (size_t i = 0; i < labels.size(); i++) {
        if(labels[i] == eps) {
            continue;
        }
        Arc arc1(eps, labels[i], Weight::One(), cur_state);
        Arc arc2(labels[i], eps, Weight::One(), cur_state);
        ofst->AddArc(cur_state, arc1);
        ofst->AddArc(cur_state, arc2);
    }
    ofst->SetFinal(cur_state, Weight::One());
}

template<class Arc, class I>
void MakeReferenceTransducer(const vector<I> &labels, MutableFst<Arc> *ofst) {
    MakeLinearAcceptor(labels, ofst);
}

template<class Arc>
void MakeHypothesisTransducer(CompactLattice &clat, vector<vector<double>> scale, MutableFst<Arc> *ofst) {
    ScaleLattice(scale, &clat); // typically scales to zero.
    fst::RemoveAlignmentsFromCompactLattice(&clat);
    kaldi::Lattice lat;
    ConvertLattice(clat, &lat); // convert to non-compact form.. won't introduce
    // extra states because already removed alignments.
    ConvertLattice(lat, ofst); // this adds up the (lm,acoustic) costs to get
    // the normal (tropical) costs.
    fst::Project(ofst, fst::PROJECT_OUTPUT); // project on words.
}

int main(int argc, char *argv[]) {
    try {

        BaseFloat acoustic_scale = 0.0;
        BaseFloat lm_scale = 0.0;

        const char *usage =
            "Combine lattices with transcripts in such way that \n"
            "they can be used in the lattice based SST training\n"
            "\n"
            "Usage: compile-reference-graph [options] <word-rxfilename> \n"
            "         <transcripts-rspecifier> <lattice-rspecifier> <fst-wspecifier>\n"
            "e.g.   compile-reference-graph 'ark:sym2int.pl --map-oov 1 -f 2- words.txt text|' \n"
            "         ark:1.lats ark:reference.fsts\n"
            "\n";
        ParseOptions po(usage);
        po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
        po.Register("lm-scale", &lm_scale, "Scaling factor for graph/lm costs");

        po.Read(argc, argv);

        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }

        string transcript_rspecifier = po.GetArg(1);
        string lats_rspecifier = po.GetArg(2);
        string fsts_wspecifier = po.GetArg(3);

        vector<vector<double> > scale = fst::LatticeScale(lm_scale, acoustic_scale);

        SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
        RandomAccessCompactLatticeReader clat_reader(lats_rspecifier);
        TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);

        int num_succeed = 0, num_fail = 0;

        for (; !transcript_reader.Done(); transcript_reader.Next()) {
            std::string key = transcript_reader.Key();
            const std::vector<int32> &transcript = transcript_reader.Value();

            if (!clat_reader.HasKey(key)) {
                num_fail++;
                continue;
            }

            CompactLattice clat = clat_reader.Value(key);

            // We do not create large Edit FST for all words
            // Instead, we create E FST only for words,
            // which occurs in both transcripts and in lattice
            vector<int32> labels;
            if (! FilterWords(transcript, clat, &labels)) {
                KALDI_WARN << "Empty Lattice for utterance "
                           << key;
                num_fail++;
                continue;
            }

            VectorFst<StdArc> edit_fst;
            MakeEditTransducer(labels, &edit_fst);

            if (edit_fst.Start() == fst::kNoStateId) {
                KALDI_WARN << "Empty edit FST for utterance "
                           << key;
                num_fail++;
                continue;
            }

            VectorFst<StdArc> reference_fst;
            MakeReferenceTransducer(transcript, &reference_fst);

            if (reference_fst.Start() == fst::kNoStateId) {
                KALDI_WARN << "Empty transcript FST for utterance "
                           << key;
                num_fail++;
                continue;
            }

            fst::VectorFst<StdArc> hypothesis_fst;
            MakeHypothesisTransducer(clat, scale, &hypothesis_fst);

            if (hypothesis_fst.Start() == fst::kNoStateId) {
                KALDI_WARN << "Empty lattice for utterance "
                           << key;
                num_fail++;
                continue;
            }

            VectorFst<StdArc> ref_edit_fst;
            fst::TableCompose(reference_fst, edit_fst, &ref_edit_fst); // TODO add cache
            fst::ArcSort(&ref_edit_fst, fst::OLabelCompare<StdArc>());

            if (ref_edit_fst.Start() == fst::kNoStateId) {
                KALDI_WARN << "Empty composition of transcripts with edit FST for utterance "
                           << key;
                num_fail++;
                continue;
            }

            VectorFst<StdArc> ref_edit_hyp_fst;
            fst::TableCompose(ref_edit_fst, hypothesis_fst, &ref_edit_hyp_fst); // TODO add cache
            fst::ArcSort(&ref_edit_hyp_fst, fst::OLabelCompare<StdArc>());

            if (ref_edit_hyp_fst.Start() == fst::kNoStateId) {
                KALDI_WARN << "Empty composition of transcripts with edit FST and hypothesis for utterance "
                           << key;
                num_fail++;
                continue;
            }

            StdArc::Weight threshold = StdArc::Weight().One();
            fst::Prune(&ref_edit_hyp_fst, threshold);
            fst::Project(&ref_edit_hyp_fst, fst::PROJECT_OUTPUT);
            fst::RmEpsilon(&ref_edit_hyp_fst);
            VectorFst<StdArc> ref_edit_hyp_fst_determinized;
            fst::DeterminizeStar(ref_edit_hyp_fst, &ref_edit_hyp_fst_determinized);
            fst::Minimize(&ref_edit_hyp_fst_determinized);

            if (ref_edit_hyp_fst_determinized.Start() != fst::kNoStateId) {
                num_succeed++;
                fst_writer.Write(key, ref_edit_hyp_fst_determinized);
            } else {
                KALDI_WARN << "Empty final graph for utterance "
                           << key;
                num_fail++;
                continue;
            }
        }

        KALDI_LOG << "compile-train-graphs: succeeded for " << num_succeed
                  << " graphs, failed for " << num_fail;
        return (num_succeed != 0 ? 0 : 1);
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
