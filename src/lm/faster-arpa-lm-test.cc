// bin/latgen-fasterlm-faster-mapped .cc

// Copyright      2018  Zhehuai Chen

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
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"
#include "lm/faster-arpa-lm.h"
#include "lm/const-arpa-lm.h"
#include "decoder/lattice-biglm-faster-decoder.h"

//echo 14207 198712 7589 175861 175861 104488 150861 139719 78075 14268 124782 61783 196158 4 20681 194454 137421 158810 161569 4 37434 50498 | awk '{for (i=1;i<=NF;i++)printf $i", "}END{print "\n"NF}'
// ~/src/kaldi/src/lm/faster-arpa-lm-test --symbol-size=200007 --bos-symbol=200005 --eos-symbol=200006 --unk-symbol=3 --verbose=7  'fstproject --project_output=true data/lang_test_tgmed/G.fst | fstarcsort --sort_type=ilabel |' data/lang_nosp_test_tgmed/G.carpa 'gunzip -c data/local/lm/3-gram.pruned.1e-7.arpa.gz| utils/map_arpa_lm.pl data/lang_test_tgsmall/words.txt|'
//
namespace kaldi {

    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::Fst;
    using fst::StdArc;
#define  Arc fst::StdArc
    using fst::ReadFstKaldi;


void get_score(fst::CacheDeterministicOnDemandFst<StdArc>* cache_dfst,
    int* word_ids, int* state_ids, float* scores, int len) {
  state_ids[0]=cache_dfst->Start();
  std::cout << "word,state,score: \n";
  for (int i =0;i<len;i++) {
  Arc lm_arc;
  assert(cache_dfst->GetArc(state_ids[i], word_ids[i], &lm_arc));
  if (i< len-1) state_ids[i+1]=lm_arc.nextstate;
  scores[i]=lm_arc.weight.Value();
  std::cout <<word_ids[i]<<","<<state_ids[i]<<","<<scores[i]<<"\n";
  }
}
}
int main(int argc, char *argv[]) {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::Fst;
    using fst::StdArc;
#define  Arc fst::StdArc
    using fst::ReadFstKaldi;

#define TEST_SIZE 39
//#define TEST_SIZE 28
//#define TEST_SIZE 25
    ParseOptions po("");
    float scores[TEST_SIZE];
    float scores2[TEST_SIZE];
    float scores3[TEST_SIZE];
    //int32 word_ids[]={14207, 198712, 7589, 175861, 171937, 124782, 36528, 175861, 104488, 150861, 139719, 78075, 14268, 124782, 61783, 196158, 4, 20681, 194454, 137421, 158810, 161569, 4, 37434, 50498};
    //int32 word_ids[] = {14207, 198712, 7589, 4, 171935, 87918, 124782, 36528, 175861, 104488, 150861, 139719, 78075, 14268, 124782, 61783, 196158, 4, 20681, 194454, 138359, 155516, 2379, 160908, 2811, 4, 37434, 50498};
    int32 word_ids[] = {78521, 148206, 178313, 175861, 144826, 28459, 25372, 62655, 138328, 175861, 72352, 76155, 152997, 4, 102911, 177031, 193231, 127711, 71590, 47932, 151710, 40606, 5411, 82074, 86219, 81505, 77097, 4, 155384, 194419, 193822, 71589, 76098, 163928, 124918, 177084, 9376, 81505, 78840};
    int32 state_ids[TEST_SIZE]={0};

    ArpaParseOptions arpa_options;
    arpa_options.Register(&po);
    int32 symbol_size;
    po.Register("symbol-size", &symbol_size, "symbol table size");
    po.Register("unk-symbol", &arpa_options.unk_symbol,
                "Integer corresponds to unknown-word in language model. -1 if "
                "no such word is provided.");
    po.Register("bos-symbol", &arpa_options.bos_symbol,
                "Integer corresponds to <s>. You must set this to your actual "
                "BOS integer.");
    po.Register("eos-symbol", &arpa_options.eos_symbol,
                "Integer corresponds to </s>. You must set this to your actual "
                "EOS integer.");

    po.Read(argc, argv);

    {
    std::string g_lm_fst_rxfilename = po.GetArg(1);
    VectorFst<StdArc> *old_lm_fst = fst::CastOrConvertToVectorFst(
        fst::ReadFstKaldiGeneric(g_lm_fst_rxfilename));
    fst::BackoffDeterministicOnDemandFst<StdArc> old_lm_dfst(*old_lm_fst);
    fst::CacheDeterministicOnDemandFst<StdArc> cache_dfst(&old_lm_dfst, 1e7);
    get_score(&cache_dfst, word_ids, state_ids, scores, TEST_SIZE);
    }
   {
    std::string g_lm_fst_rxfilename = po.GetArg(2);
    ConstArpaLm new_lm;
    ReadKaldiObject(g_lm_fst_rxfilename, &new_lm);
    ConstArpaLmDeterministicFst new_lm_dfst(new_lm);
    fst::CacheDeterministicOnDemandFst<StdArc> cache_dfst(&new_lm_dfst, 1e7);
    get_score(&cache_dfst, word_ids, state_ids, scores2, TEST_SIZE);
    }
   {
    std::string g_lm_fst_rxfilename = po.GetArg(3);
    FasterArpaLm new_lm(arpa_options, g_lm_fst_rxfilename, symbol_size);
    FasterArpaLmDeterministicFst new_lm_dfst(new_lm);
    fst::CacheDeterministicOnDemandFst<StdArc> cache_dfst(&new_lm_dfst, 1e7);
    get_score(&cache_dfst, word_ids, state_ids, scores3, TEST_SIZE);
   }
   for (int i=0;i<TEST_SIZE;i++) {
     if (abs(scores[i]-scores2[i])>1e-4) KALDI_LOG<<scores[i]<< " "<< scores2[i]<< " "<<word_ids[i]<<" "<<i;
     if (abs(scores[i]-scores3[i])>1e-4) KALDI_LOG<<scores[i]<< " "<< scores3[i]<< " "<<word_ids[i]<<" "<<i;
   }
   return 0;
}
