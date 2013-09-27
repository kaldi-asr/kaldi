// lm/kaldi-lmtable.cc
//
// Copyright 2009-2011 Gilles Boulianne.
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
/**
 * @file kaldi-lmtable.cc
 * @brief Implementation of internal representation for language model.
 *
 * See kaldi-lmtable.h for more details.
 */

#include "lm/kaldi-lmtable.h"
#include "base/kaldi-common.h"
#include <sstream>

namespace kaldi {

// typedef fst::StdArc::StateId StateId;

// newlyAdded will be updated
LmFstConverter::StateId LmFstConverter::AddStateFromSymb(
    const std::vector<string> &ngramString,
    int kstart, int kend,
    fst::StdVectorFst *pfst,
    bool &newlyAdded) {
  fst::StdArc::StateId sid;
  std::string separator;
  separator.resize(1);
  separator[0] = '\0';
  
  std::string hist;
  if (kstart == 0) {
    hist.append(separator);
  } else {
    for (int k = kstart; k >= kend; k--) {
      hist.append(ngramString[k]);
      hist.append(separator);
    }
  }

  newlyAdded = false;
  sid = FindState(hist);
  if (sid < 0) {
    sid = pfst->AddState();
    histState_[hist] = sid; 
    newlyAdded = true;
    //cerr << "Created state " << sid << " for " << hist << endl;
  } else {
    //cerr << "State symbol " << hist << " already exists" << endl;
  }

  return sid;
}

void LmFstConverter::ConnectUnusedStates(fst::StdVectorFst *pfst) {

  // go through all states with a recorded backoff destination 
  // and find out any that has no output arcs and is not final
  unsigned int connected = 0;
  // cerr << "ConnectUnusedStates has recorded "<<bkState_.size()<<" states.\n";

  for (BkStateMap::iterator bkit = bkState_.begin(); bkit != bkState_.end(); ++bkit) {
    // add an output arc to its backoff destination recorded in backoff_
    fst::StdArc::StateId src = bkit->first, dst = bkit->second;
    if (pfst->NumArcs(src)==0 && !IsFinal(pfst, src)) {
      // cerr << "ConnectUnusedStates: adding arc from "<<src<<" to "<<dst<<endl;
      pfst->AddArc(src, fst::StdArc(0, 0, fst::StdArc::Weight::One(), dst)); // epsilon arc with no cost
      connected++;
    }
  }
  cerr << "Connected " << connected << " states without outgoing arcs." << endl;
}

void LmFstConverter::AddArcsForNgramProb(
    int ilev, int maxlev,
    float logProb,
    float logBow,
    std::vector<string> &ngs,
    fst::StdVectorFst *fst,
    const string startSent,
    const string endSent) {
  fst::StdArc::StateId src, dst, dbo;
  std::string curwrd = ngs[1];
  int64 ilab, olab;
  LmWeight prob = ConvertArpaLogProbToWeight(logProb);
  LmWeight bow  = ConvertArpaLogProbToWeight(logBow);
  bool newSrc, newDbo, newDst = false;

  if (ilev >= 2) {
    // General case works from N down to 2-grams
    src = AddStateFromSymb(ngs,   ilev,   2, fst, newSrc);
    if (ilev != maxlev) {
	  // add all intermediate levels from 2 to current
	  // last ones will be current backoff source and destination
	  for (int iilev=2; iilev <= ilev; iilev++) {
		dst = AddStateFromSymb(ngs, iilev,   1, fst, newDst);
		dbo = AddStateFromSymb(ngs, iilev-1, 1, fst, newDbo);
		bkState_[dst] = dbo;
	  }
    } else {
	  // add all intermediate levels from 2 to current
	  // last ones will be current backoff source and destination
	  for (int iilev=2; iilev <= ilev; iilev++) {
		dst = AddStateFromSymb(ngs, iilev-1, 1, fst, newDst);
		dbo = AddStateFromSymb(ngs, iilev-2, 1, fst, newDbo);
		bkState_[dst] = dbo;
	  }
    }
  } else {
    // special case for 1-grams: start from 0-gram
    if (curwrd.compare(startSent) != 0) {
      src = AddStateFromSymb(ngs, 0, 1, fst, newSrc);
    } else {
      // extra special case if in addition we are at beginning of sentence
      // starts from initial state and has no cost
      src = fst->Start();
      prob = fst::StdArc::Weight::One();
    }
    dst = AddStateFromSymb(ngs, 1, 1, fst, newDst);
    dbo = AddStateFromSymb(ngs, 0, 1, fst, newDbo);
    bkState_[dst] = dbo;
  }

  // state is final if last word is end of sentence
  if (curwrd.compare(endSent) == 0) {
    fst->SetFinal(dst, fst::StdArc::Weight::One());
  }
  // add labels to symbol tables
  ilab = fst->MutableInputSymbols()->AddSymbol(curwrd);
  olab = fst->MutableOutputSymbols()->AddSymbol(curwrd);

  // add arc with weight "prob" between source and destination states
  // cerr << "n-gram prob, fstAddArc: src "<< src << " dst " << dst;
  // cerr << " lab " << ilab << endl;
  fst->AddArc(src, fst::StdArc(ilab, olab, prob, dst));

  // add backoffs to any newly created destination state
  // but only if non-final
  if (!IsFinal(fst, dst) && newDst && dbo != dst) {
    ilab = olab = 0;
    // cerr << "backoff, fstAddArc: src "<< src << " dst " << dst;
    // cerr << " lab " << ilab << endl;
    fst->AddArc(dst, fst::StdArc(ilab, olab, bow, dbo));
  }
}

#ifndef HAVE_IRSTLM

bool LmTable::ReadFstFromLmFile(std::istream &istrm,
                                fst::StdVectorFst *fst,
                                bool useNaturalOpt,
                                const string startSent,
                                const string endSent) {
#ifdef KALDI_PARANOID
  KALDI_ASSERT(fst);
  KALDI_ASSERT(fst->InputSymbols() && fst->OutputSymbols());
#endif

  conv_->UseNaturalLog(useNaturalOpt);

  // do not use state symbol table for word histories anymore
  string inpline;
  size_t pos1, pos2;
  int ilev, maxlev = 0;

  // process \data\ section
  cerr << "\\data\\" << endl;

  while (getline(istrm, inpline) && !istrm.eof()) {
    std::istringstream ss(inpline);
    std::string token;
    ss >> token >> std::ws;
    if (token == "\\data\\" && ss.eof()) break;
  }
  if (istrm.eof()) {
    KALDI_ERR << "\\data\\ token not found in arpa file.\n";
  }

  while (getline(istrm, inpline) && !istrm.eof()) {
    // break out of loop if another section is found
    if (inpline.find("-grams:") != string::npos) break;
    if (inpline.find("\\end\\") != string::npos) break;

    // look for valid "ngram N = M" lines
    pos1 = inpline.find("ngram");
    pos2 = inpline.find("=");
    if (pos1 == string::npos ||  pos2 == string::npos || pos2 <= pos1) {
      continue;  // not valid, continue looking
    }
    // found valid line
    ilev = atoi(inpline.substr(pos1+5, pos2-(pos1+5)).c_str());
    if (ilev > maxlev) {
      maxlev = ilev;
    }
  }
  if (maxlev == 0) {
    // reached end of loop without having found any n-gram
    KALDI_ERR << "No ngrams found in specified file";
  }

  // process "\N-grams:" sections, we may have already read a "\N-grams:" line
  // if so, process it, otherwise get another line
  while (inpline.find("-grams:") != string::npos
         || (getline(istrm, inpline) && !istrm.eof()) ) {
    // look for a valid "\N-grams:" section
    pos1 = inpline.find("\\");
    pos2 = inpline.find("-grams:");
    if (pos1 == string::npos || pos2 == string::npos || pos2 <= pos1) {
      continue;  // not valid line, continue looking for one
    }
    // found, set current level
    ilev = atoi(inpline.substr(pos1+1, pos2-(pos1+1)).c_str());
    cerr << "Processing " << ilev <<"-grams" << endl;

    // process individual n-grams
    while (getline(istrm, inpline) && !istrm.eof()) {
      // break out of inner loop if another section is found
      if (inpline.find("-grams:") != string::npos) break;
      if (inpline.find("\\end\\") != string::npos) break;

      // parse ngram line: first field = prob, other fields = words,
      // last field = backoff (optional)
      std::vector<string> ngramString;
      float prob, bow;

      // eat up space.
      const char *cur_cstr = inpline.c_str();
      while (*cur_cstr && isspace(*cur_cstr))
        cur_cstr++;

      if (*cur_cstr == '\0') // Ignore empty lines.
        continue;
      char *next_cstr;
      // found, parse probability from first field
      prob = STRTOF(cur_cstr, &next_cstr);
      if (next_cstr == cur_cstr)
        KALDI_ERR << "Bad line in LM file [parsing "<<(ilev)<<"-grams]: "<<inpline;
      cur_cstr = next_cstr;
      while (*cur_cstr && isspace(*cur_cstr))
        cur_cstr++;

      for (int i = 0; i < ilev; i++) {

        if (*cur_cstr == '\0')
          KALDI_ERR << "Bad line in LM file [parsing "<<(ilev)<<"-grams]: "<<inpline;

        const char *end_cstr = strpbrk(cur_cstr, " \t");
        std::string this_word;
        if (end_cstr == NULL) {
          this_word = std::string(cur_cstr);
          cur_cstr += strlen(cur_cstr);
        } else {
          this_word = std::string(cur_cstr, end_cstr-cur_cstr);
          cur_cstr = end_cstr;
          while (*cur_cstr && isspace(*cur_cstr))
            cur_cstr++;
        }

        // words are inserted so position 1 is most recent word,
        // and position N oldest word (IRSTLM convention)
        ngramString.insert(ngramString.begin(), this_word);
      }
      // reserve an element 0 so that words go from 1, ..., ng.size-1
      ngramString.insert(ngramString.begin(), "");
      bow = 0;
      if (ilev < maxlev) {
        // try converting anything left in the line to a backoff weight
        if (*cur_cstr != '\0') {
          char *end_cstr;
          bow = STRTOF(cur_cstr, &end_cstr);
          if (end_cstr != cur_cstr) {  // got something.
            while (*end_cstr != '\0' && isspace(*end_cstr))
              end_cstr++;
            if (*end_cstr != '\0')
              KALDI_ERR << "Junk "<<(end_cstr)<<" at end of line [parsing "<<(ilev)<<"-grams]"<<inpline;
          } else {
            KALDI_ERR << "Junk "<<(cur_cstr)<<" at end of line [parsing "<<(ilev)<<"-grams]"<<inpline;
          }
        }
      }
      conv_->AddArcsForNgramProb(ilev, maxlev, prob, bow,
                                 ngramString, fst,
                                 startSent, endSent);
    }  // end of loop on individual n-gram lines
  }

  conv_->ConnectUnusedStates(fst);

  // not used anymore: delete pStateSymbs;

  // input and output symbol tables will be deleted by ~fst()
  return true;
}

#else

// #ifdef HAVE_IRSTLM implementation

bool LmTable::ReadFstFromLmFile(std::istream &istrm,
                                fst::StdVectorFst *fst,
                                bool useNaturalOpt,
                                const string startSent,
                                const string endSent) {
  load(istrm, "input name?", "output name?", 0, NONE);
  ngram ng(this->getDict(), 0);

  conv_->UseNaturalLog(useNaturalOpt);
  DumpStart(ng, fst, startSent, endSent);

  // should do some check before returning true
  return true;
}

// run through all nodes in table (as in dumplm)
void LmTable::DumpStart(ngram ng,
                        fst::StdVectorFst *fst,
                        const string startSent,
                        const string endSent) {
#ifdef KALDI_PARANOID
  KALDI_ASSERT(fst);
  KALDI_ASSERT(fst->InputSymbols() && fst->OutputSymbols());
#endif
  // we need a state symbol table while traversing word contexts
  fst::SymbolTable *pStateSymbs = new fst::SymbolTable("kaldi-lm-state");

  // dump level by level
  for (int l = 1; l <= maxlev; l++) {
    ng.size = 0;
    cerr << "Processing " << l << "-grams" << endl;
    DumpContinue(ng, 1, l, 0, cursize[1],
                 fst, pStateSymbs, startSent, endSent);
  }

  delete pStateSymbs;
  // input and output symbol tables will be deleted by ~fst()
}

// run through given levels and positions in table
void LmTable::DumpContinue(ngram ng, int ilev, int elev,
                           table_entry_pos_t ipos, table_entry_pos_t epos,
                           fst::StdVectorFst *fst,
                           fst::SymbolTable *pStateSymbs,
                           const string startSent, const string endSent) {
  LMT_TYPE ndt = tbltype[ilev];
  ngram ing(ng.dict);
  int ndsz = nodesize(ndt);

#ifdef KALDI_PARANOID
  KALDI_ASSERT(ng.size == ilev - 1);
  KALDI_ASSERT(ipos >= 0 && epos <= cursize[ilev] && ipos < epos);
  KALDI_ASSERT(pStateSymbs);
#endif

  ng.pushc(0);

  for (table_entry_pos_t i = ipos; i < epos; i++) {
    *ng.wordp(1) = word(table[ilev] + (table_pos_t)i * ndsz);
    float ipr = prob(table[ilev] + (table_pos_t)i * ndsz, ndt);
    // int ipr = prob(table[ilev] + i * ndsz, ndt);
    // skip pruned n-grams
    if (isPruned && ipr == NOPROB) continue;

    if (ilev < elev) {
      // get first and last successor position
      table_entry_pos_t isucc = (i > 0 ? bound(table[ilev] +
                                               (table_pos_t) (i-1) * ndsz,
                                               ndt) : 0);
      table_entry_pos_t esucc = bound(table[ilev] +
                                      (table_pos_t) i * ndsz, ndt);
      if (isucc < esucc)  // there are successors!
        DumpContinue(ng, ilev+1, elev, isucc, esucc,
                     fst, pStateSymbs, startSent, endSent);
      // else
      // cerr << "no successors for " << ng << "\n";
    } else {
      // cerr << i << " ";  // this was just to count printed n-grams
      // cerr << ipr <<"\t";
      // cerr << (isQtable?ipr:*(float *)&ipr) <<"\t";

      // if table is inverted then revert n-gram
      if (isInverted && ng.size > 1) {
        ing.invert(ng);
        ng = ing;
      }

      // cerr << "ilev " << ilev << " ngsize " << ng.size << endl;

      // for FST creation: vector of words strings
      std::vector<string> ngramString;
      for (int k = ng.size; k >= 1; k--) {
        // words are inserted so position 1 is most recent word,
        // and position N oldest word (IRSTLM convention)
        ngramString.insert(ngramString.begin(),
                           this->getDict()->decode(*ng.wordp(k)));
      }
      // reserve index 0 so that words go from 1, .., ng.size-1
      ngramString.insert(ngramString.begin(), "");
      float ibo = 0;
      if (ilev < maxlev) {
        // Backoff
        ibo = bow(table[ilev]+ (table_pos_t)i * ndsz, ndt);
        // if (isQtable) cerr << "\t" << ibo;
        // else if (ibo != 0.0) cerr << "\t" << ibo;
      }
      conv_->AddArcsForNgramProb(ilev, maxlev, ipr, ibo,
                                 ngramString, fst, pStateSymbs,
                                 startSent, endSent);
    }
  }
}

#endif

}  // end namespace kaldi

