/******************************************************************************
IrstLM: IRST Language Model Toolkit
Copyright (C) 2010 Christian Hardmeier, FBK-irst Trento, Italy

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA

******************************************************************************/

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "cmd.h"
#include "util.h"
#include "lmtable.h"
#include "n_gram.h"

using namespace irstlm;

void print_help(int TypeFlag=0){
  std::cerr << std::endl << "score-lm - scores sentences with a language model" << std::endl;
  std::cerr << std::endl << "USAGE:"  << std::endl
            << "       score-lm -lm <model>  [options]" << std::endl;
  std::cerr << std::endl << "OPTIONS:" << std::endl;
  std::cerr << "       -lm      language model to use (must be specified)" << std::endl;
  std::cerr << "       -dub     dictionary upper bound (default: 10000000" << std::endl;
  std::cerr << "       -level   max level to load from the language models (default: 1000," << std::endl;
  std::cerr << "           meaning the actual LM order)" << std::endl;
  std::cerr << "       -mm 1    memory-mapped access to lm (default: 0)" << std::endl;
  std::cerr << std::endl;

  FullPrintParams(TypeFlag, 0, 1, stderr);
}

void usage(const char *msg = 0)
{
  if (msg){
    std::cerr << msg << std::endl;
  }
  else{
                print_help();
  }
}

int main(int argc, char **argv)
{
  int mmap = 0;
  int dub = IRSTLM_DUB_DEFAULT;
  int requiredMaxlev = IRSTLM_REQUIREDMAXLEV_DEFAULT;
  char *lm = NULL;

  bool help=false;

        DeclareParams((char*)
                "lm", CMDSTRINGTYPE|CMDMSG, &lm, "language model to use (must be specified)",
                "DictionaryUpperBound", CMDINTTYPE|CMDMSG, &dub, "dictionary upperbound to compute OOV word penalty: default 10^7",
                "dub", CMDINTTYPE|CMDMSG, &dub, "dictionary upperbound to compute OOV word penalty: default 10^7",
                "memmap", CMDINTTYPE|CMDMSG, &mmap, "uses memory map to read a binary LM",
                "mm", CMDINTTYPE|CMDMSG, &mmap, "uses memory map to read a binary LM",
                "level", CMDINTTYPE|CMDMSG, &requiredMaxlev, "maximum level to load from the LM; if value is larger than the actual LM order, the latter is taken",
                "lev", CMDINTTYPE|CMDMSG, &requiredMaxlev, "maximum level to load from the LM; if value is larger than the actual LM order, the latter is taken",
                                                                
                "Help", CMDBOOLTYPE|CMDMSG, &help, "print this help",
                "h", CMDBOOLTYPE|CMDMSG, &help, "print this help",
                                                                
                (char *)NULL
                );

  if (argc == 1){
		usage();
		exit_error(IRSTLM_NO_ERROR);
  }

  GetParams(&argc, &argv, (char*) NULL);
        
  if (help){
		usage();
		exit_error(IRSTLM_NO_ERROR);
  }


  if(lm == NULL){
		usage();
		exit_error(IRSTLM_ERROR_DATA,"Missing parameter: please, specify the LM to use (-lm)");
  }

  std::ifstream lmstr(lm);
  lmtable lmt;
  lmt.setMaxLoadedLevel(requiredMaxlev);
  lmt.load(lmstr, lm, NULL, mmap);
  lmt.setlogOOVpenalty(dub);

  for(;;) {
    std::string line;
    std::getline(std::cin, line);
    if(!std::cin.good())
      return !std::cin.eof();

    std::istringstream linestr(line);
    ngram ng(lmt.dict);

    double logprob = .0;
    while((linestr >> ng))
      logprob += lmt.lprob(ng);

    std::cout << logprob << std::endl;
  }
}
