// $Id: prune-lm.cpp 27 2010-05-03 14:33:51Z nicolabertoldi $

/******************************************************************************
 IrstLM: IRST Language Model Toolkit, prune LM
 Copyright (C) 2008 Fabio Brugnara, FBK-irst Trento, Italy

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


#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include "cmd.h"
#include "util.h"
#include "math.h"
#include "lmtable.h"

/********************************/
using namespace std;
using namespace irstlm;

void print_help(int TypeFlag=0){
  std::cerr << std::endl << "prune-lm - prunes language models" << std::endl;
  std::cerr << std::endl << "USAGE:"  << std::endl;
	std::cerr << "       prune-lm [options] <inputfile> [<outputfile>]" << std::endl;
  std::cerr << std::endl << "DESCRIPTION:" << std::endl;
	std::cerr << "       prune-lm reads a LM in either ARPA or compiled format and" << std::endl;
	std::cerr << "       prunes out n-grams (n=2,3,..) for which backing-off to the" << std::endl;
	std::cerr << "       lower order n-gram results in a small difference in probability." << std::endl;
	std::cerr << "       The pruned LM is saved in ARPA format" << std::endl;
  std::cerr << std::endl << "OPTIONS:" << std::endl;
	
	FullPrintParams(TypeFlag, 0, 1, stderr);
}

void usage(const char *msg = 0)
{
  if (msg){
    std::cerr << msg << std::endl;
	}
	if (!msg){
		print_help();
	}
}

void s2t(string	cps, float *thr)
{
  int	i;
  char *s=strdup(cps.c_str());
	char *tk;

  thr[0]=0;
  for(i=1,tk=strtok(s, ","); tk; tk=strtok(0, ","),i++) thr[i]=atof(tk);
  for(; i<MAX_NGRAM; i++) thr[i]=thr[i-1];
  free(s);
}

int main(int argc, char **argv)
{
  float thr[MAX_NGRAM];
  char *spthr=NULL;
	int	aflag=0;
  std::vector<std::string> files;
	
	bool help=false;
	
	DeclareParams((char*)								
		"threshold", CMDSTRINGTYPE|CMDMSG, &spthr,  "pruning thresholds for 2-grams, 3-grams, 4-grams,...; if less thresholds are specified, the last one is applied to all following n-gram levels; default is 0",
		"t", CMDSTRINGTYPE|CMDMSG, &spthr, "pruning thresholds for 2-grams, 3-grams, 4-grams,...; if less thresholds are specified, the last one is applied to all following n-gram levels; default is 0",
							
		"abs", CMDBOOLTYPE|CMDMSG, &aflag, "uses absolute value of weighted difference; default is 0",

		"Help", CMDBOOLTYPE|CMDMSG, &help, "print this help",
		"h", CMDBOOLTYPE|CMDMSG, &help, "print this help",
								
		(char *)NULL
		);

	if (argc == 1){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}
	
	int first_file=1;
	for (int i=1; i < argc; i++) {
		if (strcmp(argv[i],"-") == 0){ //handles /dev/stdin or /dev/stdout
			if (first_file == 1){
				files.push_back("/dev/stdin");
			}else if (first_file == 2){
				files.push_back("/dev/stdout");
			}else{
				usage("Warning: You can use the value for the input or output file only");
			}
			first_file++;
		}else if(argv[i][0] != '-'){
			files.push_back(argv[i]);
			first_file++;
		}
	}
	
	
	GetParams(&argc, &argv, (char*) NULL);
	
	if (help){
		usage();
		exit_error(IRSTLM_NO_ERROR);
	}
	
	if (files.size() > 2) {
    usage();
		exit_error(IRSTLM_ERROR_DATA,"Too many arguments");
  }

  if (files.size() < 1) {
    usage();
		exit_error(IRSTLM_ERROR_DATA,"Specify a LM file to read from");
  }

  memset(thr, 0, sizeof(thr));
  if(spthr != NULL) s2t(spthr, thr);
  std::string infile = files[0];
  std::string outfile= "";
	
  if (files.size() == 1) {
    outfile=infile;

    //remove path information
    std::string::size_type p = outfile.rfind('/');
    if (p != std::string::npos && ((p+1) < outfile.size()))
      outfile.erase(0,p+1);

    //eventually strip .gz
    if (outfile.compare(outfile.size()-3,3,".gz")==0)
      outfile.erase(outfile.size()-3,3);

    outfile+=".plm";
  } else
    outfile = files[1];
	
  lmtable lmt;
  inputfilestream inp(infile.c_str());
  if (!inp.good()) {
		std::stringstream ss_msg;
		ss_msg << "Failed to open " << infile;
		exit_error(IRSTLM_ERROR_IO, ss_msg.str());
  }

  lmt.load(inp,infile.c_str(),outfile.c_str(),0);
  std::cerr << "pruning LM with thresholds: \n";

  for (int i=1; i<lmt.maxlevel(); i++) std::cerr<< " " << thr[i];
  std::cerr << "\n";
  lmt.wdprune((float*)thr, aflag);
  lmt.savetxt(outfile.c_str());
	
	exit_error(IRSTLM_NO_ERROR);
}

