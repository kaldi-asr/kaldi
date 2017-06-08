// $Id: dict.cpp 3677 2010-10-13 09:06:51Z bertoldi $


#include <iostream>
#include "cmd.h"
#include "util.h"
#include "mfstream.h"
#include "mempool.h"
#include "dictionary.h"

using namespace std;

	
void print_help(int TypeFlag=0){
	std::cerr << std::endl << "dict - extracts a dictionary" << std::endl;
	std::cerr << std::endl << "USAGE:"  << std::endl;
	std::cerr << "       dict -i=<inputfile> [options]" << std::endl;
	std::cerr << std::endl << "DESCRIPTION:" << std::endl;
	std::cerr << "       dict extracts a dictionary from a corpus or a dictionary." << std::endl;
	std::cerr << std::endl << "OPTIONS:" << std::endl;
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
  char *inp=NULL;
  char *out=NULL;
  char *testfile=NULL;
  char *intsymb=NULL;  //must be single characters
  int freqflag=0;      //print frequency of words
  int sortflag=0;      //sort dictionary by frequency
  int curveflag=0;     //plot dictionary growth curve
  int curvesize=10;    //size of curve
  int listflag=0;      //print oov words in test file
  int size=1000000;    //initial size of table ....
  float load_factor=0;   //initial load factor, default LOAD_FACTOR
	
  int prunefreq=0;    //pruning according to freq value
  int prunerank=0;    //pruning according to freq rank
	
	bool help=false;
  
	DeclareParams((char*)
                "InputFile", CMDSTRINGTYPE|CMDMSG, &inp, "input file (Mandatory)",
                "i", CMDSTRINGTYPE|CMDMSG, &inp, "input file (Mandatory)",
                "OutputFile", CMDSTRINGTYPE|CMDMSG, &out, "output file",
                "o", CMDSTRINGTYPE|CMDMSG, &out, "output file",
                "f", CMDBOOLTYPE|CMDMSG, &freqflag,"output word frequencies; default is false",
                "Freq", CMDBOOLTYPE|CMDMSG, &freqflag,"output word frequencies; default is false",
                "sort", CMDBOOLTYPE|CMDMSG, &sortflag,"sort dictionary by frequency; default is false",
                "Size", CMDINTTYPE|CMDMSG, &size, "Initial dictionary size; default is 1000000",
                "s", CMDINTTYPE|CMDMSG, &size, "Initial dictionary size; default is 1000000",
                "LoadFactor", CMDFLOATTYPE|CMDMSG, &load_factor, "set the load factor for cache; it should be a positive real value; default is 0",
                "lf", CMDFLOATTYPE|CMDMSG, &load_factor, "set the load factor for cache; it should be a positive real value; default is 0",
                "IntSymb", CMDSTRINGTYPE|CMDMSG, &intsymb, "interruption symbol",
                "is", CMDSTRINGTYPE|CMDMSG, &intsymb, "interruption symbol",
								
                "PruneFreq", CMDINTTYPE|CMDMSG, &prunefreq, "prune words with frequency below the specified value",
                "pf", CMDINTTYPE|CMDMSG, &prunefreq, "prune words with frequency below the specified value",
                "PruneRank", CMDINTTYPE|CMDMSG, &prunerank, "prune words with frequency rank above the specified value",
                "pr", CMDINTTYPE|CMDMSG, &prunerank, "prune words with frequency rank above the specified value",
								
                "Curve", CMDBOOLTYPE|CMDMSG, &curveflag,"show dictionary growth curve; default is false",
                "c", CMDBOOLTYPE|CMDMSG, &curveflag,"show dictionary growth curve; default is false",
                "CurveSize", CMDINTTYPE|CMDMSG, &curvesize, "default 10",
                "cs", CMDINTTYPE|CMDMSG, &curvesize, "default 10",
								
                "TestFile", CMDSTRINGTYPE|CMDMSG, &testfile, "compute OOV rates on the specified test corpus",
                "t", CMDSTRINGTYPE|CMDMSG, &testfile, "compute OOV rates on the specified test corpus",
                "ListOOV", CMDBOOLTYPE|CMDMSG, &listflag, "print OOV words to stderr; default is false",
                "oov", CMDBOOLTYPE|CMDMSG, &listflag, "print OOV words to stderr; default is false",
								
								"Help", CMDBOOLTYPE|CMDMSG, &help, "print this help",
								"h", CMDBOOLTYPE|CMDMSG, &help, "print this help",
								
                (char*)NULL
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
	
  if (inp==NULL) {
		usage();
		exit_error(IRSTLM_NO_ERROR, "Warning: no input file specified");
  };
	
  // options compatibility issues:
  if (curveflag && !freqflag)
    freqflag=1;
  if (testfile!=NULL && !freqflag) {
    freqflag=1;
    mfstream test(testfile,ios::in);
    if (!test) {
      usage();
			std::string msg("Warning: cannot open testfile: ");
			msg.append(testfile);
      exit_error(IRSTLM_NO_ERROR, msg);
    }
    test.close();
		
  }
	
  //create dictionary: generating it from training corpus, or loading it from a dictionary file
  dictionary *d = new dictionary(inp,size,load_factor);
	
  // sort dictionary
  if (prunefreq>0 || prunerank>0 || sortflag) {
		dictionary *sortd=new dictionary(d,false); 
		sortd->sort();
		delete d;
    d=sortd;
  }
	
	
  // show statistics on dictionary growth and OOV rates on test corpus
  if (testfile != NULL)
    d->print_curve_oov(curvesize, testfile, listflag);
  if (curveflag)
    d->print_curve_growth(curvesize);
	
	
  //prune words according to frequency and rank
  if (prunefreq>0 || prunerank>0) {
    cerr << "pruning dictionary prunefreq:" << prunefreq << " prunerank: " << prunerank <<" \n";
    int count=0;
    int bos=d->encode(d->BoS());
    int eos=d->encode(d->EoS());
		
    for (int i=0; i< d->size() ; i++) {
      if (prunefreq && d->freq(i) <= prunefreq && i!=bos && i!=eos) {
        d->freq(i,0);
        continue;
      }
      if (prunerank>0 && count>=prunerank && i!=bos && i!=eos) {
        d->freq(i,0);
        continue;
      }
      count++;
    }
  }
  // if outputfile is provided, write the dictionary into it
  if(out!=NULL) d->save(out,freqflag);
	
}

