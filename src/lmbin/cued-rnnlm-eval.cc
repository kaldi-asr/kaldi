#include "lm/cued-rnnlm-lib.h"

using namespace std;
using namespace cued_rnnlm;

int main (int argc, char **argv)
{
    string str;
    string validfile, testfile, feafile, inputwlist, outputwlist,
           nglmstfile, inmodelname;
    int    debug, fullvocsize, nthread;
    vector<int> layersizes;
    float lambda, lognormconst, lmscale, ip;
    bool binformat, flag_feature=false;
	if (argc < 2)
	{
		printusage (argv[0]);
        return SUCCESS;
	}
	arguments arg (argc, argv);
	if (!arg.empty())
    {
        validfile = arg.find("-validfile");
        testfile = arg.find("-testfile");

        feafile = arg.find("-feafile");
        if (!isEmpty(feafile))                  flag_feature = true;
        else                                    flag_feature = false;

        inputwlist = arg.find("-inputwlist");
        outputwlist = arg.find("-outputwlist");

        str = arg.find("-lognormconst");
        if (!isEmpty(str))                      lognormconst = string2float(str);
        else                                    lognormconst = -1.0;            // default lognormconst value: -1.0;

        str = arg.find("-lambda");
        if (!isEmpty(str))                      lambda = string2float(str);
        else                                    lambda = 0.5;

        str = arg.find("-debug");
        if (!isEmpty(str))                      debug = string2int(str);
        else                                    debug = 1;          // default debug value: 1

        str = arg.find("-nthread");
        if (!isEmpty(str))                      nthread = string2int(str);
        else                                    nthread = 1;          // default nthread value: 1

        inmodelname  = arg.find("-readmodel");

        str = arg.find("-fullvocsize");
        if (!isEmpty(str))                      fullvocsize = string2int(str);
        else                                    fullvocsize = 0;

        str = arg.find("-lmscale");
        if (!isEmpty(str))                      lmscale = string2float(str);
        else                                    lmscale = 12.0;

        str = arg.find("-ip");
        if (!isEmpty(str))                      ip = string2float(str);
        else                                    ip = 0.0;

        str = arg.find("-binformat");
        if (!isEmpty(str))                      binformat = true;
        else                                    binformat = false;  // default model format: TEXT

        str = arg.find("-nglmstfile");
        if (!isEmpty(str))                      nglmstfile = str;
        else                                    lambda = 1.0;

        if (debug > 1)
        {
            for (int i=0; i<argc; i++) printf ("%s ", argv[i]);
            printf ("\n");
        }

        if (!isEmpty(arg.find ("-ppl")))
        {
            RNNLM rnnlm (inmodelname, inputwlist, outputwlist, layersizes, fullvocsize, binformat, debug);
            rnnlm.setLognormConst (lognormconst); // TODO(hxu) figure out what this does
            rnnlm.setNthread(nthread);
//            if (flag_feature)
//            {
//                rnnlm.setFeafile (feafile);
//                rnnlm.ReadFeaFile (feafile);
//            }
            rnnlm.calppl(testfile, lambda, nglmstfile);
        }
        else if (!isEmpty(arg.find ("-nbest")))
        {
            RNNLM rnnlm (inmodelname, inputwlist, outputwlist, layersizes,  fullvocsize, binformat, debug);
            rnnlm.setLognormConst (lognormconst);
            rnnlm.setNthread (nthread);
            rnnlm.setLmscale (lmscale);
            rnnlm.setIp (ip);
//            if (flag_feature)
//            {
//                rnnlm.setFeafile (feafile);
//                rnnlm.ReadFeaFile (feafile);
//            }

            rnnlm.calnbest(testfile, lambda, nglmstfile);
        }
        else
        {
            printusage(argv[0]);
        }
    }
    else
    {
        printusage(argv[0]);
    }
}
