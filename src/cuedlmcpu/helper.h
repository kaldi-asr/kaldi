#ifndef __HEAD_HELPER__
#define __HEAD_HELPER__
#include "head.h"

void printusage(char *str);

class arguments
{
protected:
    int argc;
    char **argv;
    map <string, string> argmap;
public:
    arguments(int n, char **v): argc(n-1), argv(v+1)
    {
        int i;
        string str;
        i = 0;
        while (i < argc)
        {
            if (argv[i][0] == '-')
            {
                str = argv[i];
                if (i<argc-1 && argv[i+1][0] != '-')
                {
                    argmap[str] = argv[i+1];
                    i += 2;
                }
                else
                {
                    argmap[str] = "true";
                    i += 1;
                }
            }
        }
    }
    bool empty ()
    {
        return (argc == 0);
    }
    string find (string str)
    {
        if (argmap.find(str) == argmap.end())
        {
            return string ("EMPTY");
        }
        return argmap[str];
    }
};


bool isEmpty(string str);

int string2int (string str);

float string2float (string str);

void parseArray (string str, vector<int> &layersizes);

float randomv(float min, float max);

int getline (char *line, int &max_words_line, FILE *&fptr);

float logadd (float x, float y);
#endif
