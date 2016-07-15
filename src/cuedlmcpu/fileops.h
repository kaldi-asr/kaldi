#ifndef __FILEOPS_H__
#define __FILEOPS_H__
#include "head.h"
#include "Mathops.h"

class FILEPTR
{
protected:
    FILE *fptr;
    int i;
    string filename;
public:
    FILEPTR()
    {
        fptr = NULL;
    }
    ~FILEPTR()
    {
        if (fptr)       fclose(fptr);
        fptr = NULL;
    }
    void open (string fn)
    {
        filename = fn;
        fptr = fopen (filename.c_str(), "rt");
        if (fptr == NULL)
        {
            printf ("ERROR: Failed to open file: %s\n", filename.c_str());
            exit (0);
        }
    }
    void close()
    {
        if (fptr)
        {
            fclose(fptr);
            fptr = NULL;
        }
    }
    bool eof()
    {
        return feof(fptr);
    }
    int readint ()
    {
        if (!feof(fptr))
        {
            if(fscanf (fptr, "%d", &i) != 1)
            {
                if (!feof(fptr))
                {
                    printf ("Warning: failed to read feature index from text file (%s)\n", filename.c_str());
                }
            }
            return i;
        }
        else
        {
            return INVALID_CUED_INT;
        }
    }
    void readline (vector<string> &linevec, int &cnt)
    {
        linevec.clear();
        char word[1024];
        char c;
        int index=0;
        cnt = 0;
        while (!feof(fptr))
        {
            c = fgetc(fptr);
            // getvalidchar (fptr, c);
            if (c == '\n')
            {
                if (cnt==0 && word[0] != '<')
                {
                    linevec.push_back("<s>");
                    cnt ++;
                }
                if (index > 0)
                {
                    word[index] = 0;
                    linevec.push_back(word);
                    cnt ++;
                }
                break;
            }
            else if ((c == ' ' || c == '\t') && index == 0) // space in the front of line
            {
                continue;
            }
            else if ((c == ' ' || c=='\t') && index > 0) // space in the middle of line
            {
                word[index] = 0;
                if (cnt==0 && word[0] != '<')
                {
                    linevec.push_back("<s>");
                    cnt ++;
                }
                linevec.push_back(word);
                index = 0;
                cnt ++;
            }
            else
            {
                word[index] = c;
                index ++;
            }
        }
        if (cnt>0 && word[0] != '<')
        {
            linevec.push_back("</s>");
            cnt ++;
        }
    }
};
#endif
