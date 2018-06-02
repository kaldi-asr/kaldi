//
// Interactive testing for Vocab class
//

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testVocab.cc,v 1.10 2010/06/02 05:49:58 stolcke Exp $";
#endif

#include <tcl.h>

#include "File.h"
#include "Vocab.h"

static Vocab myVocab(100);

/*ARGSUSED*/
int
VocabAddWord(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   char buffer[100];

   if (argc != 2) {
	Tcl_SetResult(interp, (char *)"word expected", TCL_STATIC);
	return TCL_ERROR;
   }

   sprintf(buffer, "%d", myVocab.addWord(argv[1]));
   Tcl_SetResult(interp, buffer, TCL_VOLATILE);
   return TCL_OK;
}

/*ARGSUSED*/
int
VocabGetWord(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   char buffer[100];

   if (argc != 2) {
	Tcl_SetResult(interp, (char *)"word expected", TCL_STATIC);
	return TCL_ERROR;
   }

   sprintf(buffer, "%d", myVocab.getIndex(argv[1]));
   Tcl_SetResult(interp, buffer, TCL_VOLATILE);
   return TCL_OK;
}

/*ARGSUSED*/
int
VocabGetIndex(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   VocabIndex index;

   if (argc != 2) {
	Tcl_SetResult(interp, (char *)"index expected", TCL_STATIC);
	return TCL_ERROR;
   }

   if (Tcl_GetInt(interp, argv[1], (int *)&index) != TCL_OK) {
	return TCL_ERROR;
   }
   Tcl_SetResult(interp, (char *)(myVocab.getWord(index)), TCL_STATIC);
   return TCL_OK;
}

/*ARGSUSED*/
int
VocabDeleteWord(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   if (argc != 2) {
	Tcl_SetResult(interp, (char *)"word expected", TCL_STATIC);
	return TCL_ERROR;
   }

   myVocab.remove(argv[1]);
   return TCL_OK;
}

/*ARGSUSED*/
int
VocabDeleteIndex(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   VocabIndex index;

   if (argc != 2) {
	Tcl_SetResult(interp, (char *)"index expected", TCL_STATIC);
	return TCL_ERROR;
   }

   if (Tcl_GetInt(interp, argv[1], (int *)&index) != TCL_OK) {
	return TCL_ERROR;
   }
   myVocab.remove(index);
   return TCL_OK;
}

/*ARGSUSED*/
int
VocabDeleteAll(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   if (argc != 1) {
	Tcl_SetResult(interp, (char *)"no args expected", TCL_STATIC);
	return TCL_ERROR;
   }

   VocabIter iter(myVocab);
   VocabIndex index;
   VocabString word;

   unsigned howmany = 0;
   while ((word = iter.next(index))) {
       myVocab.remove(word);
       howmany ++;
   }
   cerr << howmany << " removed\n";
   return TCL_OK;
}

/*ARGSUSED*/
int
VocabList(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   VocabIndex index;
   VocabString word;
   VocabIter iter(myVocab);

   if (argc != 1) {
	Tcl_SetResult(interp, (char *)"no args expected", TCL_STATIC);
	return TCL_ERROR;
   }

   printf("%u words\n", myVocab.numWords());
   while ((word = iter.next(index))) {
	printf("index = %d, word = %s\n", index, word);
   }
   return TCL_OK;
}

/*ARGSUSED*/
int
VocabRead(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   if (argc != 2) {
	Tcl_SetResult(interp, (char *)"filename expected", TCL_STATIC);
	return TCL_ERROR;
   }

   File file(argv[1], "r", 0);
   if (file.error()) {
	Tcl_SetResult(interp, (char *)"fopen failed", TCL_STATIC);
	return TCL_ERROR;
   }

   myVocab.read(file);
   return TCL_OK;
}

/*ARGSUSED*/
int
VocabWrite(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   if (argc != 2) {
	Tcl_SetResult(interp, (char *)"filename expected", TCL_STATIC);
	return TCL_ERROR;
   }

   File file(argv[1], "w", 0);
   if (file.error()) {
	Tcl_SetResult(interp, (char *)"fopen failed", TCL_STATIC);
	return TCL_ERROR;
   }

   myVocab.write(file);
   return TCL_OK;
}


extern "C" int
Tcl_AppInit(Tcl_Interp *interp)
{
   Tcl_CreateCommand(interp, "add", (Tcl_CmdProc *)VocabAddWord, 0, NULL);
   Tcl_CreateCommand(interp, "get", (Tcl_CmdProc *)VocabGetWord, 0, NULL);
   Tcl_CreateCommand(interp, "getindex", (Tcl_CmdProc *)VocabGetIndex, 0, NULL);
   Tcl_CreateCommand(interp, "delete", (Tcl_CmdProc *)VocabDeleteWord, 0, NULL);
   Tcl_CreateCommand(interp, "deleteindex", (Tcl_CmdProc *)VocabDeleteIndex, 0, NULL);
   Tcl_CreateCommand(interp, "deleteall", (Tcl_CmdProc *)VocabDeleteAll, 0, NULL);
   Tcl_CreateCommand(interp, "list", (Tcl_CmdProc *)VocabList, 0, NULL);
   Tcl_CreateCommand(interp, "read", (Tcl_CmdProc *)VocabRead, 0, NULL);
   Tcl_CreateCommand(interp, "write", (Tcl_CmdProc *)VocabWrite, 0, NULL);
   return 0;
}


