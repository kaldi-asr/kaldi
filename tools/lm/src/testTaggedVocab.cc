//
// Interactive testing for Vocab class
//

#ifndef lint
static char Copyright[] = "Copyright (c) 1995-2010 SRI International.  All Rights Reserved.";
static char RcsId[] = "@(#)$Header: /home/srilm/CVS/srilm/lm/src/testTaggedVocab.cc,v 1.5 2010/06/02 05:49:58 stolcke Exp $";
#endif

#include <tcl.h>

#include "File.h"
#include "TaggedVocab.h"

static TaggedVocab myVocab(100);

/*ARGSUSED*/
int
VocabAddWord(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   char buffer[100];

   if (argc != 2) {
	Tcl_SetResult(interp, (char *)"word expected", TCL_STATIC);
	return TCL_ERROR;
   }

   VocabIndex result = myVocab.addWord(argv[1]);
   if (result == Vocab_None) {
       sprintf(buffer, "%d", result);
   } else {
       sprintf(buffer, "%d/%d", myVocab.unTag(result), myVocab.getTag(result));
   }
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

   VocabIndex result = myVocab.getIndex(argv[1]);
   if (result == Vocab_None) {
       sprintf(buffer, "%d", result);
   } else {
       sprintf(buffer, "%d/%d", myVocab.unTag(result), myVocab.getTag(result));
   }
   Tcl_SetResult(interp, buffer, TCL_VOLATILE);
   return TCL_OK;
}

/*ARGSUSED*/
int
VocabGetIndex(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   VocabIndex index, tag;

   if (argc < 2) {
	Tcl_SetResult(interp, (char *)"index expected", TCL_STATIC);
	return TCL_ERROR;
   }

   if (Tcl_GetInt(interp, argv[1], (int *)&index) != TCL_OK) {
	return TCL_ERROR;
   }
   if (argc > 2) {
       if (Tcl_GetInt(interp, argv[2], (int *)&tag) != TCL_OK) {
	    return TCL_ERROR;
       }
   } else {
       tag = Tag_None;
   }
   VocabIndex tagged = myVocab.tagWord(index, tag);
   Tcl_SetResult(interp, (char *)(myVocab.getWord(tagged)), TCL_STATIC);
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
VocabList(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   VocabIndex index;
   VocabString word, tag;
   VocabIter iter(myVocab);
   VocabIter titer(myVocab.tags());

   if (argc != 1) {
	Tcl_SetResult(interp, (char *)"no args expected", TCL_STATIC);
	return TCL_ERROR;
   }

   printf("%d words\n", myVocab.numWords());
   while ((word = iter.next(index))) {
	printf("index = %d, word = %s\n", index, word);
   }
   printf("%d tags\n", myVocab.tags().numWords());
   while ((tag = titer.next(index))) {
	printf("index = %d, tag = %s\n", index, tag);
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
   Tcl_CreateCommand(interp, "list", (Tcl_CmdProc *)VocabList, 0, NULL);
   Tcl_CreateCommand(interp, "read", (Tcl_CmdProc *)VocabRead, 0, NULL);
   Tcl_CreateCommand(interp, "write", (Tcl_CmdProc *)VocabWrite, 0, NULL);
   return 0;
}


