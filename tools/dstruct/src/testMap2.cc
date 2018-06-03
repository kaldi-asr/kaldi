//
// Interactive testing for Map2 datastructure
//
// Copyright (c) 1999-2010 SRI International.  All Rights Reserved.
//
// $Header: /home/srilm/CVS/srilm/dstruct/src/testMap2.cc,v 1.5 2014-05-25 20:01:43 stolcke Exp $
//

#include <stdio.h>
#include <string.h>
#include <tcl.h>

//#define USE_SARRAY_MAP2

#include "Map2.cc"

#define KEY1_T int
#define KEY2_T const char *
#define DATA_T char *
#define MAP2_T Map2<KEY1_T,KEY2_T,DATA_T>
#define ITER_T Map2Iter<KEY1_T,KEY2_T,DATA_T>
#define ITER2_T Map2Iter2<KEY1_T,KEY2_T,DATA_T>

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_MAP2(KEY1_T,KEY2_T,DATA_T);
#endif

static MAP2_T myMap2;

int compKeys(KEY1_T key1, KEY1_T key2)
{
	return key2 - key1;
}

KEY1_T
get_key1 (Tcl_Interp *interp, char **argv)
{
	KEY1_T result = -999999;
	if (argv[0] != NULL) {
	       Tcl_GetInt(interp, argv[0], &result);
	}
	return result;
}

KEY2_T
get_key2 (Tcl_Interp *interp, char **argv)
{
	if (argv[0] != NULL && argv[1] != NULL) {
		return argv[1];
	} else {
		return "NOARG";
	}
}

/*ARGSUSED*/
int
Find(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   DATA_T *result;

   if (argc < 2) {
	Tcl_SetResult(interp, (char *)"2 args expected", TCL_STATIC);
	return TCL_ERROR;
   }

   KEY1_T key1 = get_key1(interp, argv + 1);
   KEY2_T key2 = get_key2(interp, argv + 1);

   result = myMap2.find(key1, key2);
   if (result) {
       Tcl_SetResult(interp, *result, TCL_STATIC);
    }
   return TCL_OK;
}

/*ARGSUSED*/
int
Insert(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   DATA_T *result;

   if (argc < 3) {
	Tcl_SetResult(interp, (char *)"3 args expected", TCL_STATIC);
	return TCL_ERROR;
   }
   char *value = strdup(argv[1]);

   KEY1_T key1 = get_key1(interp, argv + 2);
   KEY2_T key2 = get_key2(interp, argv + 2);

   result = myMap2.insert(key1, key2);
   if (result) {
       Tcl_SetResult(interp, *result, TCL_DYNAMIC);
       *result = value;
   }
   myMap2.dump();
   return TCL_OK;
}

/*ARGSUSED*/
int
Delete(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   DATA_T result;
   Boolean found;

   if (argc < 2) {
	Tcl_SetResult(interp, (char *)"2 args expected", TCL_STATIC);
	return TCL_ERROR;
   }

   KEY1_T key1 = get_key1(interp, argv + 1);
   KEY2_T key2 = get_key2(interp, argv + 1);

   found = myMap2.remove(key1, key2, &result);
   if (found) {
       Tcl_SetResult(interp, result, TCL_DYNAMIC);
   }
   return TCL_OK;
}

/*ARGSUSED*/
int
DeleteRow(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   Boolean result;

   if (argc < 1) {
	Tcl_SetResult(interp, (char *)"1 arg expected", TCL_STATIC);
	return TCL_ERROR;
   }

   KEY1_T key1 = get_key1(interp, argv + 1);

   result = myMap2.remove(key1);
   if (result) {
       Tcl_SetResult(interp, (char *)"FOUND", TCL_STATIC);
   }
   return TCL_OK;
}

/*ARGSUSED*/
int
List(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   MAP2_T *result;
   KEY1_T key1;
   ITER_T myIter(myMap2, compKeys);

   while (myIter.next(key1)) {
	ITER2_T myIter2(myMap2, key1);
	KEY2_T key2;
	DATA_T *data;

	while ((data = myIter2.next(key2))) {
	    cout << "key = " << key1 << "," << key2 
		    << ", value = " << *data << endl;
	}
   }
   MemStats stats;
   myMap2.memStats(stats);
   stats.print();

   return TCL_OK;
}

/*ARGSUSED*/
int
Quit(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   exit(0);
   return TCL_OK;
}

extern "C" int
Tcl_AppInit(Tcl_Interp *interp)
{
   Tcl_CreateCommand(interp, "find", (Tcl_CmdProc *)Find, 0, NULL);
   Tcl_CreateCommand(interp, "insert", (Tcl_CmdProc *)Insert, 0, NULL);
   Tcl_CreateCommand(interp, "delete", (Tcl_CmdProc *)Delete, 0, NULL);
   Tcl_CreateCommand(interp, "delrow", (Tcl_CmdProc *)DeleteRow, 0, NULL);
   Tcl_CreateCommand(interp, "list", (Tcl_CmdProc *)List, 0, NULL);
   Tcl_CreateCommand(interp, "quit", (Tcl_CmdProc *)Quit, 0, NULL);

   return 0;
}


