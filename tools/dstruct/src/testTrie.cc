//
// Interactive testing for Trie datastructure
//
// Copyright (c) 1995-2010 SRI International.  All Rights Reserved.
//
// $Header: /home/srilm/CVS/srilm/dstruct/src/testTrie.cc,v 1.15 2013/03/08 14:31:00 stolcke Exp $
//

#include <stdio.h>
#include <string.h>
#include <tcl.h>

//#define USE_SARRAY_TRIE

#include "Trie.cc"

#define KEY_T int
#define DATA_T char *
#define TRIE_T Trie<KEY_T,DATA_T>
#define ITER_T TrieIter<KEY_T,DATA_T>
#define ITER2_T TrieIter2<KEY_T,DATA_T>

#ifdef INSTANTIATE_TEMPLATES
INSTANTIATE_TRIE(KEY_T,DATA_T);
#endif

static TRIE_T myTrie;

int compKeys(KEY_T key1, KEY_T key2)
{
	return key2 - key1;
}

int *
get_keys (Tcl_Interp *interp, char **argv)
{
	static KEY_T keys[100];
	int i = 0;

	while (argv[i] != NULL) {
		if (Tcl_GetInt(interp, argv[i], &keys[i]) == TCL_ERROR) {
			return NULL;
		}
		i++;
	}
	Map_noKey(keys[i]);
	return keys;
}

/*ARGSUSED*/
int
Find(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   DATA_T *result;
   KEY_T *keys;

   keys = get_keys(interp, argv + 1);
   if (!keys) {
	return TCL_ERROR;
   }

   result = myTrie.find(keys);
   if (result) {
       Tcl_SetResult(interp, *result, TCL_STATIC);
    }
   return TCL_OK;
}

/*ARGSUSED*/
int
Prefix(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   DATA_T *result;
   KEY_T *keys;
   unsigned depth;

   keys = get_keys(interp, argv + 1);
   if (!keys) {
	return TCL_ERROR;
   }

   result = myTrie.findPrefix(keys, depth);

   cerr << "depth = " << depth << endl;
   
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
   KEY_T *keys;
   char *value;

   if (argc < 2) {
	Tcl_SetResult(interp, (char *)"word expected", TCL_STATIC);
	return TCL_ERROR;
   }

   value = strdup(argv[1]);
   keys = get_keys(interp, argv + 2);
   if (!keys) {
	return TCL_ERROR;
   }

   Boolean foundp;
   result = myTrie.insert(keys, foundp);
   if (result) {
       Tcl_SetResult(interp, *result, TCL_DYNAMIC);
       cerr << "foundp = " << foundp << endl;
       *result = value;
   }
   myTrie.dump();
   return TCL_OK;
}

/*ARGSUSED*/
int
Delete(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   DATA_T *result;
   KEY_T *keys;

   keys = get_keys(interp, argv + 1);
   if (!keys) {
	return TCL_ERROR;
   }

   if (myTrie.remove(keys, result)) {
       Tcl_SetResult(interp, *result, TCL_DYNAMIC);
   }
   return TCL_OK;
}

/*ARGSUSED*/
int
List(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   TRIE_T *result;
   KEY_T *keys;

   keys = get_keys(interp, argv + 1);
   if (!keys) {
	return TCL_ERROR;
   }

   if ((result = myTrie.findTrie(keys))) {
	ITER_T iter(*result, compKeys);
	KEY_T key;

	while ((result = iter.next(key))) {
	    if (result->value()) {
		cout << "key = " << key << ", value = " << result->value() << endl;
		//cout << "find = " << *(result->data()) << endl;
	    }
	}
   }
   MemStats stats;
   myTrie.memStats(stats);
   stats.print();

   return TCL_OK;
}

/*ARGSUSED*/
int
RList(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   int level;
   TRIE_T *result;
   KEY_T *keys;

   if (argc < 2) {
	Tcl_SetResult(interp, (char *)"level expected", TCL_STATIC);
	return TCL_ERROR;
   }
   if (Tcl_GetInt(interp, argv[1], &level) == TCL_ERROR) {
	return TCL_ERROR;
   }

   keys = get_keys(interp, argv + 2);
   if (!keys) {
	return TCL_ERROR;
   }

   if ((result = myTrie.findTrie(keys))) {
	KEY_T keys2[100];
	ITER2_T iter(*result, keys2, level, compKeys);

	while ((result = iter.next())) {
	    if (result->value()) {
		cout << "key =" ;
		for (int i = 0; !Map_noKeyP(keys2[i]); i++) {
		    cout << " " << keys2[i];
		}
		cout << ", value = " << result->value() << endl;
	    }
	}
   }

   return TCL_OK;
}

/*ARGSUSED*/
int
Quit(ClientData cd, Tcl_Interp *interp, int argc, char **argv)
{
   TRIE_T *anothertrie = new TRIE_T;

   /*
    * Test = operator
    */
   cerr << "*** testing = operator ***\n";
   *anothertrie = myTrie;
   anothertrie->dump();

   delete anothertrie;
   exit(0);

   return TCL_OK;
}

extern "C" int
Tcl_AppInit(Tcl_Interp *interp)
{
   Tcl_CreateCommand(interp, "find", (Tcl_CmdProc *)Find, 0, NULL);
   Tcl_CreateCommand(interp, "prefix", (Tcl_CmdProc *)Prefix, 0, NULL);
   Tcl_CreateCommand(interp, "insert", (Tcl_CmdProc *)Insert, 0, NULL);
   Tcl_CreateCommand(interp, "delete", (Tcl_CmdProc *)Delete, 0, NULL);
   Tcl_CreateCommand(interp, "list", (Tcl_CmdProc *)List, 0, NULL);
   Tcl_CreateCommand(interp, "rlist", (Tcl_CmdProc *)RList, 0, NULL);
   Tcl_CreateCommand(interp, "quit", (Tcl_CmdProc *)Quit, 0, NULL);

   return 0;
}

